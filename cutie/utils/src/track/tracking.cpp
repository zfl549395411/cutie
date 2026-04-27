/******************************************************************************
 * Copyright 2017 RoboSense All rights reserved.
 * Suteng Innovation Technology Co., Ltd. www.robosense.ai

 * This software is provided to you directly by RoboSense and might
 * only be used to access RoboSense LiDAR. Any compilation,
 * modification, exploration, reproduction and redistribution are
 * restricted without RoboSense's prior consent.

 * THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESSED OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL ROBOSENSE BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

#include "hyper_vision/perception/occupancy_objects/occupancy_detection/tracking.h"

namespace robosense {
namespace perception {

GridTracking::GridTracking(const YAML::Node &cfg_node)
    : ogm(OccGridMap::GetInstance()),
      rd(RangeDetection::GetInstance()),
      gd(GroundDetection::GetInstance()) {
  gnd_valid_map_.resize(ogm.sub_size_, ogm.sub_size_);
  gnd_info_map_.resize(ogm.sub_size_, ogm.sub_size_);
  nongnd_valid_map_.resize(ogm.sub_size_, ogm.sub_size_);
  nongnd_info_map_.resize(ogm.sub_size_, ogm.sub_size_);
  grid_row_map_.resize(ogm.sub_size_, ogm.sub_size_);
  intensity_map_.resize(ogm.sub_size_, ogm.sub_size_);
}

void GridTracking::track(const PolygonDivide::Ptr &polygon_ptr) {
  updateLabel();
  calcGauss();
  bfsDynamicObject(polygon_ptr);
  clearUnobservedGrid();
  updateObservedGrid();
  // ogm.saveMap(ogm.type_sub_map_, "/apollo/data/occupancy/type_map/");
}

void GridTracking::updateLabel() {
  for (const auto &pi : cloud_ptr->roi_idx) {
    auto &info = cloud_ptr->infos[pi];
    auto &label = info.label;
    if (label == kGroundPt || label == kDynamicPt) {
      continue;
    }
    const auto &sub_type = ogm.subType(info.sub_x, info.sub_y);
    if (sub_type == kBarrierGrid) {
      label = kBarrierPt;
    } else if (sub_type == kMistGrid) {
      label = kMistPt;
    } else if (sub_type == kBloomingGrid && label != kHighRefPt) {
      label = kBloomingPt;
    }
  }
}

void GridTracking::calcGauss() {
  gnd_valid_map_.setZero();
  nongnd_valid_map_.setZero();
  grid_row_map_.setZero();
  intensity_map_.setZero();
  const auto &track_cnt_ogm = ogm.at(kTrackCntMap);
  auto &det_frame_ogm = ogm.at(kDetFrameMap);
  for (const auto &pi : cloud_ptr->roi_idx) {
    // 过滤噪点
    const auto &info = cloud_ptr->infos[pi];
    const auto &label = info.label;
    if (label == kBloomingPt || label == kMistPt) {
      continue;
    }
    // 过滤地面点
    Eigen::Vector3d grid_pt = cloud_ptr->grid_pt(pi);
    grid_pt.x() -= ogm.gFloor(grid_pt.x());
    grid_pt.y() -= ogm.gFloor(grid_pt.y());
    int sub_x{info.sub_x}, sub_y{info.sub_y};
    if (label == kGroundPt) {
      if (!gnd_valid_map_(sub_x, sub_y)) {
        gnd_valid_map_(sub_x, sub_y) = 1;
        gnd_info_map_(sub_x, sub_y).init(grid_pt);
      } else {
        gnd_info_map_(sub_x, sub_y).update(grid_pt);
      }
      continue;
    }
    // 近距离减小高度阈值
    const auto &ego_pt = cloud_ptr->ego_pt(pi);
    const auto &dis = cloud_ptr->depth(pi);
    double max_h_th = ogm.ego_max_h_;
    int map_x{info.map_x}, map_y{info.map_y};
    if (ogm.near_dist_ > 15 && dis < ogm.near_dist_ &&
        std::abs(ego_pt.y()) < 5 &&
        track_cnt_ogm(map_x, map_y) < ogm.pub_track_cnt_) {
      max_h_th = std::max(1.0, ogm.ego_max_h_ * dis / ogm.near_dist_);
    }
    // 过滤高度
    double gnd_height = gd.getGndHeight(ego_pt.x(), ego_pt.y());
    if (ego_pt.z() - gnd_height > max_h_th) {
      continue;
    }
    // 静目标栅格
    auto &sub_type = ogm.subType(sub_x, sub_y);
    if (sub_type == kUnoccupiedGrid || sub_type == kBloomingGrid) {
      sub_type = kStaticGrid;
    }
    // 累帧
    if (nongnd_valid_map_(sub_x, sub_y) == 0) {
      nongnd_valid_map_(sub_x, sub_y) = 1;
      nongnd_info_map_(sub_x, sub_y).init(grid_pt);
      det_frame_ogm(map_x, map_y) = cloud_ptr->frame_num;
    } else {
      nongnd_info_map_(sub_x, sub_y).update(grid_pt);
    }
    // 记录栅格最大反射率
    uint8_t pt_intensity = static_cast<uint8_t>(info.intensity);
    auto &grid_intensity = intensity_map_(sub_x, sub_y);
    grid_intensity = std::max(grid_intensity, pt_intensity);
  }
}

GridRow GridTracking::calcRowCount(const int sub_x, const int sub_y) {
  // 根据距离计算两排点阈值
  Eigen::Vector2d grid_pt(ogm.xSubToPos(sub_x), ogm.ySubToPos(sub_y));
  double dis = (grid_pt - ogm.LidarInGrid().head(2)).norm();
  double two_row_th = 1.1 * dis * rd.real_res_;
  double three_row_th = 2 * two_row_th;
  double near_gnd_th = std::max(0.25, three_row_th);
  // 尝试合并邻域栅格地面点
  auto &nongnd_info = nongnd_info_map_(sub_x, sub_y);
  const auto &gnd_info = gnd_info_map_(sub_x, sub_y);
  if (!gnd_valid_map_(sub_x, sub_y)) {
    for (const auto &nb : nbs_) {
      int nb_x = sub_x + nb.x();
      int nb_y = sub_y + nb.y();
      if (!ogm.inSubRange(nb_x, nb_y) || !gnd_valid_map_(nb_x, nb_y)) {
        continue;
      }
      const auto &nb_gnd_info = gnd_info_map_(nb_x, nb_y);
      if (nongnd_info.getDiffZWith(nb_gnd_info) < near_gnd_th) {
        nongnd_info.min_z = std::min(nongnd_info.min_z, nb_gnd_info.min_z);
      }
    }
  } else if (nongnd_info.getDiffZWith(gnd_info) < near_gnd_th) {
    nongnd_info.min_z = std::min(nongnd_info.min_z, gnd_info.min_z);
  }
  // 根据高度差判断排点数
  double height_diff = nongnd_info.getDiffZ();
  if (height_diff < two_row_th) {
    return kOneRowGrid;
  } else if (height_diff < three_row_th || nongnd_info.count < 3) {
    return kTwoRowGrid;
  } else {
    return kMultiRowGrid;
  }
}

void GridTracking::hold(const BlindSpotDetection::Ptr &blind_ptr) {
  auto &track_cnt_ogm = ogm.at(kTrackCntMap);
  auto &grid_type_ogm = ogm.at(kTypeMap);
  ogm.forInSubMap([&](int sub_x, int sub_y, int map_x, int map_y,
                      double grid_px, double grid_py) {
    // 过滤未pub栅格
    auto &track_cnt = track_cnt_ogm(map_x, map_y);
    if (track_cnt < ogm.pub_track_cnt_) {
      return;
    }
    // 过滤累帧异常栅格(解决上下层问题)
    Eigen::Vector3d grid_pt{grid_px, grid_py, 0};
    grid_pt += ogm.gauss(map_x, map_y).getCentroid();
    Eigen::Vector3d ego_pt = ogm.tf_grid2ego_ * grid_pt;
    double gnd_height = gd.getGndHeight(ego_pt.x(), ego_pt.y());
    if (std::abs(ego_pt.z() - gnd_height) > 2.0) {
      track_cnt = 0;
      return;
    }
    // 过滤有观测栅格
    if (nongnd_valid_map_(sub_x, sub_y) || gnd_valid_map_(sub_x, sub_y)) {
      return;
    }
    // 过滤动目标栅格
    if (grid_type_ogm(map_x, map_y) == kDynamicGrid) {
      return;
    }
    // 盲区高度以下为盲区
    grid_pt.z() = ogm.gauss(map_x, map_y).getMaxZ();
    ego_pt = ogm.tf_grid2ego_ * grid_pt;
    if (ego_pt.z() > gnd_height &&
        ego_pt.z() < blind_ptr->getBlindHeight(ego_pt.x(), ego_pt.y())) {
      track_cnt += ogm.subType(sub_x, sub_y) == kUnoccupiedGrid ? 2 : 1;
      return;
    }
  });
}

void GridTracking::bfsDynamicObject(const PolygonDivide::Ptr &polygon_ptr) {
  auto &obj_grids = ogm.obj_grids_;
  auto &connect_info_ogm = ogm.at(kConnectDynamicInfoMap);
  while (!obj_grids.empty()) {
    int queue_size = obj_grids.size();
    for (int j = 0; j < queue_size; ++j) {
      Eigen::Vector2i seed = obj_grids.front();
      obj_grids.pop();
      // 扩散联通动目标
      int track_id = ogm.subObjId(seed.x(), seed.y());
      for (const auto &nb : nbs_) {
        const auto seed_nb = seed + nb;
        if (!ogm.inSubRange(seed_nb.x(), seed_nb.y())) {
          continue;
        }
        int map_x = ogm.xSubToMap(seed_nb.x());
        int map_y = ogm.ySubToMap(seed_nb.y());
        auto &connect_info = connect_info_ogm(map_x, map_y);
        const auto &polygon_id = ogm.at(kTrackIdMap)(map_x, map_y);
        if (connect_info < 0 ||
            ogm.subType(seed_nb.x(), seed_nb.y()) != kStaticGrid ||
            ogm.subCurbType(seed_nb.x(), seed_nb.y()) == kCurbGrid ||
            ogm.subObjId(seed_nb.x(), seed_nb.y()) != -1 ||
            !polygon_ptr->checkDynamicConnect(polygon_id, track_id)) {
          continue;
        }
        // 被不同栅格连接
        if (connect_info > 0 && connect_info != track_id) {
          connect_info = -1;
          continue;
        }
        obj_grids.push(seed_nb);
        ogm.subType(seed_nb.x(), seed_nb.y()) = kBfsDynamicGrid;
        ogm.subObjId(seed_nb.x(), seed_nb.y()) = track_id;
        ogm.staticToDynamic(map_x, map_y);
      }
    }
  }
}

void GridTracking::clearUnobservedGrid() {
  auto &track_cnt_ogm = ogm.at(kTrackCntMap);
  auto &det_frame_ogm = ogm.at(kDetFrameMap);
  auto &row_seq_ogm = ogm.at(kRowSeqMap);
  int curr_frame = cloud_ptr->frame_num;
  ogm.forInRawSubMap([&](int sub_x, int sub_y) {
    // 跳过无累帧栅格
    int map_x = ogm.xSubToMap(sub_x);
    int map_y = ogm.ySubToMap(sub_y);
    auto &track_cnt = track_cnt_ogm(map_x, map_y);
    if (track_cnt == 0) {
      return;
    }
    // 动转静
    ogm.dynamicToStatic(map_x, map_y);
    // 跳过当前帧观测栅格
    if (nongnd_valid_map_(sub_x, sub_y)) {
      return;
    }
    // 计算距上一次有效观测帧的帧数
    auto &det_frame = det_frame_ogm(map_x, map_y);
    int disappear_cnt = (curr_frame - det_frame + INT16_MAX) % INT16_MAX;
    // 更新累帧计数
    det_frame = curr_frame;
    track_cnt = ogm.subType(sub_x, sub_y) == kUnoccupiedGrid
                    ? std::max(0, track_cnt - disappear_cnt - 1)
                    : std::max(0, track_cnt - disappear_cnt);
    if (track_cnt == 0) {
      return;
    }
    // 更新排点数序列
    auto &row_cnt_seq = row_seq_ogm(map_x, map_y);
    while (disappear_cnt-- && row_cnt_seq > 0) {
      row_cnt_seq = (row_cnt_seq << 1) & (0xEE);  // 0b11101110
    }
  });
}

void GridTracking::updateObservedGrid() {
  auto &conf_ogm = ogm.at(kConfMap);
  auto &track_cnt_ogm = ogm.at(kTrackCntMap);
  auto &track_id_ogm = ogm.at(kTrackIdMap);
  auto &grid_type_ogm = ogm.at(kTypeMap);
  auto &det_dist_ogm = ogm.at(kDetDistMap);
  auto &row_seq_ogm = ogm.at(kRowSeqMap);
  ogm.forInRawSubMap([&](int sub_x, int sub_y) {
    if (!nongnd_valid_map_(sub_x, sub_y)) {
      nongnd_info_map_(sub_x, sub_y).reset();
      return;
    }
    // 加载世界栅格信息
    int map_x = ogm.xSubToMap(sub_x);
    int map_y = ogm.ySubToMap(sub_y);
    auto &track_cnt = track_cnt_ogm(map_x, map_y);
    auto &track_id = track_id_ogm(map_x, map_y);
    auto &grid_conf = conf_ogm(map_x, map_y);
    auto &grid_type = grid_type_ogm(map_x, map_y);
    auto &det_dist = det_dist_ogm(map_x, map_y);
    auto &row_cnt_seq = row_seq_ogm(map_x, map_y);

    // 1. 两排点判断(同时尝试合并地面点)
    auto &grid_row = grid_row_map_(sub_x, sub_y);
    grid_row = calcRowCount(sub_x, sub_y);

    // 2. 栅格高斯合并
    const auto &sub_type = ogm.subType(sub_x, sub_y);
    if (track_cnt == 0) {
      grid_type = (sub_type == kDynamicGrid || sub_type == kBfsDynamicGrid)
                      ? kDynamicGrid
                      : kStaticGrid;
      track_id = 0;
      grid_conf = 0;
      det_dist = 0;
      row_cnt_seq = 0;
      ogm.gauss(map_x, map_y) = nongnd_info_map_(sub_x, sub_y);
    } else {
      ogm.gauss(map_x, map_y).merge(nongnd_info_map_(sub_x, sub_y));
      // 最大高度只取最新观测: 可解近处盲区hold问题
      ogm.gauss(map_x, map_y).max_z = nongnd_info_map_(sub_x, sub_y).max_z;
    }

    // 3. 起目标 & 跟踪
    bool is_static = (grid_type == kStaticGrid) && (sub_type != kBarrierGrid);
    if (track_cnt < ogm.pub_track_cnt_) {
      // 起目标逻辑: 首帧单排点参与计数, 后续两/三排点计数+1, 单排点计数-1
      if (track_cnt == 0 || grid_row >= kTwoRowGrid) {
        track_cnt += 1;
      } else if (track_cnt > 1) {
        track_cnt -= 1;
      }
      // 若当前帧起目标: 静目标计数直接置为上限, 动目标为 pub 阈值
      if (track_cnt >= ogm.pub_track_cnt_) {
        Eigen::Vector2d grid_pt(ogm.xSubToPos(sub_x), ogm.ySubToPos(sub_y));
        double dis = (grid_pt - ogm.EgoInGrid().head(2)).norm();
        det_dist = std::max(dis, det_dist);
        track_cnt = is_static ? ogm.max_track_cnt_ : ogm.pub_track_cnt_;
      }
    } else if (grid_row >= kTwoRowGrid) {
      // 跟踪逻辑: 单排点计数不变, 静目标保持计数上限, 动目标保持 pub 阈值
      track_cnt = is_static ? ogm.max_track_cnt_ : ogm.pub_track_cnt_;
    } else if (track_cnt > ogm.pub_track_cnt_ && !is_static) {
      // 非静目标栅格: 不超过 pub 阈值
      track_cnt = ogm.pub_track_cnt_;
    }

    // 4. 置信度逻辑: 高四位代表四帧中有几帧三排点, 低四位代表四帧中有几帧两排点
    // 若已经为高, 就保持高
    if (grid_conf != kHighConfOcc) {
      row_cnt_seq = (row_cnt_seq << 1) & (0xEE);  // 0b11101110
      if (grid_row >= kTwoRowGrid) {
        row_cnt_seq |= 0x01;  // 两排点
        if (grid_row == kMultiRowGrid) {
          row_cnt_seq |= 0x10;  // 三排点
        }
        int two_row_frames = row_sequence_map_[row_cnt_seq & 0x0F];
        int three_row_frames = row_sequence_map_[row_cnt_seq >> 4];
        if (three_row_frames >= 2 || two_row_frames == 4) {
          grid_conf = kHighConfOcc;
        } else if (two_row_frames >= 2) {
          grid_conf = kMidConfOcc;
        }
      }
    }
  });
}
void GridTracking::updateConnectDynamicInfo() {
  auto &connect_info_ogm = ogm.at(kConnectDynamicInfoMap);
  auto &track_cnt_ogm = ogm.at(kTrackCntMap);
  ogm.forInSubMap([&](int sub_x, int sub_y, int map_x, int map_y,
                      double grid_px, double grid_py) {
    auto &track_cnt = track_cnt_ogm(map_x, map_y);
    auto &connect_info = connect_info_ogm(map_x, map_y);
    // 不pub清除历史动目标相关记录
    if (track_cnt == 0) {
      connect_info = 0;
    }
    // 已经染成动目标（优先级高于历史信息判断，因为历史信息自洽性在bfs等处已经完成判断）
    if ((ogm.subType(sub_x, sub_y) == kBfsDynamicGrid ||
         ogm.subType(sub_x, sub_y) == kDynamicGrid) &&
        ogm.subObjId(sub_x, sub_y) > 0) {
      connect_info = ogm.subObjId(sub_x, sub_y);
      return;
    }
    // 已经标记为动态目标，不重复判断
    if (connect_info != 0) {
      return;
    }
    Eigen::Vector3d grid_pt{grid_px, grid_py, 0};
    grid_pt += ogm.gauss(map_x, map_y).getCentroid();
    Eigen::Vector3d ego_pt = ogm.tf_grid2ego_ * grid_pt;
    // 远距离不判断是否能够融合，box可能不准
    if (std::abs(ego_pt.x()) > 50) {
      return;
    }
    if (track_cnt >= ogm.pub_track_cnt_) {
      for (const auto &box : ogm.obj_infos_) {
        Eigen::Vector2f direction(std::cos(box.second.yaw),
                                  std::sin(box.second.yaw));
        Eigen::Vector2f normal(-std::sin(box.second.yaw),
                               std::cos(box.second.yaw));
        Eigen::Vector2f dist(ego_pt[0] - box.second.center[0],
                             ego_pt[1] - box.second.center[1]);
        float horizontal_dist = dist.dot(direction);
        float longitudinal_dist = dist.dot(normal);
        float head_thresh = box.second.size[0] * 0.5,
              tail_tresh = box.second.size[0] * 0.5;
        float longitudinal_thresh = box.second.size[1] * 0.5;
        if (IsVRU(box.second.type)) {
          head_thresh = std::min(head_thresh * 4, 5.0f);
          tail_tresh = std::min(tail_tresh * 4, 5.0f);
          longitudinal_thresh = std::min(longitudinal_thresh * 4, 5.0f);
        } else if (box.second.type == ObjectType::TYPE_CAR) {
          head_thresh = head_thresh + 1.0;
          tail_tresh = tail_tresh * 2;
          longitudinal_thresh = longitudinal_thresh + 1.0;
        } else {
          head_thresh = head_thresh * 2;
          tail_tresh = tail_tresh * 2;
          longitudinal_thresh = longitudinal_thresh * 2;
        }
        if (horizontal_dist < head_thresh && horizontal_dist > -tail_tresh &&
            std::abs(longitudinal_dist) < longitudinal_thresh) {
          return;
        }
      }
      connect_info = -1;
    }
  });
}
}  // namespace perception
}  // namespace robosense
