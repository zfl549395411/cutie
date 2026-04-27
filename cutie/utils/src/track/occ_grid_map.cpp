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

#include "hyper_vision/perception/occupancy_objects/occupancy_detection/occ_grid_map.h"

#include "hyper_vision/perception/occupancy_objects/occupancy_detection/polygon.h"
#include "hyper_vision/perception/occupancy_objects/occupancy_detection/range_detection.h"

namespace robosense {
namespace perception {

void OccGridMap::init(const YAML::Node &ogm_cfg,
                      const YAML::Node &semantic_cfg) {
  // ogm config
  double x_min, x_max, y_min, y_max, z_min, z_max;
  rally::yamlRead(ogm_cfg, "res", res_);
  rally::yamlRead(ogm_cfg, "x_min", x_min);
  rally::yamlRead(ogm_cfg, "x_max", x_max);
  rally::yamlRead(ogm_cfg, "y_min", y_min);
  rally::yamlRead(ogm_cfg, "y_max", y_max);
  rally::yamlRead(ogm_cfg, "z_min", z_min);
  rally::yamlRead(ogm_cfg, "z_max", z_max);
  rally::yamlRead(ogm_cfg, "h_max", ego_max_h_);
  rally::yamlRead(ogm_cfg, "near_t", near_time_);
  grid_max_h_ = ego_max_h_;

  // semantic config
  rally::yamlRead(semantic_cfg, "object", enable_object_);
  rally::yamlRead(semantic_cfg, "curb", enable_curb_);
  rally::yamlRead(semantic_cfg, "ghost", enable_ghost_);
  rally::yamlRead(semantic_cfg, "mist", enable_mist_);
  rally::yamlRead(semantic_cfg, "blooming", enable_blooming_);
  rally::yamlRead(semantic_cfg, "barrier", enable_barrier_);

  // 分辨率
  res_inv_ = 1.0 / res_;
  neighbor_dist_ = res_ / RangeDetection::GetInstance().res_;

  // 自车 ROI
  base_roi_.min_bounds = Eigen::Vector3d(x_min, y_min, z_min);
  base_roi_.max_bounds = Eigen::Vector3d(x_max, y_max, z_max);
  base_rows_ = std::ceil(base_roi_.xRange() / res_) + 2;
  base_cols_ = std::ceil(base_roi_.yRange() / res_) + 2;
  edge_corners_.resize(8);
  edge_points_ = std::vector<Eigen::Vector3d>{
      {x_min, y_min, z_min}, {x_min, y_max, z_min}, {x_max, y_min, z_min},
      {x_max, y_max, z_min}, {x_min, y_min, z_max}, {x_min, y_max, z_max},
      {x_max, y_min, z_max}, {x_max, y_max, z_max}};

  // occupancy map for pub pillars
  map_info_.resolution = res_;
  map_info_.width = std::ceil(base_roi_.xRange() / res_);
  map_info_.height = std::ceil(base_roi_.yRange() / res_);
  map_info_.origin_coordinate.x = std::ceil(-x_min / res_);
  map_info_.origin_coordinate.y = std::ceil(-y_min / res_);

  // 世界栅格底图尺寸
  Eigen::Vector3d super_range(std::max(std::abs(x_min), std::abs(x_max)),
                              std::max(std::abs(y_min), std::abs(y_max)),
                              std::max(std::abs(z_min), std::abs(z_max)));
  double super_radius = std::ceil(super_range.norm() * res_inv_) * res_;
  super_start_ = -super_radius;
  super_range_ = 2.0 * super_radius;
  super_size_ = toGrid(super_range_) + 1;

  // 子图最大尺寸
  Eigen::Vector3d sub_range(base_roi_.xRange(), base_roi_.yRange(),
                            base_roi_.zRange());
  double sub_radius = std::ceil(sub_range.norm() * res_inv_) * res_;
  sub_size_ = toGrid(sub_radius) + 1;

  // int ogm
  Ogm<int16_t> empty_int_ogm(super_size_, super_size_);
  empty_int_ogm.setZero();
  int_ogms_.resize(__INT_OGM_SIZE__, empty_int_ogm);
  tmp_int_ogms_.resize(__INT_OGM_SIZE__, empty_int_ogm);

  // double ogm
  Ogm<double> empty_dbl_ogm(super_size_, super_size_);
  empty_dbl_ogm.setZero();
  dbl_ogms_.resize(__DBL_OGM_SIZE__, empty_dbl_ogm);
  tmp_dbl_ogms_.resize(__DBL_OGM_SIZE__, empty_dbl_ogm);

  // gauss map
  gauss_ogm_.resize(super_size_, super_size_);
  tmp_gauss_ogm_.resize(super_size_, super_size_);
  hi_gauss_ogm_.resize(super_size_, super_size_);

  // sub map
  obj_id_sub_map_.resize(sub_size_, sub_size_);
  type_sub_map_.resize(sub_size_, sub_size_);
  curb_sub_map_.resize(sub_size_, sub_size_);

  // reset
  resetWorldOrigin();
  clear_row_ = super_size_;  // 第一次切原点前不需要清理 tmp 图层

  // init world for debug
  YAML::Node world_cfg;
  if (rally::yamlSubNode(ogm_cfg, "world_origin", world_cfg)) {
    std::vector<double> pose, quat;
    rally::yamlRead(world_cfg, "pose", pose);
    rally::yamlRead(world_cfg, "quat", quat);
    world_orig_ = Eigen::Vector3d(pose[0], pose[1], pose[2]);
    world_quat_ = Eigen::Quaterniond(quat[0], quat[1], quat[2], quat[3]);
    tf_world2global_.linear() = world_quat_.toRotationMatrix();
    tf_world2global_.translation() = world_orig_;
    tf_global2world_ = tf_world2global_.inverse();
    is_origin_reset_ = false;
    is_debug_first_ = true;
  }

  // info
  AINFO << "====== World Grid Map Info =======";
  AINFO << "    x roi: [" << x_min << "m, " << x_max << "m]";
  AINFO << "    y roi: [" << y_min << "m, " << y_max << "m]";
  AINFO << "    z roi: [" << z_min << "m, " << z_max << "m]";
  AINFO << "    range: [" << -super_radius << "m, " << super_radius << "m]";
  AINFO << "    base_size: " << base_rows_ << " * " << base_cols_;
  AINFO << "    super_size: " << super_size_ << " * " << super_size_;
  AINFO << "    sub_size: " << sub_size_ << " * " << sub_size_;
  AINFO << "    resolution: " << res_ << "m";
  AINFO << "    neighbor_dist: " << neighbor_dist_ << "m";
  AINFO << "====== Semantic Info =======";
  AINFO << "    enable_object: " << enable_object_;
  AINFO << "    enable_curb: " << enable_curb_;
  AINFO << "    enable_mist: " << enable_mist_;
  AINFO << "    enable_blooming: " << enable_blooming_;
  AINFO << "    enable_barrier: " << enable_barrier_;
}

void OccGridMap::checkLocalization(const Eigen::Affine3d &tf) {
  // debug localization
  Eigen::Matrix3d R = tf.linear();
  Eigen::Vector3d t = tf.translation();
  double pitch = std::asin(-R(2, 0)) * rad_to_deg;  // 按照 ZYX 外旋计算
  double roll = std::atan(R(2, 1) / R(2, 2)) * rad_to_deg;
  double yaw = std::atan(R(1, 0) / R(0, 0)) * rad_to_deg;
  AINFO << "============== tfEgoToGlobal ==============";
  AINFO << " x[" << std::to_string(t.x()) << "m] "
        << " y[" << std::to_string(t.y()) << "m] "
        << " z[" << std::to_string(t.z()) << "m] ";
  AINFO << " Roll[" << std::to_string(roll) << "°] "
        << " Pitch[" << std::to_string(pitch) << "°] "
        << " Yaw[" << std::to_string(yaw) << "°]";

  // debug ego2world
  R = tf_ego2world_.linear();
  t = tf_ego2world_.translation();
  pitch = std::asin(-R(2, 0)) * rad_to_deg;  // 按照 ZYX 外旋计算
  roll = std::atan(R(2, 1) / R(2, 2)) * rad_to_deg;
  yaw = std::atan(R(1, 0) / R(0, 0)) * rad_to_deg;
  AINFO << "============ tfEgoToWorld(prev) ===========";
  AINFO << " x[" << std::to_string(t.x()) << "m] "
        << " y[" << std::to_string(t.y()) << "m] "
        << " z[" << std::to_string(t.z()) << "m] ";
  AINFO << " Roll[" << std::to_string(roll) << "°] "
        << " Pitch[" << std::to_string(pitch) << "°] "
        << " Yaw[" << std::to_string(yaw) << "°]";

  // return
  if (is_origin_reset_) {
    return;
  }
  if (is_debug_first_) {
    is_debug_first_ = false;
    return;
  }

  // check localization
  Eigen::Vector2d prev_pos = tf_ego2global_.translation().head(2);
  Eigen::Vector2d curr_pos = tf.translation().head(2);
  if ((curr_pos - prev_pos).norm() > super_range_) {
    AWARN << "Localization error! Reset world origin!";
    clearTmpMap(super_size_);
    resetWorldOrigin();
    return;
  }

  // check ego2world rotation
  if (!isTmpCleared()) return;  // 未准备好 tmp 图层, 不会切原点
  if (std::abs(pitch) > 10.0 || std::abs(roll) > 10.0) {
    AWARN << "Over rotation limit! Reset world origin!";
    moveToTmp();
    resetWorldOrigin();
  }
}

void OccGridMap::updateTransformer(const Eigen::Affine3d &tf) {
  // 1. update world ⇋ global (reset world origin)
  if (is_origin_reset_) {
    tf_world2global_ = tf;
    tf_global2world_ = tf.inverse();
    world_orig_ = tf.translation();
    world_quat_ = Eigen::Quaterniond(tf.linear());
    tf_reg_world_.setIdentity();  // 重置配准
  }
  // 2. update ego ⇋ global
  tf_ego2global_ = tf;
  tf_global2ego_ = tf_ego2global_.inverse();
  // 3. update ego ⇋ world
  tf_ego2world_ = tf_global2world_ * tf_ego2global_;
  world_normal_ = tf_ego2world_.linear().col(2);
  tf_ego2world_ = tf_reg_world_ * tf_ego2world_;
  tf_world2ego_ = tf_ego2world_.inverse();
  // 4. update grid ⇋ world
  tf_grid2world_.translation().x() = gFloor(tf_ego2world_.translation().x());
  tf_grid2world_.translation().y() = gFloor(tf_ego2world_.translation().y());
  tf_world2grid_.translation() = -tf_grid2world_.translation();
  // 5. update ego ⇋ grid
  tf_ego2grid_ = tf_world2grid_ * tf_ego2world_;
  tf_grid2ego_ = tf_world2ego_ * tf_grid2world_;
  // 6. update lidar ⇋ grid
  tf_lidar2grid_ = tf_ego2grid_ * tf_lidar2ego_;
  tf_grid2lidar_ = tf_ego2lidar_ * tf_grid2ego_;
  // 7. update map info
  updateLogicOffset();
  updateSubRange();
  clearTmpMap();
  // 8. update cloud
  cloud_ptr->updateLidarPoints(tf_ego2lidar_);
  cloud_ptr->updateGridPoints(tf_ego2grid_);
  cloud_ptr->roiFilter(base_roi_);
  // 9. update cloud grid
  for (const auto &pi : cloud_ptr->roi_idx) {
    const auto &grid_pt = cloud_ptr->grid_pt(pi);
    auto &info = cloud_ptr->infos[pi];
    int super_x = posToSuper(grid_pt.x());
    int super_y = posToSuper(grid_pt.y());
    info.sub_x = xSuperToSub(super_x);
    info.sub_y = ySuperToSub(super_y);
    info.map_x = xToPhysic(super_x);
    info.map_y = yToPhysic(super_y);
    if (info.sub_x < 0 || info.sub_x >= sub_size_ || info.sub_y < 0 ||
        info.sub_y >= sub_size_) {
      AWARN << "out of sub range! "
            << "xyz: " << cloud_ptr->ego_pt(pi).transpose()
            << ", sub_x: " << info.sub_x << ", sub_y: " << info.sub_y
            << ", map_x: " << info.map_x << ", map_y: " << info.map_y;
    }
  }
  // 10. world info
  AINFO << "============= tfWorldToGlobal =============";
  AINFO << " pose: [" << std::to_string(world_orig_.x()) << ", "
        << std::to_string(world_orig_.y()) << ", "
        << std::to_string(world_orig_.z()) << "]";
  AINFO << " quat: [" << std::to_string(world_quat_.w()) << ", "
        << std::to_string(world_quat_.x()) << ", "
        << std::to_string(world_quat_.y()) << ", "
        << std::to_string(world_quat_.z()) << "]";
}

void OccGridMap::updateRegistration(const Eigen::Affine3d &tf_grid_reg_delta) {
  // tf_grid_reg_delta 表示当前帧以上一帧配准结果为初值进行配准的变化量
  tf_reg_world_ =
      tf_grid2world_ * tf_grid_reg_delta * tf_world2grid_ * tf_reg_world_;
  // update ego ⇋ grid
  tf_ego2grid_ = tf_grid_reg_delta * tf_ego2grid_;
  tf_grid2ego_ = tf_ego2grid_.inverse();
  // update lidar ⇋ grid
  tf_lidar2grid_ = tf_ego2grid_ * tf_lidar2ego_;
  tf_grid2lidar_ = tf_ego2lidar_ * tf_grid2ego_;
  // update ego ⇋ world
  tf_ego2world_ = tf_grid2world_ * tf_ego2grid_;
  tf_world2ego_ = tf_grid2ego_ * tf_world2grid_;
  // update cloud
  cloud_ptr->updateGridPoints(tf_ego2grid_);
}

void OccGridMap::updateObjSubMap(
    ObjectFusionArray::Ptr objs_ptr,
    const std::shared_ptr<PolygonDivide> &polygon_ptr) {
  // reset
  obj_grids_ = std::queue<Eigen::Vector2i>();
  obj_id_sub_map_.setConstant(-1);
  type_sub_map_.setZero();
  obj_infos_.clear();
  if (!enable_object_ || objs_ptr == nullptr) {
    return;
  }

  // 时间对齐补偿
  double obj_time = (double)objs_ptr->header.time / 1e9;
  double time_diff = cloud_ptr->time_sec - obj_time;
  Eigen::Affine3d tf_sync = tfToSync(obj_time);

  // 动目标子图染色
  std::vector<std::array<int, 2>> obj_roi;
  static std::vector<Eigen::Vector2i> grid_corners(4);
  static Roi<int, 2> roi2i;
  static Roi<double, 3> roi3d;
  for (const auto &obj : objs_ptr->object_list) {
    if (!IsDynamicObj(obj.type)) {
      continue;
    }
    // 动静态补偿
    auto box3d = Box3dInfo(obj);
    box3d.staticComp(tf_sync);
    box3d.dynamicComp(time_diff);
    obj_infos_[obj.track_id] = box3d;
    box3d.transform(tf_ego2lidar_);
    // 获取在 super_map 上的范围
    auto corners3d = box3d.getCorners3D();
    for (int i = 0; i < 4; i++) {
      corners3d[i] = tf_lidar2grid_ * corners3d[i];
      grid_corners[i].x() = posToSuper(corners3d[i].x());
      grid_corners[i].y() = posToSuper(corners3d[i].y());
    }
    getGridsRoi(grid_corners, obj_roi, roi2i);
    // 根据范围标记 type & id
    int row_offset = roi2i.xMin() - sub_roi_.xMin();
    int col_offset = roi2i.yMin() - sub_roi_.yMin();
    for (int i = 0; i < obj_roi.size(); ++i) {
      int sub_x = i + row_offset;
      if (sub_x < 0 || sub_x >= sub_rows_) {
        continue;
      }
      const auto &sub_col_range = sub_roi_by_row_[sub_x];
      int col_start = std::max(obj_roi[i][0] + col_offset, sub_col_range[0]);
      int col_end = std::min(obj_roi[i][1] + col_offset, sub_col_range[1]);
      int map_x = xSubToMap(sub_x);
      for (int sub_y = col_start; sub_y <= col_end; ++sub_y) {
        int map_y = ySubToMap(sub_y);
        const auto &polygon_id = at(kTrackIdMap)(map_x, map_y);
        if (IsVRU(obj.type) &&
            !polygon_ptr->checkDynamicConnect(polygon_id, obj.track_id)) {
          continue;
        }
        staticToDynamic(map_x, map_y);
        subType(sub_x, sub_y) = kDynamicGrid;
        subObjId(sub_x, sub_y) = obj.track_id;
        obj_grids_.emplace(sub_x, sub_y);
      }
    }
  }

  // 动目标点云语义
  for (const auto &pi : cloud_ptr->roi_idx) {
    auto &info = cloud_ptr->infos[pi];
    int sub_x{info.sub_x}, sub_y{info.sub_y};
    int obj_id = subObjId(sub_x, sub_y);
    if (obj_id == -1) {
      continue;
    }
    const auto &obj_info = obj_infos_[obj_id];
    const auto &pt = cloud_ptr->ego_pt(pi);
    if (pt.z() < obj_info.getMinZ()) {
      info.label = kGhostPt;
    } else if (pt.z() < obj_info.getMaxZ()) {
      info.label = kDynamicPt;
      info.obj_id = obj_id;
    }
  }
}

void OccGridMap::updateSemantic(BevNNSegmentMatrix::Ptr bev_segment_ptr,
                                LidarSemantic::Ptr semantic_ptr) {
  if (semantic_ptr == nullptr) {
    const auto seg_map = CSRU8ToDense(bev_segment_ptr->csr_matrix, 0);
    int row, col;
    for (const auto &pi : cloud_ptr->roi_idx) {
      const auto &ego_pt = cloud_ptr->ego_pt(pi);
      if (ego_pt.z() >= 2 * ego_max_h_ ||
          !bev_segment_ptr->getIndex(ego_pt, row, col)) {
        continue;
      }
      BevSemantic semantic = static_cast<BevSemantic>(seg_map[row][col]);
      auto &info = cloud_ptr->infos[pi];
      int sub_x{info.sub_x}, sub_y{info.sub_y};
      const auto &curb_type = subCurbType(sub_x, sub_y);
      info.in_freespace = (curb_type == kInCurbGrid);
      if (semantic == BevSemantic::GROUND) {
        info.label = kGroundPt;
      } else if (semantic == BevSemantic::MIST && enable_mist_ &&
                 curb_type != kCurbGrid && ego_pt.x() < 50) {
        info.label = kMistPt;
      }
    }
  } else {
    for (const auto &pi : cloud_ptr->roi_idx) {
      Semantic semantic = semantic_ptr->get(pi);
      cloud_ptr->semantic(pi) = semantic;
      cloud_ptr->isBarrier(pi) = semantic_ptr->isBarrier(pi);
      const auto &ego_pt = cloud_ptr->ego_pt(pi);
      auto &info = cloud_ptr->infos[pi];
      int sub_x{info.sub_x}, sub_y{info.sub_y};
      const auto &curb_type = curb_sub_map_(sub_x, sub_y);
      info.in_freespace = (curb_type == kInCurbGrid);
      if (semantic == Semantic::GROUND) {
        info.label = kGroundPt;
      } else if (semantic == Semantic::MIST_NOISE && enable_mist_ &&
                 curb_type != kCurbGrid && ego_pt.x() < 50) {
        info.label = kMistPt;
      }
    }
  }
}

void OccGridMap::updateLogicOffset() {
  // 重置世界栅格底图中新物理地址的值
  int prev_total_step_x = total_step_x_;
  int prev_total_step_y = total_step_y_;
  total_step_x_ = toGrid(tf_grid2world_.translation().x());
  total_step_y_ = toGrid(tf_grid2world_.translation().y());
  clearMap(logic_offset_x_, total_step_x_ - prev_total_step_x, true);  // 清行
  clearMap(logic_offset_y_, total_step_y_ - prev_total_step_y, false);  // 清列
  // 更新物理和逻辑相对关系
  logic_offset_x_ = (total_step_x_ % super_size_ + super_size_) % super_size_;
  logic_offset_y_ = (total_step_y_ % super_size_ + super_size_) % super_size_;
  physic_offset_x_ = (super_size_ - logic_offset_x_) % super_size_;
  physic_offset_y_ = (super_size_ - logic_offset_y_) % super_size_;
}

void OccGridMap::updateGlobalOptions(const Eigen::Vector3d &ego_vel,
                                     Chassis::Ptr chassis_ptr) {
  // 根据自车速度决定近距离范围
  ego_vel_ = ego_vel.norm();
  near_dist_ = std::min(30.0, 5.0 + near_time_ * ego_vel_);
  // 世界栅格坐标系中的累帧高度范围
  h_ratio_ = gridNormal().z();
  grid_max_h_ = ego_max_h_ / h_ratio_;
  // 雨刮器信息
  if (chassis_ptr != nullptr && chassis_ptr->isRainy()) {
    rainy_scene_cnt_ = rainy_scene_hold_cnt_;
  } else {
    rainy_scene_cnt_ = std::max(rainy_scene_cnt_ - 1, 0);
  }
  is_rainy_ = rainy_scene_cnt_ > 0;
  wiper_mode_ = (chassis_ptr == nullptr) ? -1 : chassis_ptr->wiper_mode;
}

void OccGridMap::updateSubRange() {
  // 获取 roi 边界栅格并根据凸包排序
  for (size_t i = 0; i < edge_points_.size(); i++) {
    Eigen::Vector3d grid_pt = tf_ego2grid_ * edge_points_[i];
    int super_x = posToSuper(grid_pt.x());
    int super_y = posToSuper(grid_pt.y());
    edge_corners_[i].x() = limit(super_x, 0, super_size_ - 1);
    edge_corners_[i].y() = limit(super_y, 0, super_size_ - 1);
  }
  // 更新子图在逻辑地址的范围
  genConvexHull(edge_corners_, sort_corners_);
  getGridsRoi(sort_corners_, sub_roi_by_row_, sub_roi_, edge_grids_);
  sub_rows_ = sub_roi_.xRange() + 1;
  sub_cols_ = sub_roi_.yRange() + 1;
  // debug sub_size
  AINFO << "============= Sub Map Range =============";
  AINFO << " x[" << sub_roi_.xMin() << ", " << sub_roi_.xMax() << "], "
        << " y[" << sub_roi_.yMin() << ", " << sub_roi_.yMax() << "]";
  AINFO << " size: " << sub_rows_ << " * " << sub_cols_;
}

void OccGridMap::clearMap(int start, int step, bool isRow) {
  if (step == 0 || std::abs(step) >= super_size_) return;
  if (step < 0) {  // 考虑负步进量, 调换起始
    start = (start + step + super_size_) % super_size_;
    step = -step;
  }
  if (start + step > super_size_) {  // 考虑循环地址
    clearMap(start, super_size_ - start, isRow);
    clearMap(0, (start + step) % super_size_, isRow);
    return;
  }
  if (isRow) {
    for (auto &ogm : int_ogms_) ogm.middleRows(start, step).setZero();
    for (auto &ogm : dbl_ogms_) ogm.middleRows(start, step).setZero();
  } else {
    for (auto &ogm : int_ogms_) ogm.middleCols(start, step).setZero();
    for (auto &ogm : dbl_ogms_) ogm.middleCols(start, step).setZero();
  }
}

void OccGridMap::clearTmpMap(int step) {
  if (isTmpCleared()) return;
  step = std::min(step, super_size_ - clear_row_);
  for (auto &ogm : tmp_int_ogms_) ogm.middleRows(clear_row_, step).setZero();
  for (auto &ogm : tmp_dbl_ogms_) ogm.middleRows(clear_row_, step).setZero();
  clear_row_ += step;
}

void OccGridMap::resetWorldOrigin() {
  is_origin_reset_ = true;
  clear_row_ = 0;       // 重置清理图层行数
  logic_offset_x_ = 0;  // 重置逻辑地址和物理地址的偏移量x
  logic_offset_y_ = 0;  // 重置逻辑地址和物理地址的偏移量y
  total_step_x_ = 0;    // 重置总栅格步进量x
  total_step_y_ = 0;    // 重置总栅格步进量y
  std::swap(int_ogms_, tmp_int_ogms_);
  std::swap(dbl_ogms_, tmp_dbl_ogms_);
  std::swap(gauss_ogm_, tmp_gauss_ogm_);
  // std::swap(hi_gauss_ogm_, tmp_hi_gauss_ogm_);
}

void OccGridMap::moveToTmp() {
  Eigen::Matrix3d rot = tf_grid2ego_.linear();
  forInSubMap([&](int sub_x, int sub_y, int map_x, int map_y, double grid_px,
                  double grid_py) {
    // 迁移累帧信息
    if (!at(kTrackCntMap)(map_x, map_y)) {
      return;
    }
    Eigen::Vector3d gauss_pt{grid_px, grid_py, 0.0};
    gauss_pt += gauss(map_x, map_y).getCentroid();  // gauss 存的是栅格偏置
    gauss_pt = tf_grid2ego_ * gauss_pt;
    int tmp_x = posToSuper(gauss_pt.x());
    int tmp_y = posToSuper(gauss_pt.y());
    if (!inSuperRange(tmp_x, tmp_y) || !atTmp(kTrackCntMap)(tmp_x, tmp_y)) {
      return;
    }
    // todo: 多个栅格切换到同一栅格时, 暂未考虑融合, 采取先到先得策略
    // 迁移 int 图层
    atTmp(kTypeMap)(tmp_x, tmp_y) = at(kTypeMap)(map_x, map_y);
    atTmp(kConfMap)(tmp_x, tmp_y) = at(kConfMap)(map_x, map_y);
    atTmp(kTrackCntMap)(tmp_x, tmp_y) = at(kTrackCntMap)(map_x, map_y);
    atTmp(kTrackIdMap)(tmp_x, tmp_y) = at(kTrackIdMap)(map_x, map_y);
    atTmp(kRowSeqMap)(tmp_x, tmp_y) = at(kRowSeqMap)(map_x, map_y);
    atTmp(kHIOccupiedMap)(tmp_x, tmp_y) = at(kHIOccupiedMap)(map_x, map_y);
    atTmp(kDetFrameMap)(tmp_x, tmp_y) = at(kDetFrameMap)(map_x, map_y);
    atTmp(kDynamicFrameMap)(tmp_x, tmp_y) = at(kDynamicFrameMap)(map_x, map_y);
    atTmp(kConnectDynamicInfoMap)(tmp_x, tmp_y) =
        at(kConnectDynamicInfoMap)(map_x, map_y);
    // 迁移 double 图层
    atTmp(kDetDistMap)(tmp_x, tmp_y) = at(kDetDistMap)(map_x, map_y);
    // 迁移 gauss 图层
    gauss_pt.x() -= gFloor(gauss_pt.x());  // 计算新的栅格偏置距离
    gauss_pt.y() -= gFloor(gauss_pt.y());
    gaussTmp(tmp_x, tmp_y).moveFrom(gauss(map_x, map_y), rot, gauss_pt);
  });
}

void OccGridMap::getGridsRoi(std::vector<Eigen::Vector2i> &corners,
                             std::vector<std::array<int, 2>> &roi_idx,
                             Roi<int, 2> &roi2i) {
  // 去除偏置
  roi2i.fromPoints(corners);
  for (auto &corner : corners) {
    corner -= roi2i.min_bounds;
  }
  // 计算每行的索引起始
  std::array<int, 2> range_init{roi2i.yRange(), 0};
  roi_idx.assign(roi2i.xRange() + 1, range_init);
  for (size_t i = 0; i < corners.size(); i++) {
    const auto &start = corners[i];
    const auto &end = corners[(i + 1) % corners.size()];
    const auto line = generateLine(start, end);
    for (const auto &grid : line) {
      int row = grid.x();
      int col = grid.y();
      roi_idx[row][0] = std::min(col, roi_idx[row][0]);
      roi_idx[row][1] = std::max(col, roi_idx[row][1]);
    }
  }
}

void OccGridMap::getGridsRoi(std::vector<Eigen::Vector2i> &corners,
                             std::vector<std::array<int, 2>> &roi_idx,
                             Roi<int, 2> &roi2i,
                             std::vector<Eigen::Vector2i> &edge_grids) {
  // 去除偏置
  roi2i.fromPoints(corners);
  for (auto &corner : corners) {
    corner -= roi2i.min_bounds;
  }
  // 计算每行的索引起始
  std::array<int, 2> range_init{roi2i.yRange(), 0};
  roi_idx.assign(roi2i.xRange() + 1, range_init);
  edge_grids.clear();
  for (size_t i = 0; i < corners.size(); i++) {
    const auto &start = corners[i];
    const auto &end = corners[(i + 1) % corners.size()];
    const auto line = generateLine(start, end, false);
    for (const auto &grid : line) {
      int row = grid.x();
      int col = grid.y();
      roi_idx[row][0] = std::min(col, roi_idx[row][0]);
      roi_idx[row][1] = std::max(col, roi_idx[row][1]);
    }
    edge_grids.insert(edge_grids.end(), line.begin(), line.end());
  }
}

void OccGridMap::getPubPillars(const Ogm<GridRow> &grid_row_map,
                               std::vector<OccupancyPillar> &pillars) {
  const auto &track_id_ogm = at(kTrackIdMap);
  const auto &track_cnt_ogm = at(kTrackCntMap);
  const auto &connct_info_ogm = at(kConnectDynamicInfoMap);
  forInSubMap([&](int sub_x, int sub_y, int map_x, int map_y, double grid_px,
                  double grid_py) {
    // 只输出存在累帧计数的栅格
    if (track_cnt_ogm(map_x, map_y) == 0) {
      return;
    }
    // 高斯栅格投影自车坐标系
    const auto &gauss_grid = gauss(map_x, map_y);
    Eigen::Vector3d grid_pt = {grid_px, grid_py, 0.0};
    grid_pt += gauss_grid.getCentroid();
    Eigen::Vector3d ego_pt = tf_grid2ego_ * grid_pt;
    int row = map_info_.xToRow(ego_pt.x());
    int col = map_info_.yToCol(ego_pt.y());
    if (!map_info_.inRoi(row, col)) {
      return;
    }
    // 组合 uint32_t label
    uint32_t l1 = static_cast<uint32_t>(grid_row_map(sub_x, sub_y));
    uint32_t l2 = static_cast<uint32_t>(subType(sub_x, sub_y));
    uint32_t l3 =
        static_cast<uint32_t>((connct_info_ogm(map_x, map_y) + 2) % 100);
    uint32_t l4 = static_cast<uint32_t>(track_cnt_ogm(map_x, map_y));
    uint32_t pillar_label = (l1 << 24) | (l2 << 16) | (l3 << 8) | l4;
    // 构建 pillar
    OccupancyPillar pillar;
    pillar.row_index = static_cast<uint32_t>(row);
    pillar.col_index = static_cast<uint32_t>(col);
    pillar.id = static_cast<uint32_t>(track_id_ogm(map_x, map_y));
    pillar.label = pillar_label;
    pillar.start_height = ego_pt.z() - gauss_grid.getMinDiffZ();
    pillar.end_height = ego_pt.z() + gauss_grid.getMaxDiffZ();
    pillars.emplace_back(pillar);
  });
  map_info_.data_end_index = pillars.size();
}

void OccGridMap::dynamicToStatic(const int map_x, const int map_y) {
  if (at(kTypeMap)(map_x, map_y) == kDynamicGrid) {
    const auto &dynamic_frame = at(kDynamicFrameMap)(map_x, map_y);
    int interval =
        (cloud_ptr->frame_num - dynamic_frame + INT16_MAX) % INT16_MAX;
    if (interval >= dynamic_hold_cnt_) {
      at(kTypeMap)(map_x, map_y) = kStaticGrid;
    }
  }
}

void OccGridMap::staticToDynamic(const int map_x, const int map_y) {
  at(kDynamicFrameMap)(map_x, map_y) = cloud_ptr->frame_num;
  if (at(kTypeMap)(map_x, map_y) != kDynamicGrid) {
    at(kTypeMap)(map_x, map_y) = kDynamicGrid;
    if (at(kTrackCntMap)(map_x, map_y) > pub_track_cnt_) {
      at(kTrackCntMap)(map_x, map_y) = pub_track_cnt_;
    }
  }
}

void cal_covariance(Eigen::ArrayXd voxel_info_,
                    Eigen::Matrix3d &covariance_matrix_) {
  // 求解协方差矩阵
  int voxel_point_num = voxel_info_(0);
  double alpha = 1.0 / voxel_point_num;
  // 提取数组中的特定元素，避免重复访问
  auto &voxel_info_1 = voxel_info_(1);
  auto &voxel_info_2 = voxel_info_(2);
  auto &voxel_info_3 = voxel_info_(3);
  // 计算中间结果，避免重复计算
  double alpha_voxel_info_1 = alpha * voxel_info_1;
  double alpha_voxel_info_2 = alpha * voxel_info_2;
  covariance_matrix_(0, 0) = voxel_info_(4) - alpha_voxel_info_1 * voxel_info_1;
  covariance_matrix_(0, 1) = voxel_info_(7) - alpha_voxel_info_1 * voxel_info_2;
  covariance_matrix_(0, 2) = voxel_info_(8) - alpha_voxel_info_1 * voxel_info_3;
  covariance_matrix_(1, 0) = covariance_matrix_(0, 1);
  covariance_matrix_(1, 1) = voxel_info_(5) - alpha_voxel_info_2 * voxel_info_2;
  covariance_matrix_(1, 2) = voxel_info_(9) - alpha_voxel_info_2 * voxel_info_3;
  covariance_matrix_(2, 0) = covariance_matrix_(0, 2);
  covariance_matrix_(2, 1) = covariance_matrix_(1, 2);
  covariance_matrix_(2, 2) =
      voxel_info_(6) - alpha * voxel_info_3 * voxel_info_3;
  covariance_matrix_ *= alpha;
}

}  // namespace perception
}  // namespace robosense
