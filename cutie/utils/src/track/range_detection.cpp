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

#include "hyper_vision/perception/occupancy_objects/occupancy_detection/range_detection.h"

#include "hyper_vision/perception/occupancy_objects/occupancy_detection/occ_grid_map.h"

namespace robosense {
namespace perception {

void RangeDetection::init(const uint8_t sensor_id, const YAML::Node &cfg_node) {
  rally::yamlRead(cfg_node, "pitch_min", raw_pitch_min_);
  rally::yamlRead(cfg_node, "pitch_max", raw_pitch_max_);
  rally::yamlRead(cfg_node, "yaw_min", raw_yaw_min_);
  rally::yamlRead(cfg_node, "yaw_max", raw_yaw_max_);
  rally::yamlRead(cfg_node, "res", res_);
  rally::yamlRead(cfg_node, "real_res", real_res_);

  // range image config
  RangeImageConfig range_cfg;
  range_cfg.pitch_range[0] = raw_pitch_max_;
  range_cfg.pitch_range[1] = raw_pitch_min_;
  range_cfg.yaw_range[0] = raw_yaw_max_;
  range_cfg.yaw_range[1] = raw_yaw_min_;
  range_cfg.voxel_resolution[0] = res_;
  range_cfg.voxel_resolution[1] = res_;
  range_cfg.max_points_num = LidarSensorIdToPointsNum(sensor_id);
  range_img_ptr = std::make_shared<RangeImage>(range_cfg);

  // depth image
  rows_ = (int)(std::ceil((raw_pitch_max_ - raw_pitch_min_) / res_));
  cols_ = (int)(std::ceil((raw_yaw_max_ - raw_yaw_min_) / res_));
  index_img_.resize(rows_, cols_);
  depth_img_.resize(rows_, cols_);

  // debug info
  AINFO << "====== Range Image Info =======";
  AINFO << "  pitch: [" << raw_pitch_min_ << "°, " << raw_pitch_max_ << "°]";
  AINFO << "  yaw: [" << raw_yaw_min_ << "°, " << raw_yaw_max_ << "°]";
  AINFO << "  size: " << rows_ << " * " << cols_;
  AINFO << "  resolution: " << res_ << "°";
  AINFO << "  real_re: " << real_res_ << "°";

  // angle -> radian
  res_ *= deg_to_rad;
  real_res_ *= deg_to_rad;
  res_inv_ = 1.0 / res_;
  raw_pitch_min_ *= deg_to_rad;
  raw_pitch_max_ *= deg_to_rad;
  raw_yaw_min_ *= deg_to_rad;
  raw_yaw_max_ *= deg_to_rad;

  // 提前计算每行的 tan 值
  row2tan_.resize(rows_);
  for (int row = 0; row < rows_; ++row) {
    double pitch = getRawPitch(row);
    row2tan_[row] = std::tan(pitch);
  }

  // arcsin 查表法快速计算
  AsinLookupTable::GetInstanceRad();
}

void RangeDetection::updateRangeImage(RangeImage::Ptr range_ptr) {
  // 获取 RangeImage
  if (range_ptr == nullptr) {
    range_img_ptr->calcRangeImage(cloud_ptr->lidar_points);
  } else {
    range_img_ptr = range_ptr;
  }
  // 确认当前帧实际边界
  const auto &boundary = range_img_ptr->effective_boundary_;
  col_min_ = boundary[0];
  col_max_ = boundary[1];
  row_min_ = boundary[2];
  row_max_ = boundary[3];
  valid_cols_ = col_max_ - col_min_ + 1;
  valid_rows_ = row_max_ - row_min_ + 1;
  yaw_min_ = raw_yaw_max_ - (col_max_ + 1) * res_;
  yaw_max_ = raw_yaw_max_ - col_min_ * res_;
  pitch_min_ = raw_pitch_max_ - (row_max_ + 1) * res_;
  pitch_max_ = raw_pitch_max_ - row_min_ * res_;
  AINFO << "====== RangeImage =======";
  AINFO << " pitch: [" << pitch_min_ * rad_to_deg << "°, "
        << pitch_max_ * rad_to_deg << "°]";
  AINFO << " yaw: [" << yaw_min_ * rad_to_deg << "°, " << yaw_max_ * rad_to_deg
        << "°]";
  AINFO << " size: " << valid_rows_ << " * " << valid_cols_;

  // 更新点云属性和 depth_img
  std::vector<int> point_idx;
  for (int row = row_min_; row <= row_max_; row++) {
    for (int col = col_min_; col <= col_max_; col++) {
      getPointIndex(row, col, point_idx);
      int min_pi = -1, max_pi = -1;
      for (const auto &pi : point_idx) {
        if (min_pi == -1 || cloud_ptr->depth(pi) < cloud_ptr->depth(min_pi)) {
          min_pi = pi;
        }
        if (max_pi == -1 || cloud_ptr->depth(pi) > cloud_ptr->depth(max_pi)) {
          max_pi = pi;
        }
        cloud_ptr->row(pi) = row;
        cloud_ptr->col(pi) = col;
      }
      index(row, col) = min_pi;
      depth(row, col) = max_pi >= 0 ? cloud_ptr->lidarDepth(max_pi) : 0;
    }
  }
}

void RangeDetection::getPointIndex(const int row, const int col,
                                   std::vector<int> &point_idx) {
  int idx = valid_cols_ * (row - row_min_) + (col - col_min_);
  int voxel_offset = range_img_ptr->voxel_offset_[idx];
  int next_voxel_offset = range_img_ptr->voxel_offset_[idx + 1];
  const auto &point_save = range_img_ptr->point_save_;
  point_idx.resize(next_voxel_offset - voxel_offset);
  for (int i = 0; i < point_idx.size(); i++) {
    point_idx[i] = point_save[i + voxel_offset].point_idx;
  }
}

void RangeDetection::getSortedPointIndex(const int row, const int col,
                                         std::vector<int> &point_idx) {
  int idx = valid_cols_ * (row - row_min_) + (col - col_min_);
  int voxel_offset = range_img_ptr->voxel_offset_[idx];
  int next_voxel_offset = range_img_ptr->voxel_offset_[idx + 1];
  auto &point_save = range_img_ptr->point_save_;
  point_idx.resize(next_voxel_offset - voxel_offset);
  // 排序
  if (point_idx.size() > 1) {
    auto compare = [&](const RangeImage::range_image_t &t1,
                       const RangeImage::range_image_t &t2) {
      return cloud_ptr->depth(t1.point_idx) < cloud_ptr->depth(t2.point_idx);
    };
    std::sort(point_save.begin() + voxel_offset,
              point_save.begin() + next_voxel_offset, compare);
  }
  // 取索引
  for (int i = 0; i < point_idx.size(); i++) {
    point_idx[i] = point_save[i + voxel_offset].point_idx;
  }
}

void RangeDetection::detectGhost() {
  auto &ogm = OccGridMap::GetInstance();
  if (!ogm.enable_ghost_) {
    return;
  }
  std::vector<int> gnd_stack;
  std::vector<int> point_idx;
  for (int col = col_min_; col <= col_max_; ++col) {
    // 地面镜像鬼影
    gnd_stack.clear();
    for (int row = row_max_ - 1; row >= row_min_; --row) {
      int pi = index(row, col);
      if (pi == -1 || cloud_ptr->label(pi) != kGroundPt) {
        continue;
      }
      double depth = cloud_ptr->depth(pi);
      while (!gnd_stack.empty()) {
        int prev_pi = gnd_stack.back();
        if (depth > cloud_ptr->depth(prev_pi)) break;
        cloud_ptr->label(prev_pi) = kGhostPt;
        gnd_stack.pop_back();
      }
      gnd_stack.emplace_back(pi);
    }
    // 动目标鬼影
    int prev_row = -1;
    int prev_obj_id = -1;
    double prev_depth = 0;
    for (int row = row_max_ - 1; row >= row_min_; --row) {
      int pi = index(row, col);
      if (pi == -1) continue;
      // 过滤 vru
      int obj_id = cloud_ptr->obj_id(pi);
      if (obj_id == -1 || IsVRU(ogm.obj_infos_[obj_id].type)) {
        continue;
      }
      // 同一分辨单元内, 动目标后方的点均为鬼影点
      getPointIndex(row, col, point_idx);
      for (const auto &i : point_idx) {
        if (cloud_ptr->obj_id(i) != obj_id) {
          cloud_ptr->label(pi) = kGhostPt;
          cloud_ptr->obj_id(pi) = -1;
        }
      }
      // 两动目标点中间夹杂的深度更大的非动目标点为鬼影点
      double depth = cloud_ptr->depth(pi);
      if (obj_id == prev_obj_id) {
        double max_dis = std::max(depth, prev_depth);
        for (int tmp_row = prev_row - 1; tmp_row > row; --tmp_row) {
          getPointIndex(tmp_row, col, point_idx);
          for (const auto &i : point_idx) {
            if (cloud_ptr->depth(i) > max_dis) {
              cloud_ptr->label(i) = kGhostPt;
              cloud_ptr->obj_id(i) = -1;
            }
          }
        }
      }
      prev_row = row;
      prev_obj_id = obj_id;
      prev_depth = depth;
    }
  }
}

void RangeDetection::drawRangeImage() {
  static std::unordered_map<int, cv::Vec3b> colormap = {
      {(int)kOutRoiPt, cv::Vec3b(30, 30, 30)},          // 默认 灰
      {(int)kCommonPt, cv::Vec3b(0, 255, 0)},           // 普通点 绿
      {(int)kGroundPt, cv::Vec3b(255, 0, 0)},           // 地面点 蓝
      {(int)kDynamicPt, cv::Vec3b(230, 50, 227)},       // 动目标 粉
      {(int)kGhostPt, cv::Vec3b(255, 255, 0)},          // 鬼影 青
      {(int)kMistPt, cv::Vec3b(255, 255, 255)},         // 水雾 白
      {(int)kHighRefPt, cv::Vec3b(0, 0, 255)},          // 高反 红
      {(int)kBloomingPt, cv::Vec3b(0, 255, 255)},       // 膨胀 黄
      {(int)kBarrierPt, cv::Vec3b(255, 0, 255)},        // 横杆 紫
      // cv::Vec3b(30, 105, 210) 巧克力色
  };
  cv::Mat img(valid_rows_, valid_cols_, CV_8UC3);
  img.setTo(cv::Scalar(50, 50, 50));
  std::vector<int> point_idx;
  for (int row = row_min_; row <= row_max_; ++row) {
    for (int col = col_min_; col <= col_max_; ++col) {
      getPointIndex(row, col, point_idx);
      if (point_idx.empty()) continue;
      const auto &label = cloud_ptr->label(point_idx[0]);
      img.at<cv::Vec3b>(row - row_min_, col - col_min_) = colormap[(int)label];
    }
  }
  std::string save_path = "/apollo/data/range_image/" +
                          std::to_string(cloud_ptr->frame_num) + ".png";
  cv::imwrite(save_path, img);
}

}  // namespace perception
}  // namespace robosense
