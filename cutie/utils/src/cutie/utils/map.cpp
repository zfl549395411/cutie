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

#include "robot/perception/occupancy3d/occupancy_detection/occ_grid_map.h"

namespace robot {
namespace perception {

void OccGridMap::init(const YAML::Node &ogm_cfg) {
  // ogm config
  rally::yamlRead(ogm_cfg, "radius", super_radius_);
  rally::yamlRead(ogm_cfg, "res", res_);
  res_inv_ = 1.0 / res_;

  // 栅格尺寸
  super_start_ = -super_radius_;
  super_range_ = 2.0 * super_radius_;
  super_size_ = toGrid(super_range_) + 1;

  // int ogm
  Ogm<int16_t> empty_int_ogm(super_size_, super_size_);
  empty_int_ogm.setZero();
  int_ogms_.resize(__INT_OGM_SIZE__, empty_int_ogm);

  // double ogm
  Ogm<double> empty_dbl_ogm(super_size_, super_size_);
  empty_dbl_ogm.setZero();
  dbl_ogms_.resize(__DBL_OGM_SIZE__, empty_dbl_ogm);

  // reset
  resetWorldOrigin();

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
  }

  // info
  AINFO << "====== World Grid Map Info =======";
  AINFO << "    range: [" << -super_radius_ << "m, " << super_radius_ << "m]";
  AINFO << "    super_size: " << super_size_ << " * " << super_size_;
  AINFO << "    resolution: " << res_ << "m";
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
}

void OccGridMap::updateTransformer(const Eigen::Affine3d &tf) {
  // 1. update ego ⇋ global
  tf_ego2global_ = tf;
  tf_global2ego_ = tf_ego2global_.inverse();
  // 2. update ego ⇋ world
  tf_ego2world_ = tf_global2world_ * tf_ego2global_;
  tf_world2ego_ = tf_ego2world_.inverse();
  // 3. update grid ⇋ world
  tf_grid2world_.translation().x() = gFloor(tf_ego2world_.translation().x());
  tf_grid2world_.translation().y() = gFloor(tf_ego2world_.translation().y());
  tf_world2grid_.translation() = -tf_grid2world_.translation();
  // 4. update ego ⇋ grid
  tf_ego2grid_ = tf_world2grid_ * tf_ego2world_;
  tf_grid2ego_ = tf_world2ego_ * tf_grid2world_;
  // 5. update lidar ⇋ grid
  tf_lidar2grid_ = tf_ego2grid_ * tf_lidar2ego_;
  tf_grid2lidar_ = tf_ego2lidar_ * tf_grid2ego_;
  // 6. update polar ⇋ grid
  Eigen::Matrix3d R_ego2grid = tf_ego2grid_.linear();
  double yaw = std::atan2(R_ego2grid(1, 0), R_ego2grid(0, 0));
  Eigen::AngleAxisd yaw_rot(yaw, Eigen::Vector3d::UnitZ());
  tf_polar2grid_.linear() = yaw_rot.toRotationMatrix();
  tf_polar2grid_.translation() = tf_lidar2grid_.translation();
  tf_grid2polar_ = tf_polar2grid_.inverse();
  // 7. update ego ⇋ polar
  tf_ego2polar_ = tf_grid2polar_ * tf_ego2grid_;
  tf_polar2ego_ = tf_grid2ego_ * tf_polar2grid_;
  // 8. update map info
  updateLogicOffset();
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

void OccGridMap::resetWorldOrigin() {
  logic_offset_x_ = 0;  // 重置逻辑地址和物理地址的偏移量x
  logic_offset_y_ = 0;  // 重置逻辑地址和物理地址的偏移量y
  total_step_x_ = 0;    // 重置总栅格步进量x
  total_step_y_ = 0;    // 重置总栅格步进量y
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

}  // namespace perception
}  // namespace robot
