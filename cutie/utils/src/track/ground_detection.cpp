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

#include "hyper_vision/perception/occupancy_objects/occupancy_detection/ground_detection.h"

#include "hyper_vision/perception/occupancy_objects/occupancy_detection/occ_grid_map.h"
#include "hyper_vision/perception/occupancy_objects/occupancy_detection/range_detection.h"

namespace robosense {
namespace perception {

void GroundDetection::init(const YAML::Node &cfg_node) {
  // debug cost time
  bool debug_time = false;
  rally::yamlRead(cfg_node, "debug_time", debug_time);
  if (debug_time) {
    tc_.init({"RangeFit", "ExtractSeeds", "GroundICP", "Interpolation",
              "FilterGround", "GroundTotal"});
  }

  // debug ground normal
  rally::yamlRead(cfg_node, "debug_normal", debug_normal_);

  // ground filter margin
  rally::yamlRead(cfg_node, "height_th", height_th_);
  rally::yamlRead(cfg_node, "rainy_height_th", rainy_height_th_);

  // ground map params
  rally::yamlRead(cfg_node, "res", res_);
  rally::yamlRead(cfg_node, "x_min", x_min_);
  rally::yamlRead(cfg_node, "x_max", x_max_);
  rally::yamlRead(cfg_node, "y_min", y_min_);
  rally::yamlRead(cfg_node, "y_max", y_max_);
  x_size_ = (int)(std::ceil((x_max_ - x_min_) / res_));
  y_size_ = (int)(std::ceil((y_max_ - y_min_) / res_));
  map_size_ = x_size_ * y_size_;
  res_inv_ = 1.0 / res_;
  ground_map_.map_info.length = x_size_;
  ground_map_.map_info.width = y_size_;
  ground_map_.map_info.min_x = x_min_;
  ground_map_.map_info.min_y = y_min_;
  ground_map_.map_info.max_x = x_max_;
  ground_map_.map_info.max_y = y_max_;
  ground_map_.map_info.resolution = res_;
  ground_map_.map_data.resize(map_size_);

  // 图层
  nongnd_valid_map_.resize(x_size_, y_size_);
  nongnd_gauss_map_.resize(x_size_, y_size_);
  gnd_valid_map_.resize(x_size_, y_size_);
  gnd_gauss_map_.resize(x_size_, y_size_);
  gnd_height_map_.resize(x_size_, y_size_);
  integral_count_map_.resize(x_size_, y_size_);
  integral_height_map_.resize(x_size_, y_size_);

  // 权重查表
  weight_lut_.resize(std::max(x_size_, y_size_), 1.0);
  for (int r = 1; r < weight_lut_.size(); ++r) {
    weight_lut_[r] = 1.0 / std::pow(r * res_, 2);
  }
}

void GroundDetection::detectGround() {
  // time cost
  timeval time_start, time_end, time_total_start, time_total_end;
  tc_.getTime(time_total_start);

  // range image fit
  tc_.getTime(time_start);
  rangeImageFit();
  tc_.getTime(time_end);
  tc_.calTime(time_start, time_end, "RangeFit");

  // calculate feature
  tc_.getTime(time_start);
  calGroundGauss();
  extractSeeds();
  tc_.getTime(time_end);
  tc_.calTime(time_start, time_end, "ExtractSeeds");

  // register
  tc_.getTime(time_start);
  auto &ogm = OccGridMap::GetInstance();
  if (ogm.is_origin_reset_) {
    ogm.is_origin_reset_ = false;
  } else if (enable_icp_) {
    ogm.updateRegistration(tf_icp_);
  }
  tc_.getTime(time_end);
  tc_.calTime(time_start, time_end, "GroundICP");

  // ground interpolation
  tc_.getTime(time_start);
  groundInterpolation();
  tc_.getTime(time_end);
  tc_.calTime(time_start, time_end, "Interpolation");

  // ground filter
  tc_.getTime(time_start);
  filterGround();
  tc_.getTime(time_end);
  tc_.calTime(time_start, time_end, "FilterGround");

  // show time_cost
  tc_.getTime(time_total_end);
  tc_.calTime(time_total_start, time_total_end, "GroundTotal");
  tc_.showTime();
}

void GroundDetection::rangeImageFit() {
  auto &rd = RangeDetection::GetInstance();
  for (int col = rd.col_min_; col <= rd.col_max_; ++col) {
    LinearFit lf;
    double k{0}, b{0};
    double delta_d{0}, delta_z{0};
    int prev_gnd_row = -1;
    std::deque<int> gnd_window;
    std::vector<int> unfit_idx;
    for (int row = rd.row_max_ - 1; row >= rd.row_min_; --row) {
      int pi = rd.index(row, col);
      if (pi == -1) continue;
      if (!cloud_ptr->inFreespace(pi)) break;
      auto &label = cloud_ptr->label(pi);
      if (label != kGroundPt && label != kCommonPt) continue;
      // 检查相邻梯度
      const auto &pt = cloud_ptr->ego_pt(pi);
      const auto &depth = cloud_ptr->depth(pi);
      if (!gnd_window.empty()) {
        int prev_pi = gnd_window.back();
        const auto &prev_pt = cloud_ptr->ego_pt(prev_pi);
        const auto &prev_depth = cloud_ptr->depth(prev_pi);
        delta_d = depth - prev_depth;
        delta_z = std::abs(pt.z() - prev_pt.z());
        if (delta_z > delta_d * soft_tan_th_) {
          label = kCommonPt;
          continue;
        }
      }
      // 添加初始窗口
      if (gnd_window.size() < 4) {
        if (label == kGroundPt) {
          gnd_window.push_back(pi);
          lf.add(depth, pt.z());
          prev_gnd_row = row;
        }
        continue;
      }
      // 连续欠拟合地面点就清除窗口并回滚状态
      if (unfit_idx.size() >= 3) {
        for (const auto &i : unfit_idx) {
          cloud_ptr->label(i) = kGroundPt;
        }
        gnd_window.clear();
        unfit_idx.clear();
        lf.clear();
        row = prev_gnd_row;
        continue;
      }
      // 直线拟合
      lf.fit(k, b);
      double pred_z = k * depth + b;
      double err = std::abs(pt.z() - pred_z);
      double err_th = std::min(0.2, 0.1 + 0.02 * delta_d);
      if (err > err_th) {
        if (label == kGroundPt) {
          unfit_idx.emplace_back(pi);
          label = kCommonPt;
        }
        continue;
      }
      // 滑窗迭代
      unfit_idx.clear();
      prev_gnd_row = row;
      label = kGroundPt;
      int front_pi = gnd_window.front();
      const auto &front_pt = cloud_ptr->ego_pt(front_pi);
      const auto &front_depth = cloud_ptr->depth(front_pi);
      lf.remove(front_depth, front_pt.z());
      lf.add(depth, pt.z());
      gnd_window.pop_front();
      gnd_window.push_back(pi);
    }
  }
}

void GroundDetection::calGroundGauss() {
  gnd_valid_map_.setZero();
  nongnd_valid_map_.setZero();
  for (const auto &pi : cloud_ptr->roi_idx) {
    // roi & curb 过滤
    const auto &pt = cloud_ptr->ego_pt(pi);
    if (!cloud_ptr->inFreespace(pi) || !inRoi(pt.x(), pt.y())) {
      continue;
    }
    // 地面点 & 非地面点分别累帧
    int gx = xToGrid(pt.x());
    int gy = yToGrid(pt.y());
    auto &label = cloud_ptr->label(pi);
    if (label == kGroundPt) {
      if (!gnd_valid_map_(gx, gy)) {
        gnd_valid_map_(gx, gy) = 1;
        gnd_gauss_map_(gx, gy).init(pt);
      } else {
        gnd_gauss_map_(gx, gy).update(pt);
      }
    } else if (label == kCommonPt) {
      if (!nongnd_valid_map_(gx, gy)) {
        nongnd_valid_map_(gx, gy) = 1;
        nongnd_gauss_map_(gx, gy).init(pt);
      } else {
        nongnd_gauss_map_(gx, gy).update(pt);
      }
    }
  }
}

void GroundDetection::extractSeeds() {
  Eigen::Matrix3d eigen_mat;   // 特征向量
  Eigen::Vector3d eigen_vals;  // 特征值
  gnd_height_map_.setZero();
  if (debug_normal_) {
    gnd_points_.clear();
    gnd_normals_.clear();
  }
  int seeds_cnt{0};
  float thickness{0.f}, curvature{0.f};
  for (int gx = 0; gx < x_size_; ++gx) {
    for (int gy = 0; gy < y_size_; ++gy) {
      if (!gnd_valid_map_(gx, gy)) {
        continue;
      }
      // 过滤接地障碍物
      const auto &gnd_gauss = gnd_gauss_map_(gx, gy);
      if (nongnd_valid_map_(gx, gy) &&
          gnd_gauss.getDiffZWith(nongnd_gauss_map_(gx, gy)) < 0.3) {
        gnd_valid_map_(gx, gy) = 0;
        continue;
      }
      // 种子点初筛: 点数&高度差
      if (gnd_gauss.size() < 3 || gnd_gauss.getDiffZ() > height_th_) {
        gnd_valid_map_(gx, gy) = 0;
        continue;
      }
      // 种子点复筛: 曲率&法向量
      Eigen::Matrix3d cov_mat = gnd_gauss.getCovMat();
      calEigenInfo(cov_mat, eigen_vals, eigen_mat);
      // if (getCurvature(eigen_vals) > curvature_th_ ||
      //     std::abs(getNormalVector(eigen_mat).z()) < cos_normal_th_) {
      //   gnd_valid_map_(gx, gy) = 0;
      //   continue;
      // }
      // 调试法向量
      if (debug_normal_) {
        // Eigen::Matrix3d cov_mat = gnd_gauss.getCovMat();
        // calEigenInfo(cov_mat, eigen_vals, eigen_mat);
        // Eigen::Vector3d normal_vec = getNormalVector(eigen_mat);
        Eigen::Vector3d gnd_pt = gnd_gauss.getCentroid();
        Eigen::Vector3d normal_vec{0, 0, 1};
        gnd_points_.emplace_back(gnd_pt);
        gnd_normals_.emplace_back(normal_vec);
      }
      // 地面种子栅格
      gnd_height_map_(gx, gy) = gnd_gauss.getMeanZ();
      seeds_cnt += 1;
      thickness += gnd_gauss.getDiffZ();
      curvature += getCurvature(eigen_vals);
    }
  }
  // 通过平均厚度判断雨天
  if (seeds_cnt > 0) {
    thickness /= seeds_cnt;
    curvature /= seeds_cnt;
    if (thickness > rainy_thickness_ && curvature > rainy_curvature_) {
      rainy_scene_cnt_ = std::min(rainy_scene_cnt_ + 1, rainy_scene_hold_cnt_);
    } else {
      rainy_scene_cnt_ = std::max(rainy_scene_cnt_ - 1, 0);
    }
  }
  bool is_gnd_rainy = isGroundRainy();
  auto &ogm = OccGridMap::GetInstance();
  ogm.is_rainy_ |= is_gnd_rainy;
  AINFO << "GndSeeds[" << seeds_cnt << "] GndThick[" << thickness
        << "] GndCurvature[" << curvature << "] GndCnt[" << rainy_scene_cnt_
        << "] GndRainy[" << is_gnd_rainy << "] Wiper[" << ogm.wiper_mode_
        << "] WiperCnt[" << ogm.rainy_scene_cnt_ << "] Rainy[" << ogm.is_rainy_
        << "]";
}

void GroundDetection::groundInterpolation() {
  // 计算积分图
  for (int gx = 0; gx < x_size_; ++gx) {
    uint32_t y_sum_count = 0;
    double y_sum_height = 0;
    for (int gy = 0; gy < y_size_; ++gy) {
      y_sum_count += gnd_valid_map_(gx, gy);
      y_sum_height += gnd_height_map_(gx, gy);
      integral_count_map_(gx, gy) = y_sum_count;
      integral_height_map_(gx, gy) = y_sum_height;
      if (gx > 0) {
        integral_count_map_(gx, gy) += integral_count_map_(gx - 1, gy);
        integral_height_map_(gx, gy) += integral_height_map_(gx - 1, gy);
      }
    }
  }
  // 非种子栅格插值
  for (int gx = 0; gx < x_size_; ++gx) {
    for (int gy = 0; gy < y_size_; ++gy) {
      int index = gx * y_size_ + gy;
      if (gnd_valid_map_(gx, gy)) {
        ground_map_.map_data[index] = mapToU8(gnd_height_map_(gx, gy));
        continue;
      }
      // 二分查找包含种子点的最小半径
      int min_r = 0;
      int max_r = std::max(std::max(gx, x_size_ - 1 - gx),
                           std::max(gy, y_size_ - 1 - gy));
      while (min_r != max_r) {
        int mid_r = (max_r - min_r) / 2 + min_r;
        auto count = getAreaSum(integral_count_map_, gx, gy, mid_r);
        if (count > 0) {
          max_r = mid_r;
        } else {
          min_r = mid_r + 1;
        }
      }
      // 环形差分加权
      double sum_count{0}, sum_height{0};
      double prev_count{0}, prev_height{0};
      max_r = std::min(min_r + 3, static_cast<int>(weight_lut_.size()));
      for (int r = min_r; r < max_r; ++r) {
        double count = getAreaSum(integral_count_map_, gx, gy, r);
        double height = getAreaSum(integral_height_map_, gx, gy, r);
        sum_count += weight_lut_[r] * (count - prev_count);
        sum_height += weight_lut_[r] * (height - prev_height);
        prev_count = count;
        prev_height = height;
      }
      if (!isNearZero(sum_count)) {
        gnd_height_map_(gx, gy) = sum_height / sum_count;
      }
      ground_map_.map_data[index] = mapToU8(gnd_height_map_(gx, gy));
    }
  }
}

void GroundDetection::filterGround() {
  double height_th =
      OccGridMap::GetInstance().is_rainy_ ? rainy_height_th_ : height_th_;
  for (const auto &pi : cloud_ptr->roi_idx) {
    const auto &pt = cloud_ptr->ego_pt(pi);
    auto &label = cloud_ptr->label(pi);
    if (!inRoi(pt.x(), pt.y()) || label == kMistPt) {
      continue;
    }
    int gx = xToGrid(pt.x());
    int gy = yToGrid(pt.y());
    double h_from_gnd = pt.z() - gnd_height_map_(gx, gy);
    if (h_from_gnd < -height_th) {
      label = kGhostPt;
    } else if (h_from_gnd < height_th) {
      label = kGroundPt;
    } else if (label == kGroundPt) {
      label = kCommonPt;
    }
  }
  cloud_ptr->ghostFilter();
}

void GroundDetection::debugNormals(const CloudData::Ptr &debug_cloud_ptr) {
  for (size_t i = 0; i < gnd_points_.size(); i++) {
    // xyz
    const auto &ego_pt = gnd_points_[i];
    debug_cloud_ptr->ego_points.emplace_back(ego_pt);
    // normal
    const auto &ego_normal = gnd_normals_[i];
    PointProperty pp;
    pp.intensity = static_cast<uint8_t>((ego_normal.x() + 1) / 2 * 255);
    pp.ring = static_cast<uint8_t>((ego_normal.y() + 1) / 2 * 255);
    pp.depth = static_cast<uint8_t>((ego_normal.z() + 1) / 2 * 255);
    debug_cloud_ptr->infos.emplace_back(pp);
  }
}

}  // namespace perception
}  // namespace robosense
