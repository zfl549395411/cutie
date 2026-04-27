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
#include "hyper_vision/perception/common/data/range_image.h"

#include <fstream>

#include "pcl/io/pcd_io.h"
namespace robosense {
namespace perception {

void RangeImage::SetPointCloudPtr(const LidarPointCloud::Ptr &points_ptr) {
  points_ptr_ = points_ptr;
}
RangeImage::RangeImage(const RangeImageConfig &options) {
  RangeImageInit(options);
  options_ = options;
}
void RangeImage::RangeImageInit(const RangeImageConfig &options) {
  RENSURE(options.yaw_range[0] > options.yaw_range[1] &&
          options.pitch_range[0] > options.pitch_range[1]);
  RENSURE(options.voxel_resolution[0] > 0.f &&
          options.voxel_resolution[1] > 0.f);

  voxel_resolution_inv_[0] = 1.0 / options.voxel_resolution[0];
  voxel_resolution_inv_[1] = 1.0 / options.voxel_resolution[1];

  yaw_shift_ = static_cast<int>(
      std::ceil(options.yaw_range[1] * voxel_resolution_inv_[0]));
  pitch_shift_ = static_cast<int>(
      std::ceil(options.pitch_range[1] * voxel_resolution_inv_[1]));

  int32_t yaw_left = static_cast<int>(
      std::ceil(options.yaw_range[0] * voxel_resolution_inv_[0]));
  int32_t pitch_up = static_cast<int>(
      std::ceil(options.pitch_range[0] * voxel_resolution_inv_[1]));

  voxel_shape_[0] = (yaw_left - yaw_shift_) + 1;
  voxel_shape_[1] = (pitch_up - pitch_shift_) + 1;

  voxel_map_size_ = voxel_shape_[0] * voxel_shape_[1];
  voxel_num_ = std::make_unique<uint16_t[]>(voxel_map_size_);
  voxel_offset_ = std::make_unique<uint32_t[]>(voxel_map_size_ + 1);

  points_row_.resize(options.max_points_num);
  points_col_.resize(options.max_points_num);
  point_save_.resize(options.max_points_num);
  point_reserve_.reserve(options.max_points_num);
  veh_to_lidar_affine_ = options.vehicle_axis_to_lidar_pose;
  // col row
  effective_boundary_ = {0, voxel_shape_[0], 0, voxel_shape_[1]};
  // arcsin quick-calculation
  AsinLookupTable::GetInstanceDeg();
}
void RangeImage::GetLidarCoordinate(const Eigen::Vector3f &points, int &x,
                                    int &y) {
  auto yaw_max = options_.yaw_range[0];
  auto yaw_min = options_.yaw_range[1];
  auto pitch_max = options_.pitch_range[0];
  auto pitch_min = options_.pitch_range[1];
  auto res_yaw_inv = voxel_resolution_inv_[0];
  auto res_pitch_inv = voxel_resolution_inv_[1];
  float rad_to_deg = 180.0f / M_PI;

  auto xl = points.x();
  auto yl = points.y();
  auto zl = points.z();
  if (fabs(xl) < 0.01) {
    x = 0;
    y = 0;
    return;
  }
  auto yaw = std::asin(yl / std::sqrt(xl * xl + yl * yl)) * rad_to_deg;
  auto pitch =
      std::asin(zl / std::sqrt(xl * xl + yl * yl + zl * zl)) * rad_to_deg;

  if (!(yaw <= yaw_max && yaw >= yaw_min && pitch <= pitch_max &&
        pitch >= pitch_min)) {
    x = 0;
    y = 0;
    return;
  }

  x = static_cast<int>((yaw_max - yaw) * res_yaw_inv);
  y = static_cast<int>((pitch_max - pitch) * res_pitch_inv);
  x -= effective_boundary_[0];
  y -= effective_boundary_[2];
}
bool RangeImage::GetLidarCoordinate(const Eigen::Vector3f &points, int &x,
                                    int &y, bool limit) {
  const auto &yaw_max = options_.yaw_range[0];
  const auto &yaw_min = options_.yaw_range[1];
  const auto &pitch_max = options_.pitch_range[0];
  const auto &pitch_min = options_.pitch_range[1];
  const auto &res_yaw_inv = voxel_resolution_inv_[0];
  const auto &res_pitch_inv = voxel_resolution_inv_[1];
  const auto &xl = points.x();
  const auto &yl = points.y();
  const auto &zl = points.z();
  float rad_to_deg = 180.0f / M_PI;

  auto xy_squre_sum = xl * xl + yl * yl;
  if (xy_squre_sum < 0.01) {
    return false;
  }

  auto yaw = std::asin(yl / std::sqrt(xy_squre_sum)) * rad_to_deg;
  auto pitch = std::asin(zl / std::sqrt(xy_squre_sum + zl * zl)) * rad_to_deg;

  x = static_cast<int>((yaw_max - yaw) * res_yaw_inv);
  y = static_cast<int>((pitch_max - pitch) * res_pitch_inv);
  x -= effective_boundary_[0];
  y -= effective_boundary_[2];

  int range_x = effective_boundary_[1] - effective_boundary_[0];
  int range_y = effective_boundary_[3] - effective_boundary_[2];

  // debug
  // if (x < 0 || x > range_x || y < 0 || y > range_y) {
  //   std::cout << "lidar_pt: " << xl << " " << yl << " " << zl << std::endl;
  //   std::cout << "yaw: " << yaw << ", pitch: " << pitch << std::endl;
  //   std::cout << "rangeimg_x: " << x << ", rangeimg_y: " << y << std::endl;
  //   std::cout << "boundary_x: " << range_x << ", boundary_y: " << range_y <<
  //   std::endl;
  // }

  // 限幅 or 检查
  if (limit) {
    if (x < 0) {
      x = 0;
    } else if (x > range_x) {
      x = range_x;
    }
    if (y < 0) {
      y = 0;
    } else if (y > range_y) {
      y = range_y;
    }
  } else {
    if (x < 0 || x > range_x || y < 0 || y > range_y) {
      return false;
    }
  }

  return true;
}

// 输入车体坐标系
void RangeImage::GetVehCoordinate(const Eigen::Vector3f &points, int &x,
                                  int &y) {
  Eigen::Vector3f lidar_points = veh_to_lidar_affine_ * points;
  GetLidarCoordinate(lidar_points, x, y);
}

void RangeImage::calcRangeImage(const LidarPointCloud::Ptr &cloud_ptr,
                                const bool is_lidar) {
  points_ptr_ = cloud_ptr;
  const auto &affine3f = options_.vehicle_axis_to_lidar_pose;
  auto yaw_max = options_.yaw_range[0];
  auto yaw_min = options_.yaw_range[1];
  auto pitch_max = options_.pitch_range[0];
  auto pitch_min = options_.pitch_range[1];
  auto res_yaw_inv = voxel_resolution_inv_[0];
  auto res_pitch_inv = voxel_resolution_inv_[1];

  std::memset(voxel_num_.get(), 0, voxel_map_size_ * sizeof(uint16_t));

  // float deg_to_rad = M_PI / 180.0f;
  float rad_to_deg = 180.0f / M_PI;

  point_reserve_.clear();
  // sensor axis
  for (size_t i = 0; i < cloud_ptr->points.size(); ++i) {
    const auto &pt = cloud_ptr->points[i];
    Eigen::Vector3f e_pt;
    e_pt << pt.x, pt.y, pt.z;

    auto xl = e_pt.x();
    auto yl = e_pt.y();
    auto zl = e_pt.z();
    // auto a = Q_rsqrt(xl*xl + yl*yl);
    // auto b = Q_rsqrt(xl*xl + yl*yl + zl*zl);

    // auto yaw = std::asin(yl*a);
    // auto pitch = std::asin(zl*b);
    int c = 0, r = 0;
    if (is_lidar) {
      GetLidarCoordinate(e_pt, c, r);
    } else {
      GetVehCoordinate(e_pt, c, r);
    }
    if (c == 0 && r == 0) {
      continue;
    }
    if (c < 0 || r < 0) {
      continue;
    }

    auto index = static_cast<uint32_t>(voxel_shape_[0] * r + c);
    voxel_num_[index]++;
    range_image_t tmp = {static_cast<uint32_t>(i), index};
    point_reserve_.emplace_back(tmp);
  }

  uint32_t index_recd = 0;
  for (uint32_t i = 0; i < voxel_map_size_; ++i) {
    voxel_offset_[i] = index_recd;
    index_recd += voxel_num_[i];
  }
  voxel_offset_[voxel_map_size_] = index_recd;

  uint32_t size = point_reserve_.size();
  point_save_.resize(size);
  for (uint32_t i = 0; i < size; ++i) {
    auto &voxel = point_reserve_[i];
    const auto index =
        voxel_offset_[voxel.voxel_index + 1] - voxel_num_[voxel.voxel_index];
    point_save_[index].point_idx = voxel.point_idx;
    point_save_[index].voxel_index = voxel.voxel_index;
    voxel_num_[voxel.voxel_index]--;
  }

#if 0
  auto compareCoordinates = [] (const RangeImage::range_image_t& coord1,
    const RangeImage::range_image_t& coord2) {
    if (coord1.voxel_index != coord2.voxel_index) {
        return coord1.voxel_index < coord2.voxel_index;
    }
    return coord1.xl < coord2.xl;
  };
  std::sort(point_save_.begin(), point_save_.end(), compareCoordinates);
#else
  //   auto compareCoordinates = [](const RangeImage::range_image_t &coord1,
  //                                const RangeImage::range_image_t &coord2) {
  //     return false;
  //   };

  //   uint32_t test = 0;
  //   int start_index = 0;
  //   for (uint32_t i = 1; i < size; ++i) {
  //     int end_index = i;
  //     if (point_save_[start_index].voxel_index !=
  //         point_save_[end_index].voxel_index) {
  //       auto start_iter = point_save_.begin() + start_index;
  //       auto end_iter = point_save_.begin() + end_index;

  //       // 对指定范围进行排序
  //       if (end_index > start_index + 1) {
  //         std::sort(start_iter, end_iter, compareCoordinates);
  //       }
  //       start_index = end_index;
  //     }
  //   }

#endif
  effective_boundary_ = {0, voxel_shape_[0] - 1, 0, voxel_shape_[1] - 1};
}
void RangeImage::calcRangeImage(const std::vector<Eigen::Vector3d> &points) {
  auto yaw_max = options_.yaw_range[0];
  auto yaw_min = options_.yaw_range[1];
  auto pitch_max = options_.pitch_range[0];
  auto pitch_min = options_.pitch_range[1];
  auto res_yaw_inv = voxel_resolution_inv_[0];
  auto res_pitch_inv = voxel_resolution_inv_[1];

  // effective_boundary
  effective_boundary_[0] = INT32_MAX;
  effective_boundary_[1] = 0;
  effective_boundary_[2] = INT32_MAX;
  effective_boundary_[3] = 0;

  // project cloud
  const auto &asin_lut = AsinLookupTable::GetInstanceDeg();
  for (size_t pi = 0; pi < points.size(); pi++) {
    const auto &pt = points[pi];
    double yaw = asin_lut.asin(pt.y() / pt.head(2).norm());
    double pitch = asin_lut.asin(pt.z() / pt.norm());
    if (!(yaw <= yaw_max && yaw >= yaw_min && pitch <= pitch_max &&
          pitch >= pitch_min)) {
      points_row_[pi] = -1;
      continue;
    }
    int32_t row = static_cast<int32_t>((pitch_max - pitch) * res_pitch_inv);
    int32_t col = static_cast<int32_t>((yaw_max - yaw) * res_yaw_inv);
    points_row_[pi] = row;
    points_col_[pi] = col;
    effective_boundary_[0] = std::min(effective_boundary_[0], col);
    effective_boundary_[1] = std::max(effective_boundary_[1], col);
    effective_boundary_[2] = std::min(effective_boundary_[2], row);
    effective_boundary_[3] = std::max(effective_boundary_[3], row);
  }

  // point_reserve & voxel_num
  point_reserve_.clear();
  std::memset(voxel_num_.get(), 0, voxel_map_size_ * sizeof(uint16_t));
  auto yaw_size = effective_boundary_[1] - effective_boundary_[0] + 1;
  auto pitch_size = effective_boundary_[3] - effective_boundary_[2] + 1;
  for (uint32_t pi = 0; pi < points.size(); pi++) {
    if (points_row_[pi] < 0) {
      continue;
    }
    int row = points_row_[pi] - effective_boundary_[2];
    int col = points_col_[pi] - effective_boundary_[0];
    uint32_t vi = static_cast<uint32_t>(row * yaw_size + col);
    voxel_num_[vi]++;
    range_image_t tmp = {pi, vi};
    point_reserve_.emplace_back(tmp);
  }

  // voxel_offset
  uint32_t index_recd = 0;
  uint32_t voxel_size = yaw_size * pitch_size;
  for (uint32_t i = 0; i < voxel_size; ++i) {
    voxel_offset_[i] = index_recd;
    index_recd += voxel_num_[i];
  }
  voxel_offset_[voxel_size] = index_recd;

  // point_save
  uint32_t size = point_reserve_.size();
  for (uint32_t i = 0; i < size; ++i) {
    const auto &voxel = point_reserve_[i];
    const auto index =
        voxel_offset_[voxel.voxel_index + 1] - voxel_num_[voxel.voxel_index];
    point_save_[index].point_idx = voxel.point_idx;
    point_save_[index].voxel_index = voxel.voxel_index;
    voxel_num_[voxel.voxel_index]--;
  }
}
void RangeImage::DrawImage(const std::string &path) {
  if (points_ptr_ == nullptr) {
    return;
  }
  std::array<uint32_t, 2> voxel_shape = {
      effective_boundary_[1] - effective_boundary_[0] + 1,
      effective_boundary_[3] - effective_boundary_[2] + 1};
  AINFO << "boundary " << effective_boundary_[0] << " "
        << effective_boundary_[1] << " " << effective_boundary_[2] << " "
        << effective_boundary_[3] << std::endl;
  uint32_t voxel_map_size = voxel_map_size_;
  auto &point_save = point_save_;  // {yaw_size, pitch_size}
  auto &voxel_offset = voxel_offset_;
  auto &cloud = points_ptr_->points;

  // depth
  cv::Mat range_img_dist =
      cv::Mat::zeros(voxel_shape[1], voxel_shape[0], CV_32F);
  cv::Mat range_img_intensity =
      cv::Mat::zeros(voxel_shape[1], voxel_shape[0], CV_32F);
  cv::Mat mask = cv::Mat::zeros(voxel_shape[1], voxel_shape[0], CV_32F);
  for (uint32_t i = 0; i < voxel_shape[1]; ++i) {
    for (uint32_t j = 0; j < voxel_shape[0]; ++j) {
      auto idx = i * voxel_shape[0] + j;
      if (voxel_offset[idx] == voxel_offset[idx + 1]) {
        continue;
      }
      auto index = point_save[voxel_offset[idx]].point_idx;
      auto &pt = cloud[index];
      auto depth = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
      //   auto start = point_save.begin() + voxel_offset[idx];
      //   auto end = point_save.begin() + voxel_offset[idx + 1];
      //   AINFO << idx << " " << voxel_offset[idx] << " "
      //             << voxel_offset[idx + 1] << std::endl;
      // depth
      if (depth > 100) {
        depth = 100;
      }
      if (mask.at<float>(i, j) == 0.0f) {
        range_img_dist.at<float>(i, j) = depth;
        range_img_intensity.at<float>(i, j) = pt.intensity;
        mask.at<float>(i, j) = 1.0;
      }
    }
  }
  // 归一化dist image
  cv::normalize(range_img_dist, range_img_dist, 0, 1, cv::NORM_MINMAX);
  //   cv::Mat onesMatrix = cv::Mat::ones(voxel_shape[1], voxel_shape[0],
  //   CV_32FC1); range_img_dist = onesMatrix - range_img_dist;
  //   range_img_dist.mul(mask);
  cv::multiply(range_img_dist, mask, range_img_dist);
  cv::Mat channel8U(voxel_shape[1], voxel_shape[0], CV_8UC3);
  for (uint32_t i = 0; i < voxel_shape[1]; ++i) {
    for (uint32_t j = 0; j < voxel_shape[0]; ++j) {
      auto depth = range_img_dist.at<float>(i, j);
      float r, g, b;
      if (depth == 0) {
        r = 0;
        g = 0;
        b = 0;
      } else if (depth < 0.25) {
        r = 0;
        g = 4 * depth;
        b = 1;
      } else if (depth < 0.5) {
        r = 0;
        g = 1;
        b = 4 * (0.5 - depth);
      } else if (depth < 0.75) {
        r = 4 * (depth - 0.5);
        g = 1;
        b = 0;
      } else {
        r = 1;
        g = 4 * (1 - depth);
        b = 0;
      }

      cv::Vec3b pixel((uint8_t)(b * 255), (uint8_t)(g * 255),
                      (uint8_t)(r * 255));
      channel8U.at<cv::Vec3b>(i, j) = pixel;
    }
  }
  cv::imwrite(path + "_dist.jpg", channel8U);
  // 归一化intensity image
  //   onesMatrix = cv::Mat::ones(voxel_shape[1], voxel_shape[0], CV_32FC1);
  //   range_img_intensity = onesMatrix - range_img_intensity;
  //   range_img_intensity.mul(mask);
  cv::multiply(range_img_intensity, mask, range_img_intensity);
  cv::Mat channel8U_intensity(voxel_shape[1], voxel_shape[0], CV_8UC3);
  for (uint32_t i = 0; i < voxel_shape[1]; ++i) {
    for (uint32_t j = 0; j < voxel_shape[0]; ++j) {
      auto intensity = range_img_intensity.at<float>(i, j);
      float r, g, b;
      if (intensity == 0) {
        r = 0;
        g = 0;
        b = 0;
      } else if (intensity < 50) {
        r = 0;
        g = 2 * intensity;
        b = 1;
      } else if (intensity < 100) {
        r = 0;
        g = 0;
        b = 2 * (100 - intensity);
      } else if (intensity < 150) {
        r = 2 * (intensity - 150);
        g = 1;
        b = 0;
      } else {
        r = 1;
        g = 4 * (1 - intensity);
        b = 0;
      }

      cv::Vec3b pixel((uint8_t)(b * 255), (uint8_t)(g * 255),
                      (uint8_t)(r * 255));
      channel8U_intensity.at<cv::Vec3b>(i, j) = pixel;
    }
  }
  cv::imwrite(path + "_intensity.jpg", channel8U_intensity);
}
void RangeImage::DrawImage(const std::string &path,
                           const std::vector<std::vector<cv::Point>> &points,
                           const std::vector<cv::Point> &proj_points) {
  if (points_ptr_ == nullptr) {
    return;
  }
  std::array<uint32_t, 2> voxel_shape = {
      effective_boundary_[1] - effective_boundary_[0] + 1,
      effective_boundary_[3] - effective_boundary_[2] + 1};
  AINFO << "boundary " << effective_boundary_[0] << " "
        << effective_boundary_[1] << " " << effective_boundary_[2] << " "
        << effective_boundary_[3] << std::endl;
  uint32_t voxel_map_size = voxel_map_size_;
  auto &point_save = point_save_;  // {yaw_size, pitch_size}
  auto &voxel_offset = voxel_offset_;
  auto &cloud = points_ptr_->points;

  // depth
  cv::Mat range_img_dist =
      cv::Mat::zeros(voxel_shape[1], voxel_shape[0], CV_32F);
  cv::Mat range_img_intensity =
      cv::Mat::zeros(voxel_shape[1], voxel_shape[0], CV_32F);
  cv::Mat mask = cv::Mat::zeros(voxel_shape[1], voxel_shape[0], CV_32F);
  for (uint32_t i = 0; i < voxel_shape[1]; ++i) {
    for (uint32_t j = 0; j < voxel_shape[0]; ++j) {
      std::vector<int> point_idx;
      GetPointIndex(i, j, point_idx);
      if (point_idx.empty()) {
        continue;
      }
      auto &pt = cloud[point_idx[0]];
      auto depth = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
      //   auto start = point_save.begin() + voxel_offset[idx];
      //   auto end = point_save.begin() + voxel_offset[idx + 1];
      //   AINFO << idx << " " << voxel_offset[idx] << " "
      //             << voxel_offset[idx + 1] << std::endl;
      // depth
      if (depth > 100) {
        depth = 100;
      }
      if (mask.at<float>(i, j) == 0.0f) {
        range_img_dist.at<float>(i, j) = depth;
        range_img_intensity.at<float>(i, j) = pt.intensity;
        mask.at<float>(i, j) = 1.0;
      }
    }
  }
  // 归一化dist image
  cv::normalize(range_img_dist, range_img_dist, 0, 1, cv::NORM_MINMAX);
  //   cv::Mat onesMatrix = cv::Mat::ones(voxel_shape[1], voxel_shape[0],
  //   CV_32FC1); range_img_dist = onesMatrix - range_img_dist;
  //   range_img_dist.mul(mask);
  cv::multiply(range_img_dist, mask, range_img_dist);
  cv::Mat channel8U(voxel_shape[1], voxel_shape[0], CV_8UC3);
  for (uint32_t i = 0; i < voxel_shape[1]; ++i) {
    for (uint32_t j = 0; j < voxel_shape[0]; ++j) {
      auto depth = range_img_dist.at<float>(i, j);
      float r, g, b;
      if (depth == 0) {
        r = g = b = 0;
      } else if (depth < 0.167) {
        r = 0;
        g = 6 * depth;
        b = 1;
      } else if (depth < 0.333) {
        r = 0;
        g = 1;
        b = 1 - 6 * (depth - 0.167);
      } else if (depth < 0.5) {
        r = 6 * (depth - 0.333);
        g = 1;
        b = 0;
      } else if (depth < 0.667) {
        r = 1;
        g = 1 - 6 * (depth - 0.5);
        b = 0;
      } else if (depth < 0.833) {
        r = 1;
        g = 0;
        b = 6 * (depth - 0.667);
      } else {
        r = 1 - 6 * (depth - 0.833);
        g = 0;
        b = 1;
      }

      cv::Vec3b pixel((uint8_t)(b * 255), (uint8_t)(g * 255),
                      (uint8_t)(r * 255));
      channel8U.at<cv::Vec3b>(i, j) = pixel;
    }
  }
  for (auto &p : points) {
    cv::rectangle(channel8U, p[0], p[1], cv::Scalar(0, 0, 255), 1);
  }
  //   for (auto &p : proj_points) {
  //     cv::circle(channel8U, p, 5, cv::Scalar(255, 0, 0), 1);
  //     cv::circle(channel8U, p, 10, cv::Scalar(0, 255, 0), 1);
  //   }
  cv::imwrite(path + "_dist.jpg", channel8U);

  cv::multiply(range_img_intensity, mask, range_img_intensity);
  cv::Mat channel8U_intensity(voxel_shape[1], voxel_shape[0], CV_8UC3);
  for (uint32_t i = 0; i < voxel_shape[1]; ++i) {
    for (uint32_t j = 0; j < voxel_shape[0]; ++j) {
      auto intensity = range_img_intensity.at<float>(i, j);
      float r, g, b;
      if (intensity <150) {
        r = 0;
        g = 0;
        b = 0;
      } else if (intensity < 50) {
        r = 0;
        g = 2 * intensity;
        b = 1;
      } else if (intensity < 100) {
        r = 0;
        g = 0;
        b = 2 * (100 - intensity);
      } else if (intensity < 150) {
        r = 2 * (intensity - 150);
        g = 1;
        b = 0;
      } else {
        r = 1;
        g = 4 * (1 - intensity);
        b = 0;
      }

      cv::Vec3b pixel((uint8_t)(b * 255), (uint8_t)(g * 255),
                      (uint8_t)(r * 255));
      channel8U_intensity.at<cv::Vec3b>(i, j) = pixel;
    }
  }
  for (auto &p : points) {
    cv::rectangle(channel8U_intensity, p[0], p[1], cv::Scalar(0, 0, 255), 1);
  }
//   for (auto &p : proj_points) {
//     cv::circle(channel8U_intensity, p, 5, cv::Scalar(255, 0, 0), 1);
//     cv::circle(channel8U_intensity, p, 10, cv::Scalar(0, 255, 0), 1);
//   }
  cv::imwrite(path + "_intensity.jpg", channel8U_intensity);
}
void RangeImage::UpdateRangeImage(int rangeimage_pts, int voxel_offset_size,
                                  void *point_save_data,
                                  void *voxel_offset_data,
                                  void *effective_boundary_data) {
  // no points
  points_ptr_ = nullptr;
  // get point_save
  point_save_.resize(rangeimage_pts);

  memcpy(point_save_.data(), point_save_data,
         sizeof(RangeImage::range_image_t) * point_save_.size());
  memcpy(voxel_offset_.get(), voxel_offset_data, voxel_offset_size);
  memcpy(effective_boundary_.data(), effective_boundary_data,
         sizeof(int32_t) * effective_boundary_.size());
}
void RangeImage::GetPointIndex(const int &row, const int &col,
                               std::vector<int> &point_idx) {
  point_idx.clear();
  std::array<uint32_t, 2> voxel_shape = {
      effective_boundary_[1] - effective_boundary_[0] + 1,
      effective_boundary_[3] - effective_boundary_[2] + 1};
  if (row >= voxel_shape[1] || col >= voxel_shape[0]) {
    return;
  }
  uint32_t voxel_map_size = voxel_map_size_;
  auto &point_save = point_save_;  // {yaw_size, pitch_size}
  auto &voxel_offset = voxel_offset_;

  auto idx = row * voxel_shape[0] + col;
  if (voxel_offset[idx] == voxel_offset[idx + 1]) {
    return;
  }
  auto start = point_save.begin() + voxel_offset[idx];
  auto end = point_save.begin() + voxel_offset[idx + 1];
  while (start != end) {
    point_idx.emplace_back(start->point_idx);
    ++start;
  }
  return;
}
}  // namespace perception
}  // namespace robosense
