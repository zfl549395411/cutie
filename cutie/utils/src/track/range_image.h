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

#ifndef HYPER_VISION_PERCEPTION_COMMON_DATA_RANGE_IMAGE_H
#define HYPER_VISION_PERCEPTION_COMMON_DATA_RANGE_IMAGE_H

#include <array>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "hyper_vision/perception/common/utils/msg_types.hpp"
#include "hyper_vision/perception/common/utils/perception_framework_utils.hpp"
#include "rally/utils/utils.h"

namespace robosense {
namespace perception {

// 查表法计算asin
class AsinLookupTable {
 public:
  static AsinLookupTable& GetInstanceRad() {
    static AsinLookupTable asin_table(false);
    return asin_table;
  }

  static AsinLookupTable& GetInstanceDeg() {
    static AsinLookupTable asin_table(true);
    return asin_table;
  }

  AsinLookupTable(const bool to_deg) {
    const int table_size = 100001;
    const double rad_to_deg = 180.0 / M_PI;
    res_ = 2.0 / (table_size - 1);
    res_inv_ = (table_size - 1) / 2.0;
    table_.resize(table_size);
    for (int i = 0; i < table_size; ++i) {
      double sin_value = -1.0 + (i + 0.5) * res_;  // 生成[-1, 1]区间的sin
      sin_value = sin_value > 1 ? 1 : sin_value;   // 截断
      table_[i] = std::asin(sin_value);            // 计算asin(x)
      if (to_deg) table_[i] *= rad_to_deg;         // 弧度转角度
    }
  }

  inline double asin(const double& sin_value) const {
    int index = static_cast<int>((sin_value + 1.0) * res_inv_);
    return table_[index];
  }

 private:
  double res_;                 // sin分辨率
  double res_inv_;             // sin分辨率倒数
  std::vector<double> table_;  // 存储asin值的查找表
};

struct RangeImageConfig {
  // {yaw angle resolution, pitch angle resolution}
  std::array<float, 2> voxel_resolution{0.2f, 0.2f};
  // {angle upper bound, angle lower bound}
  std::array<float, 2> yaw_range = {90.0f, -90.0f};
  std::array<float, 2> pitch_range = {60.0f, -60.0f};

  // 用于开辟空间，不同雷达型号的点云点数不同
  int max_points_num = 78750;

  Eigen::Affine3f vehicle_axis_to_lidar_pose;
};
// 激光坐标系
class RangeImage {
 public:
  using Ptr = std::shared_ptr<RangeImage>;
  explicit RangeImage(const RangeImageConfig& options);
  void UpdateRangeImage(int rangeimage_pts, int voxel_offset_size,
                        void* point_save_data, void* voxel_offset_data,
                        void* effective_boundary_data);

  // 输入激光坐标系
  void GetLidarCoordinate(const Eigen::Vector3f& points, int& x, int& y);
  bool GetLidarCoordinate(const Eigen::Vector3f& points, int& x, int& y,
                          bool limit);
  // 输入车体坐标系
  void GetVehCoordinate(const Eigen::Vector3f& points, int& x, int& y);

  // 根据栅格索引获取点云索引
  void GetPointIndex(const int& row, const int& col,
                     std::vector<int>& point_idx);

  struct range_image_t {
    uint32_t point_idx;
    uint32_t voxel_index;
  };
  uint64_t timestamp;

  // rangeimage根据预设分辨率，计算出长宽像素数
  std::array<uint32_t, 2> voxel_shape_;  // {yaw_size, pitch_size} -> {col,row}
  // rangeimage总像素数
  uint32_t voxel_map_size_;
  // 按所属voxel顺序排列的点云索引
  std::vector<range_image_t> point_save_;  // {yaw_size, pitch_size}
  // voxel顺序排列的poist_save索引偏移量
  std::unique_ptr<uint32_t[]> voxel_offset_;
  // voxel_offset中实际存储的真实边界
  std::array<int32_t, 4> effective_boundary_;  // {c_min, c_max, r_min, r_max}
  LidarPointCloud::Ptr points_ptr_ = nullptr;

  void DrawImage(const std::string& path);
  void DrawImage(const std::string& path,
                 const std::vector<std::vector<cv::Point>>& points,
                 const std::vector<cv::Point>& proj_points);

  void calcRangeImage(const LidarPointCloud::Ptr& cloud_ptr,
                      const bool is_lidar);
  void calcRangeImage(const std::vector<Eigen::Vector3d>& points);
  void SetPointCloudPtr(const LidarPointCloud::Ptr& points_ptr);
  inline bool HaveLidar() { return points_ptr_ != nullptr; }

 private:
  void RangeImageInit(const RangeImageConfig& options);
  // 像素/°
  std::array<float, 2> voxel_resolution_inv_;  // {yaw_res, pitch_res}

  // rangeimage0点偏移量，用于将负值索引校正至图像0点
  int32_t yaw_shift_, pitch_shift_;
  // 用于本地计算的中间变量，记录每个voxel有多少点
  std::unique_ptr<uint16_t[]> voxel_num_;
  // 用于本地计算的中间变量，按顺序记录存储点云->rangeimage转换映射
  std::vector<range_image_t> point_reserve_;  // {yaw_size, pitch_size}
  // 用于本地计算的中间变量，记录每个点云所在的行列
  std::vector<int32_t> points_row_, points_col_;  // max_points_num

  RangeImageConfig options_;

  Eigen::Affine3f veh_to_lidar_affine_;
};

class RangeImageCacheCenter {
 public:
  bool enable = false;
  static RangeImageCacheCenter& GetInstance() {
    static RangeImageCacheCenter range_image_center;
    return range_image_center;
  }

  void Init(const RangeImageConfig& options, const int time_thresh,
            const int max_cache_len = 5) {
    if (is_inited_) {
      AINFO << "RangeImageCacheCenter has inited";
      return;
    }
    std::unique_lock<std::mutex> lck(mtx_);
    is_inited_ = true;
    max_cache_len_ = max_cache_len;
    time_thresh_ = time_thresh;
    range_image_caches_.resize(max_cache_len);
    lidar_points_caches_.resize(max_cache_len);
    for (auto& object : range_image_caches_) {
      std::get<0>(object) = 0;
      std::get<1>(object) = std::make_shared<RangeImage>(options);
      std::get<2>(object) = std::make_unique<std::mutex>();
    }
    for (auto& cloud : lidar_points_caches_) {
      std::get<0>(cloud) = 0;
      std::get<1>(cloud).reset(new LidarPointCloud);
      std::get<2>(cloud) = std::make_unique<std::mutex>();
    }
    range_image_cahces_save_idx_ = 0;
    lidar_points_caches_save_idx_ = 0;
  }
  // 获取保存对象及其索引，并对该内存上锁
  bool AcquireSaveRangeImagePtr(const int64_t timestamp, int& out_idx,
                                RangeImage::Ptr& out_msg) {
    if (!is_inited_) {
      AERROR << "RangeImageCacheCenter has not inited";
      return false;
    }
    std::unique_lock<std::mutex> lck(mtx_);
    // 阻塞锁
    std::get<2>(range_image_caches_[range_image_cahces_save_idx_])->lock();

    out_idx = range_image_cahces_save_idx_;
    std::get<0>(range_image_caches_[range_image_cahces_save_idx_]) = timestamp;
    out_msg = std::get<1>(range_image_caches_[range_image_cahces_save_idx_]);
    ++range_image_cahces_save_idx_;
    range_image_cahces_save_idx_ %= max_cache_len_;
    return true;
  }
  // 根据索引释放对象，并解锁
  bool ReleaseSaveRangeImagePtr(const int64_t timestamp, const int& idx) {
    if (!is_inited_) {
      AERROR << "RangeImageCacheCenter has not inited";
      return false;
    }
    std::unique_lock<std::mutex> lck(mtx_);
    if (idx < 0 || idx >= max_cache_len_) {
      AERROR << "rangeimage save idx is out of range";
      return false;
    }
    if (std::get<0>(range_image_caches_[idx]) != timestamp) {
      AERROR << "rangeimage save timestamp is not match";
      return false;
    }
    // 解锁
    std::get<2>(range_image_caches_[idx])->unlock();
    return true;
  }
  // 获取保存点云及其索引，并对该内存上锁
  bool AcquireSavePointCloudPtr(const int64_t timestamp, int& out_idx,
                                LidarPointCloud::Ptr& out_msg) {
    if (!is_inited_) {
      AERROR << "RangeImageCacheCenter has not inited";
      return false;
    }
    std::unique_lock<std::mutex> lck(mtx_);
    // 阻塞锁
    std::get<2>(lidar_points_caches_[lidar_points_caches_save_idx_])->lock();
    out_idx = lidar_points_caches_save_idx_;
    std::get<0>(lidar_points_caches_[lidar_points_caches_save_idx_]) =
        timestamp;
    out_msg = std::get<1>(lidar_points_caches_[lidar_points_caches_save_idx_]);
    ++lidar_points_caches_save_idx_;
    lidar_points_caches_save_idx_ %= max_cache_len_;
    return true;
  }
  //   // 根据索引释放对象，并解锁
  bool ReleaseSavePointCloudPtr(const int64_t timestamp, const int& idx,
                                const bool cal_range_image) {
    if (!is_inited_) {
      AERROR << "RangeImageCacheCenter has not inited";
      std::get<2>(lidar_points_caches_[idx])->unlock();
      return false;
    }
    std::unique_lock<std::mutex> lck(mtx_);
    if (idx < 0 || idx >= max_cache_len_) {
      AERROR << "lidar save idx is out of range";
      std::get<2>(lidar_points_caches_[idx])->unlock();
      return false;
    }
    if (std::get<0>(lidar_points_caches_[idx]) != timestamp) {
      AERROR << "lidar save timestamp is not match";
      std::get<2>(lidar_points_caches_[idx])->unlock();
      return false;
    }
    if (cal_range_image) {
      std::get<2>(range_image_caches_[range_image_cahces_save_idx_])->lock();
      std::get<0>(range_image_caches_[range_image_cahces_save_idx_]) =
          timestamp;
      std::get<1>(range_image_caches_[range_image_cahces_save_idx_])
          ->calcRangeImage(std::get<1>(lidar_points_caches_[idx]), false);
      std::get<2>(range_image_caches_[range_image_cahces_save_idx_])->unlock();
      ++range_image_cahces_save_idx_;
      range_image_cahces_save_idx_ %= max_cache_len_;
    }
    // 解锁
    std::get<2>(lidar_points_caches_[idx])->unlock();
    return true;
  }

  // 根据时间戳获取rangeimage，并在lidar缓存中匹配点云
  bool GetRangeImage(const int64_t timestamp, RangeImage::Ptr& msg_ptr,
                     int64_t& find_lidar_time) {
    std::unique_lock<std::mutex> lck(mtx_);
    if (!is_inited_ || range_image_caches_.size() == 0) {
      AERROR << "RangeImageCacheCenter has not inited";
      return false;
    }
    for (int i = range_image_cahces_save_idx_ - 1;
         i != range_image_cahces_save_idx_; --i) {
      if (i < 0) {
        i = max_cache_len_ - 1;
        if (i == range_image_cahces_save_idx_) {
          return false;
        }
      }

      int64_t msg_time = std::get<0>(range_image_caches_[i]);
      find_lidar_time = msg_time;
      // 不是最近的，是阈值内最新的
      if (abs(msg_time - timestamp) < time_thresh_) {
        std::unique_lock<std::mutex> lck(*std::get<2>(range_image_caches_[i]));
        msg_ptr = std::get<1>(range_image_caches_[i]);
        if (msg_ptr->HaveLidar()) {
          return true;
        } else {
          for (int j = lidar_points_caches_save_idx_ - 1;
               j != lidar_points_caches_save_idx_; --j) {
            if (j < 0) {
              j = max_cache_len_ - 1;
              if (j == lidar_points_caches_save_idx_) {
                return false;
              }
            }
            int64_t lidar_time = std::get<0>(lidar_points_caches_[j]);
            if (abs(lidar_time - msg_time) < 100) {
              std::unique_lock<std::mutex> lck(
                  *std::get<2>(lidar_points_caches_[j]));
              msg_ptr->SetPointCloudPtr(std::get<1>(lidar_points_caches_[j]));
              return true;
            }
          }
          return false;
        }
      }
    }
    return false;
  }

 private:
  RALLY_DISALLOW_COPY_AND_ASSIGN(RangeImageCacheCenter)
  RangeImageCacheCenter() {}
  ~RangeImageCacheCenter() {}

  // 锁住整个center不允许外部访问
  std::mutex mtx_;
  bool is_inited_ = false;
  int max_cache_len_;
  int time_thresh_ = 15000;
  int range_image_cahces_save_idx_ = 0;
  int lidar_points_caches_save_idx_ = 0;
  std::vector<std::tuple<int64_t, RangeImage::Ptr, std::unique_ptr<std::mutex>>>
      range_image_caches_;
  std::vector<
      std::tuple<int64_t, LidarPointCloud::Ptr, std::unique_ptr<std::mutex>>>
      lidar_points_caches_;
};

}  // namespace perception
}  // namespace robosense

#endif  // HYPER_VISION_PERCEPTION_COMMON_DATA_RANGE_IMAGE_H
