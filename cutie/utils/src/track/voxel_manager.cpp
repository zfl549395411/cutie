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
#include "hyper_vision/perception/segformer/voxel_manager.h"
namespace robosense {
namespace perception {
bool VoxelData::Init(const YAML::Node& config) {
  rally::yamlRead(config, "voxel_resolution", voxel_resolution);

  std::vector<float> temp_size;
  rally::yamlRead(config, "voxel_size", temp_size);
  voxel_size = Vec3D(temp_size[0], temp_size[1], temp_size[2]);
  z_resolution = voxel_size.z / 128;
  voxel_num = Eigen::Vector3i(voxel_size.x / voxel_resolution,
                              voxel_size.y / voxel_resolution, 128);
  // 初始化为全局中点
  voxel_origin =
      Eigen::Vector3i(voxel_num.x() / 2, voxel_num.y() / 2, voxel_num.z() / 2);
  voxels.assign(voxel_num.x(), std::vector<std::bitset<Z_NUM>>(
                                   voxel_num.y(), std::bitset<Z_NUM>(0)));
  return true;
}
bool VoxelData::ReInit(const SegformerCommand& command) {
  voxel_resolution = command.xy_resolution;
  voxel_size.z = 128 * command.z_resolution;
  z_resolution = voxel_size.z / 128;
  voxel_num = Eigen::Vector3i(voxel_size.x / voxel_resolution,
                              voxel_size.y / voxel_resolution, 128);
  // 初始化为全局中点
  voxel_origin =
      Eigen::Vector3i(voxel_num.x() / 2, voxel_num.y() / 2, voxel_num.z() / 2);
  voxels.assign(voxel_num.x(), std::vector<std::bitset<Z_NUM>>(
                                   voxel_num.y(), std::bitset<Z_NUM>(0)));
  return true;
}
cv::Mat VoxelData::DrawVoxels() {
  auto getHeatMapColor = [](int count) -> cv::Vec3b {
    if (count == 0) return cv::Vec3b(255, 255, 255);  // 白色

    float ratio = static_cast<float>(count) / Z_NUM;

    // 使用对数缩放来增强低值的区分度
    float log_ratio = std::log1p(ratio * 10.0f) /
                      std::log1p(10.0f);  // 将0-1映射到0-1但对数缩放

    // AINFO << " COUNT: " << count << " ratio: " << ratio << " log_ratio: " <<
    // log_ratio;

    // 热力图
    if (log_ratio < 0.15f) {
      int b = 255;
      int g = static_cast<int>(1020 * log_ratio);
      return cv::Vec3b(b, g, 0);
    } else if (log_ratio < 0.35f) {
      int g = 255;
      int b = static_cast<int>(1020 * (0.5f - log_ratio));
      return cv::Vec3b(b, g, 0);
    } else if (log_ratio < 0.55f) {
      int r = static_cast<int>(1020 * (log_ratio - 0.5f));
      return cv::Vec3b(0, 255, r);
    } else {
      int g = static_cast<int>(1020 * (1.0f - log_ratio));
      return cv::Vec3b(0, g, 255);
    }
  };

  int voxel_width = voxel_size.x / voxel_resolution;
  int voxel_height = voxel_size.y / voxel_resolution;
  if (1 || debug_img.empty()) {
    debug_img =
        cv::Mat(voxel_height, voxel_width, CV_8UC3, cv::Scalar(255, 255, 255));
  }
  auto img_origin = voxel_to_img(voxel_origin.x(), voxel_origin.y());
  cv::circle(debug_img, img_origin, 5, cv::Scalar(0, 0, 255), -1);
  for (int x = 0; x < voxels.size(); ++x) {
    for (int y = 0; y < voxels[x].size(); ++y) {
      auto color = getHeatMapColor(voxels[x][y].count());
      if (voxels[x][y].count() != 0) {
        auto img_point = voxel_to_img(x, y);
        debug_img.at<cv::Vec3b>(img_point.y, img_point.x) = color;
      }
    }
  }
  return debug_img;
}
bool VoxelManager::Init(const YAML::Node& config) {
  YAML::Node roi_config;
  rally::yamlSubNode(config, "base_roi", roi_config);
  rally::yamlRead(config, "show_image", show_image_);
  roi_.Init(roi_config);
  voxel_data_.Init(config);

  // 初始化参考点，此处用的是中点，该中心构成的odom坐标系下roi包围盒必须能够被栅格完全覆盖
  voxel_ref_center.x() = (roi_.x_limit[1] + roi_.x_limit[0]) * 0.5;
  voxel_ref_center.y() = (roi_.y_limit[1] + roi_.y_limit[0]) * 0.5;
  return true;
}
void VoxelManager::UpdateVoxels(int64_t time_stamp,
                                const Eigen::Affine3d& new_veh_to_rel) {
  // 首帧
  if (time_stamp_ == 0) {
    time_stamp_ = time_stamp;
    veh_to_rel = new_veh_to_rel;
    // 首帧初始化为中心点位置
    center_rel_pose = voxel_ref_center;
    fist_frame_to_rel_pose = new_veh_to_rel;
    return;
  }
  auto t_0 = apollo::cyber::Time::Now();
  // 当前参考点位置
  Eigen::Vector3d cur_rel_pose = new_veh_to_rel * voxel_ref_center;
  Eigen::Vector3d cur_first_frame_pose =
      fist_frame_to_rel_pose.inverse() * cur_rel_pose;
  // 当前参考点在前一帧的栅格坐标系下的位置（新的中点要移动的位置）
  Eigen::Vector3i cur_voxel_center = OdomToVoxel(cur_rel_pose);
  AINFO << "pre rel pose " << center_rel_pose.transpose() << " CUR REL POSE "
        << cur_first_frame_pose.transpose() << "pre voxel center "
        << voxel_data_.voxel_origin.transpose() << " CUR VOXEL CENTER "
        << cur_voxel_center.transpose() << " odom pose "
        << cur_rel_pose.transpose();
  int cur_x_min = wrap(cur_voxel_center.x() - voxel_data_.voxel_num.x() * 0.5,
                       voxel_data_.voxel_num.x());
  int cur_y_min = wrap(cur_voxel_center.y() - voxel_data_.voxel_num.y() * 0.5,
                       voxel_data_.voxel_num.y());
  int cur_z_min = wrap(cur_voxel_center.z() - voxel_data_.voxel_num.z() * 0.5,
                       voxel_data_.voxel_num.z());

  // 前一帧的栅格边界
  int pre_x_min =
      wrap(voxel_data_.voxel_origin.x() - voxel_data_.voxel_num.x() * 0.5,
           voxel_data_.voxel_num.x());
  int pre_y_min =
      wrap(voxel_data_.voxel_origin.y() - voxel_data_.voxel_num.y() * 0.5,
           voxel_data_.voxel_num.y());
  int pre_z_min =
      wrap(voxel_data_.voxel_origin.z() - voxel_data_.voxel_num.z() * 0.5,
           voxel_data_.voxel_num.z());
  AINFO << " CUR  MIN " << cur_x_min << " " << cur_y_min << " " << cur_z_min
        << " pre min " << pre_x_min << " " << pre_y_min << " " << pre_z_min;
  // 更新x
  if (std::abs(cur_first_frame_pose.x() - center_rel_pose.x()) >
      voxel_data_.voxel_resolution) {
    int x_step = (cur_first_frame_pose.x() - center_rel_pose.x()) > 0 ? 1 : -1;
    for (int x = pre_x_min; x != cur_x_min;
         x = wrap(x + x_step, voxel_data_.voxel_num.x())) {
      for (int y = 0; y < voxel_data_.voxel_num.y(); ++y) {
        voxel_data_.Reset(x, y);
      }
    }
  }

  // 更新y
  if (std::abs(cur_first_frame_pose.y() - center_rel_pose.y()) >
      voxel_data_.voxel_resolution) {
    int y_step = (cur_first_frame_pose.y() - center_rel_pose.y()) > 0 ? 1 : -1;
    for (int x = 0; x < voxel_data_.voxel_num.x(); ++x) {
      for (int y = pre_y_min; y != cur_y_min;
           y = wrap(y + y_step, voxel_data_.voxel_num.y())) {
        voxel_data_.Reset(x, y);
      }
    }
  }
  // 更新z
  if (std::abs(cur_first_frame_pose.z() - center_rel_pose.z()) >
      voxel_data_.z_resolution) {
    int z_step = (cur_first_frame_pose.z() - center_rel_pose.z()) > 0 ? 1 : -1;
    std::bitset<VoxelData::Z_NUM> mask;
    for (int z = pre_z_min; z != cur_z_min;
         z = wrap(z + z_step, voxel_data_.voxel_num.z())) {
      mask.set(z);
    }
    for (int x = 0; x < voxel_data_.voxel_num.x(); ++x) {
      for (int y = 0; y < voxel_data_.voxel_num.y(); ++y) {
        auto& z_bit = voxel_data_.voxels[x][y];
        z_bit &= ~mask;
      }
    }
  }
  // 更新栅格信息
  time_stamp_ = time_stamp;
  veh_to_rel = new_veh_to_rel;
  center_rel_pose = cur_first_frame_pose;
  voxel_data_.voxel_origin = cur_voxel_center;
  auto t_1 = apollo::cyber::Time::Now();
  auto cur_processing_time = (t_1 - t_0).ToSecond();
  AINFO << "voxel update time: "
        << ", cur: " << cur_processing_time * 1000 << " ms";
  if (show_image_) {
    auto voxel_img = DrawRoi();
    cv::imshow("voxel", voxel_img);
    cv::waitKey(1);
  }
}
bool VoxelManager::SaveVoxel(const Eigen::Vector3d& point_base,
                             Eigen::Vector3i& point_voxel) {
  if (!roi_.InROI(point_base)) {
    return false;
  }
  point_voxel = BaseToVoxel(point_base);
  if (voxel_data_.Check(point_voxel.x(), point_voxel.y(), point_voxel.z())) {
    return false;
  } else {
    voxel_data_.Set(point_voxel.x(), point_voxel.y(), point_voxel.z());
    return true;
  }
}
Eigen::Vector3i VoxelManager::OdomToVoxel(const Eigen::Vector3d& point_odom) {
  // 归一化到自车所在栅格坐标系
  // 转换到初始坐标系
  Eigen::Vector3d point_in_first_frame =
      fist_frame_to_rel_pose.inverse() * point_odom;
  // 归一化到以参考点（box中心点为中心的栅格坐标系）
  point_in_first_frame = point_in_first_frame - voxel_ref_center;
  // 离散参考栅格
  float x_in_first_frame =
      point_in_first_frame.x() / voxel_data_.voxel_resolution;
  float y_in_first_frame =
      point_in_first_frame.y() / voxel_data_.voxel_resolution;
  float z_in_first_frame = point_in_first_frame.z() / voxel_data_.z_resolution;
  // 增加偏移量后的实际栅格地址
  float x = x_in_first_frame + voxel_data_.voxel_num.x() / 2;
  float y = y_in_first_frame + voxel_data_.voxel_num.y() / 2;
  float z = z_in_first_frame + voxel_data_.voxel_num.z() / 2;

  // 离散回环
  int x_int = wrap(static_cast<int>(x), voxel_data_.voxel_num.x());
  int y_int = wrap(static_cast<int>(y), voxel_data_.voxel_num.y());
  int z_int = wrap(static_cast<int>(z), voxel_data_.voxel_num.z());
  return Eigen::Vector3i(x_int, y_int, z_int);
}
Eigen::Vector3d VoxelManager::VoxelToOdom(const Eigen::Vector3i& point_voxel) {
  // 和OdomToVoxel不完全一样，OdomToVoxel是变到了以初始中心点所在的真实超大栅格中再wrap到存储空间
  // 但wrap的轮数信息丢失了，所以只能以当前参考点（局部逻辑虚拟图）为中心去恢复相对位置
  // 恢复在虚拟图中的位置
  auto unwrap = [&](int center_pose, int voxel_num, int pose) {
    int offset = center_pose - voxel_num / 2;
    return wrap(pose - offset, voxel_num);
  };
  // 以中心点为原点的离散栅格坐标系
  int x = unwrap(voxel_data_.voxel_origin.x(), voxel_data_.voxel_num.x(),
                 point_voxel.x()) -
          voxel_data_.voxel_num.x() / 2;
  int y = unwrap(voxel_data_.voxel_origin.y(), voxel_data_.voxel_num.y(),
                 point_voxel.y()) -
          voxel_data_.voxel_num.y() / 2;
  int z = unwrap(voxel_data_.voxel_origin.z(), voxel_data_.voxel_num.z(),
                 point_voxel.z()) -
          voxel_data_.voxel_num.z() / 2;

  // 恢复到连续空间,相对于当前参考点的偏移量
  float x_continuous = x * voxel_data_.voxel_resolution;
  float y_continuous = y * voxel_data_.voxel_resolution;
  float z_continuous = z * voxel_data_.z_resolution;
  // 恢复到以首帧为原点的栅格坐标系
  Eigen::Vector3d point_in_first_frame =
      center_rel_pose +
      Eigen::Vector3d(x_continuous, y_continuous, z_continuous);
  // 恢复到全局坐标系
  return fist_frame_to_rel_pose * point_in_first_frame;
}
Eigen::Vector3d VoxelManager::VoxelToBase(const Eigen::Vector3i& point_voxel) {
  return veh_to_rel.inverse() * VoxelToOdom(point_voxel);
}
Eigen::Vector3i VoxelManager::BaseToVoxel(const Eigen::Vector3d& point_base) {
  // 转到 odom 坐标
  Eigen::Vector3d point_rel = veh_to_rel * point_base;
  return OdomToVoxel(point_rel);
}
cv::Mat VoxelManager::DrawRoi() {
  cv::Mat img = voxel_data_.DrawVoxels();
  std::vector<Eigen::Vector3d> roi_points = {
      Eigen::Vector3d(roi_.x_limit[0], roi_.y_limit[0], 0),
      Eigen::Vector3d(roi_.x_limit[1], roi_.y_limit[0], 0),
      Eigen::Vector3d(roi_.x_limit[1], roi_.y_limit[1], 0),
      Eigen::Vector3d(roi_.x_limit[0], roi_.y_limit[1], 0)};
  std::vector<Eigen::Vector3i> voxel_points;
  for (auto& point : roi_points) {
    voxel_points.emplace_back(BaseToVoxel(point));
  }
  // 0-1 2-3是长边
  auto DrawWrappedLine = [&](cv::Mat& img, const cv::Point& p1,
                             const cv::Point& p2, const cv::Scalar& color,
                             bool length, int thickness = 2) {
    cv::circle(img, p1, 2, color, 5);
    cv::circle(img, p2, 2, color, 5);
    // int w = img.cols, h = img.rows;
    // // cv::Point centerp_pt = voxel_data_.voxel_to_img(
    // //     voxel_data_.voxel_origin.x(), voxel_data_.voxel_origin.y());
    // bool warp_x = false;
    // bool warp_y = false;
    // if (!length) {
    // //   warp_x = std::abs(p1.x - p2.x) > w / 2;
    // //   warp_y = std::abs(p1.y - p2.y) > h / 2;
    // } else {
    //   warp_x = p1.x > p2.x;
    // //   warp_y = std::abs(p1.y - p2.y) < h / 2;
    // }

    // AINFO << " point  " << p1 << " " << p2<<" length "<<length<<" warp
    // "<<warp_x<<" "<<warp_y; if (!warp_x && !warp_y) {
    //   cv::line(img, p1, p2, color, thickness);
    //   return;
    // }
    // // 对p1的连线
    // cv::Point temp = p2;
    // if (warp_x) {
    //   if (p1.x > p2.x) {
    //     temp.x = temp.x + w;
    //   } else {
    //     temp.x = temp.x - w;
    //   }
    // }
    // if (warp_y) {
    //   if (p1.y > p2.y) {
    //     temp.y = temp.y + h;
    //   } else {
    //     temp.y = temp.y - h;
    //   }
    // }
    // cv::line(img, p1, temp, color, thickness);
    // // 对p2的连线
    // temp = p1;
    // if (warp_x) {
    //   if (p2.x > p1.x) {
    //     temp.x = temp.x + w;
    //   } else {
    //     temp.x = temp.x - w;
    //   }
    // }
    // if (warp_y) {
    //   if (p2.y > p1.y) {
    //     temp.y = temp.y + h;
    //   } else {
    //     temp.y = temp.y - h;
    //   }
    // }
    // cv::line(img, temp, p2, color, thickness);
  };
  for (int i = 0; i < 4; ++i) {
    auto& p1 = voxel_points[i];
    auto& p2 = voxel_points[(i + 1) % 4];
    cv::Point pt1 = voxel_data_.voxel_to_img(p1.x(), p1.y());
    cv::Point pt2 = voxel_data_.voxel_to_img(p2.x(), p2.y());
    bool length = (i == 0 || i == 2);
    DrawWrappedLine(img, pt1, pt2, cv::Scalar(0, 255, 0), length);
  }
  Eigen::Vector3i center_voxel = BaseToVoxel(Eigen::Vector3d(0, 0, 0));
  cv::circle(img, voxel_data_.voxel_to_img(center_voxel.x(), center_voxel.y()),
             5, cv::Scalar(0, 255, 0), -1);
  return img;
}
}  // namespace perception
}  // namespace robosense