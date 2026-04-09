/******************************************************************************
 * Copyright 2017 RoboSense All rights reserved.
 * Suteng Innovation Technology Co., Ltd. www.robot.ai

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

#include "robot/perception/common/utils/remap_utils.hpp"

namespace robot {
namespace perception {
bool RemapInfo::Init(const SensorID sensor_id,
                     const CameraCalib::Ptr& calib_ptr, const cv::Mat virtual_k,
                     const int width, const int height, RemapModel remap_model,
                     bool use_16sc2) {
  // 需要深拷贝，因为有些地方这个calib会被修改
  ori_calib_ptr = std::make_shared<CameraCalib>(*calib_ptr);
  if (ori_calib_ptr == nullptr) {
    AINFO << " SensorID " << kSensorIDToNameMap.at(sensor_id)
          << " have no calib";
    return false;
  }
  remap_size = cv::Size(width, height);
  remap_k = virtual_k;
  Eigen::Vector3d rect_vec(0, 0, 0);
  if (VirtualRectRad.find(sensor_id) != VirtualRectRad.end()) {
    rect_vec = VirtualRectRad.at(sensor_id);
  }
  // 仿真相机的rotate_to_veh
  const Eigen::AngleAxisd Rrx{rect_vec[0], Eigen::Vector3d::UnitX()};
  const Eigen::AngleAxisd Rry{rect_vec[1], Eigen::Vector3d::UnitY()};
  const Eigen::AngleAxisd Rrz{rect_vec[2], Eigen::Vector3d::UnitZ()};
  Eigen::Quaterniond rot_cam2veh_remap = Rrz * Rry * Rrx;
  remap_trans2veh =
      Eigen::Translation3d(ori_calib_ptr->GetPos()) * rot_cam2veh_remap;
  // 真实相机的逆旋转，veh_to_camera
  const Eigen::AngleAxisd Rx{-ori_calib_ptr->GetRoll(),
                             Eigen::Vector3d::UnitX()};
  const Eigen::AngleAxisd Ry{-ori_calib_ptr->GetPitch(),
                             Eigen::Vector3d::UnitY()};
  const Eigen::AngleAxisd Rz{-ori_calib_ptr->GetYaw(),
                             Eigen::Vector3d::UnitZ()};
  Eigen::Quaterniond rot_veh2fisheye = Rx * Ry * Rz;
  // 从虚拟相机转到真实相机
  Eigen::Quaterniond rot_cam2fisheye = rot_veh2fisheye * rot_cam2veh_remap;
  // 构造虚拟相机的成像
  if (remap_model == RemapModel::FISHEYE_UNFOLDED ||
      remap_model == RemapModel::FISHEYE_ORIGIN) {
    FisheyeRemap(rot_cam2fisheye, remap_model, use_16sc2);
  } else if (remap_model == RemapModel::PINHOLE_UNDISTORTED) {
    PinholeRemap(rot_cam2fisheye, remap_model);
  } else if (remap_model == RemapModel::RESIZE) {
    ResizeRemap();
  } else {
    AERROR << "remap_model not support";
    ResizeRemap();
  }

  AINFO << " SensorID " << kSensorIDToNameMap.at(sensor_id)
        << "\n remap init done. virtual k: \n"
        << remap_k << "\n remap_trans2veh: \n"
        << remap_trans2veh.matrix() << "\n remap size:" << remap_size;
  return true;
}
void RemapInfo::FisheyeRemap(const Eigen::Quaterniond rot_cam2fisheye,
                             const RemapModel remap_model, bool use_16sc2) {
  use_16sc2_ = use_16sc2;
  if (use_16sc2_) {
    AINFO << " FISHEYE REMAP USE 16SC2";
    map_x = cv::Mat::zeros(remap_size.height, remap_size.width, CV_16SC2);
    map_y;
  } else {
    map_x = cv::Mat::zeros(remap_size.height, remap_size.width, CV_32F);
    map_y = cv::Mat::zeros(remap_size.height, remap_size.width, CV_32F);
  }

  inv_map_x = cv::Mat::zeros(ori_calib_ptr->GetHeight(),
                             ori_calib_ptr->GetWidth(), CV_32F);
  inv_map_y = cv::Mat::zeros(ori_calib_ptr->GetHeight(),
                             ori_calib_ptr->GetWidth(), CV_32F);
  valid_mask = cv::Mat::ones(remap_size.height, remap_size.width, CV_8U);
  //   map_x = cv::Mat::zeros(remap_size.height, remap_size.width, CV_16SC2);
  float theta, phi, xc, yc, r, th, th_d, xd, yd;
  Eigen::Vector3d pt, pt_cam;
  // i-> x col  j -> y row
  for (int i = 0; i < remap_size.width; ++i) {
    for (int j = 0; j < remap_size.height; ++j) {
      // AINFO<<" I J "<<i<<" "<<j;
      // 映射球面展开
      if (remap_model == RemapModel::FISHEYE_UNFOLDED) {
        // 鱼眼强制使得fy=fx，保证横纵尺度一致
        theta = (i - remap_k.at<float>(0, 2)) / remap_k.at<float>(0, 0);
        phi = (j - remap_k.at<float>(1, 2)) / remap_k.at<float>(0, 0);
        // 恢复相机坐标系坐标
        pt = Eigen::Vector3d(std::sin(theta) * std::cos(phi), std::sin(phi),
                             std::cos(theta) * std::cos(phi));
      }
      // 映射标准鱼眼
      else if (remap_model == RemapModel::FISHEYE_ORIGIN) {
        // 鱼眼强制使得fy=fx，保证横纵尺度一致
        float x = (i - remap_k.at<float>(0, 2)) / remap_k.at<float>(0, 0);
        float y = (j - remap_k.at<float>(1, 2)) / remap_k.at<float>(0, 0);
        r = std::sqrt(x * x + y * y);
        theta = r;  // 等距近似
        phi = std::atan2(y, x);
        pt = Eigen::Vector3d(std::sin(theta) * std::cos(phi),
                             std::sin(theta) * std::sin(phi), std::cos(theta));
      }
      // 真实相机坐标系
      pt_cam = (rot_cam2fisheye * pt);
      if (pt_cam[2] <= 0) {
        xc = 0.;
        yc = -remap_size.width;
      } else {
        xc = pt_cam[0] / pt_cam[2];
        yc = pt_cam[1] / pt_cam[2];
      }
      // 加畸变
      r = std::sqrt(xc * xc + yc * yc);
      th = std::atan2(r, 1.0);
      th_d = th +
             ori_calib_ptr->GetDistCoeffMatf().at<float>(0) * std::pow(th, 3) +
             ori_calib_ptr->GetDistCoeffMatf().at<float>(1) * std::pow(th, 5) +
             ori_calib_ptr->GetDistCoeffMatf().at<float>(2) * std::pow(th, 7) +
             ori_calib_ptr->GetDistCoeffMatf().at<float>(3) * std::pow(th, 9);
      xd = (th_d / (r + 0.000001)) * xc;
      yd = (th_d / (r + 0.000001)) * yc;
      // 反投影回原图像
      float u_d = xd * ori_calib_ptr->GetInnerMat()(0, 0) +
                  ori_calib_ptr->GetInnerMat()(0, 2);
      float v_d = yd * ori_calib_ptr->GetInnerMat()(1, 1) +
                  ori_calib_ptr->GetInnerMat()(1, 2);
      if (use_16sc2_) {
        map_x.at<cv::Vec2s>(j, i) =
            cv::Vec2w(static_cast<int16_t>(u_d), static_cast<int16_t>(v_d));
      } else {
        map_x.at<float>(j, i) = u_d;
        map_y.at<float>(j, i) = v_d;
      }
      // 反投影回原图像
      if (v_d > 0 && v_d < inv_map_x.rows && u_d > 0 && u_d < inv_map_x.cols) {
        inv_map_x.at<float>(int(v_d), int(u_d)) = i;
        inv_map_y.at<float>(int(v_d), int(u_d)) = j;
      } else {
        valid_mask.at<uint8_t>(j, i) = 0;
      }
    }
  }
}
void RemapInfo::PinholeRemap(const Eigen::Quaterniond rot_cam2fisheye,
                             const RemapModel remap_model) {
  if (remap_model == RemapModel::PINHOLE_UNDISTORTED) {
    cv::initUndistortRectifyMap(
        ori_calib_ptr->GetInnerCvMat(), ori_calib_ptr->GetDistCoeffMatf(),
        cv::noArray(), remap_k, cv::Size(remap_size.width, remap_size.height),
        CV_32F, map_x, map_y);
  }
}
void RemapInfo::ResizeRemap() {
  map_x = cv::Mat::zeros(remap_size.height, remap_size.width, CV_32F);
  map_y = cv::Mat::zeros(remap_size.height, remap_size.width, CV_32F);
  for (int x = 0; x < remap_size.width; ++x) {
    for (int y = 0; y < remap_size.height; ++y) {
      map_x.at<float>(y, x) = (static_cast<float>(x) / remap_size.width) *
                              ori_calib_ptr->GetWidth();
      map_y.at<float>(y, x) = (static_cast<float>(y) / remap_size.height) *
                              ori_calib_ptr->GetHeight();
    }
  }
}
void CreateVirtualRay(const RemapInfo& remap_info, int input_width,
                      int input_height, int output_width, int output_height,
                      pcl::PointCloud<RsPoint>::Ptr& virtual_ray) {
  float x_scale = 1.0 * input_width / output_width;
  float y_scale = 1.0 * input_height / output_height;
  cv::Mat intrinsic = remap_info.remap_k;
  Eigen::Affine3d trans2veh = remap_info.remap_trans2veh;
  Eigen::Matrix3d rotation = trans2veh.rotation();
  virtual_ray.reset(new pcl::PointCloud<RsPoint>());
  virtual_ray->points.resize(output_width * output_height);
  for (int x = 0; x < output_width; ++x) {
    for (int y = 0; y < output_height; ++y) {
      float x_c =
          (x * x_scale - intrinsic.at<float>(0, 2)) / intrinsic.at<float>(0, 0);
      float y_c =
          (y * y_scale - intrinsic.at<float>(1, 2)) / intrinsic.at<float>(0, 0);
      float r = std::sqrt(x_c * x_c + y_c * y_c);
      float theta = r;  // 等距近似
      float phi = std::atan2(y_c, x_c);
      Eigen::Vector3d dir =
          Eigen::Vector3d(std::sin(theta) * std::cos(phi),
                          std::sin(theta) * std::sin(phi), std::cos(theta));
      dir.normalize();
      dir = rotation * dir;
      virtual_ray->points[y * output_width + x].x = dir.x();
      virtual_ray->points[y * output_width + x].y = dir.y();
      virtual_ray->points[y * output_width + x].z = dir.z();
    }
  }
}
void CreateVirtualRay(const RemapInfo& remap_info, int input_width,
                      int input_height, int output_width, int output_height,
                      std::vector<Eigen::Vector3d>& virtual_ray) {
  float x_scale = 1.0 * input_width / output_width;
  float y_scale = 1.0 * input_height / output_height;
  cv::Mat intrinsic = remap_info.remap_k;
  Eigen::Affine3d trans2veh = remap_info.remap_trans2veh;
  Eigen::Matrix3d rotation = trans2veh.rotation();
  virtual_ray.resize(output_width * output_height);
  for (int x = 0; x < output_width; ++x) {
    for (int y = 0; y < output_height; ++y) {
      float x_c =
          (x * x_scale - intrinsic.at<float>(0, 2)) / intrinsic.at<float>(0, 0);
      float y_c =
          (y * y_scale - intrinsic.at<float>(1, 2)) / intrinsic.at<float>(0, 0);
      float r = std::sqrt(x_c * x_c + y_c * y_c);
      float theta = r;  // 等距近似
      float phi = std::atan2(y_c, x_c);
      Eigen::Vector3d dir(std::sin(theta) * std::cos(phi),
                          std::sin(theta) * std::sin(phi), std::cos(theta));
      dir.normalize();
      virtual_ray[y * output_width + x] = rotation * dir;
    }
  }
}
}  // namespace perception
}  // namespace robot
