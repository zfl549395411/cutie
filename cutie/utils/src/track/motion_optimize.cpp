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

#include "robot/perception/uwb_postprocess/process/motion_optimize.h"

namespace robot {
namespace perception {
void MotionOptimize::AddObserve(
    const Eigen::Vector3f& odom_pose, const Eigen::Vector3f& base_pose,
    const float base_distance, const float odom_distance, const float base_yaw,
    const float odom_yaw, const double time_stamp, const SensorID _sensor_id,
    const UWB_POSE_CODE _uwb_pose, const TRACKER_STATE_CODE _track_state,
    const OBSERVE_SENSOR _dist_sensor, const COVER_STATE _image_cover_state) {
  if (observe_states_.size() >= max_history_len_) {
    observe_states_.pop_front();
    optimize_states_.pop_front();
  }
  observe_states_.emplace_back(std::make_shared<ObserveInfo>(
      odom_pose[0], odom_pose[1], odom_pose[2], base_pose[0], base_pose[1],
      base_pose[2], odom_distance, base_distance, odom_yaw, base_yaw,
      time_stamp, _sensor_id, _uwb_pose, _track_state, _dist_sensor,
      _image_cover_state));
  // 初始化直接更新
  if (!is_init_) {
    optimize_states_.emplace_back(std::make_shared<OptimizeInfo>(
        odom_pose[0], odom_pose[1], odom_pose[2], 0, 0, 0, time_stamp));
    is_init_ = true;
  } else {
    // 后续可以填写速度预测值
    const auto& pre_opt_info = optimize_states_.back();
    optimize_states_.emplace_back(std::make_shared<OptimizeInfo>(
        pre_opt_info->odom_x, pre_opt_info->odom_y, pre_opt_info->odom_z,
        pre_opt_info->odom_vel_x, pre_opt_info->odom_vel_y,
        pre_opt_info->odom_vel_z, time_stamp));
  }
  if (kalman_filter_ == nullptr) {
    const auto& opt_info = optimize_states_.back();
    KalmanFilter::OptimizeArray init_array{
        opt_info->odom_x, opt_info->odom_y, opt_info->odom_z,
        opt_info->odom_vel_x, opt_info->odom_vel_y};
    kalman_filter_ = std::make_shared<KalmanFilter>(init_array, time_stamp);
  }
  ++observe_frame;
  if (isCameraSensor(_sensor_id)) {
    pre_camera_observe_frame = observe_frame;
  }
  time_stamp_ = time_stamp;
  //   AINFO << " GET OBSERVE STATE: SENSOR ID " <<
  //   kSensorIDToNameMap.at(_sensor_id)
  //         << " TIME " << std::to_string(time_stamp) << " POSE ODOM "
  //         << odom_pose.transpose() << " BASE " << base_pose.transpose()
  //         << " DISTANCE ODOM " << odom_distance << " BASE" << base_distance
  //         << " YAW ODOM " << odom_yaw << " BASE " << base_yaw;
}
OutInfo MotionOptimize::Optimize() {
  if (!is_init_) {
    return optimize_states_.back()->ConvertToOutInfo();
  }
  // 获取速度
  auto& cur_optimize = optimize_states_.back();
  const auto& cur_observe = observe_states_.back();
  float vel_x, vel_y;
  if (calVelByPos(10, vel_x, vel_y)) {
    cur_optimize->odom_vel_x = vel_x;
    cur_optimize->odom_vel_y = vel_y;
  }

  // 卡尔曼滤波
  KalmanFilter::ObserveArray init_array{
      cur_observe->distance_odom, cur_observe->yaw_odom, cur_observe->odom_z,
      cur_optimize->odom_vel_x, cur_optimize->odom_vel_y};
  // dist_weight
  KalmanFilter::ObserveArray error_array;
  float extra_error = 1;
  // 超出fov范围
  if (std::abs(cur_observe->yaw_odom) > 1.0) {
    extra_error *= 9;
  }
  if (observe_states_.size() > 2) {
    float dist_diff =
        std::abs(cur_observe->distance_odom -
                 observe_states_[observe_states_.size() - 2]->distance_odom);
    if (dist_diff > 1.0) {
      extra_error *= dist_diff;
    }
  }

  error_array[0] = 0.5 * 0.5 * extra_error;  // distance误差为0.5
  error_array[1] = 0.5 * 0.5;
  error_array[2] = 1;
  // uwb位置观测权重
  if (isUwbLocationSensor(cur_observe->sensor_id)) {
    error_array[1] = 0.2 * 0.2;  // 角度误差为10°
    error_array[2] = 1;          // z基本无效
    // 超出fov范围时，重平滑
    if (cur_observe->track_state == TRACKER_STATE_CODE::EDGE_LOST_TRACK &&
        pre_camera_observe_frame != 0) {
      // 进入fov（目前为25°）
      if (std::abs(cur_observe->yaw_base) < 0.45) {
        extra_error *= 9;
      }
    }
    // 变化较大(10°)
    float yaw_diff =
        std::abs(cur_observe->yaw_odom -
                 std::atan2(cur_optimize->odom_y, cur_optimize->odom_x));
    if (yaw_diff > 0.2) {
      extra_error *= (yaw_diff / 0.2) * (yaw_diff / 0.2);
    }
    // 历史关联到相机，不相信uwb
    if (pre_camera_observe_frame != 0 &&
        observe_frame - pre_camera_observe_frame < 50) {
      extra_error *= 2 * 2;
      int frame_gap = observe_frame - pre_camera_observe_frame;
      if (frame_gap < 20) {
        extra_error *= 20 / frame_gap;
      }
    }
    error_array[1] *= extra_error;
    error_array[2] *= extra_error;
  }
  // 图像观测权重
  else if (isCameraSensor(cur_observe->sensor_id)) {
    error_array[1] = 0.02 * 0.02;  // 角度误差为2°
    error_array[2] = 0.5 * 0.5;    // z较为有效
    // 距离观测来源
    if (cur_observe->dist_sensor == OBSERVE_SENSOR::CAMERA_OBSERVE) {
      float dist_error = std::max(cur_observe->distance_base * 0.2f, 2.0f);
      error_array[0] = dist_error * dist_error;
    } else if (cur_observe->dist_sensor == OBSERVE_SENSOR::LIDAR_OBSERVE) {
      error_array[0] = cur_observe->distance_base > 10
                           ? 0.2 * cur_observe->distance_base * 0.2 *
                                 cur_observe->distance_base
                           : 1.0f;
    } else if (cur_observe->dist_sensor == OBSERVE_SENSOR::NO_OBSERVE) {
      error_array[0] = 2 * 2;
    }
    // 跟踪状态
    if (!IsTrack(cur_observe->track_state)) {
      error_array[0] *= 4;
      error_array[1] *= 4;
    }
    // 遮挡状态
    if (cur_observe->image_cover_state == COVER_STATE::FULL_COVER ||
        cur_observe->image_cover_state == COVER_STATE::SIMILAR_PART_COVER) {
      error_array[0] *= 4;
      error_array[1] *= 4;
    }
  }
  // 设置速度权重
  float vel_squre =
      init_array[3] * init_array[3] + init_array[4] * init_array[4];
  if (vel_squre >= 2 * 2) {
    error_array[3] = vel_squre * 4;
    error_array[4] = vel_squre * 4;
  }
  if (cur_optimize->vel_valid == VEL_VALID_CODE::HIGH_WEIGHT) {
    error_array[3] = vel_squre * 0.25;
    error_array[4] = vel_squre * 0.25;
  } else if (cur_optimize->vel_valid == VEL_VALID_CODE::INVALID) {
    error_array[3] = std::max(9.0f, vel_squre);
    error_array[4] = std::max(9.0f, vel_squre);
  } else {
    error_array[3] = vel_squre;
    error_array[4] = vel_squre;
    error_array[3] = std::max(1.0f, vel_squre);
    error_array[4] = std::max(1.0f, vel_squre);
  }
  auto filter_result = kalman_filter_->UpdateObserve(init_array, error_array,
                                                     cur_observe->time_stamp);
  cur_optimize->odom_x = filter_result.at<float>(0, 0);
  cur_optimize->odom_y = filter_result.at<float>(1, 0);
  cur_optimize->odom_z = filter_result.at<float>(2, 0);
  //   cur_optimize.odom_vel_x = filter_result.at<float>(3, 0);
  //   cur_optimize.odom_vel_y = filter_result.at<float>(4, 0);

  // 输出优化结果
  return cur_optimize->ConvertToOutInfo();
}
bool MotionOptimize::calVelByPos(const int window, float& vel_x, float& vel_y) {
  if (optimize_states_.size() < window) {
    return false;
  }
  auto& cur_state = optimize_states_.back();
  const auto& cur_observe = observe_states_.back();
  const auto& pre_state = optimize_states_[optimize_states_.size() - window];
  const auto& pre_observe = observe_states_[optimize_states_.size() - window];

  float distance_x_diff = cur_state->odom_x - pre_state->odom_x;
  float distance_y_diff = cur_state->odom_y - pre_state->odom_y;

  double time_diff = cur_observe->time_stamp - pre_observe->time_stamp;
  if (std::abs(distance_x_diff) < 0.1 && std::abs(distance_y_diff) < 0.1) {
    vel_x = 0;
    vel_y = 0;
    cur_state->vel_valid = VEL_VALID_CODE::HIGH_WEIGHT;
    return true;
  }
  // 平滑
  vel_x = distance_x_diff / time_diff;
  vel_y = distance_y_diff / time_diff;
  int weight = window;
  if (vel_x * vel_x + vel_y * vel_y > 9) {
    weight = 1;
    cur_state->vel_valid = VEL_VALID_CODE::LOW_WEIGHT;
  } else if (isCameraSensor(cur_observe->sensor_id) &&
             isCameraSensor(pre_observe->sensor_id) &&
             IsGoodDistObserve(cur_observe->dist_sensor,
                               cur_observe->distance_base) &&
             IsGoodDistObserve(pre_observe->dist_sensor,
                               pre_observe->distance_base)) {
    weight *= 2;
    cur_state->vel_valid = VEL_VALID_CODE::HIGH_WEIGHT;
  } else {
    cur_state->vel_valid = VEL_VALID_CODE::LOW_WEIGHT;
  }
  vel_x = vel_x * weight;
  vel_y = vel_y * weight;
  float count = weight;
  for (int i = optimize_states_.size() - 5; i < optimize_states_.size() - 1;
       ++i) {
    weight = i - optimize_states_.size() + window + 1 - 5;
    if (weight < 0) {
      continue;
    }
    if (optimize_states_[i]->vel_valid == VEL_VALID_CODE::INVALID) {
      weight = 0;
    } else if (optimize_states_[i]->vel_valid == VEL_VALID_CODE::HIGH_WEIGHT) {
      weight *= 2;
    }
    count += weight;
    vel_x = vel_x + weight * optimize_states_[i]->odom_vel_x;
    vel_y = vel_y + weight * optimize_states_[i]->odom_vel_y;
  }
  vel_x /= count;
  vel_y /= count;
  if (vel_x > 5 || vel_y > 5 || count < 10 || time_diff < 0.1) {
    return false;
  }

  return true;
}
}  // namespace perception
}  // namespace robot