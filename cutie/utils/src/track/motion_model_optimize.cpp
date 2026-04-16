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

#include "hyper_vision/perception/post_fusion/furnace/motion_model_optimize.h"

#include <rally/utils/timer/time.h>

#include "hyper_vision/perception/post_fusion/furnace/furnace_utils.h"

namespace robosense {
namespace perception {

void MotionModelOptimize::addMeasurement(
    const Vec3D& odom_xy, const Vec3D& base_xy, const BoxSize& box_size,
    const double odom_yaw, const bool is_valid_theta,
    const uint8_t ref_point_status, const Vec3D& self_car_odom_xy,
    const Vec3D& odom_vel, const bool is_vel_valid, const int is_lidar_check,
    const int is_side_fisheye_check, const int is_only_narrow_check,
    const SensorID box_observe_source, const SensorID vel_observe_source,
    const double time_stamp, const bool is_key_object,
    const bool is_lidar_box_refine, const MotionType motion_type,
    OUTSIDE_LIDAR_FOV_STATE lidar_fov_state,
    const bool is_dangerous_low_speed) {
  // check
  if (measure_states_.size() >= max_model_len_) {
    measure_states_.pop_front();
    inner_states_.pop_front();
  }
  if (is_vel_valid) {
    have_valid_vel_input_ = true;
  }
  is_key_object_ = is_key_object;
  is_dangerous_low_speed_ = is_dangerous_low_speed;
  motion_type_ = motion_type;
  auto new_theta = NormalizeAngle(odom_yaw);
  // 正负可以表示是否和观测朝向反向
  double new_vel = isSameDirection(odom_vel, new_theta)
                       ? std::hypot(odom_vel.x, odom_vel.y)
                       : -std::hypot(odom_vel.x, odom_vel.y);
  measure_states_.emplace_back(new MeasureModel(
      odom_xy.x, odom_xy.y, base_xy.x, base_xy.y, new_theta, is_valid_theta,
      box_size.length, box_size.width, ref_point_status, self_car_odom_xy.x,
      self_car_odom_xy.y, new_vel, odom_vel.x, odom_vel.y, is_vel_valid,
      is_lidar_check, is_side_fisheye_check, is_only_narrow_check,
      box_observe_source, vel_observe_source, time_stamp, is_lidar_box_refine,
      lidar_fov_state));

  // theta reverse protect
  if (!inner_states_.empty()) {
    thetaInverseCheck(inner_states_.back()->getTheta(), new_theta, new_vel);
  }

  if (!is_inited_) {
    auto input_vel = is_vel_valid ? new_vel : 0.0;

    inner_states_.emplace_back(
        new CeresModel(odom_xy.x, odom_xy.y, new_theta, input_vel,
                       box_size.length, box_size.width, 0.0, 0.0, time_stamp));
  } else {
    double pre_v = inner_states_.back()->getVel();
    double pre_theta = inner_states_.back()->getTheta();
    double pre_x = inner_states_.back()->getX();
    double pre_y = inner_states_.back()->getY();
    double acc = inner_states_.back()->getAcc();
    double omega = inner_states_.back()->getOmega();
    double delta_time = time_stamp - inner_states_.back()->getTime();

    // 填写的是预测值
    inner_states_.emplace_back(
        new CeresModel(pre_x + pre_v * std::cos(pre_theta) * delta_time,
                       pre_y + pre_v * std::sin(pre_theta) * delta_time,
                       NormalizeAngle(pre_theta + omega * delta_time), pre_v,
                       inner_states_.back()->getLength(),
                       inner_states_.back()->getWidth(), 0.0, 0.0, time_stamp));
  }

  // 仅在第三帧进行初始化时进行，对inner_state和历史输入进行了一次平均，对yaw角进行了一次翻转
  initInnerStates();

  // 初始化质心滤波器
  if (light_kalman_ptr_ == nullptr) {
    double vel_x = inner_states_.back()->getVel() *
                   std::cos(inner_states_.back()->getTheta());
    double vel_y = inner_states_.back()->getVel() *
                   std::sin(inner_states_.back()->getTheta());
    light_kalman_ptr_ = std::make_unique<KalmanFilterExtendForOdom>(
        inner_states_.back()->getX(), inner_states_.back()->getY(), vel_x,
        vel_y, inner_states_.back()->getTime(),
        BoxSize(inner_states_.back()->getLength(),
                inner_states_.back()->getWidth(), 0),
        inner_states_.back()->getTheta());
  }
  //   AINFO << "optimize (" << track_id_ << ") input: xy " << odom_xy.x << ", "
  //         << odom_xy.y << ", yaw " << odom_yaw << ", size " <<
  //         box_size.length
  //         << ", " << box_size.width << ", ref " << int(ref_point_status)
  //         << ", vel " << odom_vel.x << ", " << odom_vel.y << ", vel valid "
  //         << int(is_vel_valid) << ", self " << self_car_odom_xy.x << ", "
  //         << self_car_odom_xy.y << ", time "
  //         << rally::Time(time_stamp).toNanosecond() << " dangerous "
  //         << is_dangerous_low_speed;
}

bool MotionModelOptimize::optimize(SaveModel& optimize_state,
                                   const ObjectType& type) {
  tmp_iter_idx_++;
  if (!is_inited_) {
    if (!inner_states_.empty()) {
      pre_output_ = inner_states_.back()->convertToSaveModel();
    }
    return false;
  }
  auto t0 = apollo::cyber::Time::Now();

  // 判断是否为非关键目标
  // todo

  // size cost function
  {
    auto t_single = apollo::cyber::Time::Now();

    // 观测权重
    double length_weight = 0.0;
    double width_weight = 0.0;
    std::tie(length_weight, width_weight) = computeMeasureWeightOfLengthWidth();
    OptimizeWithAverage(
        4, 5, {static_cast<float>(measure_states_.back()->getLength())},
        {static_cast<float>(length_weight)});
    OptimizeWithAverage(
        5, 5, {static_cast<float>(measure_states_.back()->getWidth())},
        {static_cast<float>(width_weight)});
    auto t_temporal_fusion = apollo::cyber::Time::Now();
    if ((t_temporal_fusion - t_single).ToSecond() * 1000.0 >= 5) {
      AINFO << "size optimize cost time: track_id =  " << int(track_id_)
            << " cost time "
            << (t_temporal_fusion - t_single).ToSecond() * 1000.0 << " ms";
    }
  }

  // motion optimize
  bool use_theta_reverse_check = true;

  bool use_light_optimize = false;
  if (optimize_level_ == 0) {
    use_light_optimize = true;
  } else if (optimize_level_ == 1 &&
             (std::abs(measure_states_.back()->getBaseX()) > 10 ||
              std::abs(measure_states_.back()->getBaseY()) > 10 ||
              !is_key_object_)) {
    use_light_optimize = true;
  } else if (optimize_level_ == 2 && !is_key_object_) {
    use_light_optimize = true;
  }
  if (type == ObjectType::TYPE_PED || type == ObjectType::TYPE_ANIMAL) {
    pedMotionOptimize();
    use_theta_reverse_check = false;
  } else if (use_light_optimize) {
    LightWeightOptimize(type);
  } else if (std::max(inner_states_.back()->getLength(),
                      inner_states_.back()->getWidth()) > 2.0) {
    rearAxelMotionOptimize();
  } else {
    ctrvMotionOptimize();
  }

  // save optimization result
  optimize_state = inner_states_.back()->convertToSaveModel();
  // [x, y, theta, v, omega, time]
  // AINFO << "optimize (" << track_id_ << ") output: " << optimize_state[0]
  //       << ", " << optimize_state[1] << ", " << optimize_state[2] << ", "
  //       << optimize_state[3] << ", " << optimize_state[4] << ", "
  //       << optimize_state[5] << ", " << optimize_state[6] << ", "
  //       << optimize_state[7] << ", " << optimize_state[8] << ", "
  //       << rally::Time(optimize_state[9]).toNanosecond();

  // post processing
  // reverse vel protect
  if (use_theta_reverse_check) {
    reverseVelProtect(optimize_state);
    // second filter for static object
    keepStaticState(optimize_state);
  } else {
    // 不进行状态保持也要记录历史状态
    pre_output_ = optimize_state;
  }

  auto t1 = apollo::cyber::Time::Now();
  if ((t1 - t0).ToSecond() * 1000.0 > 5) {
    AINFO << "MotionModelOptimize::optimize cost time: "
          << (t1 - t0).ToSecond() * 1000.0
          << " ms, track id = " << int(track_id_);
  }
  // 静止非关键目标不允许朝向翻转
  if (std::abs(optimize_state[3]) < 0.3 &&
      (measure_states_.back()->getBaseX() < 0 ||
       std::abs(measure_states_.back()->getBaseY()) > 8)) {
    use_theta_reverse_check = false;
  }
  if (use_theta_reverse_check) {
    reviseThetaReverse();
  }

  // 更新滤波器
  double vel_x = inner_states_.back()->getVel() *
                 std::cos(inner_states_.back()->getTheta());
  double vel_y = inner_states_.back()->getVel() *
                 std::sin(inner_states_.back()->getTheta());
  light_kalman_ptr_->SetPredict(inner_states_.back()->getX(),
                                inner_states_.back()->getY(), vel_x, vel_y,
                                inner_states_.back()->getTime(),
                                BoxSize(inner_states_.back()->getLength(),
                                        inner_states_.back()->getWidth(), 0),
                                inner_states_.back()->getTheta());

  return true;
}

void MotionModelOptimize::reviseThetaReverse() {
  auto optimize_theta = inner_states_.back()->getTheta();
  auto origin_theta = measure_states_.back()->getTheta();
  // 大速度反向，直接翻转朝向
  if (inner_states_.back()->getVel() < -5) {
    if (std::cos(optimize_theta - origin_theta) < 0.0) {
      theta_consistence_count_ += 10;
    } else {
      theta_consistence_count_ += 5;
    }
  } else if (inner_states_.back()->isVelValid()) {
    if (inner_states_.back()->getVel() < -2.7 ||
        (std::abs(inner_states_.back()->getVel()) <= 2.7 &&
         std::cos(optimize_theta - origin_theta) < 0.0)) {
      theta_consistence_count_++;
    }
  } else if (std::cos(optimize_theta - origin_theta) < 0.0) {
    theta_consistence_count_++;
  } else {
    theta_consistence_count_ = 0;
  }
  constexpr int theta_consist_win = 10;
  if (theta_consistence_count_ < theta_consist_win) {
    return;
  } else {
    theta_consistence_count_ = 0;
  }

  // theta reverse, use the origin theta to optimize
  for (size_t k = 0; k < inner_states_.size(); ++k) {
    inner_states_[k]->getTheta() = inner_states_[k]->getTheta() < 0.0
                                       ? inner_states_[k]->getTheta() + M_PI
                                       : inner_states_[k]->getTheta() - M_PI;
    inner_states_[k]->getVel() = -inner_states_[k]->getVel();
    inner_states_[k]->getAcc() = -inner_states_[k]->getAcc();
  }

  if (theta_kalman_ptr_ != nullptr) {
    theta_kalman_ptr_->ResetInverseTheta();
  }

  return;
}

void MotionModelOptimize::initInnerStates() {
  if (!is_inited_ && inner_states_.size() >= 3) {
    is_inited_ = true;

    initLengthAndWidth();
    initTheta();
  }

  return;
}

void MotionModelOptimize::initLengthAndWidth() {
  if (inner_states_.empty()) {
    return;
  }

  double mean_length = 0.0;
  double mean_width = 0.0;
  for (const auto& measure : measure_states_) {
    mean_length += measure->getLength();
    mean_width += measure->getWidth();
  }

  mean_length /= measure_states_.size();
  mean_width /= measure_states_.size();

  inner_states_.back()->getLength() = mean_length;
  inner_states_.back()->getWidth() = mean_width;
}

void MotionModelOptimize::initTheta() {
  if (measure_states_.empty()) {
    return;
  }

  // statics the max theta
  std::vector<int> consistent_count(measure_states_.size(), 0);
  for (size_t m = 0; m < measure_states_.size(); ++m) {
    for (size_t n = m + 1; n < measure_states_.size(); ++n) {
      if (std::cos(measure_states_[m]->getTheta() -
                   measure_states_[n]->getTheta()) >= 0.0) {
        consistent_count[m]++;
        consistent_count[n]++;
      }
    }
  }

  int max_consistent_k =
      std::max_element(consistent_count.begin(), consistent_count.end()) -
      consistent_count.begin();
  double ref_theta = measure_states_[max_consistent_k]->getTheta();

  // check is consistent with ref_theta
  for (size_t k = 0; k < inner_states_.size(); ++k) {
    if (std::cos(inner_states_[k]->getTheta() - ref_theta) >= 0.0) {
      continue;
    }

    inner_states_[k]->getTheta() = inner_states_[k]->getTheta() < 0.0
                                       ? inner_states_[k]->getTheta() + M_PI
                                       : inner_states_[k]->getTheta() - M_PI;
    inner_states_[k]->getVel() = -inner_states_[k]->getVel();
    inner_states_[k]->getAcc() = -inner_states_[k]->getAcc();
  }
}

double MotionModelOptimize::computeMeasureVelByPos() {
  if (measure_states_.size() <= 2) {
    return 0.0;
  }

  // lidar staitc check
  int lidar_static_check_win =
      measure_states_.size() >= 10 ? measure_states_.size() - 9 : 1;
  bool is_pre_lidar_static =
      !measure_states_.back()->isLidarSource() &&
      measure_states_[lidar_static_check_win]->isLidarSource() &&
      measure_states_[lidar_static_check_win]->isVelValid() &&
      std::abs(measure_states_[lidar_static_check_win]->getVel()) < 0.3;

  if (is_pre_lidar_static) {
    return 0.0;
  }

  double vel_theta_x, vel_odom_x, vel_odom_y;
  std::tie(vel_theta_x, vel_odom_x, vel_odom_y) = calVelByPos();
  // AINFO << vel_theta_x << " " << vel_odom_x << " " << vel_odom_y;

  if (measure_states_.back()->isLidarSource()) {
    return vel_theta_x;
  } else {
    return std::abs(vel_theta_x) <= 1.0 ? 0.0 : vel_theta_x;
  }
}

std::tuple<double, double> MotionModelOptimize::computePedMeasureVelByPos() {
  if (measure_states_.size() < 5) {
    return std::make_tuple(0.0, 0.0);
  }

  // lidar staitc check
  int lidar_static_check_win =
      measure_states_.size() >= 5 ? measure_states_.size() - 4 : 1;
  bool is_pre_lidar_static =
      !measure_states_.back()->isLidarSource() &&
      measure_states_[lidar_static_check_win]->isLidarSource() &&
      measure_states_[lidar_static_check_win]->isVelValid() &&
      std::abs(measure_states_[lidar_static_check_win]->getVel()) < 0.3;

  if (is_pre_lidar_static) {
    return std::make_tuple(0.0, 0.0);
  }

  double vel_theta_x, vel_odom_x, vel_odom_y;
  std::tie(vel_theta_x, vel_odom_x, vel_odom_y) = calVelByPos();
  // AINFO << vel_theta_x << " " << vel_odom_x << " " << vel_odom_y;

  if (measure_states_.back()->isLidarSource()) {
    return std::make_tuple(vel_odom_x, vel_odom_y);
  } else {
    return vel_odom_x * vel_odom_x + vel_odom_y * vel_odom_y <= 0.36  // 0.6*0.6
               ? std::make_tuple(0.0, 0.0)
               : std::make_tuple(vel_odom_x, vel_odom_y);
  }
}

std::tuple<double, double, double> MotionModelOptimize::calVelByPos() {
  int half_win = std::floor(measure_states_.size() / 2);
  uint8_t ref_point_status = measure_states_.back()->getRefPointStatus();

  std::vector<double> first_pt_status(6, 0);   // [x, y, c_x, c_y, timestamp, n]
  std::vector<double> second_pt_status(6, 0);  // [x, y, c_x, c_y, timestamp, n]

  double pt_x;
  double pt_y;
  double pt_c_x;
  double pt_c_y;
  for (size_t k = 0; k < measure_states_.size(); k++) {
    pt_c_x = measure_states_[k]->getX();
    pt_c_y = measure_states_[k]->getY();
    pt_x = pt_c_x;
    pt_y = pt_c_y;

    // 位置计算速度只考虑车头车尾
    if (ref_point_status != REF_POINT_CENTER) {
      bool is_front =
          (measure_states_[k]->getX() - measure_states_[k]->getSelfCarX()) *
                  std::cos(measure_states_[k]->getTheta()) +
              std::sin(measure_states_[k]->getTheta()) *
                  (measure_states_[k]->getY() -
                   measure_states_[k]->getSelfCarY()) <
          0.0;

      if (is_front) {
        pt_x += measure_states_[k]->getLength() * 0.5 *
                std::cos(measure_states_[k]->getTheta());
        pt_y += measure_states_[k]->getLength() * 0.5 *
                std::sin(measure_states_[k]->getTheta());
      } else {
        pt_x -= measure_states_[k]->getLength() * 0.5 *
                std::cos(measure_states_[k]->getTheta());
        pt_y -= measure_states_[k]->getLength() * 0.5 *
                std::sin(measure_states_[k]->getTheta());
      }
    }
    // AINFO << pt_c_x << " " << pt_c_y << " " << pt_x << " " << pt_y;

    if (k < half_win) {
      first_pt_status[0] += pt_x;
      first_pt_status[1] += pt_y;
      first_pt_status[2] += pt_c_x;
      first_pt_status[3] += pt_c_y;
      first_pt_status[4] += measure_states_[k]->getTimeStamp();
      first_pt_status[5] += 1.0;
    } else {
      second_pt_status[0] += pt_x;
      second_pt_status[1] += pt_y;
      second_pt_status[2] += pt_c_x;
      second_pt_status[3] += pt_c_y;
      second_pt_status[4] += measure_states_[k]->getTimeStamp();
      second_pt_status[5] += 1.0;
    }
  }

  double delta_x = second_pt_status[0] / second_pt_status[5] -
                   first_pt_status[0] / first_pt_status[5];
  double delta_y = second_pt_status[1] / second_pt_status[5] -
                   first_pt_status[1] / first_pt_status[5];
  double delta_c_x = second_pt_status[2] / second_pt_status[5] -
                     first_pt_status[2] / first_pt_status[5];
  double delta_c_y = second_pt_status[3] / second_pt_status[5] -
                     first_pt_status[3] / first_pt_status[5];
  double delta_time = second_pt_status[4] / second_pt_status[5] -
                      first_pt_status[4] / first_pt_status[5];
  // AINFO << delta_c_x << " " << delta_c_y << " " << delta_x << " " << delta_y
  //       << " " << delta_time;

  if (std::hypot(delta_x, delta_y) > std::hypot(delta_c_x, delta_c_y)) {
    delta_x = delta_c_x;
    delta_y = delta_c_y;
  }

  double vel_theta_x =
      (delta_x * std::cos(measure_states_.back()->getTheta()) +
       delta_y * std::sin(measure_states_.back()->getTheta())) /
      delta_time;
  double vel_odom_x = delta_x / delta_time;
  double vel_odom_y = delta_y / delta_time;
  return std::make_tuple(vel_theta_x, vel_odom_x, vel_odom_y);
}

std::tuple<double, bool> MotionModelOptimize::computeMeasureVelByAvgVel(
    const int avg_vel_win) {
  if (measure_states_.empty()) {
    return {0.0, true};
  }
  int count = 0;
  std::pair<double, Vec3D> avg_vel = {0.0, {0.0, 0.0, 0.0}};
  for (auto measure_iter = measure_states_.rbegin();
       measure_iter != measure_states_.rend(); ++measure_iter) {
    if ((*measure_iter)->isVelValid()) {
      double vel_weight = 1.;
      switch (count) {
        case 0:
          vel_weight = 6.;
          break;
        case 1:
          vel_weight = 3.;
          break;
        case 2:
          vel_weight = 1.;
          break;
        default:
          vel_weight = 1.;
          break;
      }
      avg_vel.first += vel_weight;
      avg_vel.second.x += vel_weight * (*measure_iter)->getVelX();
      avg_vel.second.y += vel_weight * (*measure_iter)->getVelY();
    }

    count++;
    if (count >= avg_vel_win) {
      break;
    }
  }
  if (count == 0) {
    return {0.0, false};
  }
  if (avg_vel.first < 1e-6) {
    return {0.0, true};
  }

  avg_vel.second.x /= avg_vel.first;
  avg_vel.second.y /= avg_vel.first;

  double abs_vel = std::hypot(avg_vel.second.x, avg_vel.second.y);

  return {isSameDirection(avg_vel.second, inner_states_.back()->getTheta())
              ? abs_vel
              : -abs_vel,
          true};
}

double MotionModelOptimize::copmuteVelBound() {
  const auto& cur_measure = measure_states_.back();
  const auto& pre_measure = *(std::prev(measure_states_.end(), 2));
  double vel_thres = cur_measure->isLidarSource() ? 0.3 : 0.6;
  double initial_abs_val;
  if (cur_measure->isVelValid()) {
    double abs_vel_by_pos = std::abs(computeMeasureVelByPos());
    double abs_vel_by_vel = 0;
    bool can_be_used = true;
    std::tie(abs_vel_by_vel, can_be_used) = computeMeasureVelByAvgVel(3);
    abs_vel_by_vel = std::abs(abs_vel_by_vel);
    if (std::min(abs_vel_by_pos, abs_vel_by_vel) < vel_thres) {
      initial_abs_val = std::min(abs_vel_by_pos, abs_vel_by_vel);
    } else {
      initial_abs_val = abs_vel_by_vel;
    }
  } else {
    double pre_vel = (*std::prev(inner_states_.end(), 2))->getVel();
    initial_abs_val = std::min(std::abs(computeMeasureVelByPos()),
                               std::abs(pre_vel) + vel_thres);
  }

  return initial_abs_val * 1.2;
}

void MotionModelOptimize::keepStaticState(SaveModel& optimize_state) {
  if (std::abs(optimize_state[3]) < 0.25 &&
      (have_valid_vel_input_ || motion_type_ == MotionType::Stationary ||
       motion_type_ == MotionType::Stoped) &&
      std::max(inner_states_.back()->getLength(),
               inner_states_.back()->getWidth()) >= 1.5 &&
      !(measure_states_.back()->getBaseX() > 80 &&
        measure_states_.back()->isOnlyNarrowSource())) {  // 0.25m/s
    double alpha = 0.2;
    if (!is_key_object_) {
      // 激光区域不能压死，避免远距离误差导致的异常侵入
      alpha =
          measure_states_.size() > 5 &&
                  !IsLidarSensor(measure_states_.back()->getBoxObserveSource())
              ? 0
              : 0.1;
    } else if (is_dangerous_low_speed_) {
      alpha = 0.2;
    } else if (motion_type_ == MotionType::Stationary ||
               motion_type_ == MotionType::Stoped) {
      alpha = 0.1;
    }
    optimize_state[0] =
        pre_output_[0] * (1 - alpha) + optimize_state[0] * alpha;
    optimize_state[1] =
        pre_output_[1] * (1 - alpha) + optimize_state[1] * alpha;
    if (std::cos(optimize_state[2] - pre_output_[2]) > 0.0) {
      optimize_state[2] = NormalizeAngle(
          pre_output_[2] +
          alpha * NormalizeAngle(optimize_state[2] - pre_output_[2]));
    }

    optimize_state[4] =
        pre_output_[4] * (1 - alpha) + optimize_state[4] * alpha;
    optimize_state[5] =
        pre_output_[5] * (1 - alpha) + optimize_state[5] * alpha;
  }
  pre_output_ = optimize_state;
}

void MotionModelOptimize::reverseVelProtect(SaveModel& optimize_state) {
  inner_states_.back()->setVelValid(true);
  // compute ref vel
  constexpr int vel_valid_win = 3;
  int check_count = 0;
  int vel_count = 0;
  bool is_lidar_source = false;
  std::array<float, 2> ref_vel = {0.0f, 0.0f};
  for (auto iter = measure_states_.rbegin(); iter != measure_states_.rend();
       ++iter) {
    check_count++;
    const auto& measure_ptr = *iter;

    if (measure_ptr->isVelValid()) {
      vel_count++;
      ref_vel[0] += measure_ptr->getVelX();
      ref_vel[1] += measure_ptr->getVelY();
    }

    if (measure_ptr->isLidarSource()) {
      is_lidar_source = true;
    }

    if (check_count >= vel_valid_win) {
      break;
    }
  }

  if (vel_count == 0) {
    float base_x = measure_states_.back()->getBaseX();
    bool far_vel_invald =
        (!is_lidar_source && base_x >= 50) && optimize_state[3] < 0.0;
    if (far_vel_invald) {
      optimize_state[3] = 0.0;
      inner_states_.back()->setVelValid(false);
    }

    return;
  }

  ref_vel[0] /= vel_count;
  ref_vel[1] /= vel_count;
  float abs_ref_vel = std::hypot(ref_vel[0], ref_vel[1]);

  float o_vel_x = optimize_state[3] * std::cos(optimize_state[2]);
  float o_vel_y = optimize_state[3] * std::sin(optimize_state[2]);
  float abs_o_vel = std::hypot(o_vel_x, o_vel_y);

  // check whether need check
  float base_x = measure_states_.back()->getBaseX();
  float ref_val_norm = std::hypot(ref_vel[0], ref_vel[1]);
  // 近距离lidar观测运动目标
  bool case_1 =
      (is_lidar_source || base_x < 50) && std::abs(optimize_state[3]) > 0.3;
  // 远距离无lidar观测目标
  bool case_2 = (!is_lidar_source && base_x >= 50);

  auto pre_measure = *std::prev(measure_states_.end(), 2);
  auto cur_measure = measure_states_.back();
  // 之前的速度无效或差异较大
  bool is_optimize_vel_invalid =
      !(*std::prev(inner_states_.end(), 2))->isVelValid() ||
      std::abs(optimize_state[3] - pre_output_[3]) >
          std::max(
              std::abs(cur_measure->getVel() - pre_measure->getVel()) * 1.2,
              0.6);
  bool case_3 =
      is_optimize_vel_invalid &&
      std::abs(abs_o_vel - abs_ref_vel) >=
          std::max(0.2 * abs_ref_vel,
                   0.3);  // previous inner_vel is valid, judge cur vel

  if (!(case_1 || case_2 || case_3)) {
    return;
  }

  // std::cout << "ref_vel = " << ref_vel[0] << ", " << ref_vel[1] << ", o_vel_x
  // = " << o_vel_x << ", " << o_vel_y << std::endl;
  const auto& measure_state = measure_states_.back();
  bool is_vel_reverse =
      case_2 || case_3 ||
      ((ref_vel[0] * o_vel_x + ref_vel[1] * o_vel_y) < -0.0001);
  if (is_vel_reverse) {
    optimize_state[3] = ref_vel[0] * std::cos(optimize_state[2]) +
                        ref_vel[1] * std::sin(optimize_state[2]);
    inner_states_.back()->setVelValid(false);
  }

  return;
}

void MotionModelOptimize::rearAxelMotionOptimize() {
  thetaKalmanSmooth(1);
  ceres::Problem problem;
  ceres::LossFunction* measure_loss_function =
      new ceres::HuberLoss(1.0);  // new ceres::HuberLoss(1.0); //nullptr;
  ceres::LossFunction* motion_loss_function = new ceres::HuberLoss(
      1.0);  // nullptr; //new ceres::HuberLoss(1.0); //nullptr;
  ceres::Manifold* angle_manifold = AngleManifold::Create();
  // measure cost: [x, y, theta]
  for (size_t m = 0; m + 1 < inner_states_.size(); ++m) {
    // ceres中的参考点使用只有头尾点，只需要知道是否是center
    problem.AddResidualBlock(
        MeasureCostFunction::CreateAutoDiffCostFunction(
            measure_states_[m]->getRefPointStatus(),
            {inner_states_[m]->getX(), inner_states_[m]->getY(),
             inner_states_[m]->getTheta(), inner_states_[m]->getLength()},
            inner_states_[m]->getLength(),
            {inner_states_[m]->getX() - measure_states_[m]->getSelfCarX(),
             inner_states_[m]->getY() - measure_states_[m]->getSelfCarY()},
            {1.0, 1.0, 10.0}),
        measure_loss_function, inner_states_[m]->getXYData(),
        &(inner_states_[m]->getTheta()));
  }

  // theta reverse protect
  double origin_yaw = measure_states_.back()->getTheta();
  thetaInverseCheck(inner_states_.back()->getTheta(), origin_yaw);
  double xy_weight = measure_states_.back()->isLidarSource() ? 1.0 : 0.5;
  double theta_weight = 10.0;
  const auto& pre_inner = inner_states_.back();
  const auto& pre_measure_det = measure_states_.back();
  double pre_iou = BevRotatedRectIou(
      Vec3D(pre_inner->getX(), pre_inner->getY()),
      BoxSize(pre_inner->getLength(), pre_inner->getWidth()),
      pre_inner->getTheta(),
      Vec3D(pre_measure_det->getX(), pre_measure_det->getY()),
      BoxSize(pre_measure_det->getLength(), pre_measure_det->getWidth()),
      pre_measure_det->getTheta());
  bool is_source_change = isSourceChange();
  if (is_source_change) {
    xy_weight = 0.4;
    theta_weight = 2.0;
  } else {
    if (measure_states_.back()->getRefPointStatus() == REF_POINT_CENTER &&
        measure_states_.size() > 5 && pre_iou > 0.1) {
      if (std::abs(measure_states_.back()->getBaseX()) < 5 &&
          std::abs(measure_states_.back()->getBaseY()) < 5 &&
          measure_states_.back()->isSideFisheyeSource() &&
          std::abs(inner_states_.back()->getVel()) > 10) {
        //  &&
        // std::abs(EgoPoseInfo::GetInstance().GetNearestYawRate(
        //     measure_states_.back()->getTimeStamp())) > M_PI / 18) {
        xy_weight = 1.0;
        theta_weight = 4.0;
      } else {
        xy_weight = 0.4;
        theta_weight = 2.0;
      }
    } else if (measure_states_.back()->getBaseX() > 80 &&
               measure_states_.back()->isOnlyNarrowSource()) {
      xy_weight = 1.5;
    }
  }
  // 分解weight
  float x_weight = xy_weight, y_weight = xy_weight;
  if (std::abs(measure_states_.back()->getBaseX()) > 10 ||
      std::abs(measure_states_.back()->getBaseY()) > 10) {
    x_weight = xy_weight * std::cos(inner_states_.back()->getTheta());
    y_weight = xy_weight * std::sin(inner_states_.back()->getTheta());
    x_weight = clamp(x_weight, 0.4f, 1.5f);
    y_weight = clamp(y_weight, 0.4f, 1.5f);
  }

  problem.AddResidualBlock(
      MeasureCostFunction::CreateAutoDiffCostFunction(
          measure_states_.back()->getRefPointStatus(),
          {measure_states_.back()->getX(), measure_states_.back()->getY(),
           origin_yaw, measure_states_.back()->getLength()},
          inner_states_.back()->getLength(),
          {measure_states_.back()->getX() -
               measure_states_.back()->getSelfCarX(),
           measure_states_.back()->getY() -
               measure_states_.back()->getSelfCarY()},
          {x_weight, y_weight, theta_weight}),
      measure_loss_function, inner_states_.back()->getXYData(),
      &(inner_states_.back()->getTheta()));

  // RearAxelMotion
  for (size_t m = 0; m + 1 < inner_states_.size(); ++m) {
    double delta_time =
        inner_states_[m + 1]->getTime() - inner_states_[m]->getTime();
    problem.AddResidualBlock(
        RearAxelMotion::CreateAutoDiffCostFunction(
            measure_states_[m]->getRefPointStatus(),
            inner_states_[m]->getLength(), inner_states_[m + 1]->getLength(),
            delta_time,
            {inner_states_[m]->getX() - measure_states_[m]->getSelfCarX(),
             inner_states_[m]->getY() - measure_states_[m]->getSelfCarY()},
            {1.0 * 0.1 / delta_time, 1.0 * 0.1 / delta_time,
             10.0 * 0.1 / delta_time, 1.0 * 0.1 / delta_time}),
        motion_loss_function, inner_states_[m]->getXYData(),
        &(inner_states_[m]->getTheta()), &(inner_states_[m]->getVel()),
        inner_states_[m + 1]->getXYData(), &(inner_states_[m + 1]->getTheta()),
        &(inner_states_[m + 1]->getVel()), &(inner_states_.back()->getAcc()),
        &(inner_states_.back()->getSteerAngle()));
  }

  // theta manifold
  for (size_t m = 0; m < inner_states_.size(); ++m) {
    // if (measure_states_.back()->getRefPointStatus() == REF_POINT_CENTER) {
    problem.SetManifold(&(inner_states_[m]->getTheta()), angle_manifold);
    // } else {
    //     problem.SetParameterBlockConstant(&(inner_states_[m]->getTheta()));
    // }
  }

  // // acc and steer angle
  ceres::LossFunction* acc_steer_angle_loss_function =
      nullptr;  // new ceres::SoftLOneLoss(1.0);
  problem.AddResidualBlock(
      EqualCostFusion::CreateAutoDiffCostFunction(0.0, 0.1),
      acc_steer_angle_loss_function, &(inner_states_.back()->getAcc()));
  problem.AddResidualBlock(
      EqualCostFusion::CreateAutoDiffCostFunction(0.0, 1.0),
      acc_steer_angle_loss_function, &(inner_states_.back()->getSteerAngle()));

  // set the bound
  double vel_thres = measure_states_.back()->isLidarSource() ? 0.3 : 0.6;
  double max_val = copmuteVelBound();
  bool back_interest_obj = (measure_states_.back()->getBaseX() > 4 &&
                            measure_states_.back()->getBaseX() < 30 &&
                            std::abs(measure_states_.back()->getBaseY()) < 5 &&
                            motion_type_ != MotionType::Stationary &&
                            motion_type_ != MotionType::Stoped);
  if (!back_interest_obj && max_val < vel_thres) {  // 0.3 m/s
    inner_states_.back()->getVel() = 0.0;
    inner_states_.back()->getSteerAngle() = 0.0;
    inner_states_.back()->getAcc() = 0.0;
    problem.SetParameterBlockConstant(&inner_states_.back()->getVel());
    problem.SetParameterBlockConstant(&inner_states_.back()->getSteerAngle());
    problem.SetParameterBlockConstant(&(inner_states_.back()->getAcc()));
  } else if (!back_interest_obj) {
    problem.SetParameterLowerBound((&inner_states_.back()->getVel()), 0,
                                   -max_val);
    problem.SetParameterUpperBound((&inner_states_.back()->getVel()), 0,
                                   max_val);

    const auto& pre_inner_state = *(std::prev(inner_states_.end(), 2));
    double max_steer_angle =
        std::atan(3.0 * pre_inner_state->getLength() /
                  (pre_inner_state->getVel() * pre_inner_state->getVel()));
    problem.SetParameterLowerBound(&inner_states_.back()->getSteerAngle(), 0,
                                   -max_steer_angle);
    problem.SetParameterUpperBound(&inner_states_.back()->getSteerAngle(), 0,
                                   max_steer_angle);

    problem.SetParameterLowerBound(&(inner_states_.back()->getAcc()), 0, -6.0);
    problem.SetParameterUpperBound(&(inner_states_.back()->getAcc()), 0, 10.0);
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 25;
  options.max_solver_time_in_seconds = 0.0015;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // // vel optimize
  // inner_states_.back()->getVel() =
  //     0.1 * (*std::prev(inner_states_.end(), 2))->getVel() +
  //     0.2 * inner_states_.back()->getVel() +
  //     0.7 * measure_states_.back()->getVel();

  // vel cost function
  {
    ceres::Problem problem;
    ceres::LossFunction* vel_loss_function = new ceres::HuberLoss(1.0);

    // double measure_vel = measure_states_.back()->getVel();
    double measure_vel = 0;
    bool can_be_used = false;
    std::tie(measure_vel, can_be_used) = computeMeasureVelByAvgVel(2);

    // 静止目标速度压0
    if (motion_type_ == MotionType::Stationary ||
        motion_type_ == MotionType::Stoped) {
      if (std::abs(measure_vel) < 0.3) {
        measure_vel = 0;
      }
    }
    // 非关键目标
    if (!back_interest_obj && std::abs(measure_vel) < 0.2) {
      measure_vel = 0.;
    }
    double measure_vel_weight = 1.0;
    double inner_vel_weight = 1.0;
    if (!measure_states_.back()->isVelValid()) {
      measure_vel_weight = 0.0;
    } else {
      if (motion_type_ == MotionType::Stationary ||
          motion_type_ == MotionType::Stoped) {
        inner_vel_weight = 2.0;
        if (std::abs(measure_states_.back()->getVel()) > 0.3) {
          measure_vel_weight = 0.5;
        }
      } else if (std::abs(std::abs(measure_vel) -
                          std::abs(inner_states_.back()->getVel())) > 0.6) {
        // if (std::abs(measure_states_.back()->getVel()) < 2.0 &&
        //     measure_states_.back()->isLidarSource()) {
        measure_vel_weight = 2.0;
      } else if (std::abs(std::abs(measure_vel) -
                          std::abs(inner_states_.back()->getVel())) > 0.3) {
        measure_vel_weight = 1.5;
      } else if (back_interest_obj) {
        measure_vel_weight = 1.5;
      } else if (std::abs(inner_states_.back()->getVel()) < 0.3 &&
                 std::abs(measure_states_.back()->getVel()) < 0.3) {
        inner_vel_weight = 3.0;
      } else if (measure_states_.back()->isLidarSource()) {
        measure_vel_weight = 1.5;
      }

      if (isFisheyeSensor(measure_states_.back()->getVelObserveSource())) {
        measure_vel_weight = 0.5;
      }
    }

    problem.AddResidualBlock(
        VelChangeCostFunction::CreateCostFunction(
            inner_states_.back()->getVel(), {inner_vel_weight}),
        vel_loss_function, &(inner_states_.back()->getVel()));
    problem.AddResidualBlock(VelChangeCostFunction::CreateCostFunction(
                                 measure_vel, {measure_vel_weight}),
                             vel_loss_function,
                             &(inner_states_.back()->getVel()));

    ceres::Solver::Options options;
    options.max_num_iterations = 12;
    options.max_solver_time_in_seconds = 0.0015;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (summary.total_time_in_seconds * 1000.0 >= 5) {
      AINFO << "vel optimize: track_id = " << int(track_id_)
            << ", solver cost time: " << summary.total_time_in_seconds * 1000.0
            << " ms, iter size = " << summary.iterations.size();
    }
  }

  // check kalman theta error
  double kalman_theta_check_error =
      NormalizeAngle(measure_states_.back()->getTheta() -
                     theta_kalman_ptr_->GetValue());         // measure
  if (std::abs(kalman_theta_check_error) > DegToRad(0.3) &&  // measure
      std::abs(pre_kalman_theta_check_error_) > DegToRad(0.3) &&
      kalman_theta_check_error * pre_kalman_theta_check_error_ > 0.0) {
    kalman_theta_check_count_ += 1;
  } else {
    kalman_theta_check_count_ = 0;
  }
  pre_kalman_theta_check_error_ = kalman_theta_check_error;

  if (kalman_theta_check_count_ >= 2) {
    double kalman_inner_theta_check_error =
        NormalizeAngle(inner_states_.back()->getTheta() -
                       theta_kalman_ptr_->GetValue());  // measure
    theta_kalman_ptr_->SetValue(
        NormalizeAngle(theta_kalman_ptr_->GetValue() +
                       0.6 * kalman_inner_theta_check_error));  // inner
  }
  inner_states_.back()->getTheta() = theta_kalman_ptr_->GetValue();
  if (summary.total_time_in_seconds * 1000.0 > 5) {
    AINFO << "rearAxelMotion: track_id = " << int(track_id_)
          << ", solver cost time: " << summary.total_time_in_seconds * 1000.0
          << " ms, iter size = " << summary.iterations.size()
          << ", final cost = " << summary.final_cost
          << ", frame size = " << inner_states_.size();
  }

  // 计算omega
  inner_states_.back()->getOmega() =
      GetAccWithWindow(8, 2, 0.1, 0.8, true, 5, 10);
  OptimizeWithAverage(
      8, 5, {static_cast<float>(inner_states_.back()->getOmega())}, {2});
  // 计算加速度
  inner_states_.back()->getAcc() = GetAccWithWindow(6, 3, 1, 7, false, 5, 10);
  OptimizeWithAverage(
      6, 3, {static_cast<float>(inner_states_.back()->getAcc())}, {2});
}
void MotionModelOptimize::LightWeightOptimize(const ObjectType& type) {
  // 无观测时间越长，越相信观测
  float hold_observe_weight = 1;
  if (measure_states_.size() >= 2) {
    double optime_time_gap =
        measure_states_.back()->getTimeStamp() -
        measure_states_[measure_states_.size() - 2]->getTimeStamp();
    if (optime_time_gap < 0.2) {
      hold_observe_weight = 1;
    } else if (type == ObjectType::TYPE_CYC) {
      hold_observe_weight = clamp(optime_time_gap * 10, 1.0, 10.0);
    } else {
      hold_observe_weight = clamp(optime_time_gap * 5, 1.0, 5.0);
    }
  }
  bool is_source_change = isSourceChange();
  // theta角卡尔曼滤波
  {
    float yaw_weight = 1;
    if (is_source_change) {
      yaw_weight *= 0.5;
    }
    float pre_angle_ratio = 0;
    if (inner_states_.size() > 5) {
      pre_angle_ratio =
          std::abs(inner_states_[inner_states_.size() - 2]->getOmega());
    }
    if (pre_angle_ratio > 0.1) {
      yaw_weight = yaw_weight * 2 * (pre_angle_ratio / 0.1);
    }
    float yaw_change_thresh = 0.1;
    float yaw_jump_thresh = 0.8;
    if (!isVehObject(type)) {
      yaw_weight = std::max(2.0, yaw_weight * 1.5);
      yaw_jump_thresh = 1.5;
    }
    yaw_weight *= hold_observe_weight;
    thetaKalmanSmooth(yaw_weight);
    // 骑行者yaw角变化更快
    // 连续三帧都有激光观测
    bool consistent_lidar = true;
    if (measure_states_.size() > 3) {
      for (int i = measure_states_.size() - 3; i < measure_states_.size();
           ++i) {
        if (!measure_states_[i]->isLidarSource()) {
          consistent_lidar = false;
          break;
        }
      }
    } else {
      consistent_lidar = false;
    }
    if (consistent_lidar) {
      inner_states_.back()->getOmega() = GetAccWithWindow(
          8, 2, yaw_change_thresh, yaw_jump_thresh, true, 3, 6);
      OptimizeWithAverage(
          8, 3, {static_cast<float>(inner_states_.back()->getOmega())}, {2});
    } else {
      inner_states_.back()->getOmega() = GetAccWithWindow(
          8, 2, yaw_change_thresh, yaw_jump_thresh, true, 5, 10);
      OptimizeWithAverage(
          8, 5, {static_cast<float>(inner_states_.back()->getOmega())}, {2});
    }
  }

  // opitime
  float dist_base = std::hypot(measure_states_.back()->getBaseX(),
                               measure_states_.back()->getBaseY());

  // 速度加权平均滤波
  {
    // 高斯窗口平均得到一个平均观测
    bool slow_obj = false;
    if (inner_states_.size() > 5 &&
        std::abs(inner_states_[inner_states_.size() - 2]->getVel()) < 1 &&
        std::abs(measure_states_.back()->getVel()) < 1) {
      slow_obj = true;
    }
    float pre_acc = 0;
    if (inner_states_.size() > 5) {
      pre_acc = std::abs(inner_states_[inner_states_.size() - 2]->getAcc());
    }
    double measure_vel = 0;
    bool can_be_used = true;
    if (!slow_obj && std::abs(pre_acc) < 1) {
      std::tie(measure_vel, can_be_used) = computeMeasureVelByAvgVel(3);
    } else if (std::abs(pre_acc) > 3) {
      std::tie(measure_vel, can_be_used) = computeMeasureVelByAvgVel(1);
    } else {
      std::tie(measure_vel, can_be_used) = computeMeasureVelByAvgVel(2);
    }
    bool back_interest_obj =
        (measure_states_.back()->getBaseX() > 4 &&
         measure_states_.back()->getBaseX() < 30 &&
         std::abs(measure_states_.back()->getBaseY()) < 5 &&
         motion_type_ != MotionType::Stationary &&
         motion_type_ != MotionType::Stoped);

    // 静止目标速度压0
    if (motion_type_ == MotionType::Stationary ||
        motion_type_ == MotionType::Stoped) {
      if (std::abs(measure_vel) < 0.3) {
        measure_vel = 0;
      }
    }
    // 非关键目标
    if ((!back_interest_obj) && std::abs(measure_vel) < 0.2) {
      measure_vel = 0.;
    }
    double measure_vel_weight = 3;
    if (!measure_states_.back()->isVelValid() || !can_be_used) {
      measure_vel_weight = 0.1;
    } else {
      // 传感器相关
      if (measure_states_.back()->isLidarSource()) {
        measure_vel_weight = 5;
      } else if (isRadarSensor(measure_states_.back()->getVelObserveSource())) {
        measure_vel_weight = 5;
      } else if (isCameraSensor(
                     measure_states_.back()->getVelObserveSource())) {
        if (isFisheyeSensor(measure_states_.back()->getVelObserveSource()) &&
            dist_base > 5) {
          measure_vel_weight *= 0.5;
        } else if (dist_base < 20) {
          measure_vel_weight = 5;
        } else {
          measure_vel_weight = std::max(2.5 * 40 / (dist_base), 2.0);
        }
      }
      if (is_key_object_) {
        measure_vel_weight *= 1.5;
      }
      // 加速度很大
      if (is_key_object_ && pre_acc > 0.5 &&
          (measure_states_.back()->isLidarSource() ||
           isRadarSensor(measure_states_.back()->getVelObserveSource()))) {
        measure_vel_weight = std::max(5.0, measure_vel_weight * pre_acc * 2);
      } else if (pre_acc > 1) {
        measure_vel_weight = std::max(5.0, measure_vel_weight * pre_acc);
      }
      // 小速度
      if (slow_obj) {
        if (motion_type_ == MotionType::Stationary ||
            motion_type_ == MotionType::Stoped) {
          measure_vel_weight *= 0.5;
          if (std::abs(measure_states_.back()->getVel()) > 0.3) {
            measure_vel_weight *= 0.5;
          }
        } else if (std::abs(inner_states_.back()->getVel() > 0.3)) {
          measure_vel_weight = std::max(3.0, measure_vel_weight);
        } else if (!is_key_object_ &&
                   std::abs(inner_states_.back()->getVel()) < 0.2 &&
                   isFisheyeSensor(
                       measure_states_.back()->getVelObserveSource())) {
          measure_vel_weight *= 0.1;
        }
      }

      // 误差比较大
      if (std::abs(std::abs(measure_vel) -
                   std::abs(inner_states_.back()->getVel())) > 1) {
        measure_vel_weight *= 0.5;
      }

      // 速度反向
      if (measure_states_.back()->getVel() < -3) {
        measure_vel_weight *= 0.1;
      }
    }
    std::vector<float> observe;
    std::vector<float> observe_weight;

    observe.emplace_back(measure_vel);
    observe_weight.emplace_back(measure_vel_weight);

    // 后向静止目标加一个0输入
    if ((motion_type_ == MotionType::Stationary ||
         motion_type_ == MotionType::Stoped) &&
        measure_states_.back()->getBaseX() < 0) {
      observe.emplace_back(0);
      observe_weight.emplace_back(2);
    }
    // 速度窗口小一点，要不然平滑太重了
    OptimizeWithAverage(3, 3, observe, observe_weight);
    // 后向的静止目标再给一个0的观测？

    // 计算加速度
    inner_states_.back()->getAcc() = GetAccWithWindow(6, 3, 1, 7, false, 5, 10);
    OptimizeWithAverage(
        6, 3, {static_cast<float>(inner_states_.back()->getAcc())}, {2});
  }

  // 位置卡尔曼滤波
  // 实际对应了观测误差
  float pose_error = 1;
  bool is_camera_to_lidar_change = false;
  // sensor相关
  if (IsLidarSensor(measure_states_.back()->getBoxObserveSource())) {
    pose_error = 0.5;
    // 远距离切源
    if (measure_states_.size() >= 3 && dist_base > 80 &&
        !IsLidarSensor(
            (*(std::prev(measure_states_.end(), 2)))->getBoxObserveSource())) {
      is_camera_to_lidar_change = true;
      pose_error = 0.25;
    }
    // 大角速度大车
    if (measure_states_.back()->getLength() > 5 &&
        std::abs(inner_states_.back()->getOmega()) > 0.1) {
      pose_error /= (std::abs(inner_states_.back()->getOmega()) / 0.05);
    }
  } else if (isCameraSensor(measure_states_.back()->getBoxObserveSource())) {
    pose_error = std::min(std::max(0.1f * dist_base, 1.0f), 5.0f);
  }
  // 切源
  if (is_source_change && !is_camera_to_lidar_change) {
    pose_error *= 2;
  }

  // 分解到xy上,weight越大误差应该越小，ortho对应了y方向的观测误差
  // 分解认为他车坐标下非朝向方向完全不存在抖动，只允许在朝向方向运动，那么权重高斯分布是一个短轴为0长轴等于总权重的椭圆
  // 当想要增加垂向权重时，需要增加高斯椭球的短轴长度，那么分解到xy时就需要增加一个角度偏置
  float ortho_dir_weight = 0.1;
  float paral_dir_weight = 1;
  // 大角速度增加垂向权重 0.05 3°
  if (measure_states_.size() > 5 &&
      (IsLidarSensor(measure_states_.back()->getBoxObserveSource()) ||
       dist_base < 10) &&
      std::abs(inner_states_.back()->getOmega()) > 0.02) {
    ortho_dir_weight =
        clamp(ortho_dir_weight * std::abs(inner_states_.back()->getOmega()) /
                  0.05 * 10,
              0.1, 2.0);
  }
  if (IsLidarSensor(measure_states_.back()->getBoxObserveSource())) {
    ortho_dir_weight = std::max(0.2f, ortho_dir_weight);
  }
  float extra_theta = std::atan2(ortho_dir_weight, paral_dir_weight);

  // 将权重椭球分解到世界坐标系的x y 方向上，总权重为pose_error
  float y_weight =
      std::abs(std::sin(inner_states_.back()->getTheta() + extra_theta)) +
      0.0001;
  float x_weight =
      std::abs(std::cos(inner_states_.back()->getTheta() + extra_theta)) +
      0.0001;

  float y_observ_error = pose_error / y_weight;
  float x_observe_error = pose_error / x_weight;
  float min_limit = 3.0f;
  // 角速度大于0.01时最小值限制线性变化，0.05时最小值为1.0
  if (std::abs(inner_states_.back()->getOmega()) > 0.01 &&
      (IsLidarSensor(measure_states_.back()->getBoxObserveSource()) ||
       dist_base < 10)) {
    min_limit = 3.0f - std::abs(inner_states_.back()->getOmega()) * 40.0f;
  }
  if (std::abs(measure_states_.back()->getVel()) > 15 &&
      (IsLidarSensor(measure_states_.back()->getBoxObserveSource()) ||
       dist_base < 10)) {
    min_limit = 3.0f - std::abs(measure_states_.back()->getVel()) * 0.05f;
  }
  if (!is_key_object_ && !is_camera_to_lidar_change) {
    y_observ_error = clamp(y_observ_error, 5.0f, 20.0f);
    x_observe_error = clamp(x_observe_error, 5.0f, 20.0f);
  } else {
    y_observ_error = clamp(y_observ_error, min_limit, 10.0f);
    x_observe_error = clamp(x_observe_error, min_limit, 10.0f);
  }
  y_observ_error /= hold_observe_weight;
  x_observe_error /= hold_observe_weight;

  float vel_x = inner_states_.back()->getVel() *
                std::cos(inner_states_.back()->getTheta());
  float vel_y = inner_states_.back()->getVel() *
                std::sin(inner_states_.back()->getTheta());

  float vel_weight = 1;
  // 初始化帧
  if (measure_states_.size() < 3) {
    vel_weight = 0.2;
    y_observ_error /= 2;
    x_observe_error /= 2;
  }
  //   AINFO << " CUR ERROR INFO y x  " << y_observ_error << " " <<
  //   x_observe_error
  //         << " pose error  " << pose_error << " weight " << y_weight << " "
  //         << x_weight << " theta " << extra_theta << " ori weight ortho"
  //         << ortho_dir_weight << " " << paral_dir_weight << " vel y x " <<
  //         vel_y
  //         << " " << vel_x;
  // 当前帧参考点位置
  uint8_t ref_pos = measure_states_.back()->getRefPointStatus();  // center
  // 无高度信息
  BoxSize measure_size = BoxSize(measure_states_.back()->getLength(),
                                 measure_states_.back()->getWidth(), 0);
  BoxSize cur_size = BoxSize(inner_states_.back()->getLength(),
                             inner_states_.back()->getWidth(), 0);
  auto fileter_resuilt = light_kalman_ptr_->UpdateObserve(
      {measure_states_.back()->getX(), 0}, {measure_states_.back()->getY(), 0},
      x_observe_error, y_observ_error, vel_x, vel_y, vel_weight, vel_weight,
      inner_states_.back()->getTime(), measure_size, cur_size,
      inner_states_.back()->getTheta(), ref_pos);
  inner_states_.back()->getX() = *fileter_resuilt.ptr<float>(0, 0);
  inner_states_.back()->getY() = *fileter_resuilt.ptr<float>(1, 0);

  // 计算角速度和速度
}
void MotionModelOptimize::ctrvMotionOptimize() {
  thetaKalmanSmooth(1);

  ceres::Problem problem;
  ceres::LossFunction* measure_loss_function =
      new ceres::HuberLoss(1.0);  // new ceres::HuberLoss(1.0); //nullptr;
  ceres::LossFunction* motion_loss_function = new ceres::HuberLoss(
      1.0);  // nullptr; //new ceres::HuberLoss(1.0); //nullptr;
  ceres::Manifold* angle_manifold = AngleManifold::Create();
  // measure cost: [x, y, theta]
  for (size_t m = 0; m + 1 < inner_states_.size(); ++m) {
    problem.AddResidualBlock(
        MeasureCostFunction::CreateAutoDiffCostFunction(
            measure_states_[m]->getRefPointStatus(),
            {inner_states_[m]->getX(), inner_states_[m]->getY(),
             inner_states_[m]->getTheta(), inner_states_[m]->getLength()},
            inner_states_[m]->getLength(),
            {inner_states_[m]->getX() - measure_states_[m]->getSelfCarX(),
             inner_states_[m]->getY() - measure_states_[m]->getSelfCarY()},
            {5.0, 5.0, 60.0}),
        measure_loss_function, inner_states_[m]->getXYData(),
        &(inner_states_[m]->getTheta()));
  }

  // theta reverse protect
  double origin_yaw = measure_states_.back()->getTheta();
  thetaInverseCheck(inner_states_.back()->getTheta(), origin_yaw);
  problem.AddResidualBlock(
      MeasureCostFunction::CreateAutoDiffCostFunction(
          measure_states_.back()->getRefPointStatus(),
          {measure_states_.back()->getX(), measure_states_.back()->getY(),
           origin_yaw, measure_states_.back()->getLength()},
          inner_states_.back()->getLength(),
          {measure_states_.back()->getX() -
               measure_states_.back()->getSelfCarX(),
           measure_states_.back()->getY() -
               measure_states_.back()->getSelfCarY()},
          {5.0, 5.0, 60.0}),
      measure_loss_function, inner_states_.back()->getXYData(),
      &(inner_states_.back()->getTheta()));

  // ctrv smooth
  double ctrv_weight = 2.0;
  for (size_t m = 0; m + 1 < inner_states_.size(); ++m) {
    double delta_time =
        inner_states_[m + 1]->getTime() - inner_states_[m]->getTime();
    problem.AddResidualBlock(
        CTRV::CreateAutoDiffCostFunction(
            measure_states_[m]->getRefPointStatus(),
            inner_states_[m]->getLength(), inner_states_[m + 1]->getLength(),
            delta_time,
            {inner_states_[m]->getX() - measure_states_[m]->getSelfCarX(),
             inner_states_[m]->getY() - measure_states_[m]->getSelfCarY()},
            {0.2 / delta_time * ctrv_weight, 0.2 / delta_time * ctrv_weight,
             6.0 / delta_time * ctrv_weight}),
        motion_loss_function, inner_states_[m]->getXYData(),
        &(inner_states_[m]->getTheta()), inner_states_[m + 1]->getXYData(),
        &(inner_states_[m + 1]->getTheta()),
        &(inner_states_.back()->getOmega()), &(inner_states_.back()->getVel()));
  }

  // set the bound
  double vel_thres = measure_states_.back()->isLidarSource() ? 0.3 : 0.6;
  double max_val = copmuteVelBound();

  if (max_val < vel_thres) {  // 0.3 m/s
    inner_states_.back()->getVel() = 0.0;
    inner_states_.back()->getOmega() = 0.0;

    problem.SetParameterBlockConstant(&inner_states_.back()->getVel());
    problem.SetParameterBlockConstant(&inner_states_.back()->getOmega());
  } else {
    problem.SetParameterLowerBound((&inner_states_.back()->getVel()), 0,
                                   -max_val);
    problem.SetParameterUpperBound((&inner_states_.back()->getVel()), 0,
                                   max_val);

    const auto& pre_inner_state = *(std::prev(inner_states_.end(), 2));
    problem.SetParameterLowerBound(&inner_states_.back()->getOmega(), 0,
                                   -std::abs(3.0 / pre_inner_state->getVel()));
    problem.SetParameterUpperBound(&inner_states_.back()->getOmega(), 0,
                                   std::abs(3.0 / pre_inner_state->getVel()));
  }

  // theta manifold
  for (size_t m = 0; m < inner_states_.size(); ++m) {
    problem.SetParameterBlockConstant(&(inner_states_[m]->getTheta()));
    // problem.SetManifold(&(inner_states_[m]->getTheta()), angle_manifold);
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 25;
  options.max_solver_time_in_seconds = 0.0015;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  if (summary.total_time_in_seconds * 1000.0 > 5) {
    AINFO << "ctrvlMotion: track_id = " << int(track_id_)
          << ", solver cost time: " << summary.total_time_in_seconds * 1000.0
          << " ms, iter size = " << summary.iterations.size()
          << ", final cost = " << summary.final_cost
          << ", frame size = " << inner_states_.size();
  }

  // post process
  // 计算omega
  inner_states_.back()->getOmega() =
      GetAccWithWindow(8, 2, 0.1, 1.5, true, 5, 10);
  OptimizeWithAverage(
      8, 5, {static_cast<float>(inner_states_.back()->getOmega())}, {2});
  // 计算加速度
  inner_states_.back()->getAcc() = GetAccWithWindow(6, 3, 1, 7, false, 5, 10);
  OptimizeWithAverage(
      6, 3, {static_cast<float>(inner_states_.back()->getAcc())}, {2});
}

void MotionModelOptimize::ctraMotionOptimize() {
  ceres::Problem problem;
  ceres::LossFunction* measure_loss_function =
      new ceres::HuberLoss(1.0);  // new ceres::HuberLoss(1.0); //nullptr;
  ceres::LossFunction* motion_loss_function = new ceres::HuberLoss(
      1.0);  // nullptr; //new ceres::HuberLoss(1.0); //nullptr;
  ceres::Manifold* angle_manifold = AngleManifold::Create();
  // measure cost: [x, y, theta]
  for (size_t m = 0; m + 1 < inner_states_.size(); ++m) {
    problem.AddResidualBlock(
        MeasureCostFunction::CreateAutoDiffCostFunction(
            measure_states_[m]->getRefPointStatus(),
            {inner_states_[m]->getX(), inner_states_[m]->getY(),
             inner_states_[m]->getTheta(), inner_states_[m]->getLength()},
            inner_states_[m]->getLength(),
            {inner_states_[m]->getX() - measure_states_[m]->getSelfCarX(),
             inner_states_[m]->getY() - measure_states_[m]->getSelfCarY()},
            {5.0, 5.0, 60.0}),
        measure_loss_function, inner_states_[m]->getXYData(),
        &(inner_states_[m]->getTheta()));
  }

  // theta reverse protect
  double origin_yaw = measure_states_.back()->getTheta();
  thetaInverseCheck(inner_states_.back()->getTheta(), origin_yaw);
  problem.AddResidualBlock(
      MeasureCostFunction::CreateAutoDiffCostFunction(
          measure_states_.back()->getRefPointStatus(),
          {measure_states_.back()->getX(), measure_states_.back()->getY(),
           origin_yaw, measure_states_.back()->getLength()},
          inner_states_.back()->getLength(),
          {measure_states_.back()->getX() -
               measure_states_.back()->getSelfCarX(),
           measure_states_.back()->getY() -
               measure_states_.back()->getSelfCarY()},
          {5.0, 5.0, 60.0}),
      measure_loss_function, inner_states_.back()->getXYData(),
      &(inner_states_.back()->getTheta()));

  // ctra smooth
  double ctrv_weight = 2.0;
  for (size_t m = 0; m + 1 < inner_states_.size(); ++m) {
    double delta_time =
        inner_states_[m + 1]->getTime() - inner_states_[m]->getTime();
    problem.AddResidualBlock(
        CTRA::CreateAutoDiffCostFunction(
            measure_states_[m]->getRefPointStatus(),
            inner_states_[m]->getLength(), inner_states_[m + 1]->getLength(),
            delta_time,
            {inner_states_[m]->getX() - measure_states_[m]->getSelfCarX(),
             inner_states_[m]->getY() - measure_states_[m]->getSelfCarY()},
            {0.2 / delta_time * ctrv_weight, 0.2 / delta_time * ctrv_weight,
             6.0 / delta_time * ctrv_weight, 0.2 / delta_time * ctrv_weight}),
        motion_loss_function, inner_states_[m]->getXYData(),
        &(inner_states_[m]->getTheta()), &(inner_states_[m]->getVel()),
        inner_states_[m + 1]->getXYData(), &(inner_states_[m + 1]->getTheta()),
        &(inner_states_[m + 1]->getVel()), &(inner_states_.back()->getOmega()),
        &(inner_states_.back()->getAcc()));
  }

  // set the bound
  const auto& cur_measure = measure_states_.back();
  const auto& pre_measure = *(std::prev(measure_states_.end(), 2));
  double initial_abs_val =
      cur_measure->isVelValid()
          ? std::abs(cur_measure->getVel())
          : std::hypot(cur_measure->getX() - pre_measure->getX(),
                       cur_measure->getY() - pre_measure->getY()) /
                (cur_measure->getTimeStamp() - pre_measure->getTimeStamp());
  double max_val = std::max(initial_abs_val * 1.2,
                            std::abs(inner_states_.back()->getVel()) * 1.2);

  if (max_val < 0.3) {  // 0.3 m/s
    inner_states_.back()->getVel() = 0.0;
    inner_states_.back()->getOmega() = 0.0;

    problem.SetParameterBlockConstant(&inner_states_.back()->getVel());
    problem.SetParameterBlockConstant(&inner_states_.back()->getOmega());
    problem.SetParameterBlockConstant(&(inner_states_.back()->getAcc()));
  } else {
    problem.SetParameterLowerBound((&inner_states_.back()->getVel()), 0,
                                   -max_val);
    problem.SetParameterUpperBound((&inner_states_.back()->getVel()), 0,
                                   max_val);

    const auto& pre_inner_state = *(std::prev(inner_states_.end(), 2));
    problem.SetParameterLowerBound(&inner_states_.back()->getOmega(), 0,
                                   -std::abs(3.0 / pre_inner_state->getVel()));
    problem.SetParameterUpperBound(&inner_states_.back()->getOmega(), 0,
                                   std::abs(3.0 / pre_inner_state->getVel()));

    problem.SetParameterLowerBound(&(inner_states_.back()->getAcc()), 0, -6.0);
    problem.SetParameterUpperBound(&(inner_states_.back()->getAcc()), 0, 10.0);
  }

  // theta manifold
  for (size_t m = 0; m < inner_states_.size(); ++m) {
    problem.SetManifold(&(inner_states_[m]->getTheta()), angle_manifold);
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 25;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
}

void MotionModelOptimize::pedMotionOptimize() {
  int opti_win = 3;
  double mean_vel_x = 0.0;
  double mean_vel_y = 0.0;
  double mean_yaw = 0.0;
  int count_valid_vel = 0;
  int count_valid_theta = 0;
  for (size_t m = std::max(0, (int)inner_states_.size() - opti_win);
       m < inner_states_.size(); ++m) {
    if (measure_states_[m]->isVelValid()) {
      count_valid_vel++;
      if (std::abs(measure_states_[m]->getVel()) >= 0.1) {
        mean_vel_x += measure_states_[m]->getVelX();
        mean_vel_y += measure_states_[m]->getVelY();
        // AINFO << "Vel: " << measure_states_[m]->getVel() << " "
        //       << measure_states_[m]->getVelX() << " "
        //       << measure_states_[m]->getVelY();
      }
    }

    if (measure_states_[m]->isValidTheta()) {
      count_valid_theta++;
      // AINFO << "Theta: " << measure_states_[m]->getTheta();
      auto tmp_measure_theta = measure_states_[m]->getTheta();
      thetaInverseCheck((*(std::prev(inner_states_.end(), 2)))->getTheta(),
                        tmp_measure_theta);
      // AINFO << "InverseCheckTheta: " << tmp_measure_theta << " prev theta: "
      //       << (*(std::prev(inner_states_.end(), 2)))->getTheta();
      mean_yaw += tmp_measure_theta;
    }
  }
  double vel_norm = 0.0;
  double vel_norm_x = 0.0;
  double vel_norm_y = 0.0;
  if (count_valid_vel >= 1) {
    // AINFO << "mean_vel_x_y: " << mean_vel_x << " " << mean_vel_y << " "
    //       << mean_vel_x / count_valid_vel << " " << mean_vel_y /
    //       count_valid_vel
    //       << " "
    //       << std::hypot(mean_vel_x / count_valid_vel,
    //                     mean_vel_y / count_valid_vel);
    mean_vel_x /= count_valid_vel;
    mean_vel_y /= count_valid_vel;
    vel_norm = std::hypot(mean_vel_x, mean_vel_y);
  } else {
    std::tie(vel_norm_x, vel_norm_y) = computePedMeasureVelByPos();
    vel_norm = std::hypot(vel_norm_x, vel_norm_y);
    if (std::abs(vel_norm) > 3) {
      vel_norm = std::min(vel_norm, inner_states_.back()->getVel());
    }
  }

  if (count_valid_theta > 0) {
    // AINFO << mean_yaw << " " << mean_yaw / count_valid_theta << " "
    //       << (*(std::prev(inner_states_.end(), 2)))->getTheta();
    mean_yaw /= count_valid_theta;
    thetaInverseCheck((*(std::prev(inner_states_.end(), 2)))->getTheta(),
                      mean_yaw);
    // AINFO << mean_yaw;
  }

  // 如果连续三帧观测都没有180°翻转，就对当前帧朝向做统一
  int reverse_count = 0;
  for (size_t m = std::max(0, (int)measure_states_.size() - 3);
       m < measure_states_.size(); ++m) {
    if (measure_states_[m]->isValidTheta() &&
        std::cos(measure_states_[m]->getTheta() - mean_yaw) < 0) {
      ++reverse_count;
    }
  }
  if (reverse_count >= 3) {
    thetaInverseCheck(measure_states_.back()->getTheta(), mean_yaw);
  }

  if (std::abs(vel_norm) < 0.3) {
    if (count_valid_theta == 0) {
      inner_states_.back()->getTheta() =
          (*(std::prev(inner_states_.end(), 2)))->getTheta();
    } else {
      inner_states_.back()->getTheta() = mean_yaw;
    }
    inner_states_.back()->getVel() = 0.0;
  } else {
    inner_states_.back()->getVel() = std::abs(vel_norm);
    if (count_valid_vel >= 1) {
      inner_states_.back()->getTheta() = std::atan2(mean_vel_y, mean_vel_x);
    } else {
      inner_states_.back()->getTheta() = std::atan2(vel_norm_y, vel_norm_x);
    }
  }

  inner_states_.back()->getX() = measure_states_.back()->getX();
  inner_states_.back()->getY() = measure_states_.back()->getY();
  inner_states_.back()->getOmega() = 0.0;
}

std::tuple<double, double>
MotionModelOptimize::computeMeasureWeightOfLengthWidth() {
  // compute the width and length var
  double mean_length = 0.0;
  double mean_width = 0.0;

  for (const auto& measure_ptr : measure_states_) {
    mean_length += measure_ptr->getLength();
    mean_width += measure_ptr->getWidth();
  }

  mean_length /= measure_states_.size();
  mean_width /= measure_states_.size();

  double var_length = 0.0;
  double var_width = 0.0;
  for (const auto& measure_ptr : measure_states_) {
    var_length += std::pow(measure_ptr->getLength() - mean_length, 2);
    var_width += std::pow(measure_ptr->getWidth() - mean_width, 2);
  }

  var_length = std::sqrt(var_length / (measure_states_.size() - 1));
  var_width = std::sqrt(var_width / (measure_states_.size() - 1));

  // compute the weight
  double measure_length_weight = 1.0;
  // double measure_width_weight = 1.0;

  // 平均观测差值小于历史观测方差,减少权重
  if (std::abs(mean_length - inner_states_.back()->getLength()) <
      std::max(0.3, var_length)) {
    measure_length_weight = 0.2;
  }
  // 超长车完全不信fov激光目标长度
  if (IsLidarSensor(measure_states_.back()->getBoxObserveSource()) &&
      measure_states_.back()->getLidarFovState() !=
          OUTSIDE_LIDAR_FOV_STATE::INSIDE_LIDAR_FOV &&
      inner_states_.back()->getLength() > 10) {
    measure_length_weight /= 10;
  }

  // if (std::abs(mean_width - inner_states_.back()->getWidth()) <
  //     std::max(0.3, var_width)) {
  //   measure_width_weight = 0.2;
  // }

  return std::tuple<double, double>(measure_length_weight,
                                    measure_length_weight);
}

void MotionModelOptimize::thetaKalmanSmooth(float observe_weight) {
  if (theta_kalman_ptr_ == nullptr) {
    double var_w = 1;
    double var_v = std::pow(2.5, 2);
    double p = std::pow(2.5, 2);
    double init_val = inner_states_[inner_states_.size() - 1]->getTheta();
    theta_kalman_ptr_ =
        std::make_unique<ThetaManifoldKalman>(var_w, p, var_v, init_val);
  }

  const int sz = 5;
  const double thr = 0.5;
  double var_w = 1.0;

  if (inner_states_.size() >= sz) {
    Eigen::Vector2d line_coeff;
    thetaLinePredict(sz, line_coeff);
    double yaw_rate = line_coeff[0] * 180.0 / M_PI;
    if (std::abs(yaw_rate) < thr) {
      var_w = std::pow(thr, 2);
    } else {
      var_w = std::min(std::pow(yaw_rate / thr, 2), 1.3);
    }
  }
  // 状态误差，角速度越大越贴近观测
  theta_kalman_ptr_->SetQ(var_w * observe_weight);
  double measure_theta = measure_states_.back()->getTheta();
  thetaInverseCheck(inner_states_.back()->getTheta(), measure_theta);
  inner_states_.back()->getTheta() = theta_kalman_ptr_->Filter(measure_theta);
}

void MotionModelOptimize::thetaLinePredict(int win_size,
                                           Eigen::Vector2d& theta_line_coeff) {
  std::vector<double> x_times(std::min(win_size, (int)measure_states_.size()),
                              0);
  std::vector<double> y_thetas(std::min(win_size, (int)measure_states_.size()),
                               0);

  int idx = 0;
  for (size_t k = std::max(0, (int)measure_states_.size() - win_size);
       k < measure_states_.size(); ++k) {
    x_times[idx] =
        inner_states_[k]->getTime() - inner_states_.back()->getTime();
    y_thetas[idx] = inner_states_[k]->getTheta();
    idx++;
  }

  double line_sigma = LineFiting2(x_times, y_thetas, theta_line_coeff);
}

bool MotionModelOptimize::isSourceChange() {
  if (measure_states_.size() >= 3) {
    auto current_source = measure_states_.back()->getBoxObserveSource();
    if ((*(std::prev(measure_states_.end(), 2)))->getBoxObserveSource() !=
            current_source ||
        (*(std::prev(measure_states_.end(), 3)))->getBoxObserveSource() !=
            current_source) {
      return true;
    }
    if (IsLidarSensor(current_source)) {
      auto is_lidar_box_refine = measure_states_.back()->getLidarBoxRefine();
      if ((*(std::prev(measure_states_.end(), 2)))->getLidarBoxRefine() !=
              is_lidar_box_refine ||
          (*(std::prev(measure_states_.end(), 3)))->getLidarBoxRefine() !=
              is_lidar_box_refine) {
        return true;
      }
    }
  }
  return false;
}
void MotionModelOptimize::OptimizeWithAverage(
    const size_t idx, const int window, const std::vector<float>& observe,
    const std::vector<float>& observe_weight) {
  if (idx > inner_states_.size() || idx <= 0) {
    return;
  }
  if (inner_states_.size() <= 1) {
    return;
  }
  int start_pos = std::max(0, static_cast<int>(inner_states_.size()) - window);
  float value_sum = 0;
  float weight_sum = 0;

  // 计入观测 可以无观测，无观测退化为历史平均
  for (int i = 0; i < observe.size(); ++i) {
    value_sum += observe[i] * observe_weight[i];
    weight_sum += observe_weight[i];
  }
  // 统计历史优化结果，不考虑当前帧的初始化输入
  for (int i = start_pos; i + 1 < inner_states_.size(); ++i) {
    value_sum += inner_states_[i]->getIdx(idx);
    weight_sum += 1;
  }
  if (weight_sum < 0.001) {
    return;
  }
  inner_states_.back()->getIdx(idx) = value_sum / weight_sum;
}
// 设置
double MotionModelOptimize::GetAccWithWindow(
    const size_t treat_idx, const size_t idx, const float change_thresh,
    const float jump_thresh, const bool is_angle, const int short_window,
    const int long_window) {
  // 前几帧yaw角不稳定
  if (inner_states_.size() < short_window) {
    return 0;
  } else if (inner_states_.size() < long_window) {
    double time_gap =
        inner_states_.back()->getTime() - inner_states_[0]->getTime();
    double value_diff =
        inner_states_.back()->getIdx(idx) - inner_states_[0]->getIdx(idx);
    if (is_angle) {
      value_diff = NormalizeAngle(value_diff);
    }
    if (time_gap < 0.001) {
      return 0;
    } else {
      double tmp = value_diff / time_gap;
      return std::abs(tmp) > jump_thresh
                 ? inner_states_[inner_states_.size() - 2]->getIdx(treat_idx)
                 : tmp;
    }
  }
  // 1s窗口和0.5s窗口
  double time_gap_10 =
      inner_states_.back()->getTime() -
      inner_states_[inner_states_.size() - long_window]->getTime();
  double value_diff_10 =
      inner_states_.back()->getIdx(idx) -
      inner_states_[inner_states_.size() - long_window]->getIdx(idx);
  double time_gap_5 =
      inner_states_.back()->getTime() -
      inner_states_[inner_states_.size() - short_window]->getTime();
  double value_diff_5 =
      inner_states_.back()->getIdx(idx) -
      inner_states_[inner_states_.size() - short_window]->getIdx(idx);

  if (is_angle) {
    value_diff_10 = NormalizeAngle(value_diff_10);
    value_diff_5 = NormalizeAngle(value_diff_5);
  }
  if (time_gap_10 < 0.001 || time_gap_5 < 0.001) {
    return 0;
  }
  double acc_10 = value_diff_10 / time_gap_10;
  double acc_5 = value_diff_5 / time_gap_5;

  // 异常值
  if (std::abs(acc_10) > jump_thresh && std::abs(acc_5) > jump_thresh) {
    return inner_states_[inner_states_.size() - 2]->getIdx(treat_idx);
  } else if (std::abs(acc_10) > jump_thresh) {
    return acc_5;
  } else if (std::abs(acc_5) > jump_thresh) {
    return acc_10;
  }
  // 窗口方向不一致
  if (acc_10 * acc_5 < 0 && std::abs(acc_5) < change_thresh) {
    return 0;
  } else if (acc_10 * acc_5 < 0 && std::abs(acc_5) >= change_thresh) {
    return acc_5;
  }
  // 同向高速变化选择较大的,低速变化选择小的
  if (std::abs(acc_10) > change_thresh && std::abs(acc_5) > change_thresh) {
    return std::abs(acc_10) > std::abs(acc_5) ? acc_10 : acc_5;
  } else {
    return std::abs(acc_10) > std::abs(acc_5) ? acc_5 : acc_10;
  }
}
}  // namespace perception
}  // namespace robosense
