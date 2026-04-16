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

#include "hyper_vision/perception/post_fusion/furnace/kalman_filter_optimize.h"

#include "cyber/cyber.h"

namespace robosense {
namespace perception {

void KalmanFilterOptimize::initKalmanFilter() {
  // 状态11，测量8，控制0
  kal_filter_ptr.reset(new rally::KalmanFilter<float, 11U, 8U, 0U>);
  std::int64_t state_dim = 11;
  std::int64_t measure_dim = 8;

  // step A. initial state transition matrix F
  // 以0.1ms固定时间设置状态转移矩阵 11*11 A
  setKFTransitionMatrix(DEFAULT_TIME_INTERVAL);  // 默认初始的时间间隔为0.1s

  // step B. measurement matrix H矩阵，认为只和当前观测有关
  Eigen::MatrixXf measure_matrix(measure_dim, state_dim);
  measure_matrix <<  //
      1.F,
      0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F,       // x
      0.F, 1.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F,  // y
      0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F,  // yaw
      0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F,  // width
      0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F,  // length
      0.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F, 0.F,  // v_x
      0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F,  // v_y
      0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F;  // v_yaw
  kal_filter_ptr->setMeasurementMatrix(measure_matrix);

  // step C. process noise Q 状态噪声矩阵 11*11
  Eigen::VectorXf process_noise_cov;
  process_noise_cov.resize(state_dim);
  // process_noise_cov << 0.4F, 0.4F, 0.4F, 0.4F, 0.4F, 0.8F, 0.8F, 0.8F, 0.8F,
  //     0.8F, 0.8F;
  process_noise_cov << 0.2F, 0.2F, 0.2F, 0.2F, 0.2F, 0.4F, 0.4F, 0.4F, 0.1F,
      0.1F, 0.1F;
  Eigen::MatrixXf process_noise_mat(process_noise_cov.asDiagonal());
  kal_filter_ptr->setProcessNoise(process_noise_mat);

  // step D. measurement noise R 观测误差矩阵 8*8 实际就是C
  Eigen::VectorXf measure_noise_cov;
  measure_noise_cov.resize(measure_dim);
  // measure_noise_cov << 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F;
  measure_noise_cov << 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F;
  Eigen::MatrixXf measure_noise_mat(measure_noise_cov.asDiagonal());
  kal_filter_ptr->setMeasurementNoise(measure_noise_mat);

  // step E. init state cov 状态方差 11*11
  Eigen::VectorXf post_err_cov;
  post_err_cov.resize(state_dim);
  post_err_cov << 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F;
  Eigen::MatrixXf err_cov_post_mat(post_err_cov.asDiagonal());

  // step F. initial state X
  Eigen::VectorXf init_state;
  init_state.resize(state_dim);
  init_state.setZero();
  kal_filter_ptr->setStateEstimate(init_state, err_cov_post_mat);
}

void KalmanFilterOptimize::setKFTransitionMatrix(double frame_time_interval) {
  float deltaT{static_cast<float>(frame_time_interval)};
  float acc_T{0.5F * deltaT * deltaT};
  // State:   x, y, direction, width, length, velocity_x, velocity_y,
  // velocity_direction, acc_x, acc_y, acc_direction Predict: x, y, direction,
  // width, length, velocity_x, velocity_y, velocity_direction
  Eigen::MatrixXf transition_matrix(11U, 11U);
  transition_matrix <<  //
      1.F,
      0.F, 0.F, 0.F, 0.F, deltaT, 0.F, 0.F, acc_T, 0.F, 0.F,       // x
      0.F, 1.F, 0.F, 0.F, 0.F, 0.F, deltaT, 0.F, 0.F, acc_T, 0.F,  // y
      0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F, deltaT, 0.F, 0.F, acc_T,  // yaw
      0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F,       // width
      0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F,       // length
      0.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, deltaT, 0.F, 0.F,    // v_x
      0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, deltaT, 0.F,    // v_y
      0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, deltaT,    // v_yaw
      0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F,       // acc_x
      0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F,       // acc_y
      0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 1.F;       // acc_yaw
  kal_filter_ptr->setTransitionMatrix(transition_matrix);
}

std::array<double, 11UL> KalmanFilterOptimize::convertToKalmanResult(
    Eigen::Matrix<float, 11, 1> system_state) {
  //
  return std::array<double, 11UL>({
      system_state(0),
      system_state(1),
      system_state(2),
      system_state(3),
      system_state(4),
      system_state(5),
      system_state(6),
      system_state(7),
      system_state(8),
      system_state(9),
      system_state(10),
  });
}

std::array<double, 2UL> KalmanFilterOptimize::convertToKalmanResult(
    Eigen::Matrix<float, 2, 1> system_state) {
  //
  return std::array<double, 2UL>({
      system_state(0),
      system_state(1),
  });
}

void KalmanFilterOptimize::createNewKalmanObject(
    const Vec3D& box_center_odom, const float yaw_odom,
    const Vec3D& velocity_odom, const bool is_velocity_valid,
    const bool is_lidar_source, uint8_t ref_point_pos, const BoxSize& box_size,
    const double time_stamp, Vec3D& sef_car_odom_xy,
    std::array<double, 11UL>& kalman_optimize_state) {
  auto new_theta = NormalizeAngle(yaw_odom);
  // double new_vel = isSameDirection(velocity_odom, new_theta)
  //                      ? std::hypot(velocity_odom.x, velocity_odom.y)
  //                      : -std::hypot(velocity_odom.x, velocity_odom.y);
  measure_states_.emplace_back(new MeasureObject(
      velocity_odom, box_center_odom, new_theta, box_size, time_stamp,
      is_lidar_source, is_velocity_valid, sef_car_odom_xy, ref_point_pos));
  initKalmanFilter();
  Eigen::VectorXf init_state;
  init_state.resize(11);
  init_state << box_center_odom.x, box_center_odom.y, new_theta, box_size.width,
      box_size.length, velocity_odom.x, velocity_odom.y, 0, 0, 0, 0;
  kal_filter_ptr->setStateEstimate(
      init_state,
      kal_filter_ptr
          ->getStateCovariance());  // 协方差矩阵是否要根据当前的状态计算？
                                    // 目前是初始化为单位矩阵
  trajectory.emplace_back(box_center_odom);
  auto system_state = kal_filter_ptr->getStateEstimate();
  kalman_optimize_state = convertToKalmanResult(system_state);
  pre_time_stamp = time_stamp;
  return;
}

void KalmanFilterOptimize::optimize(
    const Vec3D& box_center_odom, const float yaw_odom,
    const Vec3D& velocity_odom, const bool is_velocity_valid,
    const bool is_lidar_source, uint8_t ref_point_pos, const BoxSize& box_size,
    const double time_stamp, Vec3D& sef_car_odom_xy,
    std::array<double, 11UL>& kalman_optimize_state) {
  //
  auto new_theta = NormalizeAngle(yaw_odom);
  if (measure_states_.size() >= max_model_len_) {
    measure_states_.pop_front();
  }
  measure_states_.emplace_back(new MeasureObject(
      velocity_odom, box_center_odom, new_theta, box_size, time_stamp,
      is_lidar_source, is_velocity_valid, sef_car_odom_xy, ref_point_pos));

  double time_interval = time_stamp - pre_time_stamp;
  auto time_circle = static_cast<float>(1. / time_interval);
  auto pre_state = kal_filter_ptr->getStateEstimate();

  Vec3D vel_measure = calMeasureVel();

  float rotate = new_theta - pre_state(2);
  float sign = rotate > 0.F ? -1.F : 1.F;
  while ((rotate > rally::R_M_HALF_PI + rally::R_F_EPS) ||
         (rotate < -rally::R_M_HALF_PI - rally::R_F_EPS)) {
    rotate += sign * rally::R_M_PI;
  }
  float yaw_rate = rotate * time_circle;

  // // 度量加速度
  // Vec3D tmp_acc_odom;
  // tmp_acc_odom.x = (vel_odom_optimize.x - pre_state(5)) * time_circle;
  // tmp_acc_odom.y = (vel_odom_optimize.y - pre_state(6)) * time_circle;

  if (trajectory.size() < MIN_TRACK_COUNT_) {
    Eigen::VectorXf init_state;
    init_state.resize(11);
    init_state << box_center_odom.x, box_center_odom.y, new_theta,
        box_size.width, box_size.length, vel_measure.x, vel_measure.y, yaw_rate,
        0, 0, 0;
    kal_filter_ptr->setStateEstimate(init_state,
                                     kal_filter_ptr->getStateCovariance());
  } else {
    setKFTransitionMatrix(time_interval);
    kal_filter_ptr->predict();
    auto predict_state = kal_filter_ptr->getStateEstimate();

    float predict_yaw_odom = predict_state(2);
    float yaw_diff{new_theta - predict_yaw_odom};
    float sign_tmp{yaw_diff > std::numeric_limits<float>::epsilon() ? -1.F
                                                                    : 1.F};
    while ((yaw_diff > rally::R_M_HALF_PI + rally::R_F_EPS) ||
           (yaw_diff < -rally::R_M_HALF_PI - rally::R_F_EPS)) {
      yaw_diff += sign_tmp * rally::R_M_PI;
    }
    float measure_yaw_odom = predict_yaw_odom + yaw_diff;

    Eigen::VectorXf measure(8U);
    measure << box_center_odom.x, box_center_odom.y, measure_yaw_odom,
        box_size.width, box_size.length, vel_measure.x, vel_measure.y, yaw_rate;
    kal_filter_ptr->update(measure);
  }
  trajectory.emplace_back(box_center_odom);
  auto system_state = kal_filter_ptr->getStateEstimate();
  kalman_optimize_state = convertToKalmanResult(system_state);
  pre_time_stamp = time_stamp;
  return;
}

void KalmanFilterOptimize::addMeasureObj() {
  //
}

Vec3D KalmanFilterOptimize::calMeasureVel() {
  //
  // double time_interval = measure_states_.back()->timestamp - pre_time_stamp;
  // auto time_circle = static_cast<float>(1. / time_interval);
  // auto pre_state = kal_filter_ptr->getStateEstimate();

  // Vec3D vel_odom_measure;
  // vel_odom_measure.x =
  //     (measure_states_.back()->box_center_odom.x - pre_state(0)) *
  //     time_circle;
  // vel_odom_measure.y =
  //     (measure_states_.back()->box_center_odom.y - pre_state(1)) *
  //     time_circle;
  // Vec3D vel_odom_optimize;

  // if (measure_states_.back()->is_vel_valid) {
  //   vel_odom_optimize.x =
  //       std::abs(vel_odom_measure.x) >=
  //               std::abs(measure_states_.back()->velocity_odom.x)
  //           ? measure_states_.back()->velocity_odom.x
  //           : vel_odom_measure.x;
  //   vel_odom_optimize.y =
  //       std::abs(vel_odom_measure.y) >=
  //               std::abs(measure_states_.back()->velocity_odom.y)
  //           ? measure_states_.back()->velocity_odom.y
  //           : vel_odom_measure.y;
  //   Eigen::VectorXf measure_noise_cov;
  //   measure_noise_cov.resize(8);
  //   measure_noise_cov << 1.F, 1.F, 1.F, 1.F, 1.F, 0.2F, 0.2F, 1.F;
  //   Eigen::MatrixXf measure_noise_mat(measure_noise_cov.asDiagonal());
  //   kal_filter_ptr->setMeasurementNoise(measure_noise_mat);
  // } else {
  //   vel_odom_optimize = vel_odom_measure;
  //   Eigen::VectorXf measure_noise_cov;
  //   measure_noise_cov.resize(8);
  //   measure_noise_cov << 1.F, 1.F, 1.F, 1.F, 1.F, 2.F, 2.F, 1.F;
  //   Eigen::MatrixXf measure_noise_mat(measure_noise_cov.asDiagonal());
  //   kal_filter_ptr->setMeasurementNoise(measure_noise_mat);
  //   // step C. process noise Q
  //   Eigen::VectorXf process_noise_cov;
  //   process_noise_cov.resize(11);
  //   process_noise_cov << 0.2F, 0.2F, 0.2F, 0.2F, 0.2F, 0.F, 0.F, 0.F, 0.1F,
  //       0.1F, 0.1F;
  //   Eigen::MatrixXf process_noise_mat(process_noise_cov.asDiagonal());
  //   kal_filter_ptr->setProcessNoise(process_noise_mat);
  // }

  // // bug，velocity_odom（即上游速度）可能不可用
  // float max_vel_abs =
  //     std::max(std::hypot(vel_odom_measure.x, vel_odom_measure.y),
  //              std::hypot(measure_states_.back()->velocity_odom.x,
  //                         measure_states_.back()->velocity_odom.y));
  // if (max_vel_abs < 0.5) {
  //   vel_odom_optimize.x = 0.;
  //   vel_odom_optimize.y = 0.;
  // }

  Eigen::VectorXf measure_noise_cov;
  measure_noise_cov.resize(8);
  Eigen::VectorXf process_noise_cov;
  process_noise_cov.resize(11);

  const auto& cur_measure = measure_states_.back();
  const auto& pre_measure = *(std::prev(measure_states_.end(), 2));
  Vec3D vel_odom_measure;
  Vec3D vel_odom_box_measure = computeBoxMeasureVel();
  if (cur_measure->is_vel_valid) {
    vel_odom_measure =
        std::hypot(cur_measure->velocity_odom.x,
                   cur_measure->velocity_odom.y) <=
                std::hypot(vel_odom_box_measure.x, vel_odom_box_measure.y)
            ? cur_measure->velocity_odom
            : vel_odom_box_measure;
    measure_noise_cov << 1.F, 1.F, 1.F, 1.F, 1.F, 0.2F, 0.2F, 1.F;
    Eigen::MatrixXf measure_noise_mat(measure_noise_cov.asDiagonal());
    kal_filter_ptr->setMeasurementNoise(measure_noise_mat);
  } else {
    vel_odom_measure = vel_odom_box_measure;
    measure_noise_cov << 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F;
    Eigen::MatrixXf measure_noise_mat(measure_noise_cov.asDiagonal());
    kal_filter_ptr->setMeasurementNoise(measure_noise_mat);
    process_noise_cov << 0.2F, 0.2F, 0.2F, 0.2F, 0.2F, 0.2F, 0.2F, 0.2F, 0.1F,
        0.1F, 0.1F;
    Eigen::MatrixXf process_noise_mat(process_noise_cov.asDiagonal());
    kal_filter_ptr->setProcessNoise(process_noise_mat);
  }

  // y方向速度限制
  // std::cout << "vel measurements: " << vel_odom_measure.x << " "
  //           << vel_odom_measure.y << std::endl;
  double vel_obj_base_x = vel_odom_measure.x * std::cos(cur_measure->yaw_odom) +
                          vel_odom_measure.y * std::sin(cur_measure->yaw_odom);
  double vel_obj_base_y = vel_odom_measure.y * std::cos(cur_measure->yaw_odom) -
                          vel_odom_measure.x * std::sin(cur_measure->yaw_odom);
  // std::cout << "vel_obj_base: " << vel_obj_base_x << " " << vel_obj_base_y
  //           << std::endl;
  if (std::abs(vel_obj_base_x) > 5) {
    vel_obj_base_y = 0;
  } else if (std::abs(vel_obj_base_y) > 0.3 * std::abs(vel_obj_base_x)) {
    //
    vel_obj_base_y *= 0.3 * std::abs(vel_obj_base_x) / std::abs(vel_obj_base_y);
  }
  vel_odom_measure.x = vel_obj_base_x * std::cos(cur_measure->yaw_odom) -
                       vel_obj_base_y * std::sin(cur_measure->yaw_odom);
  vel_odom_measure.y = vel_obj_base_y * std::cos(cur_measure->yaw_odom) +
                       vel_obj_base_x * std::sin(cur_measure->yaw_odom);

  // std::cout << "vel measurements: " << vel_odom_measure.x << " "
  //           << vel_odom_measure.y << std::endl;

  return vel_odom_measure;
}

Vec3D KalmanFilterOptimize::computeBoxMeasureVel() {
  if (measure_states_.size() < 2) {
    return Vec3D(0.0, 0.0, 0.0);
  }

  // // lidar staitc check
  // int lidar_static_check_win =
  //     measure_states_.size() > 10 ? measure_states_.size() - 10 : 0;
  // bool is_pre_lidar_static =
  //     !measure_states_.back()->is_lidar_source &&
  //     measure_states_[lidar_static_check_win]->is_lidar_source &&
  //     std::hypot(measure_states_[lidar_static_check_win]->velocity_odom.x,
  //                measure_states_[lidar_static_check_win]->velocity_odom.y) <
  //         0.3;

  // if (is_pre_lidar_static) {
  //   return Vec3D(0.0, 0.0, 0.0);
  // }

  int half_win = std::floor(measure_states_.size() / 2);
  uint8_t ref_point_status = measure_states_.back()->ref_point_pos;

  std::vector<double> first_pt_status(6, 0);   // [x, y, c_x, c_y, timestamp, n]
  std::vector<double> second_pt_status(6, 0);  // [x, y, c_x, c_y, timestamp, n]

  double pt_x;
  double pt_y;
  double pt_c_x;
  double pt_c_y;
  for (size_t k = 0; k < measure_states_.size(); k++) {
    pt_c_x = measure_states_[k]->box_center_odom.x;
    pt_c_y = measure_states_[k]->box_center_odom.y;
    pt_x = pt_c_x;
    pt_y = pt_c_y;

    if (ref_point_status != REF_POINT_CENTER) {
      bool is_front = (measure_states_[k]->box_center_odom.x -
                       measure_states_[k]->sef_car_odom_xy.x) *
                              std::cos(measure_states_[k]->yaw_odom) +
                          std::sin(measure_states_[k]->yaw_odom) *
                              (measure_states_[k]->box_center_odom.y -
                               measure_states_[k]->sef_car_odom_xy.y) <
                      0.0;

      if (is_front) {
        pt_x += measure_states_[k]->box_size.length * 0.5 *
                std::cos(measure_states_[k]->yaw_odom);
        pt_y += measure_states_[k]->box_size.length * 0.5 *
                std::sin(measure_states_[k]->yaw_odom);
      } else {
        pt_x -= measure_states_[k]->box_size.length * 0.5 *
                std::cos(measure_states_[k]->yaw_odom);
        pt_y -= measure_states_[k]->box_size.length * 0.5 *
                std::sin(measure_states_[k]->yaw_odom);
      }
    }

    if (k < half_win) {
      first_pt_status[0] += pt_x;
      first_pt_status[1] += pt_y;
      first_pt_status[2] += pt_c_x;
      first_pt_status[3] += pt_c_y;
      first_pt_status[4] += measure_states_[k]->timestamp;
      first_pt_status[5] += 1.0;
    } else {
      second_pt_status[0] += pt_x;
      second_pt_status[1] += pt_y;
      second_pt_status[2] += pt_c_x;
      second_pt_status[3] += pt_c_y;
      second_pt_status[4] += measure_states_[k]->timestamp;
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

  if (std::hypot(delta_x, delta_y) > std::hypot(delta_c_x, delta_c_y)) {
    delta_x = delta_c_x;
    delta_y = delta_c_y;
  }

  double vel_theta_x = (delta_x * std::cos(measure_states_.back()->yaw_odom) +
                        delta_y * std::sin(measure_states_.back()->yaw_odom)) /
                       delta_time;
  Vec3D vel_measure(delta_x / delta_time, delta_y / delta_time, 0.0);

  if (measure_states_.back()->is_lidar_source) {
    return vel_measure;
  } else {
    return std::hypot(vel_measure.x, vel_measure.y) <= 1.0
               ? Vec3D(0.0, 0.0, 0.0)
               : vel_measure;
  }
}

void KalmanFilterOptimize::initStaticObjKalmanFilter() {
  staticobj_kalman_filter_ptr.reset(new rally::KalmanFilter<float, 2U, 2U, 0U>);
  std::int64_t state_dim = 2;
  std::int64_t measure_dim = 2;

  // step A. initial state transition matrix F
  setStaticObjKFTransitionMatrix();  // 默认初始的时间间隔为0.1s

  // step B. measurement matrix H
  Eigen::MatrixXf measure_matrix(measure_dim, state_dim);
  measure_matrix << 1.F, 0.F, 0.F, 1.F;
  staticobj_kalman_filter_ptr->setMeasurementMatrix(measure_matrix);

  // step C. process noise Q
  Eigen::VectorXf process_noise_cov;
  process_noise_cov.resize(state_dim);
  process_noise_cov << 1.F, 1.F;
  Eigen::MatrixXf process_noise_mat(process_noise_cov.asDiagonal());
  staticobj_kalman_filter_ptr->setProcessNoise(process_noise_mat);

  // step D. measurement noise R
  Eigen::VectorXf measure_noise_cov;
  measure_noise_cov.resize(measure_dim);
  measure_noise_cov << 1.F, 1.F;
  Eigen::MatrixXf measure_noise_mat(measure_noise_cov.asDiagonal());
  staticobj_kalman_filter_ptr->setMeasurementNoise(measure_noise_mat);

  // step E. init state cov
  Eigen::VectorXf post_err_cov;
  post_err_cov.resize(state_dim);
  post_err_cov << 1.F, 1.F;
  Eigen::MatrixXf err_cov_post_mat(post_err_cov.asDiagonal());

  // step F. initial state X
  Eigen::VectorXf init_state;
  init_state.resize(state_dim);
  init_state.setZero();
  staticobj_kalman_filter_ptr->setStateEstimate(init_state, err_cov_post_mat);
}

void KalmanFilterOptimize::setStaticObjKFTransitionMatrix() {
  Eigen::MatrixXf transition_matrix(2U, 2U);
  transition_matrix << 1.F, 0.F, 0.F, 1.F;
  staticobj_kalman_filter_ptr->setTransitionMatrix(transition_matrix);
}

void KalmanFilterOptimize::createNewKalmanStaticObject(
    const Vec3D& box_center_odom, const double time_stamp,
    uint8_t exist_confidence, uint8_t det_exist_confidence, SensorID sensor_id,
    Vec3D& sef_car_odom_xy, std::array<double, 2UL>& kalman_optimize_state) {
  initStaticObjKalmanFilter();
  Eigen::VectorXf init_state;
  init_state.resize(2);
  init_state << box_center_odom.x, box_center_odom.y;
  staticobj_kalman_filter_ptr->setStateEstimate(
      init_state,
      staticobj_kalman_filter_ptr
          ->getStateCovariance());  // 协方差矩阵是否要根据当前的状态计算？
                                    // 目前是初始化为单位矩阵
  trajectory.emplace_back(box_center_odom);
  auto system_state = staticobj_kalman_filter_ptr->getStateEstimate();
  kalman_optimize_state = convertToKalmanResult(system_state);
  return;
}

void KalmanFilterOptimize::optimizeStaticObject(
    const Vec3D& box_center_odom, const double time_stamp,
    uint8_t exist_confidence, uint8_t det_exist_confidence, SensorID sensor_id,
    Vec3D& sef_car_odom_xy, std::array<double, 2UL>& kalman_optimize_state) {
  staticobj_measure_states_.emplace_back(new MeasureStaticObject(
      box_center_odom, time_stamp, det_exist_confidence, sensor_id));
  if (staticobj_kalman_filter_ptr == nullptr) {
    createNewKalmanStaticObject(box_center_odom, time_stamp, exist_confidence,
                                det_exist_confidence, sensor_id,
                                sef_car_odom_xy, kalman_optimize_state);
    return;
  }
  //
  Eigen::VectorXf measure_noise_cov;
  measure_noise_cov.resize(2);
  Eigen::VectorXf process_noise_cov;
  process_noise_cov.resize(2);

  if (exist_confidence == 0) {
    // 高置信度目标尽量压在原来的位置
    measure_noise_cov << 1.F, 1.F;
    Eigen::MatrixXf measure_noise_mat(measure_noise_cov.asDiagonal());
    staticobj_kalman_filter_ptr->setMeasurementNoise(measure_noise_mat);
    process_noise_cov << 0.02F, 0.02F;
    Eigen::MatrixXf process_noise_mat(process_noise_cov.asDiagonal());
    staticobj_kalman_filter_ptr->setProcessNoise(process_noise_mat);
  } else if (det_exist_confidence == 0) {
    measure_noise_cov << 0.1F, 0.1F;
    Eigen::MatrixXf measure_noise_mat(measure_noise_cov.asDiagonal());
    staticobj_kalman_filter_ptr->setMeasurementNoise(measure_noise_mat);
    process_noise_cov << 1.F, 1.F;
    Eigen::MatrixXf process_noise_mat(process_noise_cov.asDiagonal());
    staticobj_kalman_filter_ptr->setProcessNoise(process_noise_mat);
  } else if (isPinholeSensor(sensor_id) && exist_confidence <= 1) {
    measure_noise_cov << 1.F, 1.F;
    Eigen::MatrixXf measure_noise_mat(measure_noise_cov.asDiagonal());
    staticobj_kalman_filter_ptr->setMeasurementNoise(measure_noise_mat);
    process_noise_cov << 0.2F, 0.2F;
    Eigen::MatrixXf process_noise_mat(process_noise_cov.asDiagonal());
    staticobj_kalman_filter_ptr->setProcessNoise(process_noise_mat);
  } else {
    measure_noise_cov << 1.F, 1.F;
    Eigen::MatrixXf measure_noise_mat(measure_noise_cov.asDiagonal());
    staticobj_kalman_filter_ptr->setMeasurementNoise(measure_noise_mat);
    process_noise_cov << 0.1F, 0.1F;
    Eigen::MatrixXf process_noise_mat(process_noise_cov.asDiagonal());
    staticobj_kalman_filter_ptr->setProcessNoise(process_noise_mat);
  }

  staticobj_kalman_filter_ptr->predict();
  Eigen::VectorXf measure(2U);
  measure << box_center_odom.x, box_center_odom.y;
  staticobj_kalman_filter_ptr->update(measure);

  auto system_state = staticobj_kalman_filter_ptr->getStateEstimate();
  kalman_optimize_state = convertToKalmanResult(system_state);
}

cv::Mat KalmanFilterExtendForOdom::UpdateObserve(
    const std::pair<float, double>& x_obs,
    const std::pair<float, double>& y_obs, const float& w_x, const float& w_y,
    const float& v_x_obs, const float& v_y_obs, const float& w_v_x,
    const float& w_v_y, const double cur_time, const BoxSize observe_size,
    const BoxSize cur_size, const float yaw, const uint8_t ref_status) {
  ++frame_count;
  time_gap = cur_time - pre_update_time;
  pre_update_time = cur_time;

  *A.ptr<float>(0, 2) = time_gap;
  *A.ptr<float>(1, 3) = time_gap;

  // 重构输入情况
  float input_ref_x =
      x_obs.first + *pre_predict_res.ptr<float>(2, 0) * x_obs.second;
  float input_ref_y =
      y_obs.first + *pre_predict_res.ptr<float>(3, 0) * y_obs.second;
  float pre_ref_x = pre_center.x;
  float pre_ref_y = pre_center.y;

  Eigen::Vector2f input_ref =
      ComputeRefPt(input_ref_x, input_ref_y, observe_size, yaw,
                   static_cast<RefPointPos>(ref_status));
  Eigen::Vector2f pre_ref =
      ComputeRefPt(pre_ref_x, pre_ref_y, pre_optim_size, pre_yaw,
                   static_cast<RefPointPos>(ref_status));

  input_ref_x = input_ref.x();
  input_ref_y = input_ref.y();
  pre_ref_x = pre_ref.x();
  pre_ref_y = pre_ref.y();

  //   AINFO << " REF LOG " << int(ref_status) << " OBSRVE " << x_obs.first << "
  //   "
  //         << y_obs.first << " INPUT " << input_ref_x << " " << input_ref_y
  //         << " pre center " << pre_center.x << " " << pre_center.y << " ref"
  //         << pre_ref_x << " " << pre_ref_y << " size " << observe_size.length
  //         << " " << observe_size.width << " " << cur_size.length << " "
  //         << cur_size.width << " pre " << pre_optim_size.length << " "
  //         << pre_optim_size.width;
  *pre_predict_res.ptr<float>(0, 0) = pre_ref_x;
  *pre_predict_res.ptr<float>(1, 0) = pre_ref_y;
  *observe_mat.ptr<float>(0, 0) = input_ref_x;
  *observe_mat.ptr<float>(1, 0) = input_ref_y;

  *observe_mat.ptr<float>(2, 0) = v_x_obs;
  *observe_mat.ptr<float>(3, 0) = v_y_obs;

  *C.ptr<float>(0, 0) = w_x;
  *C.ptr<float>(1, 1) = w_y;
  *C.ptr<float>(2, 2) = w_v_x;
  *C.ptr<float>(3, 3) = w_v_y;

  // 预测
  //   AINFO << " PRE PREDICT " << pre_predict_res;
  //   AINFO << "A " << A;
  //   AINFO << " OBSER " << observe_mat;
  //   AINFO << "C " << C;
  pre_predict_res = A * pre_predict_res;

  // 更新方差
  //   AINFO << "Q " << Q;
  M = A.t() * M * A + Q;
  //   AINFO << "M " << M;

  // 更新增益
  cv::Mat tmp = (H * M * H.t() + C);
  //   AINFO << "TMP " << tmp;
  K = M * H.t() * tmp.inv();
  //   AINFO << "k" << " " << K;

  // 更新预测
  //   AINFO << " PREDICT OBSERVE" << H * pre_predict_res;
  //   AINFO << " DIFF " << observe_mat - H * pre_predict_res;
  pre_predict_res = pre_predict_res + K * (observe_mat - H * pre_predict_res);

  //   AINFO << "CUR PREDICT " << pre_predict_res.t();

  // 更新方差
  M = (cv::Mat::eye(M.rows, M.cols, CV_32F) - K * H) * M;

  // 重构输出用实际平均长度
  Eigen::Vector2f out_center = ComputeCenterPt(
      *pre_predict_res.ptr<float>(0, 0), *pre_predict_res.ptr<float>(1, 0),
      cur_size, yaw, static_cast<RefPointPos>(ref_status));
  pre_center.x = out_center.x();
  pre_center.y = out_center.y();
  SetPredict(pre_center.x, pre_center.y, *pre_predict_res.ptr<float>(2, 0),
             *pre_predict_res.ptr<float>(3, 0), cur_time, cur_size, yaw);
  //   AINFO << "recorver  PREDICT " << pre_predict_res.t();

  return pre_predict_res;
}
}  // namespace perception
}  // namespace robosense
