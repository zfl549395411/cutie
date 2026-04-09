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

#include "robot/perception/uwb_postprocess/process/track_obj.h"
namespace robot {
namespace perception {
void TrackObj::UpdateObserve(const ObserveObj::Ptr& uwb_obj,
                             const Eigen::Affine3f& rel_to_base_trans) {
  // 图像观测直接替换
  if (isCameraSensor(uwb_obj->sensor_id_)) {
    box_size = uwb_obj->box_size_;
  }
  double diff_time = uwb_obj->time_stamp_ - motion_optimize_->time_stamp_;
  if (diff_time <= 0.005) {
    return;
  }
  sensor_id_ = uwb_obj->sensor_id_;
  // 转到参考点坐标系
  Eigen::Affine3f inner_base_to_odom_trans =
      rel_to_ref_origin_ * rel_to_base_trans.inverse();
  Eigen::Vector3f pose_odom_decompostion =
      inner_base_to_odom_trans * uwb_obj->pose_base_decomposition;
  Eigen::Vector3f pose_odom_label =
      inner_base_to_odom_trans * uwb_obj->pose_base_label;
  float distance_base = uwb_obj->pose_base_decomposition.norm();
  float distance_odom = pose_odom_decompostion.norm();
  float angle_base = std::atan2(uwb_obj->pose_base_decomposition[1],
                                uwb_obj->pose_base_decomposition[0]);
  float angle_odom =
      std::atan2(pose_odom_decompostion[1], pose_odom_decompostion[0]);

  // 选择最接近的
  Eigen::Vector2f decom_diff(pose_odom_decompostion[0] - box_center_odom[0],
                             pose_odom_decompostion[1] - box_center_odom[1]);
  Eigen::Vector2f label_diff(pose_odom_label[0] - box_center_odom[0],
                             pose_odom_label[1] - box_center_odom[1]);
  if (decom_diff.norm() < label_diff.norm()) {
    motion_optimize_->AddObserve(
        pose_odom_decompostion, uwb_obj->pose_base_decomposition, distance_base,
        distance_odom, angle_base, angle_odom, uwb_obj->time_stamp_,
        uwb_obj->sensor_id_, uwb_pose_code, track_state, observe_sensor,
        cover_state);
  } else {
    motion_optimize_->AddObserve(
        pose_odom_label, uwb_obj->pose_base_label, distance_base, distance_odom,
        angle_base, angle_odom, uwb_obj->time_stamp_, uwb_obj->sensor_id_,
        uwb_pose_code, track_state, observe_sensor, cover_state);
  }

  // 无odom，直接透传base坐标系数据，不做优化处理
  if (inner_base_to_odom_trans.matrix() == Eigen::Matrix4f::Identity()) {
    box_center_odom = uwb_obj->pose_base_decomposition;
  } else {
    auto optimize_out = motion_optimize_->Optimize();
    box_center_odom = optimize_out.GetPose();
    velocity_odom = optimize_out.GetVel();
    yaw_odom = std::atan2(velocity_odom[1], velocity_odom[0]);
  }
  time_stamp = uwb_obj->time_stamp_;

  box_center_base = inner_base_to_odom_trans.inverse() * box_center_odom;
  yaw_base = transformTo(inner_base_to_odom_trans.inverse(), yaw_odom);
  velocity_base =
      velTransformTo(inner_base_to_odom_trans.inverse(), velocity_odom);
  //   history_pose["observe"].emplace_back(pose_odom_decompostion);
  //   if (isCameraSensor(uwb_obj->sensor_id_) &&
  //       checkBitsetOfSensorIds(SensorID::UWB_LOCATION_FRONT,
  //                              fusion_obj->sensor_ids_)) {
  //     for (const auto& obj : fusion_obj->observe_list) {
  //       if (obj->sensor_id_ == SensorID::UWB_LOCATION_FRONT) {
  //         Eigen::Vector3f uwb_pose =
  //             inner_base_to_odom_trans * obj->pose_base_decomposition;
  //         history_pose["uwb_pose"].emplace_back(uwb_pose);
  //       }
  //     }
  //   }
  //   history_pose["out"].emplace_back(box_center_odom);
  //   Eigen::Vector3f translation = inner_base_to_odom_trans.translation();
  //   history_pose["pose"].emplace_back(translation);
  //   if (history_pose["out"].size() % 100 == 0) {
  //     BevDraw tmp_draw;
  //     tmp_draw.init();
  //     tmp_draw.drawTraj(history_pose["observe"], cv::Scalar(255, 0, 0));
  //     if (history_pose.find("uwb_pose") != history_pose.end()) {
  //       tmp_draw.drawTraj(history_pose["uwb_pose"], cv::Scalar(0, 255, 0));
  //     }
  //     tmp_draw.drawTraj(history_pose["out"], cv::Scalar(255, 0, 255));
  //     tmp_draw.drawTraj(history_pose["pose"], cv::Scalar(0, 0, 0));
  //     rally::ensureDirectory("/apollo/mydata/uwb_debug/");
  //     tmp_draw.save("/apollo/mydata/uwb_debug/" +
  //                   std::to_string(uwb_obj->time_stamp_) + ".jpg");
  //   }
  pre_observe_time_stamp = time_stamp;
}
void TrackObj::predict(const double predict_time_stamp,
                       const Eigen::Affine3f& rel_to_base_trans) {
  if (predict_time_stamp < time_stamp + 0.01) {  // 10 ms
    return;
  }

  auto inner_odom_to_base_trans =
      rel_to_base_trans * rel_to_ref_origin_.inverse();
  // update the odom info
  float delta_time = predict_time_stamp - time_stamp;
  double predict_time = time_stamp - pre_observe_time_stamp;
  float vel = std::hypotf(velocity_odom.x(), velocity_odom.y());
  // 静止目标预测不调整位置，近距离目标和丢失时间不时间前推
  if (vel > 0.3 && delta_time < 0.5 && box_center_base.norm() > 1.0f &&
      (track_state != TRACKER_STATE_CODE::EDGE_LOST_TRACK ||
       predict_time < 1)) {
    box_center_odom.x() = box_center_odom.x() + velocity_odom.x() * delta_time;
    box_center_odom.y() = box_center_odom.y() + velocity_odom.y() * delta_time;
  }
  yaw_odom = NormalizeAngle(yaw_odom + yaw_rate * delta_time);

  vel = isSameDirection(velocity_odom, yaw_odom) ? vel : -vel;
  float acc = std::hypot(acceleration_odom.x(), acceleration_odom.y());
  acc = isSameDirection(acceleration_odom, yaw_odom) ? acc : -acc;

  // update the base info
  box_center_base = inner_odom_to_base_trans * box_center_odom;

  yaw_base = transformTo(inner_odom_to_base_trans, yaw_odom);
  velocity_base.x() = vel * std::cos(yaw_base);
  velocity_base.y() = vel * std::sin(yaw_base);
  acceleration_base.x() = acc * std::cos(yaw_base);
  acceleration_base.y() = acc * std::sin(yaw_base);

  // update the polygon info

  time_stamp = predict_time_stamp;
}
void TrackObj::ChooseObserve(const Eigen::Affine3f& rel_to_base_trans) {
  CheckCurPose();
  float nearest_dist = 100000;
  int best_idx = -1;
  for (int i = 0; i < fusion_obj->observe_list.size(); ++i) {
    const auto& observe_obj = fusion_obj->observe_list[i];
    if (isCameraSensor(observe_obj->sensor_id_)) {
      if (track_state == TRACKER_STATE_CODE::EDGE_LOST_TRACK ||
          track_state == TRACKER_STATE_CODE::ERROR_TRACK) {
        continue;
      } else if (track_state == TRACKER_STATE_CODE::STABLE_TRACK) {
        best_idx = i;
        break;
      }
    }
    if (isUwbLocationSensor(observe_obj->sensor_id_) &&
        ((uwb_pose_code != UWB_POSE_CODE::BACK &&
          observe_obj->uwb_msg_->sensor_id != SensorID::UWB_LOCATION_FRONT) ||
         (uwb_pose_code != UWB_POSE_CODE::BACK &&
          observe_obj->uwb_msg_->sensor_id == SensorID::UWB_LOCATION_BACK))) {
      continue;
    }
    Eigen::Vector3f pose_diff =
        box_center_base - observe_obj->pose_base_decomposition;
    if (pose_diff.norm() < nearest_dist) {
      best_idx = i;
      nearest_dist = pose_diff.norm();
    }
  }
  if (best_idx >= 0) {
    UpdateObserve(fusion_obj->observe_list[best_idx], rel_to_base_trans);
  }
  PrintInfo();
}

void TrackObj::CheckCurPose() {
  // 前视视觉正常跟踪，那一定是前向
  if (checkCameraChannlesOfSensorIds(fusion_obj->sensor_ids_) != 0) {
    uwb_pose_code == UWB_POSE_CODE::FRONT;
    return;
  }
  ObserveObj::Ptr front_uwb_ptr, back_uwb_ptr;
  for (const auto& observe_obj : fusion_obj->observe_list) {
    if (observe_obj->uwb_msg_ == nullptr) {
      continue;
    }
    if (observe_obj->uwb_msg_->sensor_id == SensorID::UWB_LOCATION_FRONT) {
      front_uwb_ptr = observe_obj;
    } else if (observe_obj->uwb_msg_->sensor_id ==
               SensorID::UWB_LOCATION_BACK) {
      back_uwb_ptr = observe_obj;
    }
  }
  if (front_uwb_ptr == nullptr && back_uwb_ptr == nullptr) {
    return;
  } else if (front_uwb_ptr == nullptr) {
    if ((uwb_pose_code == UWB_POSE_CODE::FRONT && box_center_base.norm() < 5) ||
        uwb_pose_code == UWB_POSE_CODE::UNKNOWN) {
      uwb_pose_code = UWB_POSE_CODE::BACK;
    }
    return;
  } else if (back_uwb_ptr == nullptr) {
    uwb_pose_code = UWB_POSE_CODE::FRONT;
    return;
  } else {
    if (front_uwb_ptr->uwb_msg_->uwb_data.distance <
        back_uwb_ptr->uwb_msg_->uwb_data.distance) {
      if ((uwb_pose_code == UWB_POSE_CODE::BACK &&
           box_center_base.norm() < 1) ||
          uwb_pose_code == UWB_POSE_CODE::UNKNOWN) {
        uwb_pose_code = UWB_POSE_CODE::FRONT;
      }
    } else if ((uwb_pose_code == UWB_POSE_CODE::FRONT &&
                box_center_base.norm() < 1) ||
               uwb_pose_code == UWB_POSE_CODE::UNKNOWN) {
      uwb_pose_code = UWB_POSE_CODE::BACK;
    }
  }
}
ObjectFusion TrackObj::ToPubObject(const double pub_time_stamp,
                                   const Eigen::Affine3f& rel_to_base_trans) {
  ObjectFusion pub_obj;
  pub_obj.track_id = label_id_;
  pub_obj.type = type;

  predict(pub_time_stamp, rel_to_base_trans);
  pub_obj.box_center_base =
      Vec3D(box_center_base[0], box_center_base[1], box_center_base[2]);
  pub_obj.box_size = box_size;
  pub_obj.yaw_base = yaw_base;

  pub_obj.yaw_rate = yaw_rate;
  pub_obj.velocity_base =
      Vec3D(velocity_base[0], velocity_base[1], velocity_base[2]);
  pub_obj.acceleration_base =
      Vec3D(acceleration_base[0], acceleration_base[1], acceleration_base[2]);
  pub_obj.motion_type = motion_type;

  // update the polygon
  pub_obj.polygon_base.clear();
  if (polygon_base.size() >= 3) {
    // 当前帧被激光检测到并且polygon_base有效
    pub_obj.polygon_base = polygon_base;
  } else {
    rally::Box2d box(rally::ShortArray2f(pub_obj.box_center_base.x,
                                         pub_obj.box_center_base.y),
                     yaw_base, box_size.length, box_size.width);
    std::vector<rally::ShortArray2f> pts;
    box.getAllCorners(pts);
    pub_obj.polygon_base.reserve(pts.size());
    for (const auto& pt : pts) {
      pub_obj.polygon_base.emplace_back(pt.x, pt.y);
    }
  }

  addOdomInfo(rel_to_base_trans.inverse(), pub_obj);
  if (track_state == TRACKER_STATE_CODE::EDGE_LOST_TRACK ||
      track_state == TRACKER_STATE_CODE::RECOVER_TRACK ||
      cover_state == COVER_STATE::FULL_COVER) {
    pub_obj.exist_confidence = 2;
  } else if (track_state == TRACKER_STATE_CODE::LOW_SCORE_LOST_TRACK ||
             cover_state != COVER_STATE::NO_COVER) {
    pub_obj.exist_confidence = 1;
  } else {
    pub_obj.exist_confidence = 0;
  }

  return pub_obj;
}
}  // namespace perception
}  // namespace robot