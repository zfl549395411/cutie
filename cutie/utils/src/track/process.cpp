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

#include "robot/perception/uwb_postprocess/process/process.h"
namespace robot {
namespace perception {
bool UwbPostProcess::Init(
    const YAML::Node& cfg_node, const BaseTracker::Ptr& image_track_ptr,
    const std::unordered_set<SensorID>& calib_sensor_list) {
  for (const auto& sensor : calib_sensor_list) {
    auto calib_ptr = robot::perception::CalibCenter::GetInstance().Get(
        static_cast<uint8_t>(sensor));
    if (calib_ptr == nullptr) {
      AERROR << "sensor_calib is nullptr! " << kSensorIDToNameMap.at(sensor);
      return false;
    }
    sensor_to_base_[sensor] = calib_ptr->TransToVeh().cast<float>();
    AINFO << "Load sensor calib! " << kSensorIDToNameMap.at(sensor) << " "
          << sensor_to_base_[sensor].matrix();
  }
  rally::yamlRead(cfg_node, "pub_image", pub_image_);
  image_track_ptr_ = image_track_ptr;
  track_list_ptr_ = std::make_shared<TrackList>();
  return true;
}
void UwbPostProcess::Perception(const MsgCarrier::Ptr& msg_ptr) {
  msg_ptr->output_msg_ptr->result.object_list.clear();
  if (!SetTrans(msg_ptr->pub_time)) {
    AERROR << "SetTrans error! No localization info !";
    return;
  }
  InitTrackList();
  FusionUwb(msg_ptr->input_msg_ptr);
  FusionImage(msg_ptr->pub_time);
  UpdateObserve();
  Pub(msg_ptr->output_msg_ptr);
}
// 更新当前帧转换矩阵
bool UwbPostProcess::SetTrans(const double pub_time) {
  Eigen::Affine3d base_to_rel = Eigen::Affine3d::Identity();
  UwbPostProcessEgoPoseInfo::GetInstance().GetBaseToRel(pub_time, base_to_rel);
  if (base_to_rel.matrix() == Eigen::Matrix4d::Identity()) {
    AERROR << " Not find base to rel!!!!";
    return false;
  }
  base_to_rel_ = base_to_rel.cast<float>();
  //   AINFO << " GET TRANS " << base_to_rel_.matrix();
  rally::Time timer(pub_time);
  track_list_ptr_->header.time = timer.toNanosecond();
  ++track_list_ptr_->header.seq;
  return true;
}
void UwbPostProcess::InitTrackList() {
  if (image_track_ptr_->GetTrackState() == TRACKER_STATE_CODE::ERROR_TRACK) {
    ++error_track_count;
  } else {
    error_track_count = 0;
  }
  if (error_track_count > 10) {
    if (image_track_ptr_->GetTrackMode() == 0) {
      track_list_ptr_->object_list.clear();
      AERROR
          << " Too many error track !!! need reset track list !!!! FOR OUTSIDE"
          << " " << error_track_count;
    } else {
      ReInit();
      AERROR << " Too many error track !!! need reset track list !!!!" << " "
             << error_track_count;
    }
  }
  for (auto& track_obj : track_list_ptr_->object_list) {
    track_obj->fusion_obj = std::make_shared<SingleFusionObj>();
    track_obj->GetObserveSensor() = OBSERVE_SENSOR::NO_OBSERVE;
    track_obj->GetCoverState() = COVER_STATE::NO_COVER;
    track_obj->GetTrackState() = TRACKER_STATE_CODE::STABLE_TRACK;
  }
}
void UwbPostProcess::FusionUwb(const InputMsg::Ptr& input_msg_ptr) {
  for (const auto& uwb_info : input_msg_ptr->uwb_obj) {
    const auto& uwb_msg = uwb_info.second;
    ObserveObj::Ptr uwb_observe = std::make_shared<ObserveObj>(
        uwb_msg, sensor_to_base_[uwb_msg->sensor_id], dog_info_,
        uwb_msg->sensor_id);
    bool match_label = false;
    // match
    for (auto& track_obj : track_list_ptr_->object_list) {
      if (track_obj->GetLabelId() == uwb_msg->uwb_data.tag_id) {
        track_obj->fusion_obj->observe_list.emplace_back(uwb_observe);
        setBitsetOfSensorIds(uwb_observe->sensor_id_,
                             track_obj->fusion_obj->sensor_ids_);

        match_label = true;
      }
    }
    // creat
    if (!match_label) {
      SingleFusionObj::Ptr new_fusion_obj =
          std::make_shared<SingleFusionObj>(uwb_observe);
      setBitsetOfSensorIds(uwb_observe->sensor_id_,
                           new_fusion_obj->sensor_ids_);
      track_list_ptr_->object_list.emplace_back(
          std::make_shared<TrackObj>(new_fusion_obj, base_to_rel_.inverse()));
    }
  }
}
void UwbPostProcess::FusionImage(const double& pub_time) {
  if (!image_track_ptr_->GetRectInitLabe()) {
    return;
  }
  double image_track_time = image_track_ptr_->GetTimeStamp();
  if (std::abs(image_track_time - pub_time) > 0.5) {
    AWARN << std::fixed << " image track time error! " << image_track_time
          << " " << pub_time;
    return;
  }
  for (auto& track_obj : track_list_ptr_->object_list) {
    if (track_obj->GetUwbPoseCode() != UWB_POSE_CODE::FRONT) {
      continue;
    }
    // 只有在有uwb时利用uwb更新跟踪距离
    for (const auto& obj : track_obj->fusion_obj->observe_list) {
      if (obj->sensor_id_ == SensorID::UWB_LOCATION_FRONT) {
        float distance = obj->uwb_msg_->uwb_data.distance;
        image_track_ptr_->SetDistance(distance, obj->time_stamp_,
                                      OBSERVE_SENSOR::UWB_OBSERVE);
      }
    }
    track_obj->GetTrackState() = image_track_ptr_->GetTrackState();
    track_obj->GetObserveSensor() = image_track_ptr_->GetObserveSensor();
    track_obj->GetCoverState() = image_track_ptr_->GetCoverState();
    // 这是设置时间为pub time，否则会无法用于优化（始终严重滞后于UWB）
    ObserveObj::Ptr image_observe = std::make_shared<ObserveObj>(
        image_track_ptr_, pub_time, image_track_ptr_->GetImageCalibPtr(), 0);
    setBitsetOfSensorIds(image_observe->sensor_id_,
                         track_obj->fusion_obj->sensor_ids_);
    track_obj->fusion_obj->observe_list.emplace_back(image_observe);
  }
  // 利用图像初始化
  if (track_list_ptr_->object_list.size() == 0) {
    ObserveObj::Ptr image_observe = std::make_shared<ObserveObj>(
        image_track_ptr_, pub_time, image_track_ptr_->GetImageCalibPtr(), 0);
    SingleFusionObj::Ptr new_fusion_obj =
        std::make_shared<SingleFusionObj>(image_observe);

    track_list_ptr_->object_list.emplace_back(
        std::make_shared<TrackObj>(new_fusion_obj, base_to_rel_.inverse()));
    setBitsetOfSensorIds(image_observe->sensor_id_,
                         new_fusion_obj->sensor_ids_);
    track_list_ptr_->object_list.back()->GetTrackState() =
        image_track_ptr_->GetTrackState();
    track_list_ptr_->object_list.back()->GetObserveSensor() =
        image_track_ptr_->GetObserveSensor();
    track_list_ptr_->object_list.back()->GetCoverState() =
        image_track_ptr_->GetCoverState();
  }
}
void UwbPostProcess::UpdateObserve() {
  for (const auto& track_obj : track_list_ptr_->object_list) {
    track_obj->ChooseObserve(base_to_rel_.inverse());
    if (!image_track_ptr_->GetRectInitLabe() &&
        image_track_ptr_->GetTrackMode() == 2) {
      float dist = track_obj->GetBoxCenterBase().norm();
      float angle = std::atan2(track_obj->GetBoxCenterBase().y(),
                               track_obj->GetBoxCenterBase().x());
      image_track_ptr_->ChoosePed(dist, angle);
    }
    image_track_ptr_->SetCenterBase(track_obj->GetBoxCenterBase(),
                                    track_obj->GetTimeStamp());
  }
}
void UwbPostProcess::UpdateServer() {
  if (!image_track_ptr_->GetRectInitLabe()) {
    return;
  }
  auto& server_output = UWBServer::GetInstance().output_data;
  cv::Size cur_image_size = image_track_ptr_->GetBgrSize();
  rally::Time cur_time(track_list_ptr_->header.time);
  server_output.time_stamp = cur_time.toSecond();
  if (track_list_ptr_->object_list.size() == 0) {
    if (server_output.track_state != TRACKER_STATE_CODE::NO_INIT) {
      server_output.track_state = TRACKER_STATE_CODE::ERROR_TRACK;
      AWARN << " OUT STATE IS ERROR!!!";
    }
  } else {
    const auto& track_obj = track_list_ptr_->object_list[0];
    // 更新速度
    image_track_ptr_->SetVelNorm(track_obj->GetVelocityBase().norm());
    server_output.track_state = image_track_ptr_->GetTrackState();
    auto box_center_base = track_obj->GetBoxCenterBase();
    Eigen::Vector2f center_base(box_center_base.x(), box_center_base.y());
    server_output.dist_base = center_base.norm();
    server_output.angle_base = std::atan2(center_base.y(), center_base.x());

    // rect先恢复带畸变原图再设置为和输入等大
    cv::Rect detect_rect = image_track_ptr_->GetRectNoWait();
    server_output.sensor_id = image_track_ptr_->GetOutSensorID();
    UWBServer::GetInstance().calib_mutex.lock();
    server_output.rect = RecoverRect(
        detect_rect, cur_image_size.width, cur_image_size.height,
        UWBServer::GetInstance().ori_image_width,
        UWBServer::GetInstance().ori_image_height,
        UWBServer::GetInstance().ori_K, UWBServer::GetInstance().ori_D);
    UWBServer::GetInstance().calib_mutex.unlock();
    float width_scale = UWBServer::GetInstance().input_data.image_width * 1.0f /
                        UWBServer::GetInstance().ori_image_width;
    float height_scale = UWBServer::GetInstance().input_data.image_height *
                         1.0f / UWBServer::GetInstance().ori_image_height;
    if (UWBServer::GetInstance().input_data.image_width == 0) {
      width_scale = 0.5;
      height_scale = 0.5;
    }
    server_output.image_height =
        UWBServer::GetInstance().input_data.image_height;
    server_output.image_width = UWBServer::GetInstance().input_data.image_width;
    server_output.undistort = false;
    server_output.rect.x = server_output.rect.x * width_scale;
    server_output.rect.y = server_output.rect.y * height_scale;
    server_output.rect.width = server_output.rect.width * width_scale;
    server_output.rect.height = server_output.rect.height * height_scale;
    AINFO << " set server output  " << server_output.rect;
  }
}
void UwbPostProcess::Pub(const OutputMsg::Ptr& output_msg_ptr) {
  // 更新server
  UpdateServer();
  output_msg_ptr->result.header = track_list_ptr_->header;
  rally::Time pub_time(track_list_ptr_->header.time);

  for (const auto& track_obj : track_list_ptr_->object_list) {
    output_msg_ptr->result.object_list.emplace_back(
        track_obj->ToPubObject(pub_time.toSecond(), base_to_rel_.inverse()));
    auto center_odom =
        output_msg_ptr->result.object_list.back().box_center_odom;
    image_track_ptr_->SetCenterOdom(
        Eigen::Vector3f(center_odom.x, center_odom.y, center_odom.z));
  }
  if (pub_image_) {
    // 清空图像
    for (auto& out_msg : output_msg_ptr->debug_image_) {
      out_msg.second = nullptr;
    }
    cv::Mat cur_image = image_track_ptr_->GetBrgImage().clone();
    image_track_ptr_->DrawDetectionResult(cur_image);
    cv::Rect image_rect = image_track_ptr_->GetRectNoWait();
    cv::rectangle(cur_image, image_rect, cv::Scalar(0, 255, 0), 2);
    auto calib_ptr_ = image_track_ptr_->GetImageCalibPtr();
    for (const auto& track_obj : track_list_ptr_->object_list) {
      for (const auto& observe : track_obj->fusion_obj->observe_list) {
        Eigen::Vector3d observe_pt =
            observe->pose_base_decomposition.cast<double>();
        Eigen::Vector3d observe_camera_pt =
            calib_ptr_->VehPtToCameraPt(observe_pt);
        int observe_u = 0, observe_v = 0;
        calib_ptr_->CameraPtToUV(observe_camera_pt, observe_u, observe_v, false,
                                 false);
        cv::circle(cur_image, cv::Point(observe_u, observe_v), 5,
                   cv::Scalar(0, 0, 255), -1);
      }
      Eigen::Vector3d uwb_pt = track_obj->GetBoxCenterBase().cast<double>();
      Eigen::Vector3d camera_pt = calib_ptr_->VehPtToCameraPt(uwb_pt);
      int u = 0, v = 0;
      calib_ptr_->CameraPtToUV(camera_pt, u, v, false, false);
      if (isCameraSensor(track_obj->GetSensorID())) {
        cv::circle(cur_image, cv::Point(u, v), 5, cv::Scalar(0, 255, 0), -1);
      } else {
        cv::circle(cur_image, cv::Point(u, v), 5, cv::Scalar(255, 0, 0), -1);
      }
    }
    cv::Rect predict_rect = image_track_ptr_->GetPredictRect();
    cv::rectangle(cur_image, predict_rect, cv::Scalar(0, 0, 255), 2);
    output_msg_ptr->debug_image_["/tracker_compressedimage"] =
        std::make_shared<perception::CompressedImage>();
    MatToCompressedImage(
        cur_image, track_list_ptr_->header,
        output_msg_ptr->debug_image_["/tracker_compressedimage"]);
    if (image_track_ptr_->show_image_) {
      cv::imshow("Window1", cur_image);
      cv::waitKey(1);
    }
  }
}
}  // namespace perception
}  // namespace robot