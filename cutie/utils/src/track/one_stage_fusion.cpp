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

#include "hyper_vision/perception/post_fusion/furnace/one_stage_fusion.h"

namespace robosense {
namespace perception {
void OneStageFusion::Perception(TrackingObjectList& temporal_tracking_list,
                                const PostFusionMsg::Ptr& msg_ptr) {
  history_obj_num_ = temporal_tracking_list.object_list.size();
  ++frame_count;
  // 找到lidar时间戳
  for (const auto& v : msg_ptr->input_msg_ptr->obj_map) {
    std::string topic = v.first;
    if (topic != msg_ptr->trigger_topic_) {
      continue;
    }
    auto time_stamp = apollo::cyber::Time(v.second->header.time).ToSecond();
    if (temporal_tracking_list.header.time != 0 &&
        (time_stamp < lidar_time_stamp_ ||
         std::abs(time_stamp - lidar_time_stamp_) > 0.4)) {
      temporal_tracking_list.reset();
      history_obj_num_ = 0;
      AINFO << "time sudden change, reset the temporal_tracking_list";
    }

    lidar_time_stamp_ = time_stamp;
    temporal_tracking_list.header = v.second->header;
    // break;
  }
  // 初始化当前帧的tracking_object
  //   auto t_fusion_start = apollo::cyber::Time::Now();
  InitTrack(temporal_tracking_list);
  //   auto t__fusion_end = apollo::cyber::Time::Now();
  //   AINFO << "inti time  " << (t__fusion_end - t_fusion_start).ToSecond() *
  //   1000.0
  //         << " ms"
  //         << ", tracking obj size = "
  //         << temporal_tracking_list.object_list.size();

  //   t_fusion_start = apollo::cyber::Time::Now();
  FusionLidarObj(msg_ptr, temporal_tracking_list);
  //   t__fusion_end = apollo::cyber::Time::Now();
  //   AINFO << "lidar time  "
  //         << (t__fusion_end - t_fusion_start).ToSecond() * 1000.0 << " ms"
  //         << ", tracking obj size = "
  //         << temporal_tracking_list.object_list.size();

  //   t_fusion_start = apollo::cyber::Time::Now();
  FusionVisionObj(msg_ptr, temporal_tracking_list);
  //   t__fusion_end = apollo::cyber::Time::Now();
  //   AINFO << "vision time  "
  //         << (t__fusion_end - t_fusion_start).ToSecond() * 1000.0 << " ms"
  //         << ", tracking obj size = "
  //         << temporal_tracking_list.object_list.size();

  //   t_fusion_start = apollo::cyber::Time::Now();
  FusionRadarObj(msg_ptr, temporal_tracking_list);
  //   t__fusion_end = apollo::cyber::Time::Now();
  //   AINFO << "radar time  "
  //         << (t__fusion_end - t_fusion_start).ToSecond() * 1000.0 << " ms"
  //         << ", tracking obj size = "
  //         << temporal_tracking_list.object_list.size();

  //   t_fusion_start = apollo::cyber::Time::Now();
  UpdateTrackingObject(temporal_tracking_list);
  //   t__fusion_end = apollo::cyber::Time::Now();
  //   AINFO << "update time  "
  //         << (t__fusion_end - t_fusion_start).ToSecond() * 1000.0 << " ms"
  //         << ", tracking obj size = "
  //         << temporal_tracking_list.object_list.size();
}
void OneStageFusion::InitTrack(TrackingObjectList& temporal_tracking_list) {
  Eigen::Affine3d base_to_rel_trans;
  PostFusionEgoPoseInfo::GetInstance().GetBaseToRel(lidar_time_stamp_,
                                                    base_to_rel_trans);
  Eigen::Affine3d rel_to_base_trans = base_to_rel_trans.inverse();
  for (auto& obj : temporal_tracking_list.object_list) {
    obj->observe_obj_ptr_ = std::make_shared<SingleFusionObject>();
    obj->MotionCompensation(lidar_time_stamp_, rel_to_base_trans);
    if (obj->pv_proj_box_map == nullptr) {
      obj->pv_proj_box_map = std::make_shared<PvProjMap>();
    }
    MultiSensorPvProj(*obj, *obj->pv_proj_box_map);
    obj->bev_box_base = rally::Box2d(
        rally::ShortArray2f(obj->box_center_base.x, obj->box_center_base.y),
        obj->yaw_base, obj->box_size.length, obj->box_size.width);
    obj->bev_box_base_compensation = rally::Box2d(
        rally::ShortArray2f(obj->box_center_base_compensation.x,
                            obj->box_center_base_compensation.y),
        obj->yaw_base_compensation, obj->box_size.length, obj->box_size.width);
  }
}

bool OneStageFusion::MultiSensorPvProj(
    const TrackingObject& obj,
    std::map<SensorID, std::array<std::pair<bool, PvBBox2D>, 2>>&
        pv_proj_box_map) {
  bool obj_valid = true;
  if (using_virtual_camera_) {
    // 虚拟投影只投影本相机和虚拟相机
    virtual_proj_ptr_->GetProjObject(
        obj, pv_proj_box_map[SensorID::VIRTUAL_CAMERA],
        pv_proj_box_map[SensorID::VIRTUAL_CAMERA_EXTENDED]);
  }
  for (const auto& proj_sensor : pv_proj_list_) {
    if (using_virtual_camera_) {
      // 虚拟投影只投影本相机和虚拟相机
      if (proj_sensor == SensorID::VIRTUAL_CAMERA_EXTENDED ||
          proj_sensor == SensorID::VIRTUAL_CAMERA) {
        continue;
      } else {
        pv_proj_box_map[proj_sensor][0] =
            std::pair<bool, PvBBox2D>(false, PvBBox2D());
        pv_proj_box_map[proj_sensor][1] =
            std::pair<bool, PvBBox2D>(false, PvBBox2D());
        continue;
      }
    } else {
      // 不进行虚拟投影
      if (proj_sensor == SensorID::VIRTUAL_CAMERA_EXTENDED ||
          proj_sensor == SensorID::VIRTUAL_CAMERA) {
        pv_proj_box_map[proj_sensor][0] =
            std::pair<bool, PvBBox2D>(false, PvBBox2D());
        pv_proj_box_map[proj_sensor][1] =
            std::pair<bool, PvBBox2D>(false, PvBBox2D());
        continue;
      }
    }

    if (!PvProj(proj_sensor, obj, pv_proj_box_map[proj_sensor])) {
      obj_valid = false;
    }
  }
  return obj_valid;
}
bool OneStageFusion::PvProj(const SensorID& target_sensor_id,
                            const TrackingObject& det_obj,
                            std::array<std::pair<bool, PvBBox2D>, 2>& prj_ret) {
  std::vector<Eigen::Vector3d> obj_corner_ponits =
      CalCornerPoints(det_obj, false);
  std::vector<Eigen::Vector3d> obj_comp_corner_ponits =
      CalCornerPoints(det_obj, true);
  PvBBox2D pv_box;
  PvBBox2D comp_pv_box;
  bool is_valid_pv_proj = false;
  // 投影原始3D框
  is_valid_pv_proj = Trans3DboxToPv(
      obj_corner_ponits, cam_calibs_ptr_.at(target_sensor_id), pv_box);
  if (isVehObject(det_obj.type)) {
    is_valid_pv_proj =
        is_valid_pv_proj && float(pv_box.height) / pv_box.width <= 3;
  }
  // 用原始2D框取代

  bool is_valid_comp_pv_proj = false;
  if (!using_virtual_camera_) {
    is_valid_comp_pv_proj =
        Trans3DboxToPv(obj_comp_corner_ponits,
                       cam_calibs_ptr_.at(target_sensor_id), comp_pv_box);
    if (isVehObject(det_obj.type)) {
      is_valid_comp_pv_proj =
          is_valid_comp_pv_proj &&
          float(comp_pv_box.height) / comp_pv_box.width <= 3;
    }
  }

  prj_ret[0] = std::pair<bool, PvBBox2D>(is_valid_pv_proj, pv_box);
  prj_ret[1] = std::pair<bool, PvBBox2D>(is_valid_comp_pv_proj, comp_pv_box);
  return true;
}
void OneStageFusion::GreedyMatch(
    const std::vector<SingleFusionCorrelationObject::Ptr>& observe_objs,
    const SensorID sensor_id, TrackingObjectList& temporal_tracking_list,
    std::vector<bool>& matched_label) {
  matched_label.resize(observe_objs.size(), false);
  for (int i = 0; i < observe_objs.size(); ++i) {
    const auto& correaltion_obj = observe_objs[i];
    const auto& observe_obj = correaltion_obj->obj_ptr;
    const auto& observe_box = correaltion_obj->bev_box_base_compensation;
    for (int j = 0; j < temporal_tracking_list.object_list.size(); ++j) {
      auto& tracking_obj = temporal_tracking_list.object_list[j];
      // 准备消亡目标
      if (lidar_time_stamp_ - tracking_obj->getMeasureTimeStamp() >
          tracking_obj->hold_time) {
        continue;
      }

      float bev_thresh = 0.5;
      float pv_thresh = 0.5;

      if (!isAllowedMatchType(observe_obj->type, tracking_obj->type)) {
        continue;
      }
      // 设置关联阈值 类别不同时需要很大bev阈值
      if (!isPreliminarySameType(observe_obj->type, tracking_obj->type)) {
        bev_thresh = 0.7;
      } else if (isCameraSensor(sensor_id)) {
        // pv_post id相同且bev下存在重叠
        if (tracking_obj->pv_post_id[sensor_id].find(observe_obj->object_id) !=
            tracking_obj->pv_post_id[sensor_id].end()) {
          pv_thresh = 0.5;
          bev_thresh = 0.05;
        } else {
          bev_thresh = 0.2;
          pv_thresh = 0.7;
        }
      } else if (isRadarSensor(sensor_id)) {
        bev_thresh = 0.05;
      } else if (isSmallObject(observe_obj->box_size, observe_obj->type) &&
                 isSmallObject(tracking_obj->box_size, tracking_obj->type)) {
        pv_thresh = 0.3;
        bev_thresh = 0.3;
      }
      // 激光行人目标类型不同（主要是和cycle）要增加iou
      if (isLidarSensor(sensor_id) &&
          (tracking_obj->type == ObjectType::TYPE_PED) ^
              (observe_obj->type == ObjectType::TYPE_PED)) {
        bev_thresh = 0.7;
      }

      // pv_iou
      float pv_iou = 0, bev_iou = 0;
      if (isRadarSensor(sensor_id)) {
        if (!IsRadarInPvBox(SensorID::VIRTUAL_CAMERA,
                            *correaltion_obj->radar_proj_point_map,
                            *tracking_obj->pv_proj_box_map) &&
            !IsRadarInPvBox(SensorID::VIRTUAL_CAMERA_EXTENDED,
                            *correaltion_obj->radar_proj_point_map,
                            *tracking_obj->pv_proj_box_map)) {
          continue;
        }
      } else {
        // 近距离更严格使用标准IOU

        pv_iou = std::max(CalPvProjIou(SensorID::VIRTUAL_CAMERA,
                                       *correaltion_obj->pv_proj_box_map,
                                       *tracking_obj->pv_proj_box_map, false),
                          CalPvProjIou(SensorID::VIRTUAL_CAMERA_EXTENDED,
                                       *correaltion_obj->pv_proj_box_map,
                                       *tracking_obj->pv_proj_box_map, false));
        if (pv_iou < pv_thresh) {
          continue;
        }
      }

      const auto& tracking_box = tracking_obj->bev_box_base_compensation;
      std::tie(bev_iou, std::ignore, std::ignore, std::ignore) =
          calcIou(observe_box, tracking_box);
      if (bev_iou < bev_thresh) {
        continue;
      }
      // 关联成功
      matched_label[i] = true;
      // 进行速度检查，如果速度检查不通过只覆盖
      if (temporal_fusion_ptr_->isValidarMatchingVelDir(tracking_obj,
                                                        correaltion_obj)) {
        AddObserve(tracking_obj, correaltion_obj);
        // AINFO << " track id " << tracking_obj->getId() << " Gmatch with "
        //       << "SENSOR ID " << kSensorIDToNameMap.at(sensor_id)
        //       << " object id " << observe_obj->object_id;
      } else {
        // AINFO << " track id " << tracking_obj->getId() << " NOT Gmatch with "
        //       << "SENSOR ID " << kSensorIDToNameMap.at(sensor_id)
        //       << " object id " << observe_obj->object_id;
      }
      break;
    }
  }
}
void OneStageFusion::HungarianMatch(
    const std::vector<SingleFusionCorrelationObject::Ptr>& observe_objs,
    TrackingObjectList& temporal_tracking_list,
    std::vector<bool>& matched_label) {
  std::vector<uint32_t> associated_id;
  if (observe_objs.size() > 0 &&
      temporal_tracking_list.object_list.size() > 0) {
    auto associated_score = ComputeAssociatedScore(temporal_tracking_list,
                                                   observe_objs, matched_label);
    auto n_objs =
        temporal_tracking_list.object_list.size() + observe_objs.size();
    associated_id =
        rally::HungarianGraphOptimizer().hungarianMinimumWeightPerfectMatching(
            n_objs, associated_score);
  }
  uint32_t num_history_objs = temporal_tracking_list.object_list.size();
  uint32_t num_current_objs = observe_objs.size();
  for (size_t i = 0; i < num_history_objs; ++i) {
    if (associated_id.empty()) {
      break;
    }

    if (associated_id[i] >= num_current_objs) {
      continue;
    }

    matched_label[associated_id[i]] = true;
    auto& track_obj = temporal_tracking_list.object_list[i];
    auto& current_obj = observe_objs[associated_id[i]];
    AddObserve(track_obj, current_obj);
    // AINFO << " track id " << track_obj->getId() << " Hmatch with "
    //       << "SENSOR ID " << kSensorIDToNameMap.at(current_obj->sensor_id)
    //       << " object id " << current_obj->obj_ptr->object_id;
  }
}
void OneStageFusion::CreateTrackObject(
    const std::vector<SingleFusionCorrelationObject::Ptr>& observe_objs,
    TrackingObjectList& temporal_tracking_list,
    std::vector<bool>& matched_label) {
  Eigen::Affine3d base_to_rel_trans;
  PostFusionEgoPoseInfo::GetInstance().GetBaseToRel(lidar_time_stamp_,
                                                    base_to_rel_trans);
  for (int i = 0; i < observe_objs.size(); ++i) {
    if (!matched_label[i]) {
      if (observe_objs[i]->correlation_state_ == CORRELATION_STATE::VALID &&
          IsValidTrackFilter(observe_objs[i], lidar_calibs_ptr_)) {
        TrackingObject::Ptr tmp = std::make_shared<TrackingObject>(
            temporal_fusion_ptr_->id_generate_ptr_, observe_objs[i],
            base_to_rel_trans, optimize_level_, allow_only_lidar_fusion_);
        AddObserve(tmp, observe_objs[i]);
        temporal_tracking_list.object_list.emplace_back(tmp);
      }
      matched_label[i] = true;
    }
  }
}
std::vector<rally::WeightedBipartiteEdge>
OneStageFusion::ComputeAssociatedScore(
    const TrackingObjectList& temporal_tracking_list,
    const std::vector<SingleFusionCorrelationObject::Ptr>& observe_objs,
    std::vector<bool>& matched_label) {
  std::vector<rally::WeightedBipartiteEdge> associate_score;
  associate_score.reserve(
      temporal_tracking_list.object_list.size() * observe_objs.size() +
      temporal_tracking_list.object_list.size() + observe_objs.size());
  for (size_t left_id = 0; left_id < temporal_tracking_list.object_list.size();
       ++left_id) {
    const auto& ref_obj = temporal_tracking_list.object_list[left_id];
    for (size_t right_id = 0; right_id < observe_objs.size(); ++right_id) {
      if (matched_label[right_id]) {
        continue;
      }
      const auto& current_obj = observe_objs[right_id];

      // 粗匹配
      if (!PreliminaryMatching(ref_obj, current_obj)) {
        continue;
      }
      float cur_density = calProDensity(ref_obj, current_obj);
      for (const auto& obj : ref_obj->observe_obj_ptr_->correlation_objs) {
        cur_density = std::max(calProDensity(obj, current_obj), cur_density);
        if (isPinholeSensor(obj->sensor_id) &&
            ref_obj->pv_post_id[obj->sensor_id].find(obj->obj_ptr->object_id) !=
                ref_obj->pv_post_id[obj->sensor_id].end()) {
          cur_density *= 1.2;
        } else if (isRadarSensor(obj->sensor_id) &&
                   ref_obj->radar_id[obj->sensor_id].find(
                       obj->obj_ptr->object_id) !=
                       ref_obj->radar_id[obj->sensor_id].end()) {
          cur_density *= 1.2;
        }
      }
      if (ref_obj->pub_state == PUB_CODE_INVALID_FRONT_FRONT_CAR ||
          ref_obj->pub_state == PUB_CODE_VALID_FRONT_FRONT_CAR) {
        cur_density *= 0.8;
      } else if (ref_obj->pub_state == PUB_CODE_UNKNOWN) {
        // 优先关联已经pub的目标。
        // 锥桶可能一开始是一个目标并且pub了，有一帧起出双框，第二帧又单帧关联上了。
        // 此时新的框匹配的更好，导致旧框没匹配上。然后新框被pub，而旧框被hold，导致双框
        cur_density *= 0.5;
      } else if (ref_obj->hold_state == HOLD_STATE::HOLDING) {
        // 准备消亡目标
        if (lidar_time_stamp_ - ref_obj->getMeasureTimeStamp() >
            ref_obj->hold_time) {
          cur_density *= 0.5;
        }
      }
      //   AINFO << "TRACK " << ref_obj->getId() << " match "
      //         << kSensorIDToNameMap.at(current_obj->sensor_id) << " id "
      //         << current_obj->obj_ptr->object_id << " DENSITY " <<
      //         cur_density
      //         << " " << lidar_time_stamp_ - ref_obj->getMeasureTimeStamp() <<
      //         " "
      //         << ref_obj->hold_time << " " << int(ref_obj->pub_state);
      if (cur_density > 1e-6) {
        associate_score.emplace_back(
            rally::WeightedBipartiteEdge(left_id, right_id, 1 - cur_density));
        associate_score.emplace_back(
            temporal_tracking_list.object_list.size() + right_id,
            observe_objs.size() + left_id, 0);
      }
    }
  }
  for (uint32_t left_id = 0;
       left_id < temporal_tracking_list.object_list.size(); ++left_id) {
    associate_score.emplace_back(left_id, left_id + observe_objs.size(), 0);
  }

  for (uint32_t right_id = 0; right_id < observe_objs.size(); ++right_id) {
    associate_score.emplace_back(
        right_id + temporal_tracking_list.object_list.size(), right_id, 1);
  }
  return associate_score;
}
bool OneStageFusion::PreliminaryMatching(
    const TrackingObject::Ptr& ref_obj,
    const SingleFusionCorrelationObject::Ptr& target_obj) {
  if (!isAllowedMatchType(ref_obj->type, target_obj->obj_ptr->type)) {
    return false;
  }
  bool matched_sensor = checkBitsetOfSensorIds(
      target_obj->sensor_id, ref_obj->observe_obj_ptr_->sensor_ids);
  if (matched_sensor && !isValidVelArea(ref_obj, target_obj)) {
    return false;
  }
  if (!isValidMatchingDistance(ref_obj, target_obj)) {
    return false;
  }
  if (!temporal_fusion_ptr_->isValidarMatchingVelDir(ref_obj, target_obj)) {
    return false;
  }
  // 前一帧无激光观测、有侧前观测,速度不匹配
  if (isCarObject(ref_obj->type) && IsLidarSensor(target_obj->sensor_id) &&
      ref_obj->lidar_observe_frame != ref_obj->observe_frame &&
      (checkBitsetOfSensorIds(SensorID::CAM_LEFT_FRONT, ref_obj->sensor_ids) ||
       checkBitsetOfSensorIds(SensorID::CAM_RIGHT_FRONT,
                              ref_obj->sensor_ids))) {
    if (ref_obj->observe_count > 5 &&
        (ref_obj->velocity_base.x * ref_obj->velocity_base.x +
         ref_obj->velocity_base.y * ref_obj->velocity_base.y) < 1) {
      float observe_vel_square = target_obj->obj_ptr->velocity_base.x *
                                     target_obj->obj_ptr->velocity_base.x +
                                 target_obj->obj_ptr->velocity_base.y *
                                     target_obj->obj_ptr->velocity_base.y;
      if (observe_vel_square > 25 && observe_vel_square < 500 * 500) {
        return false;
      }
    }
  }
  // 有pv iou且theta符合，不用计算bev iou
  // 锥桶类目标如果历史关联过不走这个逻辑
  if (target_obj->obj_ptr->type == ObjectType::TYPE_TRAFFIC_CONE &&
      isPinholeSensor(target_obj->sensor_id) &&
      ref_obj->pv_post_id[target_obj->sensor_id].find(
          target_obj->obj_ptr->object_id) !=
          ref_obj->pv_post_id[target_obj->sensor_id].end()) {
    return true;
  }
  // 车辆类目标和小目标关联必须保证有bev_iou
  if (IsCar(ref_obj->type) &&
      isSmallObject(target_obj->obj_ptr->box_size, target_obj->obj_ptr->type)) {
    float bev_iou = GetBevIou(ref_obj, target_obj);
    return bev_iou > 0.05;
  }

  // 小目标关联只需要有PV IOU
  bool iou_valid = isValidMatchingPvIou(ref_obj, target_obj);
  if (!iou_valid && !isVehObject(target_obj->obj_ptr->type) &&
      isSmallObject(ref_obj->box_size, ref_obj->type)) {
    return false;
  }
  bool theta_valid = isValidMatchingTheta(ref_obj, target_obj);
  if (iou_valid && theta_valid) {
    return true;
  } else {
    float bev_iou = GetBevIou(ref_obj, target_obj);
    iou_valid = (iou_valid || bev_iou > 0.05);
    if (iou_valid && (bev_iou > 0.1 || theta_valid)) {
      return true;
    } else {
      return false;
    }
  }

  // AINFO << "CAN'T MATCH " << ref_obj->getId() << " "
  //       << kSensorIDToNameMap.at(target_obj->sensor_id) << " id "
  //       << target_obj->obj_ptr->object_id << " "
  //       << isValidMatchingPvIou(ref_obj, target_obj) << " "
  //       << isValidMatchingBevIou(ref_obj, target_obj) << " "
  //       << isValidMatchingTheta(ref_obj, target_obj) << " "
  //       << isValidMatchingDistance(ref_obj, target_obj) << " "
  //       << temporal_fusion_ptr_->isValidarMatchingVelDir(ref_obj,
  //       target_obj);
  return false;
}
float OneStageFusion::GetBevIou(
    const TrackingObject::Ptr& ref_obj,
    const SingleFusionCorrelationObject::Ptr& target_obj) {
  const auto& ref_obj_box_compensation = ref_obj->bev_box_base_compensation;
  const auto& ref_obj_box = ref_obj->bev_box_base;
  const auto& target_obj_box_compensation =
      target_obj->bev_box_base_compensation;
  const auto& target_obj_box = target_obj->bev_box_base;

  float max_iou = 0, max_iou_1 = 0, max_iou_2 = 0, max_iou_3 = 0, max_iou_4 = 0;
  std::tie(max_iou_1, std::ignore, std::ignore, std::ignore) =
      calcIou(ref_obj_box, target_obj_box);
  std::tie(max_iou_2, std::ignore, std::ignore, std::ignore) =
      calcIou(ref_obj_box, target_obj_box_compensation);
  std::tie(max_iou_3, std::ignore, std::ignore, std::ignore) =
      calcIou(ref_obj_box_compensation, target_obj_box);
  std::tie(max_iou_4, std::ignore, std::ignore, std::ignore) =
      calcIou(ref_obj_box_compensation, target_obj_box_compensation);
  max_iou = std::max(max_iou, max_iou_1);
  max_iou = std::max(max_iou, max_iou_2);
  max_iou = std::max(max_iou, max_iou_3);
  max_iou = std::max(max_iou, max_iou_4);
  return max_iou;
}
bool OneStageFusion::isValidMatchingBevIou(
    const TrackingObject::Ptr& ref_obj,
    const SingleFusionCorrelationObject::Ptr& target_obj) {
  const auto& ref_obj_box = ref_obj->bev_box_base;
  const auto& target_obj_box_compensation =
      target_obj->bev_box_base_compensation;
  const auto& target_obj_box = target_obj->bev_box_base;

  // ref_obj必须和target_obj有重叠才继续计算距离进行匹配
  float max_iou = 0, min_iou = 0, a_iou = 0, b_iou = 0, max_iou_compensation;
  std::tie(max_iou, min_iou, a_iou, b_iou) =
      calcIou(ref_obj_box, target_obj_box);
  std::tie(max_iou_compensation, min_iou, a_iou, b_iou) =
      calcIou(ref_obj_box, target_obj_box_compensation);
  max_iou = std::max(max_iou, max_iou_compensation);
  float associated_iou_th = 0.;
  if (isSmallObject(target_obj->obj_ptr->box_size, target_obj->obj_ptr->type) &&
      isSmallObject(ref_obj->box_size, ref_obj->type)) {
    associated_iou_th = 0.05;
  } else {
    associated_iou_th = 0.05;
  }
  if (max_iou < associated_iou_th) {
    return false;
  } else {
    return true;
  }
}
// pv iou检查
bool OneStageFusion::isValidMatchingPvIou(
    const TrackingObject::Ptr& ref_obj,
    const SingleFusionCorrelationObject::Ptr& target_obj) {
  if (isRadarSensor(target_obj->sensor_id)) {
    return IsRadarInPvBox(SensorID::VIRTUAL_CAMERA_EXTENDED,
                          *target_obj->radar_proj_point_map,
                          *ref_obj->pv_proj_box_map) ||
           IsRadarInPvBox(SensorID::VIRTUAL_CAMERA,
                          *target_obj->radar_proj_point_map,
                          *ref_obj->pv_proj_box_map);
  }
  float pv_iou = 0;
  if (ref_obj->box_center_base.x * ref_obj->box_center_base.x +
          ref_obj->box_center_base.y * ref_obj->box_center_base.y <
      400) {
    pv_iou = std::max(
        CalPvProjIou(SensorID::VIRTUAL_CAMERA, *ref_obj->pv_proj_box_map,
                     *target_obj->pv_proj_box_map, true),
        pv_iou);
    pv_iou = std::max(CalPvProjIou(SensorID::VIRTUAL_CAMERA_EXTENDED,
                                   *ref_obj->pv_proj_box_map,
                                   *target_obj->pv_proj_box_map, true),
                      pv_iou);
  } else {
    pv_iou = std::max(
        CalPvProjIou(SensorID::VIRTUAL_CAMERA, *ref_obj->pv_proj_box_map,
                     *target_obj->pv_proj_box_map, false),
        pv_iou);
    pv_iou = std::max(CalPvProjIou(SensorID::VIRTUAL_CAMERA_EXTENDED,
                                   *ref_obj->pv_proj_box_map,
                                   *target_obj->pv_proj_box_map, false),
                      pv_iou);
  }

  float pv_thresh = 0.2;
  if (isStaticObject(ref_obj->type)) {
    pv_thresh = 0.1;
  } else if (isSmallObject(target_obj->obj_ptr->box_size, ref_obj->type) &&
             isSmallObject(target_obj->obj_ptr->box_size, ref_obj->type)) {
    pv_thresh = 0.1;
  }
  if (pv_iou < pv_thresh) {
    return false;
  } else {
    return true;
  }
}
bool OneStageFusion::isValidMatchingDistance(
    const TrackingObject::Ptr& ref_obj,
    const SingleFusionCorrelationObject::Ptr& target_obj) {
  Eigen::Vector2f ref_pt_diff_compensation(
      DistanceOfRefPoints(ref_obj->box_center_base_compensation,
                          ref_obj->box_size, ref_obj->yaw_base_compensation,
                          target_obj->obj_ptr->box_center_base_compensation,
                          target_obj->obj_ptr->box_size,
                          target_obj->obj_ptr->yaw_base_compensation));
  Eigen::Vector2f ref_pt_diff(DistanceOfRefPoints(
      ref_obj->box_center_base, ref_obj->box_size, ref_obj->yaw_base,
      target_obj->obj_ptr->box_center_base, target_obj->obj_ptr->box_size,
      target_obj->obj_ptr->yaw_base));
  Eigen::Vector2f center_pt_diff_compensation(
      target_obj->obj_ptr->box_center_base_compensation.x -
          ref_obj->box_center_base_compensation.x,
      target_obj->obj_ptr->box_center_base_compensation.y -
          ref_obj->box_center_base_compensation.y);
  Eigen::Vector2f center_pt_diff(
      target_obj->obj_ptr->box_center_base.x - ref_obj->box_center_base.x,
      target_obj->obj_ptr->box_center_base.y - ref_obj->box_center_base.y);
  // 距离偏差连线
  Eigen::Vector2f measurement_predict_diff = ref_pt_diff_compensation;
  if (measurement_predict_diff.norm() > ref_pt_diff.norm()) {
    measurement_predict_diff = ref_pt_diff;
  }
  if (measurement_predict_diff.norm() > center_pt_diff_compensation.norm()) {
    measurement_predict_diff = center_pt_diff_compensation;
  }
  if (measurement_predict_diff.norm() > center_pt_diff.norm()) {
    measurement_predict_diff = center_pt_diff;
  }

  bool matched_sensor = checkBitsetOfSensorIds(
      target_obj->sensor_id, ref_obj->observe_obj_ptr_->sensor_ids);

  // 距离大小筛选
  float observe_dist =
      std::hypot(target_obj->obj_ptr->box_center_base_compensation.x,
                 target_obj->obj_ptr->box_center_base_compensation.y);
  double dist_thresh = std::max(
      std::max(ref_obj->box_size.length, target_obj->obj_ptr->box_size.length),
      1.0f);

  if (!matched_sensor) {
    if (isStaticObject(ref_obj->type)) {
      if (ref_obj->exist_confidence == 0 &&
          target_obj->static_obj_exist_confidence == 0) {
        dist_thresh = clamp(0.06 * observe_dist, 1.0, 2.5);
      } else {
        dist_thresh = clamp(0.15 * observe_dist, 1.5, 10.0);
      }
    } else if (isPinholeSensor(target_obj->sensor_id) &&
               ref_obj->pv_post_id[target_obj->sensor_id].find(
                   target_obj->obj_ptr->object_id) !=
                   ref_obj->pv_post_id[target_obj->sensor_id].end()) {
      dist_thresh = std::max(0.3 * observe_dist, dist_thresh);
    } else if (isRadarSensor(target_obj->sensor_id) &&
               ref_obj->radar_id[target_obj->sensor_id].find(
                   target_obj->obj_ptr->object_id) !=
                   ref_obj->radar_id[target_obj->sensor_id].end()) {
      dist_thresh = std::max(0.15 * observe_dist, dist_thresh);
    } else {
      // 先不按传感器来设置，那样设置的阈值太严格
      dist_thresh = std::max(0.3 * observe_dist, dist_thresh);
    }
  } else {
    if (IsLidarSensor(target_obj->sensor_id)) {
      return false;
    } else {
      dist_thresh = std::min(ref_obj->box_size.length,
                             target_obj->obj_ptr->box_size.length);
    }
  }
  if (measurement_predict_diff.norm() > dist_thresh) {
    return false;
  }
  return true;
}
bool OneStageFusion::isValidMatchingTheta(
    const TrackingObject::Ptr& ref_obj,
    const SingleFusionCorrelationObject::Ptr& target_obj) {
  if (std::max(ref_obj->box_size.length, target_obj->obj_ptr->box_size.length) >
          2.0 &&
      std::abs(std::cos(ref_obj->yaw_base - target_obj->obj_ptr->yaw_base)) <
          0.7071) {
    return false;
  } else {
    return true;
  }
}
bool OneStageFusion::isValidVelArea(
    const TrackingObject::Ptr& ref_obj_tmp,
    const SingleFusionCorrelationObject::Ptr& target_obj) {
  if (!isVehObject(ref_obj_tmp->type) ||
      !isVehObject(target_obj->obj_ptr->type) ||
      isRadarSensor(target_obj->sensor_id)) {
    return true;
  }

  // 扇形区域判断
  Eigen::Affine3d ref_obj_base_to_rel_trans;
  temporal_fusion_ptr_->GetBaseToRelFromCache(ref_obj_tmp->time_stamp,
                                              ref_obj_base_to_rel_trans);
  auto ref_obj_box_center_odom =
      transformTo(ref_obj_base_to_rel_trans, ref_obj_tmp->box_center_base);
  auto ref_obj_vel_odom =
      velTransformTo(ref_obj_base_to_rel_trans, ref_obj_tmp->velocity_base);
  // 速度方向
  auto ref_obj_yaw_odom =
      transformTo(ref_obj_base_to_rel_trans, ref_obj_tmp->yaw_base);

  // 运动补偿，补偿至target_obj->time_stamp的前0.1s
  double delta_t = target_obj->time_stamp - ref_obj_tmp->time_stamp - 0.1;
  ref_obj_box_center_odom.x += ref_obj_vel_odom.x * delta_t;
  ref_obj_box_center_odom.y += ref_obj_vel_odom.y * delta_t;

  Eigen::Affine3d target_obj_base_to_rel_trans;
  temporal_fusion_ptr_->GetBaseToRelFromCache(target_obj->time_stamp,
                                              target_obj_base_to_rel_trans);
  ref_obj_yaw_odom =
      NormalizeAngle(ref_obj_yaw_odom + ref_obj_tmp->yaw_rate * delta_t * 0.1);
  auto tar_obj_box_center_odom = transformTo(
      target_obj_base_to_rel_trans, target_obj->obj_ptr->box_center_base);

  Eigen::Matrix2d rot_matrix;
  rot_matrix << std::cos(-ref_obj_yaw_odom), -std::sin(-ref_obj_yaw_odom),
      std::sin(-ref_obj_yaw_odom), std::cos(ref_obj_yaw_odom);
  Eigen::Vector2d ref_vec(ref_obj_box_center_odom.x, ref_obj_box_center_odom.y);
  Eigen::Vector2d tar_vec(tar_obj_box_center_odom.x, tar_obj_box_center_odom.y);
  Eigen::Vector2d tar_transformed_box_center_odom =
      rot_matrix * (tar_vec - ref_vec);
  // 速度方向扇形区域判断
  float vel_direction_th = 30. / 180. * M_PI;
  if (tar_transformed_box_center_odom[0] < -1. / std::tan(vel_direction_th) ||
      std::abs(tar_transformed_box_center_odom[1]) >
          std::abs(std::tan(vel_direction_th) *
                       tar_transformed_box_center_odom[0] +
                   1)) {
    return false;
  }
  return true;
}

float OneStageFusion::CalPvProjIou(
    const SensorID sensor_id,
    std::map<SensorID, std::array<std::pair<bool, PvBBox2D>, 2>>&
        fusion_obj_pv_proj_box_map,
    std::map<SensorID, std::array<std::pair<bool, PvBBox2D>, 2>>&
        det_obj_pv_proj_box_map,
    const bool use_IOU) {
  float iou = 0;
  for (int i = 0; i < 2; i++) {
    if (fusion_obj_pv_proj_box_map[sensor_id][i].first &&
        det_obj_pv_proj_box_map[sensor_id][i].first) {
      float max_iou = 0;
      if (use_IOU) {
        std::tie(std::ignore, std::ignore, std::ignore, std::ignore, max_iou) =
            calcPvIou2(fusion_obj_pv_proj_box_map[sensor_id][i].second,
                       det_obj_pv_proj_box_map[sensor_id][i].second);
      } else {
        std::tie(max_iou, std::ignore, std::ignore, std::ignore, std::ignore) =
            calcPvIou2(fusion_obj_pv_proj_box_map[sensor_id][i].second,
                       det_obj_pv_proj_box_map[sensor_id][i].second);
      }

      iou = std::max(max_iou, iou);
    }
  }
  return iou;
}
bool OneStageFusion::isValidAttribute(
    const TrackingObject::Ptr& ref_obj,
    const SingleFusionCorrelationObject::Ptr& target_obj) {
  // 近距离1s内有激光观测目标，不使用速度有巨大异常的目标
  if (ref_obj->lidar_observe_count != 0 &&
      ref_obj->live_frame - ref_obj->lidar_observe_frame < 10 &&
      ref_obj->box_center_base.x < 10) {
    if (std::hypot(
            ref_obj->velocity_base.x - target_obj->obj_ptr->velocity_base.x,
            ref_obj->velocity_base.y - target_obj->obj_ptr->velocity_base.y) >
        10) {
      return false;
    }
  }
  // 上一帧有激光观测目标且类型不同，不使用朝向变化过大的目标
  if (isCarObject(ref_obj->type) &&
      (ref_obj->observe_source == LIDAR_CAMERA_SOURCE ||
       ref_obj->observe_source == LIDAR_SOURCE) &&
      IsLidarSensor(target_obj->sensor_id) &&
      ref_obj->type != target_obj->obj_ptr->type) {
    if (std::abs(std::cos(ref_obj->yaw_base - target_obj->obj_ptr->yaw_base)) <
        0.866) {
      return false;
    }
  }
  return true;
}
bool OneStageFusion::IsRadarInPvBox(const SensorID sensor_id,
                                    RadarProjMap& radar_proj_map,
                                    PvProjMap& pv_proj_max) {
  for (int i = 0; i < 2; i++) {
    if (radar_proj_map[sensor_id].first && pv_proj_max[sensor_id][i].first) {
      auto& radar_point = radar_proj_map[sensor_id].second;
      auto& box = pv_proj_max[sensor_id][i].second;
      if (radar_point.x < box.x - virtual_proj_ptr_->GetAngleToWidth(0.5) ||
          radar_point.x >
              box.x + box.width + virtual_proj_ptr_->GetAngleToWidth(0.5) ||
          radar_point.y < box.y || radar_point.y > box.y + box.height) {
        continue;
      } else {
        return true;
      }
    }
  }
  return false;
}

bool OneStageFusion::IsValidTrackFilter(
    const SingleFusionCorrelationObject::Ptr& correlation_obj,
    std::map<SensorID, LidarCalib::Ptr>& lidar_calibs_ptr_) {
  auto& nn_state = correlation_obj->obj_ptr->nn_state;
  auto& sensor_id = correlation_obj->sensor_id;
  bool is_valid_nn_state = false;
  if (isFisheyeSensor(sensor_id)) {
    is_valid_nn_state = nn_state == 0 || nn_state == 4;
  } else if (isPinholeSensor(sensor_id)) {
    // 2: 截断，3：遮挡
    is_valid_nn_state =
        nn_state == 0 || nn_state == 2 || nn_state == 3 || nn_state == 4;
  } else if (IsLidarSensor(sensor_id) ||
             sensor_id == SensorID::INVALID_SENSOR_ID ||
             isRadarSensor(sensor_id)) {
    is_valid_nn_state = true;
  }
  if (!is_valid_nn_state) {
    return false;
  }

  auto& is_box_refine = correlation_obj->obj_ptr->is_box_refine;
  auto& center_pt = correlation_obj->obj_ptr->box_center_base;
  auto& obj_size = correlation_obj->obj_ptr->box_size;
  auto& tmp_vel = correlation_obj->obj_ptr->velocity_base;
  Vec3D back_pt;
  back_pt.x = correlation_obj->obj_ptr->box_center_base.x -
              0.5 * correlation_obj->obj_ptr->box_size.length *
                  std::abs(std::cos(correlation_obj->obj_ptr->yaw_base));

  switch (sensor_id) {
    case SensorID::LIDAR_M1_FRONT:
    case SensorID::LIDAR_EM4_FRONT:
    case SensorID::LIDAR_M1_LEFT:
    case SensorID::LIDAR_M1_RIGHT:
    case SensorID::LIDAR_M1_BACK: {
      Eigen::Vector3d box_center_veh_pt(center_pt.x, center_pt.y, center_pt.z);
      Eigen::Vector3d box_center_lidar_pt =
          lidar_calibs_ptr_[sensor_id]->VehPtToLidarPt(box_center_veh_pt);
      return box_center_lidar_pt(0) * box_center_lidar_pt(0) +
                 box_center_lidar_pt(1) * box_center_lidar_pt(1) >=
             25;  // 5*5
    }
    case SensorID::LIDAR_EMX_FRONT: {
      Eigen::Vector3d box_center_veh_pt(center_pt.x, center_pt.y, center_pt.z);
      Eigen::Vector3d box_center_lidar_pt =
          lidar_calibs_ptr_[sensor_id]->VehPtToLidarPt(box_center_veh_pt);
      return box_center_lidar_pt(0) * box_center_lidar_pt(0) +
                 box_center_lidar_pt(1) * box_center_lidar_pt(1) >=
             4;  // 2*2
    }
    case SensorID::LIDAR_E1_LEFT:
    case SensorID::LIDAR_E1_RIGHT:
    case SensorID::LIDAR_E1_FRONT:
    case SensorID::LIDAR_E1_BACK: {
      Eigen::Vector3d box_center_veh_pt(center_pt.x, center_pt.y, center_pt.z);
      Eigen::Vector3d box_center_lidar_pt =
          lidar_calibs_ptr_[sensor_id]->VehPtToLidarPt(box_center_veh_pt);
      float dist_square = box_center_lidar_pt(0) * box_center_lidar_pt(0) +
                          box_center_lidar_pt(1) * box_center_lidar_pt(1);
      return dist_square >= 25 && dist_square <= 900;  // 5*5 30*30
    }
    case SensorID::LIDAR_RUBY_TOP: {
      Eigen::Vector3d box_center_veh_pt(center_pt.x, center_pt.y, center_pt.z);
      Eigen::Vector3d box_center_lidar_pt =
          lidar_calibs_ptr_[SensorID::LIDAR_RUBY_TOP]->VehPtToLidarPt(
              box_center_veh_pt);
      return std::abs(box_center_lidar_pt(0)) >= 3 ||
             std::abs(box_center_lidar_pt(1)) >= 2;
    }
    case SensorID::LIDAR_FAIRY_TOP: {
      Eigen::Vector3d box_center_veh_pt(center_pt.x, center_pt.y, center_pt.z);
      Eigen::Vector3d box_center_lidar_pt =
          lidar_calibs_ptr_[SensorID::LIDAR_FAIRY_TOP]->VehPtToLidarPt(
              box_center_veh_pt);
      return std::abs(box_center_lidar_pt(0)) >= 3 ||
             std::abs(box_center_lidar_pt(1)) >= 2;
    }
    case SensorID::INVALID_SENSOR_ID: {
      Eigen::Vector3d box_center_veh_pt(center_pt.x, center_pt.y, center_pt.z);
      Eigen::Vector3d box_center_lidar_pt =
          lidar_calibs_ptr_[SensorID::LIDAR_M1_FRONT]->VehPtToLidarPt(
              box_center_veh_pt);
      return box_center_lidar_pt(0) * box_center_lidar_pt(0) +
                 box_center_lidar_pt(1) * box_center_lidar_pt(1) >=
             25;  // 5*5
    }
    case SensorID::CAM_FISHEYE_LEFT:
      return std::abs(center_pt.x) <= 10.0 && center_pt.y >= 1.0 &&
             center_pt.y <= 15.0;
    case SensorID::CAM_FISHEYE_RIGHT:
      return std::abs(center_pt.x) <= 10.0 && center_pt.y <= -1.0 &&
             center_pt.y >= -15.0;
    case SensorID::CAM_FISHEYE_FRONT:
      return (std::abs(center_pt.y) <= 1.5 && center_pt.x <= 15.0 &&
              center_pt.x > 4.0) ||
             (obj_size.length <= 2 && std::abs(center_pt.y) <= 5 &&
              center_pt.x <= 15.0 &&
              center_pt.x > 4.0);  // 如果是小目标则扩大过滤的范围
    case SensorID::CAM_FISHEYE_BACK:
      return (std::abs(center_pt.y) <= 5 && center_pt.x >= -15.0 &&
              center_pt.x <= 0.0) ||
             (obj_size.length <= 2 && std::abs(center_pt.y) <= 5 &&
              center_pt.x >= -10.0 &&
              center_pt.x <= 0.0);  // 如果是小目标则扩大过滤的范围
    case SensorID::CAM_RIGHT_BACK:
      return center_pt.y < 0 && nn_state != 2;  // 不用截断起目标
    case SensorID::CAM_LEFT_BACK:
      return center_pt.y > 0 && nn_state != 2;
    case SensorID::CAM_RIGHT_FRONT:
      return center_pt.x > 0 && center_pt.y < -5;
    case SensorID::CAM_LEFT_FRONT:
      return center_pt.x > 0 && center_pt.y > 5;
    case SensorID::CAM_BACK:
      return center_pt.x <= 0 && center_pt.x >= -100 &&
             std::abs(center_pt.y) <= 30;
    case SensorID::CAM_FRONT_NARROW:
      return (center_pt.x >= 10 && std::abs(center_pt.y) <= 30) ||
             (isStaticObject(correlation_obj->obj_ptr->type) &&
              center_pt.x >= 5 && std::abs(center_pt.y) <= 30);
    case SensorID::CAM_FRONT_WIDE:
      return center_pt.x >= 0 && center_pt.x <= 50 &&
             (std::abs(center_pt.y) <= 10 ||
              (std::abs(center_pt.y) <= 10 &&
               isStaticObject(correlation_obj->obj_ptr->type)));
    case SensorID::RADAR_FRONT_MRR: {
      // 初始化为未跟踪目标
      if (radar_id_status[sensor_id].find(
              correlation_obj->obj_ptr->object_id) ==
              radar_id_status[sensor_id].end() ||
          is_box_refine == 1 || is_box_refine == 5 || is_box_refine == 4) {
        radar_id_status[sensor_id][correlation_obj->obj_ptr->object_id] = -5;
      }
      if (is_box_refine != 2 && is_box_refine != 0) {
        return false;
      }
      if (is_box_refine != 0 &&
          correlation_obj->obj_ptr->type_confidence < 0.9) {
        return false;
      }
      // 10帧内关联到过其他目标
      if (frame_count -
              radar_id_status[sensor_id][correlation_obj->obj_ptr->object_id] <
          10) {
        return false;
      }
      // 前向仅补CIPV目标
      return center_pt.x > 10.0 && std::abs(center_pt.y) < 5.0 &&
             std::abs(tmp_vel.x) >= 5.0 && std::abs(tmp_vel.y) < 5.0 &&
             correlation_obj->velocity_base_relative.x > -11.5 &&
             std::abs(std::cos(correlation_obj->obj_ptr->yaw_base)) >
                 std::cos(M_PI / 12);
    }
    case SensorID::RADAR_BACK_MRR: {
      // 初始化为未跟踪目标
      if (radar_id_status[sensor_id].find(
              correlation_obj->obj_ptr->object_id) ==
              radar_id_status[sensor_id].end() ||
          is_box_refine == 1 || is_box_refine == 5 || is_box_refine == 4) {
        radar_id_status[sensor_id][correlation_obj->obj_ptr->object_id] = -5;
      }
      if (is_box_refine != 2 && is_box_refine != 0) {
        return false;
      }
      if (correlation_obj->obj_ptr->type_confidence < 0.9) {
        return false;
      }
      // 10帧内关联到过其他目标
      if (frame_count -
              radar_id_status[sensor_id][correlation_obj->obj_ptr->object_id] <
          10) {
        return false;
      }
      return center_pt.x < 0.0 && std::abs(center_pt.y) < 10.0 &&
             std::abs(tmp_vel.x) >= 5.0 && std::abs(tmp_vel.y) < 5.0 &&
             correlation_obj->velocity_base_relative.x > -11.5 &&
             std::abs(std::cos(correlation_obj->obj_ptr->yaw_base)) >
                 std::cos(M_PI / 12);
    }
    default:
      return false;
  }
}

float OneStageFusion::calProDensity(
    const TrackingObject::Ptr& obj_1,
    const SingleFusionCorrelationObject::Ptr& obj_2) {
  float pro_density = 0;
  // const float distance_thresh = 3.0;
  Eigen::Vector2f ref_pt_diff(
      DistanceOfRefPoints(obj_1->box_center_base, obj_1->box_size,
                          obj_1->yaw_base, obj_2->obj_ptr->box_center_base,
                          obj_2->obj_ptr->box_size, obj_2->obj_ptr->yaw_base));
  Eigen::Vector2f center_pt_diff(
      obj_2->obj_ptr->box_center_base.x - obj_1->box_center_base.x,
      obj_2->obj_ptr->box_center_base.y - obj_1->box_center_base.y);

  Eigen::Vector2f measurement_predict_diff;
  if (ref_pt_diff.norm() < center_pt_diff.norm()) {
    measurement_predict_diff = ref_pt_diff;
  } else {
    measurement_predict_diff = center_pt_diff;
  }
  float location_distance{measurement_predict_diff.norm()};

  Eigen::Vector2f track_motion_dir{obj_1->velocity_base.x,
                                   obj_1->velocity_base.y};
  float track_speed{track_motion_dir.norm()};
  Eigen::Matrix2f s_inverse{Eigen::Matrix2f::Identity()};

  if (track_speed > 2.F) {
    track_motion_dir.normalize();

    Eigen::RowVector2f dir_part;
    Eigen::RowVector2f orthogonal_dir_part;
    Eigen::RowVector2f measurement_predict_diff_mat;
    dir_part << track_motion_dir(0), track_motion_dir(1);
    orthogonal_dir_part << track_motion_dir(1), -track_motion_dir(0);
    measurement_predict_diff_mat << measurement_predict_diff(0),
        measurement_predict_diff(1);

    s_inverse = dir_part.transpose() * dir_part * 0.5F +
                orthogonal_dir_part.transpose() * orthogonal_dir_part * 2.F;

    auto square_location_distance = measurement_predict_diff_mat * s_inverse *
                                    measurement_predict_diff_mat.transpose();
    location_distance = std::sqrt(square_location_distance(0));
  }

  float result_distance = location_distance * 0.6F;
  pro_density = static_cast<float>(
      (1. / std::sqrt(2. * rally::R_M_PI *
                      static_cast<double>(s_inverse.determinant()))) *
      std::exp(-0.5 * static_cast<double>(location_distance)));

  return pro_density;
}
float OneStageFusion::calProDensity(
    const SingleFusionCorrelationObject::Ptr& obj_1,
    const SingleFusionCorrelationObject::Ptr& obj_2) {
  float pro_density = 0;
  // const float distance_thresh = 3.0;
  Eigen::Vector2f ref_pt_diff(DistanceOfRefPoints(
      obj_1->obj_ptr->box_center_base, obj_1->obj_ptr->box_size,
      obj_1->obj_ptr->yaw_base, obj_2->obj_ptr->box_center_base,
      obj_2->obj_ptr->box_size, obj_2->obj_ptr->yaw_base));
  Eigen::Vector2f center_pt_diff(
      obj_2->obj_ptr->box_center_base.x - obj_1->obj_ptr->box_center_base.x,
      obj_2->obj_ptr->box_center_base.y - obj_1->obj_ptr->box_center_base.y);

  Eigen::Vector2f measurement_predict_diff;
  if (ref_pt_diff.norm() < center_pt_diff.norm()) {
    measurement_predict_diff = ref_pt_diff;
  } else {
    measurement_predict_diff = center_pt_diff;
  }
  float location_distance{measurement_predict_diff.norm()};

  Eigen::Vector2f track_motion_dir{obj_1->obj_ptr->velocity_base.x,
                                   obj_1->obj_ptr->velocity_base.y};
  float track_speed{track_motion_dir.norm()};
  Eigen::Matrix2f s_inverse{Eigen::Matrix2f::Identity()};

  if (track_speed > 2.F) {
    track_motion_dir.normalize();

    Eigen::RowVector2f dir_part;
    Eigen::RowVector2f orthogonal_dir_part;
    Eigen::RowVector2f measurement_predict_diff_mat;
    dir_part << track_motion_dir(0), track_motion_dir(1);
    orthogonal_dir_part << track_motion_dir(1), -track_motion_dir(0);
    measurement_predict_diff_mat << measurement_predict_diff(0),
        measurement_predict_diff(1);

    s_inverse = dir_part.transpose() * dir_part * 0.5F +
                orthogonal_dir_part.transpose() * orthogonal_dir_part * 2.F;

    auto square_location_distance = measurement_predict_diff_mat * s_inverse *
                                    measurement_predict_diff_mat.transpose();
    location_distance = std::sqrt(square_location_distance(0));
  }

  float result_distance = location_distance * 0.6F;
  pro_density = static_cast<float>(
      (1. / std::sqrt(2. * rally::R_M_PI *
                      static_cast<double>(s_inverse.determinant()))) *
      std::exp(-0.5 * static_cast<double>(location_distance)));

  return pro_density;
}
void OneStageFusion::AddObserve(
    const TrackingObject::Ptr& tracking_object,
    const SingleFusionCorrelationObject::Ptr& observe_obj) {
  // 可用性判断
  if (tracking_object->observe_frame > 3 &&
      !isValidAttribute(tracking_object, observe_obj)) {
    return;
  }

  // 添加目标
  tracking_object->observe_obj_ptr_->correlation_objs.emplace_back(observe_obj);

  // 记录真实观测源
  setBitsetOfSensorIds(SensorID(observe_obj->sensor_id),
                       tracking_object->observe_obj_ptr_->sensor_ids);

  // 记录id
  if (isPinholeSensor(observe_obj->sensor_id)) {
    tracking_object->observe_obj_ptr_->pv_post_id[observe_obj->sensor_id]
        .insert(observe_obj->obj_ptr->object_id);
  } else if (isRadarSensor(observe_obj->sensor_id)) {
    tracking_object->observe_obj_ptr_
        ->correlation_radar_id[observe_obj->sensor_id]
        .insert(observe_obj->obj_ptr->object_id);
  }

  // 记录视觉属性信息
  tracking_object->observe_obj_ptr_->SetLightInfo(
      observe_obj->sensor_id,
      BRAKE_LIGHT_STATE(observe_obj->obj_ptr->brake_light),
      SIGNAL_LIGHT_STATE(observe_obj->obj_ptr->signal_light));
  if (isPinholeSensor(observe_obj->sensor_id)) {
    tracking_object->observe_obj_ptr_->SetDoorInfo(
        observe_obj->sensor_id, observe_obj->obj_ptr->door_state,
        observe_obj->obj_ptr->yaw_base);
  }
}
void OneStageFusion::FusionLidarObj(
    const PostFusionMsg::Ptr& msg_ptr,
    TrackingObjectList& temporal_tracking_list) {
  for (const auto& topic : kMatchLidarOrder) {
    const auto& msg_iter = msg_ptr->input_msg_ptr->obj_map.find(topic);
    if (msg_iter == msg_ptr->input_msg_ptr->obj_map.end()) {
      continue;
    }
    auto time_stamp = apollo::cyber::Time(msg_iter->second->header.time);
    auto sensor_id = msg_iter->second->sensor_id;
    bool is_valid_data = IsLidarSensor(sensor_id) &&
                         msg_iter->first.find("occupancy") == std::string::npos;
    if (!is_valid_data) {
      continue;
    }
    if (msg_iter->second->object_list.size() == 0) {
      continue;
    }
    if (lidar_calibs_ptr_.find(sensor_id) == lidar_calibs_ptr_.end()) {
      AWARN << "Can't find calib for sensor_id = "
            << kSensorIDToNameMap.at(sensor_id) << " topic= " << topic
            << " exit!";
      continue;
    }
    // 生成传感器的correlation_obj
    std::vector<SingleFusionCorrelationObject::Ptr> observe_objs;
    observe_objs.reserve(msg_iter->second->object_list.size());
    for (int i = 0; i < msg_iter->second->object_list.size(); ++i) {
      auto& obj = msg_iter->second->object_list[i];
      if (!IsValidLidarObj(obj)) {
        continue;
      }

      if (obj.type == ObjectType::TYPE_PED &&
          (std::abs(obj.velocity_base.x) > 0.1 ||
           std::abs(obj.velocity_base.y) > 0.1)) {
        obj.yaw_base = std::atan2(obj.velocity_base.y, obj.velocity_base.x);
      }
      MotionCompensation(sensor_id, obj, time_stamp.ToSecond(),
                         lidar_time_stamp_);
      std::shared_ptr<PvProjMap> tmp_pv_proj_box_map_ptr =
          std::make_shared<PvProjMap>();
      multi_sensor_fusion_ptr_->MultiSensorPvProj(
          obj, cam_calibs_ptr_, sensor_id, *tmp_pv_proj_box_map_ptr);
      auto new_obj = std::make_shared<SingleFusionCorrelationObject>(obj);
      new_obj->pv_proj_box_map = tmp_pv_proj_box_map_ptr;
      new_obj->sensor_id = sensor_id;
      new_obj->time_stamp = time_stamp.ToSecond();
      copyCorrelationObjectInfo(obj, *new_obj);
      observe_objs.emplace_back(new_obj);
      if (IsM1PSensor(sensor_id) || IsE1Sensor(sensor_id) ||
          IsEM4Sensor(sensor_id)) {
        new_obj->is_outside_lidar_fov =
            IsLidarFovObj(obj, sensor_id, 60, lidar_calibs_ptr_);
      } else if (IsEMXSensor(sensor_id)) {
        new_obj->is_outside_lidar_fov =
            IsLidarFovObj(obj, sensor_id, 65, lidar_calibs_ptr_);
      }
    }

    // 粗匹配
    std::vector<bool> matched_label;
    GreedyMatch(observe_objs, sensor_id, temporal_tracking_list, matched_label);
    // 匈牙利匹配
    HungarianMatch(observe_objs, temporal_tracking_list, matched_label);
    CreateTrackObject(observe_objs, temporal_tracking_list, matched_label);
  }
}
void OneStageFusion::UpdateTrackingObject(
    TrackingObjectList& temporal_tracking_list) {
  std::vector<bool> valid_observe_label(
      temporal_tracking_list.object_list.size(), false);
  // 获取车道yaw信息
  Eigen::Affine3d cur_base_to_rel_trans;
  PostFusionEgoPoseInfo::GetInstance().GetBaseToRel(lidar_time_stamp_,
                                                    cur_base_to_rel_trans);
  Eigen::Affine3d cur_rel_to_base_trans = cur_base_to_rel_trans.inverse();
  float yaw_temp = 0, cur_yaw_base = 0, left_yaw_base = 0, right_yaw_base = 0;
  if (SceneDetection::GetInstance().GetCurYaw(yaw_temp)) {
    cur_yaw_base = transformTo(cur_rel_to_base_trans, yaw_temp);
  }
  left_yaw_base = cur_yaw_base;
  right_yaw_base = cur_yaw_base;
  if (SceneDetection::GetInstance().GetLeftYaw(yaw_temp)) {
    left_yaw_base = transformTo(cur_rel_to_base_trans, yaw_temp);
  }
  if (SceneDetection::GetInstance().GetRightYaw(yaw_temp)) {
    right_yaw_base = transformTo(cur_rel_to_base_trans, yaw_temp);
  }
  AINFO << " CUR LANE YAW " << cur_yaw_base << " LEFT " << left_yaw_base
        << " RIGHT " << right_yaw_base;

  // 历史目标选择
  for (int i = 0; i < history_obj_num_; ++i) {
    auto& track_obj = temporal_tracking_list.object_list[i];
    auto& current_obj = track_obj->observe_obj_ptr_;
    // 之前记录的是当前帧的真实观测信息
    current_obj->sensor_ids = 0x0;
    bool valid_observe = false;

    // 连续观测目标统一yaw角
    if (track_obj->observe_count > 10) {
      if (track_obj->type == ObjectType::TYPE_PED) {
        current_obj->YawQuadrantUnity(track_obj->yaw_base, false,
                                      allow_only_lidar_fusion_);
      } else {
        current_obj->YawQuadrantUnity(track_obj->yaw_base, true,
                                      allow_only_lidar_fusion_);
      }
    }

    // 补偿后的box
    const auto& track_obj_box = track_obj->bev_box_base;
    // 历史yaw信息
    bool is_cut_in_obj = false;
    if (track_obj->observe_count > 5 && track_obj->box_center_base.y >= 0) {
      float pre_yaw_diff = track_obj->yaw_base - left_yaw_base;
      if (pre_yaw_diff < -M_PI / 60 && track_obj->yaw_rate < -0.01) {
        is_cut_in_obj = true;
      }
    } else if (track_obj->observe_count > 5) {
      float pre_yaw_diff = track_obj->yaw_base - right_yaw_base;
      if (pre_yaw_diff > M_PI / 60 && track_obj->yaw_rate > 0.01) {
        is_cut_in_obj = true;
      }
    }

    // 最接近IOU，最平yaw角
    float max_correlation_obj_iou = 0.F, min_correlation_obj_yaw = M_PI;
    // 最佳匹配、最佳iou匹配、最佳yaw匹配
    uint32_t index = 0, best_iou_index = 0, best_yaw_index = 0;
    // 高置信度激光结果
    bool is_bev_post_refine = false;
    bool has_lidar_observe_car = false;
    VISION_COVER_STATE cur_vision_state = VISION_COVER_STATE::NO_OBSERVE;

    // 超长车目标
    bool is_long_truck = false;
    if (track_obj->box_size.length > 5 &&
        std::abs(track_obj->box_center_base.y) < 5) {
      float head_x =
          track_obj->box_center_base.x +
          track_obj->box_size.length * 0.5 * std::cos(track_obj->yaw_base);
      float tail_x =
          track_obj->box_center_base.x -
          track_obj->box_size.length * 0.5 * std::cos(track_obj->yaw_base);
      if ((head_x > 2 && tail_x < -3) || (head_x < -3 && tail_x > 2)) {
        is_long_truck = true;
      }
    }
    for (size_t k = 0; k < current_obj->correlation_objs.size(); k++) {
      auto& tmp_correlation_obj = current_obj->correlation_objs[k];
      // 增加观测源信息 lidar只简单存储是否是fov信息
      if (IsLidarSensor(tmp_correlation_obj->sensor_id)) {
        if (tmp_correlation_obj->obj_ptr->lidar_camera_check != 0) {
          setBitsetOfSensorIds(
              SensorID(tmp_correlation_obj->obj_ptr->lidar_camera_check),
              current_obj->sensor_ids);
          setBitsetOfSensorIds(SensorID::LIDAR_M1_FRONT,
                               current_obj->sensor_ids);
        } else if (tmp_correlation_obj->sensor_id == SensorID::LIDAR_M1_FRONT &&
                   tmp_correlation_obj->obj_ptr->lidar_camera_check == 0 &&
                   tmp_correlation_obj->is_outside_lidar_fov !=
                       OUTSIDE_LIDAR_FOV_STATE::INSIDE_LIDAR_FOV) {
          setBitsetOfSensorIds(SensorID::INVALID_SENSOR_ID,
                               current_obj->sensor_ids);
        } else {
          setBitsetOfSensorIds(SensorID::LIDAR_M1_FRONT,
                               current_obj->sensor_ids);
        }
      } else {
        setBitsetOfSensorIds(SensorID(tmp_correlation_obj->sensor_id),
                             current_obj->sensor_ids);
      }
      // 只使用属性的不参与
      if (tmp_correlation_obj->correlation_state_ ==
              CORRELATION_STATE::ONLY_ATTRIBUTES_VALID &&
          !is_long_truck) {
        continue;
      }
      // 低分、遮挡、截断目标为低权重目标，而且低分目标不参与未pub目标关联
      bool is_low_priority_obj = false;
      if (isCameraSensor(tmp_correlation_obj->sensor_id) &&
          tmp_correlation_obj->obj_ptr->nn_state != 0) {
        if (track_obj->pub_state == PUB_CODE::PUB_CODE_UNKNOWN) {
          continue;
        }
        is_low_priority_obj = true;
      }
      // 近距离侧前目标为低权重目标
      if (isSmallObject(tmp_correlation_obj->obj_ptr->box_size,
                        tmp_correlation_obj->obj_ptr->type) &&
          std::abs(tmp_correlation_obj->obj_ptr->box_center_base.y) < 10) {
        if (tmp_correlation_obj->sensor_id == SensorID::CAM_RIGHT_FRONT ||
            tmp_correlation_obj->sensor_id == SensorID::CAM_LEFT_FRONT) {
          is_low_priority_obj = true;
        }
      }
      // 对于框来说近距离radar是低权重目标
      if (isRadarSensor(tmp_correlation_obj->sensor_id) &&
          std::abs(tmp_correlation_obj->obj_ptr->box_center_base.x) < 50) {
        is_low_priority_obj = true;
      }
      // radar目标记录相对速度
      if (isRadarSensor(tmp_correlation_obj->sensor_id)) {
        radar_id_status[tmp_correlation_obj->sensor_id]
                       [tmp_correlation_obj->obj_ptr->object_id] = frame_count;
        // 前向非cipv且比自车块的纯radar目标不跟踪
        if (current_obj->correlation_objs.size() == 1 &&
            std::abs(tmp_correlation_obj->obj_ptr->box_center_base.y) > 2.5 &&
            tmp_correlation_obj->obj_ptr->box_center_base.x > 0 &&
            tmp_correlation_obj->velocity_base_relative.x *
                        tmp_correlation_obj->velocity_base_relative.x +
                    tmp_correlation_obj->velocity_base_relative.y *
                        tmp_correlation_obj->velocity_base_relative.y >
                1) {
          continue;
        }
        current_obj->correlation_radar_id[tmp_correlation_obj->sensor_id]
            .insert(tmp_correlation_obj->obj_ptr->object_id);
        track_obj->velocity_base_relative =
            tmp_correlation_obj->velocity_base_relative;
      }
      valid_observe = true;
      // 记录视觉遮挡状态
      if (isCameraSensor(tmp_correlation_obj->sensor_id)) {
        if (track_obj->vision_cover_state_ == VISION_COVER_STATE::NO_OBSERVE) {
          cur_vision_state = VISION_COVER_STATE::NORMAL;
        }
        // 目前只记录narrow和wide
        if (tmp_correlation_obj->sensor_id == SensorID::CAM_FRONT_NARROW ||
            tmp_correlation_obj->sensor_id == SensorID::CAM_FRONT_WIDE) {
          // 上下遮挡通常为硬隔离，阈值设置小一点
          cur_vision_state =
              GetPvCoverState(tmp_correlation_obj->obj_ptr->box_full,
                              tmp_correlation_obj->obj_ptr->box_visible);
        }
      }
      // 选择优化源
      if (!allow_only_lidar_fusion_ &&
          tmp_correlation_obj->sensor_id == SensorID::LIDAR_M1_FRONT &&
          ((std::abs(track_obj->box_center_base.x) < 70 &&
            std::abs(track_obj->box_center_base.y) < 10 &&
            std::abs(track_obj->box_center_base.x) > 10) ||
           (tmp_correlation_obj->obj_ptr->is_box_refine == 1 &&
            tmp_correlation_obj->is_outside_lidar_fov ==
                OUTSIDE_LIDAR_FOV_STATE::INSIDE_LIDAR_FOV &&
            std::abs(track_obj->box_center_base.x) > 10 &&
            std::abs(track_obj->box_center_base.y) > 10))) {
        // 激光观测较好区域或激光精修目标直接使用激光框
        index = k;
        is_bev_post_refine = true;
      } else if (!is_bev_post_refine) {
        const auto& current_obj_box = tmp_correlation_obj->bev_box_base;
        float iou = calcIou2(track_obj_box, current_obj_box);
        if (track_obj->observe_count > 5 &&
            IsLidarSensor(tmp_correlation_obj->sensor_id)) {
          const auto& compensation_track_box =
              track_obj->bev_box_base_compensation;
          const auto& compensation_obj_box =
              tmp_correlation_obj->bev_box_base_compensation;
          iou = std::max(calcIou2(compensation_track_box, compensation_obj_box),
                         iou);
        }
        // 降低低权目标iou
        if (is_low_priority_obj) {
          iou /= 10;
        }
        // 激光边缘区域的激光精修目标，提高iou
        if (IsLidarSensor(tmp_correlation_obj->sensor_id) &&
            tmp_correlation_obj->obj_ptr->is_box_refine == 1) {
          iou *= 10;
        }
        // 小目标信激光
        if (isSmallObject(tmp_correlation_obj->obj_ptr->box_size,
                          tmp_correlation_obj->obj_ptr->type) &&
            isSmallObject(track_obj->box_size, track_obj->type) &&
            IsLidarSensor(tmp_correlation_obj->sensor_id) &&
            tmp_correlation_obj->is_outside_lidar_fov ==
                OUTSIDE_LIDAR_FOV_STATE::INSIDE_LIDAR_FOV) {
          iou *= 2;
        }
        // cutin时反向朝向目标根据差值降低IOU
        float yaw_diff_with_history =
            tmp_correlation_obj->obj_ptr->yaw_base - track_obj->yaw_base;
        if (is_cut_in_obj && yaw_diff_with_history * track_obj->yaw_rate < 0 &&
            !IsLidarSensor(tmp_correlation_obj->sensor_id)) {
          float weight =
              std::max(1.0, std::abs(yaw_diff_with_history * 180 / M_PI));
          iou *= (0.9 / weight);
        }
        if (iou > max_correlation_obj_iou) {
          // 如果之前最优是radar就直接用这个iou关联
          if (!(isRadarSensor(tmp_correlation_obj->sensor_id) &&
                !isRadarSensor(
                    current_obj->correlation_objs[best_iou_index]->sensor_id) &&
                max_correlation_obj_iou > 0.2)) {
            max_correlation_obj_iou = iou;
            best_iou_index = k;
          }
          // 使用激光观测box时不使用yaw最平
          if (IsLidarSensor(tmp_correlation_obj->sensor_id) &&
              isCarObject(tmp_correlation_obj->obj_ptr->type)) {
            has_lidar_observe_car = true;
          } else {
            has_lidar_observe_car = false;
          }
        }

        // 和所在车道最平
        float yaw_lane_diff = 0;
        if (tmp_correlation_obj->obj_ptr->box_center_base.y >= 0) {
          yaw_lane_diff =
              std::abs(tmp_correlation_obj->obj_ptr->yaw_base - left_yaw_base);
        } else {
          yaw_lane_diff =
              std::abs(tmp_correlation_obj->obj_ptr->yaw_base - right_yaw_base);
        }
        // 低权重目标和路口的可能cutin场景不进入yaw最平逻辑
        bool intern_cut_out_scene = false;
        if (SceneDetection::GetInstance().GetCurrentScene() !=
            SceneType::NORMAL_SCENE) {
          if (track_obj->box_center_base.y >= 0 &&
              track_obj->yaw_base > 0.015) {
            intern_cut_out_scene = true;
          } else if (track_obj->box_center_base.y < 0 &&
                     track_obj->yaw_base < -0.015) {
            intern_cut_out_scene = true;
          }
        }
        if (!is_cut_in_obj &&
            (SceneDetection::GetInstance().GetCurrentScene() ==
                 SceneType::NORMAL_SCENE ||
             intern_cut_out_scene) &&
            yaw_lane_diff < min_correlation_obj_yaw && iou > 0.2 &&
            !is_low_priority_obj) {
          min_correlation_obj_yaw = yaw_lane_diff;
          best_yaw_index = k;
        }
      }
    }

    //  之前无观测或者是正常观测就跟随当前帧观测变化
    if (track_obj->vision_cover_state_ == VISION_COVER_STATE::NO_OBSERVE ||
        track_obj->vision_cover_state_ == VISION_COVER_STATE::NORMAL) {
      track_obj->vision_cover_state_ = cur_vision_state;
    }
    // 之前为遮挡观测时无观测不改变遮挡状态
    else {
      if (cur_vision_state != VISION_COVER_STATE::NO_OBSERVE) {
        track_obj->vision_cover_state_ = cur_vision_state;
      }
    }
    if (valid_observe) {
      float best_yaw =
          current_obj->correlation_objs[best_iou_index]->obj_ptr->yaw_base;
      if (!is_bev_post_refine && !has_lidar_observe_car) {
        if (track_obj->box_center_base.x > -15 &&
            track_obj->box_center_base.x < 4 &&
            std::abs(track_obj->box_center_base.y) < 5 &&
            track_obj->velocity_base.x * track_obj->velocity_base.x +
                    track_obj->velocity_base.y * track_obj->velocity_base.y >
                1) {
          float yaw_thresh = M_PI / 120;
          // 当最平yaw是cutin自车时，放宽限制
          if (track_obj->box_center_base.y > 0) {
            if (current_obj->correlation_objs[best_iou_index]
                        ->obj_ptr->yaw_base > 0 &&
                current_obj->correlation_objs[best_yaw_index]
                        ->obj_ptr->yaw_base < 0) {
              yaw_thresh = M_PI / 60;
            }
          } else {
            if (current_obj->correlation_objs[best_iou_index]
                        ->obj_ptr->yaw_base < 0 &&
                current_obj->correlation_objs[best_yaw_index]
                        ->obj_ptr->yaw_base > 0) {
              yaw_thresh = M_PI / 60;
            }
          }
          if (min_correlation_obj_yaw < yaw_thresh) {
            index = best_iou_index;
            best_yaw = current_obj->correlation_objs[best_yaw_index]
                           ->obj_ptr->yaw_base;
          } else {
            index = best_iou_index;
          }

        } else {
          index = best_iou_index;
        }
      } else {
        best_yaw = current_obj->correlation_objs[index]->obj_ptr->yaw_base;
      }
      if (index < current_obj->correlation_objs.size()) {
        temporal_fusion_ptr_->copyCorrelationObjectInfo(
            *(current_obj->correlation_objs[index]), *current_obj);
        current_obj->yaw_base = best_yaw;
      }
    }
    valid_observe_label[i] = valid_observe;
  }
  // 新创建的
  for (int i = history_obj_num_; i < temporal_tracking_list.object_list.size();
       ++i) {
    auto& track_obj = temporal_tracking_list.object_list[i];
    auto& current_obj = track_obj->observe_obj_ptr_;
    current_obj->sensor_ids = 0x0;
    current_obj->YawQuadrantUnity(track_obj->yaw_base, true,
                                  allow_only_lidar_fusion_);
    bool is_bev_post_refine = false;
    bool valid_observe = false;

    for (size_t k = 0; k < current_obj->correlation_objs.size(); k++) {
      auto correlation_obj = current_obj->correlation_objs[k];

      if (IsLidarSensor(correlation_obj->sensor_id)) {
        if (correlation_obj->obj_ptr->lidar_camera_check != 0) {
          setBitsetOfSensorIds(
              SensorID(correlation_obj->obj_ptr->lidar_camera_check),
              current_obj->sensor_ids);
          setBitsetOfSensorIds(SensorID::LIDAR_M1_FRONT,
                               current_obj->sensor_ids);
        } else if (correlation_obj->sensor_id == SensorID::LIDAR_M1_FRONT &&
                   correlation_obj->obj_ptr->lidar_camera_check == 0 &&
                   correlation_obj->is_outside_lidar_fov !=
                       OUTSIDE_LIDAR_FOV_STATE::INSIDE_LIDAR_FOV) {
          setBitsetOfSensorIds(SensorID::INVALID_SENSOR_ID,
                               current_obj->sensor_ids);
        } else {
          setBitsetOfSensorIds(SensorID::LIDAR_M1_FRONT,
                               current_obj->sensor_ids);
        }
      } else {
        setBitsetOfSensorIds(SensorID(correlation_obj->sensor_id),
                             current_obj->sensor_ids);
      }
      if (correlation_obj->correlation_state_ ==
          CORRELATION_STATE::ONLY_ATTRIBUTES_VALID) {
        continue;
      }
      if (IsLidarSensor(correlation_obj->sensor_id) &&
          correlation_obj->obj_ptr->is_box_refine == 1) {
        temporal_fusion_ptr_->copyCorrelationObjectInfo(*correlation_obj,
                                                        *current_obj);
        is_bev_post_refine = true;
        valid_observe = true;
      }
      // 还没有过观测源
      if (!valid_observe) {
        temporal_fusion_ptr_->copyCorrelationObjectInfo(*correlation_obj,
                                                        *current_obj);
        valid_observe = true;
      } else if (!is_bev_post_refine) {
        bool is_small_veh_valid =
            current_obj->box_size.length < 6.0 &&
            current_obj->box_size.length > 2 &&
            current_obj->box_center_base.x < 5.0 &&
            std::min(
                std::abs(current_obj->yaw_base),
                std::abs(std::abs(current_obj->yaw_base) - rally::R_M_PI)) >
                std::min(std::abs(correlation_obj->obj_ptr->yaw_base),
                         std::abs(std::abs(correlation_obj->obj_ptr->yaw_base) -
                                  rally::R_M_PI));
        bool is_large_veh_valid =
            current_obj->box_size.length >= 6.0 &&
            std::min(
                std::abs(current_obj->yaw_base),
                std::abs(std::abs(current_obj->yaw_base) - rally::R_M_PI)) >
                std::min(std::abs(correlation_obj->obj_ptr->yaw_base),
                         std::abs(std::abs(correlation_obj->obj_ptr->yaw_base) -
                                  rally::R_M_PI));
        // 如果新目标yaw角更平，用新目标替换fusion_obj。如果是radar目标并且有非radar源的目标，则不做此判断
        if ((is_small_veh_valid || is_large_veh_valid) &&
            !(isRadarSensor(correlation_obj->sensor_id) &&
              !isRadarSensor(current_obj->sensor_id))) {
          temporal_fusion_ptr_->copyCorrelationObjectInfo(*correlation_obj,
                                                          *current_obj);
        }
      }
      // 记录radar信息
      if (isRadarSensor(correlation_obj->sensor_id)) {
        radar_id_status[correlation_obj->sensor_id]
                       [correlation_obj->obj_ptr->object_id] = frame_count;
      }
    }
    valid_observe_label[i] = valid_observe;
  }
  // 删掉要删除的无效tracking_obj
  for (int i = temporal_tracking_list.object_list.size() - 1; i >= 0; --i) {
    auto& track_obj = temporal_tracking_list.object_list[i];
    if (valid_observe_label[i]) {
      track_obj->pv_post_id = track_obj->observe_obj_ptr_->pv_post_id;
      track_obj->radar_id = track_obj->observe_obj_ptr_->correlation_radar_id;
      Eigen::Affine3d base_to_rel_trans;
      PostFusionEgoPoseInfo::GetInstance().GetBaseToRel(
          track_obj->observe_obj_ptr_->time_stamp, base_to_rel_trans);
      track_obj->updateMeasurement(track_obj->observe_obj_ptr_,
                                   base_to_rel_trans);
    } else {
      if (i >= history_obj_num_) {
        temporal_tracking_list.object_list.erase(
            temporal_tracking_list.object_list.begin() + i);
      } else {
        auto& track_obj = temporal_tracking_list.object_list[i];
        if (lidar_time_stamp_ - track_obj->getMeasureTimeStamp() >
            track_obj->dead_time) {
          temporal_tracking_list.object_list.erase(
              temporal_tracking_list.object_list.begin() + i);
        } else {
          Eigen::Affine3d base_to_rel_trans;
          track_obj->updateMeasurement(nullptr, base_to_rel_trans);
        }
      }
    }
  }
  std::sort(temporal_tracking_list.object_list.begin(),
            temporal_tracking_list.object_list.end(),
            [](const TrackingObject::Ptr a, const TrackingObject::Ptr b) {
              return a->box_center_base.x * a->box_center_base.x +
                         a->box_center_base.y * a->box_center_base.y <
                     b->box_center_base.x * b->box_center_base.x +
                         b->box_center_base.y * b->box_center_base.y;
            });
}
void OneStageFusion::FusionVisionObj(
    const PostFusionMsg::Ptr& msg_ptr,
    TrackingObjectList& temporal_tracking_list) {
  // 表示访问顺序，所以不是从msg_ptr中直接获取
  for (const auto& topic : kMatchCameraOrder) {
    const auto& msg_iter = msg_ptr->input_msg_ptr->obj_map.find(topic);
    if (msg_iter == msg_ptr->input_msg_ptr->obj_map.end()) {
      continue;
    }

    auto time_stamp = apollo::cyber::Time(msg_iter->second->header.time);
    auto sensor_id = msg_iter->second->sensor_id;
    if (msg_iter->second->object_list.size() == 0) {
      continue;
    }
    if (cam_calibs_ptr_.find(sensor_id) == cam_calibs_ptr_.end()) {
      AWARN << "Can't find calib for sensor_id = "
            << kSensorIDToNameMap.at(sensor_id) << " exit!";
      continue;
    }

    std::vector<bool> valid_label(msg_iter->second->object_list.size(), true);
    if (isFisheyeSensor(sensor_id)) {
      CheckFisheyeObjValid(msg_iter->second, valid_label);
      CheckFisheyeVelValid(msg_iter->second);
    }

    // 生成传感器的correlation_obj
    std::vector<SingleFusionCorrelationObject::Ptr> observe_objs;
    observe_objs.reserve(msg_iter->second->object_list.size());
    for (int i = 0; i < msg_iter->second->object_list.size(); ++i) {
      if (!valid_label[i]) {
        continue;
      }
      auto& obj = msg_iter->second->object_list[i];
      bool is_only_attributes_valid = false;
      if (!IsValidVisionCorrelationRangeFilter(sensor_id, obj,
                                               is_only_attributes_valid) &&
          !is_only_attributes_valid) {
        continue;
      }

      MotionCompensation(sensor_id, obj, time_stamp.ToSecond(),
                         lidar_time_stamp_);
      std::shared_ptr<PvProjMap> tmp_pv_proj_box_map_ptr =
          std::make_shared<PvProjMap>();
      bool obj_valid = multi_sensor_fusion_ptr_->MultiSensorPvProj(
          obj, cam_calibs_ptr_, sensor_id, *tmp_pv_proj_box_map_ptr);
      if (!obj_valid) {
        continue;
      }
      auto new_obj = std::make_shared<SingleFusionCorrelationObject>(obj);
      new_obj->pv_proj_box_map = tmp_pv_proj_box_map_ptr;
      new_obj->sensor_id = sensor_id;
      new_obj->time_stamp = time_stamp.ToSecond();
      if (is_only_attributes_valid) {
        new_obj->correlation_state_ = CORRELATION_STATE::ONLY_ATTRIBUTES_VALID;
      }
      copyCorrelationObjectInfo(obj, *new_obj);
      observe_objs.emplace_back(new_obj);
    }

    // 粗匹配
    std::vector<bool> matched_label;
    GreedyMatch(observe_objs, sensor_id, temporal_tracking_list, matched_label);
    HungarianMatch(observe_objs, temporal_tracking_list, matched_label);
    CreateTrackObject(observe_objs, temporal_tracking_list, matched_label);
  }
}
void OneStageFusion::FusionRadarObj(
    const PostFusionMsg::Ptr& msg_ptr,
    TrackingObjectList& temporal_tracking_list) {
  const std::vector<std::string> handle_radar_order = {
      "radar_back_postprocess_object_inner",
      "radar_front_postprocess_object_inner",
  };

  for (const auto& topic : handle_radar_order) {
    const auto& msg_iter = msg_ptr->input_msg_ptr->obj_map.find(topic);
    if (msg_iter == msg_ptr->input_msg_ptr->obj_map.end()) {
      continue;
    }

    auto time_stamp = apollo::cyber::Time(msg_iter->second->header.time);
    auto sensor_id = msg_iter->second->sensor_id;

    std::vector<SingleFusionCorrelationObject::Ptr> observe_objs;
    observe_objs.reserve(msg_iter->second->object_list.size());

    for (int i = 0; i < msg_iter->second->object_list.size(); ++i) {
      auto& obj = msg_iter->second->object_list[i];
      if (!IsValidRadarFilter(sensor_id, obj)) {
        continue;
      }
      obj.box_size.length = 4.8;
      obj.box_size.width = 1.8;
      obj.box_size.height = 1.6;
      MotionCompensation(sensor_id, obj, time_stamp.ToSecond(),
                         lidar_time_stamp_);
      std::shared_ptr<RadarProjMap> radar_pv_proj_points_ptr =
          std::make_shared<RadarProjMap>();
      multi_sensor_fusion_ptr_->MultiSensorRadarPvProj(
          obj, cam_calibs_ptr_, sensor_id, *radar_pv_proj_points_ptr);
      // 也用虚拟框投影一下，后续也可能直接改成无效
      std::shared_ptr<PvProjMap> pv_proj_box_ptr =
          std::make_shared<PvProjMap>();
      multi_sensor_fusion_ptr_->MultiSensorPvProj(obj, cam_calibs_ptr_,
                                                  sensor_id, *pv_proj_box_ptr);

      auto new_obj = std::make_shared<SingleFusionCorrelationObject>(obj);
      new_obj->radar_proj_point_map = radar_pv_proj_points_ptr;
      new_obj->pv_proj_box_map = pv_proj_box_ptr;
      new_obj->sensor_id = sensor_id;
      new_obj->time_stamp = time_stamp.ToSecond();
      copyRadarCorrelationObjectInfo(obj, *new_obj, sensor_id, nullptr);
      observe_objs.emplace_back(new_obj);
    }
    // 粗匹配
    std::vector<bool> matched_label;
    GreedyMatch(observe_objs, sensor_id, temporal_tracking_list, matched_label);
    HungarianMatch(observe_objs, temporal_tracking_list, matched_label);
    int radar_start = temporal_tracking_list.object_list.size();
    CreateTrackObject(observe_objs, temporal_tracking_list, matched_label);
    RadarObjectNMS(temporal_tracking_list, radar_start);
    // radar去重
  }
}
void OneStageFusion::RadarObjectNMS(TrackingObjectList& tracking_list,
                                    const int radar_start_idx) {
  std::vector<bool> delete_label(tracking_list.object_list.size(), false);
  for (int i = tracking_list.object_list.size() - 1; i >= radar_start_idx;
       --i) {
    if (delete_label[i]) {
      continue;
    }
    const auto& radar_obj_a = tracking_list.object_list[i];
    const auto& radar_obj_a_box = radar_obj_a->bev_box_base;
    for (int j = radar_start_idx; j < tracking_list.object_list.size(); ++j) {
      if (delete_label[i]) {
        break;
      }
      if (j == i || delete_label[j]) {
        continue;
      }
      const auto& radar_obj_b = tracking_list.object_list[j];
      const auto& radar_obj_b_box = radar_obj_b->bev_box_base;
      // nms
      auto distance_tmp = std::hypot(
          radar_obj_a->box_center_base.x - radar_obj_b->box_center_base.x,
          radar_obj_a->box_center_base.y - radar_obj_b->box_center_base.y);
      auto distance_x_tmp = std::abs(radar_obj_a->box_center_base.x -
                                     radar_obj_b->box_center_base.x);
      auto distance_y_tmp = std::abs(radar_obj_a->box_center_base.y -
                                     radar_obj_b->box_center_base.y);
      float tmp_iou_bev;
      std::tie(tmp_iou_bev, std::ignore, std::ignore, std::ignore) =
          calcIou(radar_obj_a_box, radar_obj_b_box);
      if (tmp_iou_bev > 0. ||
          (distance_x_tmp < 8 && distance_y_tmp < 2.5 &&
           !((radar_obj_a->velocity_base.x > 0) ^
             (radar_obj_b->velocity_base.x > 0)) &&
           std::abs(std::cos(radar_obj_a->yaw_base - radar_obj_b->yaw_base)) >
               std::cos(M_PI / 6))) {
        if (std::abs(radar_obj_a->box_center_base.x) >
            std::abs(radar_obj_b->box_center_base.x)) {
          delete_label[i] = true;
        } else {
          delete_label[j] = true;
        }
      }
    }
  }
  for (int i = delete_label.size() - 1; i >= 0; --i) {
    if (delete_label[i]) {
      tracking_list.object_list.erase(tracking_list.object_list.begin() + i);
    }
  }
}
}  // namespace perception
}  // namespace robosense