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

#include "ref/reference/tracking/tracking_task.h"

#include "ref/reference/tracking/motion_optimize.h"
#include "ref/reference/utils/utils.h"

namespace ref {
void TrackingTask::Init() { log_file.open("./data/Tracking.txt"); }
void TrackingTask::MultiFrameProcess(
    const std::vector<ObjectFusionArray>& fusion_object,
    std::unordered_map<int, ref::ObjectTrajectory>& trajectory_list) {
  for (auto& frame : fusion_object) {
    pos_map_[frame.frame.frame_head.frame_id] =
        frame.frame.frame_head.reference_to_world.toMat();
    // track single frame
    SingleFrameGenerateId(frame, trajectory_list);
  }
}
void TrackingTask::SingleFrameGenerateId(
    const ObjectFusionArray& frame,
    std::unordered_map<int, ref::ObjectTrajectory>& trajectory_list) {
  // associated
  pos_map_[frame.frame.frame_head.frame_id] =
      frame.frame.frame_head.reference_to_world.toMat();
  time_map_[frame.frame.frame_head.frame_id] =
      frame.frame.frame_head.frame_timestamp;

  log_file << std::fixed << frame.frame.frame_head.frame_timestamp
           << "*******************************" << std::endl;
  std::vector<uint32_t> associated_id;
  track_list_.timestamp = frame.frame.frame_head.frame_timestamp;
  track_list_.frame_idx = frame.frame.frame_head.frame_id;
  max_idx = frame.frame.frame_head.frame_id;
  auto& cur_pos = pos_map_[frame.frame.frame_head.frame_id];
  if (frame.frame.object_list.size() != 0 &&
      track_list_.object_list.size() != 0) {
    std::vector<rally::WeightedBipartiteEdge> associate_score;
    ComputeScore(frame, associate_score);
    auto n_objs =
        frame.frame.object_list.size() + track_list_.object_list.size();
    associated_id =
        rally::HungarianGraphOptimizer().hungarianMinimumWeightPerfectMatching(
            n_objs, associate_score);
  }

  // update
  uint32_t num_history_objs = track_list_.object_list.size();
  uint32_t num_current_objs = frame.frame.object_list.size();
  std::vector<bool> is_tracked(num_current_objs, false);
  std::vector<bool> is_missing(num_history_objs, true);
  for (size_t i = 0; i < num_history_objs; ++i) {
    if (associated_id.empty()) {
      break;
    }

    if (associated_id[i] >= num_current_objs) {
      continue;
    }

    is_tracked[associated_id[i]] = true;
    is_missing[i] = false;

    auto& track_obj = track_list_.object_list[i];
    auto& current_obj = frame.frame.object_list[associated_id[i]];
    track_obj->Update(
        cur_pos, current_obj, frame.observe_source[associated_id[i]],
        frame.obj_points[associated_id[i]], frame.long_bins[associated_id[i]],
        frame.short_bins[associated_id[i]]);
    log_file << "track_id = " << track_obj->obj.track_id
             << ", det_obj = " << current_obj.track_id << std::endl;
  }

  // process the missing object and remove the dead tracking obj
  for (size_t k = num_history_objs; k > 0; --k) {
    if (!is_missing[k - 1]) {
      continue;
    }

    // remove the dead tracking obj
    auto& track_obj = track_list_.object_list[k - 1];
    if (track_obj->Predict() ||
        frame.frame.frame_head.frame_timestamp - track_obj->obj.timestamp >=
            0.5) {  // 0.5s
      track_list_.object_list.erase(track_list_.object_list.begin() + k - 1);
      log_file << "delete " << track_obj->obj.track_id << std::endl;
      continue;
    }
  }

  // process untracked object and create
  for (size_t k = 0; k < num_current_objs; ++k) {
    if (is_tracked[k]) {
      continue;
    }

    auto& current_obj = frame.frame.object_list[k];
    TrackingObject::Ptr tmp_tracking_obj = std::make_shared<TrackingObject>(
        id_counts_++, cur_pos, current_obj, frame.observe_source[k],
        frame.obj_points[k], frame.long_bins[k], frame.short_bins[k]);
    track_list_.object_list.push_back(tmp_tracking_obj);
    log_file << "create " << tmp_tracking_obj->obj.track_id << std::endl;
  }
  // generate trajectory
  track_list_.UpdateTrajectory(trajectory_list);
  for (auto& obj : track_list_.object_list) {
    log_file << " id " << obj->obj.track_id
             << " vel: " << obj->obj_world.velocity[0] << " "
             << obj->obj_world.velocity[1] << std::endl;
  }
}
void TrackingTask::ComputeScore(
    const ObjectFusionArray& frame,
    std::vector<rally::WeightedBipartiteEdge>& associate_score) {
  auto& cur_pos = pos_map_[frame.frame.frame_head.frame_id];
  associate_score.reserve(
      frame.frame.object_list.size() * track_list_.object_list.size() +
      frame.frame.object_list.size() + track_list_.object_list.size());
  for (size_t track_id = 0; track_id < track_list_.object_list.size();
       ++track_id) {
    auto& track_obj = track_list_.object_list[track_id];
    for (size_t obj_id = 0; obj_id < frame.frame.object_list.size(); ++obj_id) {
      auto& target_obj = frame.frame.object_list[obj_id];
      if (is_motionless_type(track_obj->obj.type) ^
          is_motionless_type(target_obj.type)) {
        continue;
      }
      float pro_density = 0;
      double time_interval = target_obj.timestamp - track_obj->obj.timestamp;

      // theta threshold
      if (std::max(track_obj->obj.size_3d.length,
                   track_obj->obj.size_3d.width) > 2.0 &&
          std::cos(CalYawDiff90(track_obj->obj.orientation_3d.yaw,
                                target_obj.orientation_3d.yaw)) < 0.7071) {
        continue;
      }

      // ----------
      Eigen::Vector4d target_position_world;
      target_position_world << target_obj.position_3d.head(3), 1;
      target_position_world = cur_pos * target_position_world;
      Eigen::Vector2f measured_center{target_position_world[0],
                                      target_position_world[1]};
      Eigen::Vector2f predicted_center{
          track_obj->obj_world.position_3d[0] +
              track_obj->obj_world.velocity[0] * time_interval,
          track_obj->obj_world.position_3d[1] +
              track_obj->obj_world.velocity[1] * time_interval};
      Eigen::Vector2f measurement_predict_diff{measured_center -
                                               predicted_center};
      float location_distance{measurement_predict_diff.norm()};
      Eigen::Vector2f track_motion_dir{track_obj->obj_world.velocity[0],
                                       track_obj->obj_world.velocity[1]};
      float track_speed{track_motion_dir.norm()};
      Eigen::Matrix2f s_inverse{Eigen::Matrix2f::Identity()};
      if (is_motionless_type(track_obj->obj.type)) {
        if (fabs(location_distance) > 0.5) {
          continue;
        }
      } else if (track_obj->obj.is_static ||
                 fabs(track_obj->obj_world.velocity[0]) > 1e-10) {
        // have speed
        track_motion_dir.normalize();
        float dir_distance = measurement_predict_diff.dot(track_motion_dir);
        float orhogonal_distance = sqrt(location_distance * location_distance -
                                        dir_distance * dir_distance);
        // 由于点云贴合较好，要尽可能截断无效匹配，给速度放宽一些
        // 1m/s速度差值
        if (fabs(orhogonal_distance) > 0.5 * track_obj->obj.size_3d.width ||
            fabs(dir_distance) > 0.5 * track_obj->obj.size_3d.length) {
          continue;
        }
        Eigen::RowVector2f dir_part;
        Eigen::RowVector2f orthogonal_dir_part;
        Eigen::RowVector2f measurement_predict_diff_mat;
        dir_part << track_motion_dir(0), track_motion_dir(1);
        orthogonal_dir_part << track_motion_dir(1), -track_motion_dir(0);
        measurement_predict_diff_mat << measurement_predict_diff(0),
            measurement_predict_diff(1);

        s_inverse = dir_part.transpose() * dir_part * 0.5F +
                    orthogonal_dir_part.transpose() * orthogonal_dir_part * 2.F;

        auto square_location_distance =
            measurement_predict_diff_mat * s_inverse *
            measurement_predict_diff_mat.transpose();
        location_distance = std::sqrt(square_location_distance(0));
      } else if (IsVehicle(target_obj.type)) {
        // 可能是纯视觉目标无速度
        track_motion_dir << std::cos(track_obj->obj_world.orientation_3d.yaw),
            std::sin(track_obj->obj_world.orientation_3d.yaw);

        float dir_distance = measurement_predict_diff.dot(track_motion_dir);
        float orhogonal_distance = sqrt(location_distance * location_distance -
                                        dir_distance * dir_distance);
        if (fabs(orhogonal_distance) >
                2 * (target_obj.timestamp - track_obj->obj.timestamp) +
                    0.5 * track_obj->obj.size_3d.width ||
            fabs(dir_distance) >
                (10 * 1.2) * (target_obj.timestamp - track_obj->obj.timestamp) +
                    0.5 * track_obj->obj.size_3d.length) {
          continue;
        }
        Eigen::RowVector2f dir_part;
        Eigen::RowVector2f orthogonal_dir_part;
        Eigen::RowVector2f measurement_predict_diff_mat;
        dir_part << track_motion_dir(0), track_motion_dir(1);
        orthogonal_dir_part << track_motion_dir(1), -track_motion_dir(0);
        measurement_predict_diff_mat << measurement_predict_diff(0),
            measurement_predict_diff(1);

        s_inverse = dir_part.transpose() * dir_part * 0.5F +
                    orthogonal_dir_part.transpose() * orthogonal_dir_part * 2.F;

        auto square_location_distance =
            measurement_predict_diff_mat * s_inverse *
            measurement_predict_diff_mat.transpose();
        location_distance = std::sqrt(square_location_distance(0));
      } else if (ObjectTypeNameToType(target_obj.type) !=
                 rally::ObjectType::PEDESTRIAN) {
        // 非行人，基本不能移动
        if (fabs(location_distance) > 0.5) {
          continue;
        }
      } else {
        // 行人，速度有限
        if (fabs(location_distance) >
            2 * (target_obj.timestamp - track_obj->obj.timestamp)) {
          continue;
        }
      }
      float result_distance = location_distance * 0.6F;
      if (result_distance <= distance_thresh) {
        pro_density = static_cast<float>(
            (1. / std::sqrt(2. * rally::R_M_PI *
                            static_cast<double>(s_inverse.determinant()))) *
            std::exp(-0.5 * static_cast<double>(location_distance)));
      }
      if (pro_density > 0) {
        log_file << target_obj.type << " "
                 << "track id " << track_obj->obj.track_id << " "
                 << "obj id " << target_obj.track_id << " score "
                 << 1 - pro_density << std::endl;
        associate_score.emplace_back(
            rally::WeightedBipartiteEdge(track_id, obj_id, 1 - pro_density));
        associate_score.emplace_back(track_list_.object_list.size() + obj_id,
                                     frame.frame.object_list.size() + track_id,
                                     0);
      }
    }
  }
  for (uint32_t left_id = 0; left_id < track_list_.object_list.size();
       ++left_id) {
    associate_score.emplace_back(left_id,
                                 left_id + frame.frame.object_list.size(), 0);
  }

  for (uint32_t right_id = 0; right_id < frame.frame.object_list.size();
       ++right_id) {
    associate_score.emplace_back(right_id + track_list_.object_list.size(),
                                 right_id, 1);
  }
}
void TrackingTask::GetSingleTrackResult(std::vector<ObjectFusion>& obj_vec) {
  for (auto& obj : track_list_.object_list) {
    obj_vec.emplace_back(obj->obj);
  }
}
void TrackingTask::ConnectTrajectory(
    std::unordered_map<int, ref::ObjectTrajectory>& trajectory_list) {
  // 先按起点排序，然后连接终点
  std::vector<ref::ObjectTrajectory> trajectory_vec;
  for (auto& traj : trajectory_list) {
    trajectory_vec.emplace_back(traj.second);
  }
  std::sort(trajectory_vec.begin(), trajectory_vec.end(),
            [](const ref::ObjectTrajectory& traj1,
               const ref::ObjectTrajectory& traj2) {
              return std::get<0>(traj1.position_list[0]) <
                     std::get<0>(traj2.position_list[0]);
            });
  std::vector<bool> valid_label(trajectory_vec.size(), true);
  for (size_t i = 0; i < trajectory_vec.size(); ++i) {
    if (!valid_label[i]) {
      continue;
    }

    for (size_t j = i + 1; j < trajectory_vec.size(); ++j) {
      if (!valid_label[j]) {
        continue;
      }
      // 有效位置的尾部
      auto tail_it = trajectory_vec[i].position_list.end();
      --tail_it;
      while (std::get<1>(*tail_it) != ObjectTrajectory::valid) {
        --tail_it;
      }
      // 有效位置的头部
      auto head_it = trajectory_vec[j].position_list.begin();
      while (std::get<1>(*head_it) != ObjectTrajectory::valid) {
        ++head_it;
      }
      // idx
      if (std::get<4>(*tail_it) >= std::get<4>(*head_it)) {
        continue;
      }
      // type
      if (std::get<2>(*tail_it).obj_world.type !=
          std::get<2>(*head_it).obj_world.type) {
        continue;
      }
      if (is_motionless_type(std::get<2>(*tail_it).obj_world.type)) {
        continue;
      }
      if (IsVehicle(std::get<2>(*tail_it).obj_world.type) &&
          trajectory_vec[i].position_list.size() >= 2 &&
          trajectory_vec[j].position_list.size() >= 2 &&
          std::cos(CalYawDiff180(
              std::get<2>(*head_it).obj_world.orientation_3d.yaw,
              std::get<2>(*tail_it).obj_world.orientation_3d.yaw)) < 0.7) {
        continue;
      }

      // 正向pos
      auto tail_pos = std::get<2>(*tail_it).obj_world.position_3d;
      auto speed = std::get<2>(*tail_it).obj_world.velocity;
      auto predict_pos =
          (std::get<0>(*head_it) - std::get<0>(*tail_it)) * speed + tail_pos;
      auto head_pos = std::get<2>(*head_it).obj_world.position_3d;
      // 从前一段指向后一段
      Eigen::Vector3d pos_diff = head_pos - predict_pos;
      pos_diff[2] = 0;
      Eigen::Vector3d track_motion_dir_1 = Eigen::Vector3d(
          std::cos(std::get<2>(*tail_it).obj_world.orientation_3d.yaw),
          std::sin(std::get<2>(*tail_it).obj_world.orientation_3d.yaw), 0);
      float dir_distance1 = pos_diff.dot(track_motion_dir_1);
      float orhogonal_distance1 = sqrt(pos_diff.norm() * pos_diff.norm() -
                                       dir_distance1 * dir_distance1);
      // 逆向pos
      auto speed2 = std::get<2>(*head_it).obj_world.velocity;
      auto predict_pos2 =
          (std::get<0>(*tail_it) - std::get<0>(*head_it)) * speed2 + head_pos;
      Eigen::Vector3d pos_diff2 = predict_pos2 - tail_pos;
      pos_diff2[2] = 0;
      Eigen::Vector3d track_motion_dir_2 = Eigen::Vector3d(
          std::cos(std::get<2>(*head_it).obj_world.orientation_3d.yaw),
          std::sin(std::get<2>(*head_it).obj_world.orientation_3d.yaw), 0);
      float dir_distance2 = pos_diff2.dot(track_motion_dir_2);
      float orhogonal_distance2 = sqrt(pos_diff2.norm() * pos_diff2.norm() -
                                       dir_distance2 * dir_distance2);

      bool can_be_merged = false;
      float tmp_length = std::get<2>(*head_it).obj_world.size_3d.length;
      float tmp_width = std::get<2>(*head_it).obj_world.size_3d.width;
      // 行人只判断是否在速度误差范围内
      if (!IsVehicle(std::get<2>(*tail_it).obj_world.type)) {
        if (pos_diff2.norm() < std::max(1.0, 0.3 * (std::get<0>(*head_it) -
                                                    std::get<0>(*tail_it))) ||
            pos_diff.norm() < std::max(1.0, 0.3 * (std::get<0>(*head_it) -
                                                   std::get<0>(*tail_it)))) {
          can_be_merged = true;
        }
      } else {
        // dir_distance正表示前一段的预测位于后一段预测的（朝向位置）后面，负表示超前，负可以表示占据
        if ((trajectory_vec[j].position_list.size() >=
                 0.5 * trajectory_vec[i].position_list.size() &&
             dir_distance2 < tmp_length && dir_distance2 > -1.5 * tmp_length &&
             fabs(orhogonal_distance2) < 0.5 * tmp_width) ||
            (trajectory_vec[i].position_list.size() >=
                 0.5 * trajectory_vec[j].position_list.size() &&
             dir_distance1 < tmp_length && dir_distance1 > -1.5 * tmp_length &&
             fabs(orhogonal_distance1) < 0.5 * tmp_width)) {
          can_be_merged = true;
        }
      }
      //   can_be_merged = false;
      // merge
      if (can_be_merged) {
        // std::cout << trajectory_vec[i].track_id << " merge with "
        //           << trajectory_vec[j].track_id << " " << pos_diff.norm()
        //           << " "
        //           << (std::get<0>(*head_it) - std::get<0>(*tail_it))
        //           << std::endl;
        while (tail_it != trajectory_vec[i].position_list.end() &&
               std::get<4>(*tail_it) < std::get<4>(*head_it)) {
          ++tail_it;
        }
        // 删除重叠的无效位置
        trajectory_vec[i].position_list.erase(
            tail_it, trajectory_vec[i].position_list.end());
        // 补全中间时间
        int cur_idx = std::get<4>(trajectory_vec[i].position_list.back());
        ++cur_idx;
        while (cur_idx <= max_idx && cur_idx < std::get<4>(*head_it)) {
          trajectory_vec[i].position_list.emplace_back(
              trajectory_vec[i].position_list.back());
          std::get<0>(trajectory_vec[i].position_list.back()) =
              time_map_[cur_idx];
          std::get<1>(trajectory_vec[i].position_list.back()) =
              ObjectTrajectory::PostionState::invalid;
          std::get<4>(trajectory_vec[i].position_list.back()) = cur_idx;
          ++cur_idx;
        }
        for (; head_it != trajectory_vec[j].position_list.end(); ++head_it) {
          trajectory_vec[i].position_list.emplace_back(*head_it);
          std::get<2>(trajectory_vec[i].position_list.back())
              .obj_world.track_id = trajectory_vec[i].track_id;
          std::get<2>(trajectory_vec[i].position_list.back()).obj.track_id =
              trajectory_vec[i].track_id;
        }
        valid_label[j] = false;
      }
    }
  }
  trajectory_list.clear();
  // 静目标外推 由于存在重框可能未开启
  for (int i = 0; i < trajectory_vec.size(); ++i) {
    if (!valid_label[i]) {
      continue;
    }
    // if (is_motionless_type(
    //         std::get<2>(trajectory_vec[i].position_list[0]).obj.type)) {
    //   int start_idx = std::get<4>(trajectory_vec[i].position_list[0]);
    //   auto template_obj = trajectory_vec[i].position_list[0];
    //   std::cout << std::get<2>(template_obj).obj.type << std::endl;
    //   while (start_idx > 0) {
    //     --start_idx;
    //     trajectory_vec[i].position_list.insert(
    //         trajectory_vec[i].position_list.begin(), template_obj);
    //     std::get<0>(trajectory_vec[i].position_list[0]) =
    //     time_map_[start_idx];
    //     // std::get<2>(trajectory_vec[i].position_list[0]).obj.type =
    //     //     std::get<2>(template_obj).obj.type;
    //     std::get<1>(trajectory_vec[i].position_list[0]) =
    //         ObjectTrajectory::PostionState::invalid;
    //     std::get<4>(trajectory_vec[i].position_list[0]) = start_idx;
    //   }
    //   int end_idx = std::get<4>(trajectory_vec[i].position_list.back());
    //   while (end_idx < max_idx) {
    //     ++end_idx;
    //     trajectory_vec[i].position_list.emplace_back(template_obj);
    //     std::get<0>(trajectory_vec[i].position_list.back()) =
    //         time_map_[end_idx];
    //     std::get<1>(trajectory_vec[i].position_list.back()) =
    //         ObjectTrajectory::PostionState::invalid;
    //     std::get<4>(trajectory_vec[i].position_list.back()) = end_idx;
    //     if
    //     (std::get<2>(trajectory_vec[i].position_list.back()).obj.track_id
    //     ==
    //         98) {
    //       std::cout << end_idx << " "
    //                 <<
    //                 int(std::get<1>(trajectory_vec[i].position_list.back()))
    //                 << std::endl;
    //     }
    //   }
    // }

    auto& traj = trajectory_vec[i];
    trajectory_list.insert({traj.track_id, traj});
  }
}
void TrackingTask::DeleteCoverTrajectory(
    std::unordered_map<int, ref::ObjectTrajectory>& trajectory_list) {
  std::unordered_set<int> delete_id;
  for (auto cur_it = trajectory_list.begin(); cur_it != trajectory_list.end();
       ++cur_it) {
    if (is_motionless_type(
            std::get<2>(cur_it->second.position_list[0]).obj_world.type)) {
      continue;
    }
    if (delete_id.find(cur_it->first) != delete_id.end()) {
      continue;
    }
    auto next_it = cur_it;
    ++next_it;
    while (next_it != trajectory_list.end()) {
      if (delete_id.find(next_it->first) != delete_id.end()) {
        ++next_it;
        continue;
      }
      if (cur_it->second.ChekctTrajCover(next_it->second)) {
        auto cur_score = cur_it->second.GetObserveScores();
        auto next_score = next_it->second.GetObserveScores();
        if (cur_score >= next_score) {
          delete_id.insert(next_it->first);
        } else {
          delete_id.insert(cur_it->first);
          break;
        }
      }
      ++next_it;
    }
  }
  for (auto id : delete_id) {
    trajectory_list.erase(id);
  }
}
void TrackingTask::PostProcess(
    std::unordered_map<int, ref::ObjectTrajectory>& trajectory_list) {
  double optimize_time = 0;
  auto start = rally::getNowInMillSeconds();
  for (auto& traj : trajectory_list) {
    MotionModelOptimize motion_optimizer(traj.first);
    motion_optimizer.InitType(traj.second);
    motion_optimizer.InitYawInfo(traj.second, true, false);
    motion_optimizer.InitGeometryInfo(traj.second);
    // traj.second.RemoveOutsideBox();
    // traj.second.CompletionTrajectory(pos_map_);
    traj.second.BSplineFitOptimize(pos_map_, 1, 3);
    traj.second.UpdateVehObj(pos_map_);
    if (motion_optimizer.CheckStatic(traj.second, false)) {
      motion_optimizer.StaticOptimize(traj.second);
      traj.second.UpdateVehObj(pos_map_);
      continue;
    }
    auto optimizer_start = rally::getNowInMillSeconds();
    motion_optimizer.Process(traj.second, pos_map_);
    auto optimizer_end = rally::getNowInMillSeconds();
    optimize_time += (optimizer_end - optimizer_start);
    // traj.second.UpdateVehObj(pos_map_);
  }
  ConnectTrajectory(trajectory_list);
  for (auto& traj : trajectory_list) {
    MotionModelOptimize motion_optimizer(traj.first);
    traj.second.UpdateVehObj(pos_map_);
    traj.second.BSplineFitOptimize(pos_map_, 1, 3);
    traj.second.CompletionTrajectory(pos_map_);
    motion_optimizer.InitYawInfo(traj.second, true, true);
    if (motion_optimizer.CheckStatic(traj.second, true)) {
      motion_optimizer.StaticOptimize(traj.second);
    }
    motion_optimizer.InitGeometryInfo(traj.second);
    traj.second.UpdateVehObj(pos_map_);
    // traj.second.UpdateVehObj(pos_map_);
  }
  DeleteCoverTrajectory(trajectory_list);
  auto end = rally::getNowInMillSeconds();
  RINFO << "all optimize time " << end - start << " ceres time "
        << optimize_time;
}
}  // namespace ref