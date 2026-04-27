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

#include "ref/reference/utils/reference_type.h"

#include "ref/reference/utils/utils.h"

namespace ref {
TrackingObject::TrackingObject(
    const int id, const Eigen::Matrix4d& reference_to_world,
    const clip::Object& objserve_obj,
    const std::vector<clip::SensorType>& observe_source,
    const pcl::PointCloud<RsPoint>::Ptr cloud,
    const std::vector<robosense::perception::BoxBin>& long_bins,
    const std::vector<robosense::perception::BoxBin>& short_bins) {
  obj = objserve_obj;
  obj.track_id = id;
  obj_world = obj;
  obj_world.transform(reference_to_world);
  live_score = 0;
  observe_score = 0;
  for (auto sensor : observe_source) {
    observe_score += clip::kLidarSet.find(sensor) == clip::kLidarSet.end()
                         ? kVISION_SCORE
                         : kLIDAR_SCORE * 5;
    live_score += clip::kLidarSet.find(sensor) == clip::kLidarSet.end()
                      ? kVISION_SCORE
                      : kLIDAR_SCORE;
  }
  int bins_score = 0;
  for (auto& bin : long_bins) {
    if (bin.valid) bins_score += 5;
  }
  for (auto& bin : short_bins) {
    if (bin.valid) bins_score += 5;
  }
  if (bins_score == 0) {
    observe_score = 1;
  } else {
    observe_score += bins_score;
  }
  live_score = std::min(live_score, 100);
  live_score = std::max(live_score, 0);
  if (cloud != nullptr) {
    world_cloud.reset(new pcl::PointCloud<RsPoint>);
    veh_cloud = cloud;
    pcl::transformPointCloud(*veh_cloud, *world_cloud, reference_to_world);
  } else {
    veh_cloud = nullptr;
    world_cloud = nullptr;
  }
}
void TrackingObject::Update(
    const Eigen::Matrix4d& reference_to_world, const clip::Object& objserve_obj,
    const std::vector<clip::SensorType>& observe_source,
    const pcl::PointCloud<RsPoint>::Ptr cloud,
    const std::vector<robosense::perception::BoxBin>& long_bins,
    const std::vector<robosense::perception::BoxBin>& short_bins) {
  int id = obj.track_id;
  obj = objserve_obj;
  obj.track_id = id;
  obj_world = obj;
  obj_world.transform(reference_to_world);
  observe_score = 0;
  for (auto sensor : observe_source) {
    observe_score += clip::kLidarSet.find(sensor) == clip::kLidarSet.end()
                         ? kVISION_SCORE
                         : kLIDAR_SCORE;
    live_score += clip::kLidarSet.find(sensor) == clip::kLidarSet.end()
                      ? kVISION_SCORE
                      : kLIDAR_SCORE;
    ;
  }
  int bins_score = 0;
  for (auto& bin : long_bins) {
    if (bin.valid) bins_score += 10;
  }
  for (auto& bin : short_bins) {
    if (bin.valid) bins_score += 5;
  }
  if (bins_score == 0) {
    observe_score = 1;
  } else {
    observe_score += bins_score;
  }
  live_score = std::min(live_score, 100);
  live_score = std::max(live_score, 0);
  if (cloud != nullptr) {
    world_cloud.reset(new pcl::PointCloud<RsPoint>);
    veh_cloud = cloud;
    pcl::transformPointCloud(*veh_cloud, *world_cloud, reference_to_world);
  } else {
    veh_cloud = nullptr;
    world_cloud = nullptr;
  }
}
void TrackingObject::UpdateVehPos(const double& timestamp,
                                  const Eigen::Vector3d pos_veh,
                                  const Eigen::Matrix4d& reference_to_world) {
  obj.timestamp = timestamp;
  obj_world.timestamp = timestamp;
  obj.position_3d = pos_veh;
  Eigen::Vector4d tt_pt;
  tt_pt << pos_veh.x(), pos_veh.y(), pos_veh.z(), 1.;
  tt_pt = reference_to_world * tt_pt;
  obj_world.position_3d << tt_pt[0], tt_pt[1], tt_pt[2];
}
void TrackingObject::UpdateWorldPos(const double& timestamp,
                                    const Eigen::Vector3d pos_world,
                                    const Eigen::Matrix4d& reference_to_world) {
  obj.timestamp = timestamp;
  obj_world.timestamp = timestamp;
  obj_world.position_3d = pos_world;
  Eigen::Vector4d tt_pt;
  tt_pt << pos_world.x(), pos_world.y(), pos_world.z(), 1.;
  tt_pt = reference_to_world.inverse() * tt_pt;
  obj.position_3d << tt_pt[0], tt_pt[1], tt_pt[2];
}
void ObjectTrajectory::Draw(const std::string& save_path) {
  if (position_list.size() < 3) return;
  Drawer trajectory_draw;
  //   std::cout << track_id << " " << position_list.size() << std::endl;
  for (size_t i = 0; i < position_list.size(); ++i) {
    auto& obj = position_list[i];
    if (std::get<1>(obj) == PostionState::invalid)
      continue;
    else if (std::get<1>(obj) == PostionState::predict) {
      trajectory_draw.DrawBev(std::get<2>(obj).obj, cv::Scalar(0, 0, 255));
    } else {
      trajectory_draw.DrawBev(std::get<2>(obj).obj, cv::Scalar(255, 0, 0));
    }
  }
  trajectory_draw.SaveBev(save_path + std::to_string(track_id) + ".jpg");
}
RefPointPos computeRefPointStatus(const TrackingObject& obj) {
  if (std::abs(obj.obj.position_3d[0]) < 10.0 || !IsVehicle(obj.obj.type)) {
    return RefPointPos::REF_POINT_CENTER;
  }

  // refine compute
  Eigen::Vector2f pos_vec(obj.obj.position_3d[0], obj.obj.position_3d[1]);
  float x_offset =
      0.5 * obj.obj.size_3d.length * std::cos(obj.obj.orientation_3d.yaw);
  if ((pos_vec[0] * std::cos(obj.obj.orientation_3d.yaw) +
       pos_vec[1] * std::sin(obj.obj.orientation_3d.yaw)) > 0.0) {
    x_offset = -x_offset;
  }

  float ref_x = obj.obj.position_3d[0] + x_offset;
  if (ref_x > 5.0 || ref_x < -2.0) {
    if ((pos_vec[0] * std::cos(obj.obj.orientation_3d.yaw) +
         pos_vec[1] * std::sin(obj.obj.orientation_3d.yaw)) > 0.0) {
      return RefPointPos::REF_POINT_BACK;
    } else {
      return RefPointPos::REF_POINT_FRONT;
    }

  } else {
    return RefPointPos::REF_POINT_CENTER;
  }
}

// for world
void ObjectTrajectory::Draw(const Eigen::Vector3d& ori_pos,
                            const std::string& save_path) {
  if (position_list.size() < 3) return;
  Drawer trajectory_draw;
  //   std::cout << track_id << " " << position_list.size() << std::endl;
  for (size_t i = 0; i < position_list.size(); ++i) {
    auto& obj = position_list[i];
    auto obj_world_norm = std::get<2>(obj).obj_world;
    obj_world_norm.position_3d = obj_world_norm.position_3d - ori_pos;
    if (std::get<1>(obj) == PostionState::invalid)
      continue;
    else if (std::get<1>(obj) == PostionState::predict) {
      trajectory_draw.DrawBev(obj_world_norm, cv::Scalar(0, 0, 255));
    } else {
      trajectory_draw.DrawBev(obj_world_norm, cv::Scalar(255, 0, 0));
    }
    // points
    trajectory_draw.DrawBev(ori_pos, std::get<2>(obj).world_cloud);
    // line
    if (i != 0) {
      auto pre_obj_world_norm = std::get<2>(position_list[i - 1]).obj_world;
      pre_obj_world_norm.position_3d = pre_obj_world_norm.position_3d - ori_pos;
      trajectory_draw.DrawBev(
          obj_world_norm.position_3d[0], obj_world_norm.position_3d[1],
          pre_obj_world_norm.position_3d[0], pre_obj_world_norm.position_3d[1],
          cv::Scalar(0, 255, 0));
    }
  }
  //   std::vector<double> time(position_list.size()),
  //   x_pos(position_list.size()),
  //       y_pos(position_list.size());
  //   for (int i = 0; i < position_list.size(); ++i) {
  //     if(std::get<1>(position_list[i])==PostionState::invalid)
  //         continue;
  //     time[i] = std::get<0>(position_list[i]);
  //     x_pos[i] = std::get<2>(position_list[i]).obj_world.position_3d[0];
  //     y_pos[i] = std::get<2>(position_list[i]).obj_world.position_3d[1];
  //   }
  //   std::vector<double> out_pos_x, out_pos_y;
  //   BSplineFIt(time, x_pos, out_pos_x, 0.5, 3);
  //   BSplineFIt(time, y_pos, out_pos_y, 0.5, 3);
  //   for (int i = 1; i < out_pos_x.size(); ++i) {
  //     trajectory_draw.DrawBev(
  //         out_pos_x[i] - ori_pos[0], out_pos_y[i] - ori_pos[1],
  //         out_pos_x[i - 1] - ori_pos[0], out_pos_y[i - 1] - ori_pos[1],
  //         cv::Scalar(0, 255, 255));
  //   }
  trajectory_draw.SaveBev(save_path + std::to_string(track_id) + ".jpg");
}

void ObjectTrajectory::DrawMiddle(const Eigen::Vector3d& ori_pos,
                                  const std::string& save_root_dir,
                                  const int id) {
  if (position_list.size() < 3) return;

  std::vector<double> time(position_list.size()), x_pos(position_list.size()),
      y_pos(position_list.size());
  for (size_t i = 0; i < position_list.size(); ++i) {
    // auto ref_point_pos =
    // computeRefPointStatus(std::get<2>(position_list[i]));
    GetRefPointFromCenter(
        std::get<3>(position_list[i]),
        std::get<2>(position_list[i]).obj_world.orientation_3d.yaw,
        std::get<2>(position_list[i]).obj_world.size_3d.length,
        std::get<2>(position_list[i]).obj_world.position_3d[0],
        std::get<2>(position_list[i]).obj_world.position_3d[1], x_pos[i],
        y_pos[i]);
    time[i] = std::get<0>(position_list[i]);
  }
  std::vector<double> ori_ref_x = x_pos;
  std::vector<double> ori_ref_y = y_pos;
  BSplineFIt(time, ori_ref_x, x_pos, 0.5, 3);
  BSplineFIt(time, ori_ref_y, y_pos, 0.5, 3);
  // std::cout << track_id << " " << position_list.size() << std::endl;
  for (size_t i = 0; i < position_list.size(); ++i) {
    // line
    if (id != -1 && id != static_cast<int>(i)) continue;
    if (std::get<1>(position_list[i]) == PostionState::invalid) continue;
    Drawer trajectory_draw;
    for (size_t j = 1; j <= i; ++j) {
      auto& obj = position_list[j];
      auto obj_world_norm = std::get<2>(obj).obj_world;
      obj_world_norm.position_3d = obj_world_norm.position_3d - ori_pos;
      float x_ave = 0, y_ave = 0;
      GetCenter(std::get<2>(obj).world_cloud, x_ave, y_ave);
      // points
      trajectory_draw.DrawBev(ori_pos, std::get<2>(obj).world_cloud);
      // line
      if (i != 0) {
        float x_ave2 = 0, y_ave2 = 0;
        GetCenter(std::get<2>(position_list[j - 1]).world_cloud, x_ave2,
                  y_ave2);
        auto pre_obj_world_norm = std::get<2>(position_list[j - 1]).obj_world;
        pre_obj_world_norm.position_3d =
            pre_obj_world_norm.position_3d - ori_pos;
        // box_center
        // trajectory_draw.DrawBev(
        //     obj_world_norm.position_3d[0], obj_world_norm.position_3d[1],
        //     pre_obj_world_norm.position_3d[0],
        //     pre_obj_world_norm.position_3d[1], cv::Scalar(0, 255, 0));
        // cloud center
        // if (!(x_ave2 == 0 && y_ave2 == 0) && !(x_ave == 0 && y_ave == 0)) {
        //   trajectory_draw.DrawBev(x_ave - ori_pos[0], y_ave - ori_pos[1],
        //                           x_ave2 - ori_pos[0], y_ave2 - ori_pos[1],
        //                           cv::Scalar(255, 0, 255));
        // }
        trajectory_draw.DrawBev(
            ori_ref_x[j] - ori_pos[0], ori_ref_y[j] - ori_pos[1],
            ori_ref_x[j - 1] - ori_pos[0], ori_ref_y[j - 1] - ori_pos[1],
            cv::Scalar(0, 255, 0));
        // trajectory_draw.DrawBev(x_pos[j] - ori_pos[0], y_pos[j] - ori_pos[1],
        //                         x_pos[j - 1] - ori_pos[0],
        //                         y_pos[j - 1] - ori_pos[1],
        //                         cv::Scalar(0, 255, 255));
      }
    }
    auto& obj = position_list[i];
    auto obj_world_norm = std::get<2>(obj).obj_world;
    obj_world_norm.position_3d = obj_world_norm.position_3d - ori_pos;
    if (std::get<1>(obj) == PostionState::predict) {
      trajectory_draw.DrawBev(obj_world_norm, cv::Scalar(0, 0, 255));
    } else {
      trajectory_draw.DrawBev(obj_world_norm, cv::Scalar(255, 0, 0));
    }
    rally::ensureDirectory(save_root_dir + '/' + std::to_string(track_id));
    trajectory_draw.SaveBev(save_root_dir + '/' + std::to_string(track_id) +
                            '/' + std::to_string(i) + " " +
                            std::to_string(std::get<0>(obj)) + ".jpg");
  }
}

void ObjectTrajectory::SaveTxt(const std::string& file_name) {
  cv::FileStorage out_file;
  out_file.open(file_name, cv::FileStorage::WRITE);
  out_file << "result"
           << "[";
  for (auto& obj : position_list) {
    if (std::get<1>(obj) == invalid) continue;
    out_file << "{"
             << "time_stamp" << std::get<0>(obj);
    auto& obj_veh = std::get<2>(obj).obj;
    auto& obj_world = std::get<2>(obj).obj_world;
    out_file << "obj_world"
             << "{"
             << "position"
             << "{"
             << "x" << obj_world.position_3d[0] << "y"
             << obj_world.position_3d[1] << "}";
    out_file << "vel"
             << "{"
             << "x" << obj_world.velocity[0] << "y" << obj_world.velocity[1]
             << "}";
    out_file << "acc"
             << "{"
             << "x" << obj_world.acceleration[0] << "y"
             << obj_world.acceleration[1] << "}";
    out_file << "yaw" << obj_world.orientation_3d.yaw;
    out_file << "}";

    out_file << "obj_vel"
             << "{"
             << "position"
             << "{"
             << "x" << obj_veh.position_3d[0] << "y" << obj_veh.position_3d[1]
             << "}";
    out_file << "vel"
             << "{"
             << "x" << obj_veh.velocity[0] << "y" << obj_veh.velocity[1] << "}";
    out_file << "acc"
             << "{"
             << "x" << obj_veh.acceleration[0] << "y" << obj_veh.acceleration[1]
             << "}";
    out_file << "yaw" << obj_veh.orientation_3d.yaw;
    out_file << "}";
    out_file << "}";
  }
  out_file << "]";
}
bool ObjectTrajectory::ChekctTrajCover(const ObjectTrajectory& traj) {
  int cur_idx = 0;
  int i = 0;
  while (cur_idx < position_list.size()) {
    if (std::get<1>(position_list[cur_idx]) == PostionState::invalid) {
      ++cur_idx;
      continue;
    }
    while (i < traj.position_list.size() &&
           std::get<4>(traj.position_list[i]) <
               std::get<4>(position_list[cur_idx])) {
      ++i;
    }
    if (i >= traj.position_list.size()) {
      break;
    }
    if (std::get<4>(traj.position_list[i]) ==
            std::get<4>(position_list[cur_idx]) &&
        std::get<1>(traj.position_list[i]) != PostionState::invalid) {
      auto cur_obj = std::get<2>(position_list[cur_idx]).obj_world;
      auto other_obj = std::get<2>(traj.position_list[i]).obj_world;
      rally::Box2d cur_box2d(
          rally::ShortArray2f(cur_obj.position_3d[0], cur_obj.position_3d[1]),
          cur_obj.orientation_3d.yaw, cur_obj.size_3d.length,
          cur_obj.size_3d.width);
      rally::Box2d other_box2d(rally::ShortArray2f(other_obj.position_3d[0],
                                                   other_obj.position_3d[1]),
                               other_obj.orientation_3d.yaw,
                               other_obj.size_3d.length,
                               other_obj.size_3d.width);
      auto area1 = cur_box2d.area();
      auto area2 = other_box2d.area();
      auto inter_area = cur_box2d.getOverlapArea(other_box2d);
      if (inter_area / area1 > 0.1 || inter_area / area2 > 0.1) {
        return true;
      }
    }
    ++cur_idx;
  }
  return false;
}
void ObjectTrajectory::RemoveOutsideBox() {
  if (position_list.size() < 3) return;
  for (size_t i = 1; i < position_list.size() - 1; ++i) {
    // 三点自车校验，五点世界校验
    if (std::get<1>(position_list[i]) == PostionState::invalid) continue;
    int left = i - 1;
    while (left >= 0 &&
           std::get<1>(position_list[left]) == PostionState::invalid) {
      --left;
    }
    int right = i + 1;
    while (right < static_cast<int>(position_list.size()) &&
           std::get<1>(position_list[right]) == PostionState::invalid) {
      ++right;
    }
    if (left < 0 || right >= static_cast<int>(position_list.size())) {
      continue;
    }
    double left_time_diff =
        std::get<0>(position_list[i]) - std::get<0>(position_list[left]);
    double right_time_diff =
        std::get<0>(position_list[right]) - std::get<0>(position_list[i]);
    double left_ratio = left_time_diff / (left_time_diff + right_time_diff);
    Eigen::Vector3d ave_pos_veh =
        left_ratio * std::get<2>(position_list[left]).obj.position_3d +
        (1 - left_ratio) * std::get<2>(position_list[right]).obj.position_3d;
    Eigen::Vector3d diff =
        ave_pos_veh - std::get<2>(position_list[i]).obj.position_3d;
    if (diff[0] > 0.5 * std::get<2>(position_list[i]).obj.size_3d.length ||
        diff[1] > 0.3 * std::get<2>(position_list[i]).obj.size_3d.width) {
      std::get<1>(position_list[i]) = PostionState::predict;
      std::get<2>(position_list[i]).observe_score = 0;
    }
  }
}
void ObjectTrajectory::CompletionTrajectory(
    const std::unordered_map<int, Eigen::Matrix4d>& self_pos) {
  if (position_list.size() < 3) return;
  if (is_motionless_type(std::get<2>(position_list[0]).obj.type)) {
    int ref_idx = 0;
    while (std::get<1>(position_list[ref_idx]) != valid) {
      ++ref_idx;
    }
    for (size_t i = 0; i < position_list.size(); ++i) {
      if (std::get<1>(position_list[i]) == valid) continue;
      std::get<2>(position_list[i]).obj_world.timestamp =
          std::get<0>(position_list[i]);
      std::get<2>(position_list[i]).obj_world.velocity =
          std::get<2>(position_list[ref_idx]).obj_world.velocity;
      std::get<2>(position_list[i]).obj_world.acceleration =
          std::get<2>(position_list[ref_idx]).obj_world.acceleration;
      std::get<2>(position_list[i]).obj_world.size_3d =
          std::get<2>(position_list[ref_idx]).obj_world.size_3d;
      std::get<2>(position_list[i]).obj_world.position_3d =
          std::get<2>(position_list[ref_idx]).obj_world.position_3d;
      std::get<2>(position_list[i]).obj_world.orientation_3d.yaw =
          std::get<2>(position_list[ref_idx]).obj_world.orientation_3d.yaw;
      std::get<1>(position_list[i]) = ObjectTrajectory::PostionState::predict;
    }
    // for (auto& obj : position_list) {
    //   std::cout << int(std::get<1>(obj)) << std::endl;
    //   if (std::get<1>(obj) == invalid) continue;
    //   std::cout << std::get<2>(obj).obj.track_id << " " << std::get<4>(obj)
    //             << " " << int(std::get<1>(obj)) << std::endl;
    // }
    return;
  }
  for (size_t i = 1; i < position_list.size() - 1; ++i) {
    if (std::get<1>(position_list[i]) == valid) continue;
    int left = i - 1;
    std::vector<cv::Point> near_points;
    // left 1
    while (left >= 0 &&
           std::get<1>(position_list[left]) != PostionState::valid) {
      --left;
    }
    if (left < 0) {
      std::get<1>(position_list[i]) = PostionState::invalid;
      continue;
    }
    // right 1
    int right = i + 1;
    while (right < static_cast<int>(position_list.size()) &&
           std::get<1>(position_list[right]) != PostionState::valid) {
      ++right;
    }
    if (right >= static_cast<int>(position_list.size())) {
      std::get<1>(position_list[i]) = PostionState::invalid;
      continue;
    }

    if (left >= 0 && right < static_cast<int>(position_list.size())) {
      double left_time_diff =
          std::get<0>(position_list[i]) - std::get<0>(position_list[left]);
      double right_time_diff =
          std::get<0>(position_list[right]) - std::get<0>(position_list[i]);
      double left_ratio = right_time_diff / (left_time_diff + right_time_diff);
      Eigen::Vector3d ave_pos_wolrd =
          left_ratio * std::get<2>(position_list[left]).obj_world.position_3d +
          (1 - left_ratio) *
              std::get<2>(position_list[right]).obj_world.position_3d;

      std::get<2>(position_list[i]).obj_world.timestamp =
          std::get<0>(position_list[i]);
      std::get<2>(position_list[i]).obj_world.velocity =
          std::get<2>(position_list[left]).obj_world.velocity;
      std::get<2>(position_list[i]).obj_world.acceleration =
          std::get<2>(position_list[left]).obj_world.acceleration;
      std::get<2>(position_list[i]).obj_world.size_3d =
          std::get<2>(position_list[left]).obj_world.size_3d;
      std::get<2>(position_list[i]).obj_world.position_3d = ave_pos_wolrd;
      std::get<2>(position_list[i]).obj_world.orientation_3d.yaw =
          std::get<2>(position_list[left]).obj_world.orientation_3d.yaw;
      std::get<1>(position_list[i]) = ObjectTrajectory::PostionState::predict;
    } else {
      std::get<1>(position_list[i]) = PostionState::invalid;
    }
  }
}

void ObjectTrajectory::BSplineFitOptimize(
    const std::unordered_map<int, Eigen::Matrix4d>& self_pos,
    const double delta_step, const int order) {
  std::vector<double> input_time, input_x, input_y, input_yaw, input_idx;
  std::vector<int> angle_dir;  // 表示角度在-pai,0 || 0 pai
  double ave_error;
  // fit and remove noise
  for (size_t i = 0; i < position_list.size(); ++i) {
    if (std::get<1>(position_list[i]) == PostionState::invalid) continue;
    double ref_x, ref_y;
    GetRefPointFromCenter(
        ref_point_pos_,
        std::get<2>(position_list[i]).obj_world.orientation_3d.yaw,
        std::get<2>(position_list[i]).obj_world.size_3d.length,
        std::get<2>(position_list[i]).obj_world.position_3d[0],
        std::get<2>(position_list[i]).obj_world.position_3d[1], ref_x, ref_y);
    input_time.emplace_back(std::get<0>(position_list[i]));
    input_x.emplace_back(ref_x);
    input_y.emplace_back(ref_y);
    input_idx.emplace_back(i);
    double yaw_temp =
        std::get<2>(position_list[i]).obj_world.orientation_3d.yaw;
    if (yaw_temp < 0) {
      angle_dir.emplace_back(-1);
    } else {
      angle_dir.emplace_back(1);
    }
    input_yaw.emplace_back(std::cos(yaw_temp));
  }
  std::vector<double> out_x, out_y, out_yaw;
  auto x_fit_result = BSplineFIt(input_time, input_x, out_x, delta_step, order);
  auto y_fit_result = BSplineFIt(input_time, input_y, out_y, delta_step, order);
  auto yaw_fit_result =
      BSplineFIt(input_time, input_yaw, out_yaw, delta_step, order);

  while (x_fit_result.max_error > 0.5 || y_fit_result.max_error > 0.5) {
    int delete_idx = x_fit_result.max_error >= y_fit_result.max_error
                         ? x_fit_result.max_error_idx
                         : y_fit_result.max_error_idx;
    std::get<1>(position_list[input_idx[delete_idx]]) = PostionState::invalid;
    input_time.erase(input_time.begin() + delete_idx);
    input_x.erase(input_x.begin() + delete_idx);
    input_y.erase(input_y.begin() + delete_idx);
    input_idx.erase(input_idx.begin() + delete_idx);
    input_yaw.erase(input_yaw.begin() + delete_idx);
    angle_dir.erase(angle_dir.begin() + delete_idx);
    x_fit_result = BSplineFIt(input_time, input_x, out_x, delta_step, order);
    y_fit_result = BSplineFIt(input_time, input_y, out_y, delta_step, order);
    yaw_fit_result =
        BSplineFIt(input_time, input_yaw, out_yaw, delta_step, order);
  }
  if (input_time.size() == 0) return;
  // update and complete
  int cur_idx = 0;
  for (size_t i = 0; i < position_list.size(); ++i) {
    while (input_idx[cur_idx] < i && cur_idx < input_idx.size()) {
      ++cur_idx;
    }
    auto& cur_pos = self_pos.find(std::get<4>(position_list[i]))->second;
    if (cur_idx < input_idx.size() && input_idx[cur_idx] == i) {
      Eigen::Vector3d new_pos =
          std::get<2>(position_list[i]).obj_world.position_3d;
      GetCenterFromRefPoint(
          ref_point_pos_,
          std::get<2>(position_list[i]).obj_world.orientation_3d.yaw,
          std::get<2>(position_list[i]).obj_world.size_3d.length,
          out_x[cur_idx], out_y[cur_idx], new_pos[0], new_pos[1]);
      std::get<2>(position_list[i])
          .UpdateWorldPos(std::get<0>(position_list[i]), new_pos, cur_pos);
      out_yaw[cur_idx] = std::min(1.0, std::max(-1.0, out_yaw[cur_idx]));
      std::get<2>(position_list[i]).obj_world.orientation_3d.yaw =
          angle_dir[cur_idx] * std::acos(out_yaw[cur_idx]);
    } else {
      auto ref_point_pos_ = std::get<3>(position_list[i]);
      double inter_ref_x = x_fit_result.GetInter(std::get<0>(position_list[i]));
      double inter_ref_y = y_fit_result.GetInter(std::get<0>(position_list[i]));
      double inter_center_x, inter_center_y;
      GetCenterFromRefPoint(
          ref_point_pos_,
          std::get<2>(position_list[i]).obj_world.orientation_3d.yaw,
          std::get<2>(position_list[i]).obj_world.size_3d.length, inter_ref_x,
          inter_ref_y, inter_center_x, inter_center_y);
      Eigen::Vector3d new_pos(
          inter_center_x, inter_center_y,
          std::get<2>(position_list[i]).obj_world.position_3d[2]);
      std::get<2>(position_list[i])
          .UpdateWorldPos(std::get<0>(position_list[i]), new_pos, cur_pos);
      if (std::get<1>(position_list[i]) != PostionState::invalid &&
          inter_ref_x != 0 && inter_ref_y != 0)
        std::get<1>(position_list[i]) = PostionState::predict;
      double inter_yaw_cos =
          yaw_fit_result.GetInter(std::get<0>(position_list[i]));
      inter_yaw_cos = std::min(1.0, std::max(-1.0, inter_yaw_cos));
      if (std::get<2>(position_list[i]).obj_world.orientation_3d.yaw < 0)
        std::get<2>(position_list[i]).obj_world.orientation_3d.yaw =
            -std::acos(inter_yaw_cos);
      else {
        std::get<2>(position_list[i]).obj_world.orientation_3d.yaw =
            std::acos(inter_yaw_cos);
      }
    }

    // // vel
    // std::get<2>(position_list[i]).obj_world.velocity[0] =
    //     x_fit_result.GetVel(std::get<0>(position_list[i]));
    // std::get<2>(position_list[i]).obj_world.velocity[1] =
    //     y_fit_result.GetVel(std::get<0>(position_list[i]));
    // std::get<2>(position_list[i]).obj_world.yawrate =
    //     yaw_fit_result.GetVel(std::get<0>(position_list[i]));
    // // acc
    // std::get<2>(position_list[i]).obj_world.acceleration[0] =
    //     x_fit_result.GetAcc(std::get<0>(position_list[i]));
    // std::get<2>(position_list[i]).obj_world.acceleration[1] =
    //     y_fit_result.GetAcc(std::get<0>(position_list[i]));
  }
}
void ObjectTrajectory::BSplineFitOptimizeVel(
    const std::unordered_map<int, Eigen::Matrix4d>& self_pos,
    const double delta_step, const int order) {
  if (position_list.size() < 3) return;
  std::vector<double> input_time, input_vel_x, input_vel_y, input_idx;
  double ave_error;
  // fit and remove noise
  for (size_t i = 0; i < position_list.size(); ++i) {
    if (std::get<1>(position_list[i]) == PostionState::invalid) continue;
    input_vel_x.emplace_back(
        std::get<2>(position_list[i]).obj_world.velocity[0]);
    input_vel_y.emplace_back(
        std::get<2>(position_list[i]).obj_world.velocity[1]);
    input_idx.emplace_back(i);
  }
  std::vector<double> out_x, out_y;
  auto x_fit_result =
      BSplineFIt(input_time, input_vel_x, out_x, delta_step, order);
  auto y_fit_result =
      BSplineFIt(input_time, input_vel_y, out_y, delta_step, order);
  if (input_time.size() < 3) return;
  // update and complete
  int cur_idx = 0;
  for (size_t i = 0; i < position_list.size(); ++i) {
    while (cur_idx < input_idx.size() && input_idx[cur_idx] < i) {
      ++cur_idx;
    }
    auto& cur_pos = self_pos.find(std::get<4>(position_list[i]))->second;
    if (cur_idx < input_idx.size() && input_idx[cur_idx] == i) {
      std::get<2>(position_list[i]).obj_world.velocity[0] = out_x[cur_idx];
      std::get<2>(position_list[i]).obj_world.velocity[1] = out_y[cur_idx];
    } else {
      std::get<2>(position_list[i]).obj_world.velocity[0] =
          x_fit_result.GetInter(std::get<0>(position_list[i]));
      std::get<2>(position_list[i]).obj_world.velocity[1] =
          y_fit_result.GetInter(std::get<0>(position_list[i]));
    }
    std::get<2>(position_list[i]).obj_world.acceleration[0] =
        x_fit_result.GetVel(std::get<0>(position_list[i]));
    std::get<2>(position_list[i]).obj_world.acceleration[1] =
        y_fit_result.GetVel(std::get<0>(position_list[i]));
  }
}
void ObjectTrajectory::SetRefPointPos() {
  int center_num = 0, back_num = 0, front_num = 0;
  for (auto& obj : position_list) {
    switch (std::get<3>(obj)) {
      case RefPointPos::REF_POINT_CENTER:
        ++center_num;
        break;
      case RefPointPos::REF_POINT_BACK:
        ++back_num;
        break;
      case RefPointPos::REF_POINT_FRONT:
        ++front_num;
        break;
      default:
        break;
    }
  }
  if (back_num > center_num + front_num) {
    ref_point_pos_ = RefPointPos::REF_POINT_BACK;
  } else if (front_num > center_num + back_num) {
    ref_point_pos_ = RefPointPos::REF_POINT_FRONT;
  } else {
    ref_point_pos_ = RefPointPos::REF_POINT_CENTER;
  }
}
void ObjectTrajectory::UpdateVehObj(
    const std::unordered_map<int, Eigen::Matrix4d>& self_pos) {
  for (auto& obj : position_list) {
    auto& reference_to_world = self_pos.find(std::get<4>(obj))->second;
    auto& obj_world = std::get<2>(obj).obj_world;
    auto& obj_veh = std::get<2>(obj).obj;
    obj_veh = obj_world;
    obj_veh.transform(reference_to_world.inverse());
    obj_veh.orientation_3d.yaw = NormalizeAngleStd(obj_veh.orientation_3d.yaw);
  }
}
void ObjectTrajectory::UpdateWorldObj(
    const std::unordered_map<int, Eigen::Matrix4d>& self_pos) {
  for (auto& obj : position_list) {
    auto& reference_to_world = self_pos.find(std::get<4>(obj))->second;
    auto& obj_world = std::get<2>(obj).obj_world;
    auto& obj_veh = std::get<2>(obj).obj;
    obj_world = obj_veh;
    obj_world.transform(reference_to_world);
  }
}

void TrackingObjectList::UpdateTrajectory(
    std::unordered_map<int, ObjectTrajectory>& trajectory_list) {
  for (auto& obj : object_list) {
    int id = obj->obj.track_id;
    auto ref_point_state = computeRefPointStatus(*obj.get());
    if (trajectory_list.count(id) == 0) {
      trajectory_list[id].track_id = id;
      trajectory_list[id].position_list.emplace_back(
          std::tuple<double, ObjectTrajectory::PostionState, TrackingObject,
                     RefPointPos, int>{timestamp,
                                       ObjectTrajectory::PostionState::valid,
                                       *obj.get(), ref_point_state, frame_idx});
    } else {
      if ((timestamp - obj->obj.timestamp) > 0.01) {
        // 未跟踪到，后续需要填充,不使用拷贝会导致update
        trajectory_list[id].position_list.emplace_back(
            std::tuple<double, ObjectTrajectory::PostionState, TrackingObject,
                       RefPointPos, int>{
                timestamp, ObjectTrajectory::PostionState::invalid, *obj.get(),
                ref_point_state, frame_idx});
        std::get<2>(trajectory_list[id].position_list.back()).observe_score = 0;
      } else {
        trajectory_list[id].position_list.emplace_back(
            std::tuple<double, ObjectTrajectory::PostionState, TrackingObject,
                       RefPointPos, int>{
                timestamp, ObjectTrajectory::PostionState::valid, *obj.get(),
                ref_point_state, frame_idx});
      }
    }
  }
}

}  // namespace ref
