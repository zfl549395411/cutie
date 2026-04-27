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

#include "ref/reference/tracking/motion_optimize.h"

#include "ref/reference/utils/utils.h"

namespace ref {

void MotionModelOptimize::Process(
    ObjectTrajectory& trajectory,
    const std::unordered_map<int, Eigen::Matrix4d>& pos_map) {
  InputTrajectory(trajectory);
  std::vector<SaveModel> out_put_model;
  if (!Optimize(out_put_model)) {
    return;
  }
  if (out_put_model.size() != trajectory.position_list.size()) {
    RERROR << "Optimeze result don't match to input";
    return;
  }
  InitYawInfo(trajectory, true, false);
  InitGeometryInfo(trajectory);
  for (size_t i = 0; i < out_put_model.size(); ++i) {
    if (std::get<1>(trajectory.position_list[i]) == ObjectTrajectory::invalid)
      continue;

    auto& cur_pos =
        pos_map.find(std::get<4>(trajectory.position_list[i]))->second;
    auto& obj_world = std::get<2>(trajectory.position_list[i]).obj_world;
    obj_world.position_3d[0] = out_put_model[i][0];
    obj_world.position_3d[1] = out_put_model[i][1];
    obj_world.orientation_3d.yaw = out_put_model[i][2];
    obj_world.velocity[0] = out_put_model[i][3] * std::cos(out_put_model[i][2]);
    obj_world.velocity[1] = out_put_model[i][3] * std::sin(out_put_model[i][2]);
    obj_world.size_3d.length = ave_length;
    obj_world.size_3d.width = ave_width;
    obj_world.size_3d.height = ave_height;
    obj_world.acceleration[0] =
        out_put_model[i][6] * std::cos(out_put_model[i][2]);
    obj_world.acceleration[1] =
        out_put_model[i][6] * std::sin(out_put_model[i][2]);

    obj_world.yawrate = out_put_model[i][7];
    std::get<2>(trajectory.position_list[i]).obj = obj_world;
    std::get<2>(trajectory.position_list[i]).obj.transform(cur_pos.inverse());
    // std::cout << cur_pos << std::endl;
  }
}
void MotionModelOptimize::InputTrajectory(const ObjectTrajectory& trajectory) {
  for (auto& track_obj : trajectory.position_list) {
    auto& obj_world = std::get<2>(track_obj).obj_world;
    auto new_theta = NormalizeAngle(obj_world.orientation_3d.yaw);
    // init measure
    measure_states_.emplace_back(
        MeasureModel{obj_world.position_3d[0], obj_world.position_3d[1],
                     new_theta, 0.0, obj_world.size_3d.length,
                     obj_world.size_3d.width, double(trajectory.ref_point_pos_),
                     obj_world.timestamp, double(std::get<1>(track_obj))});
    // init output
    inner_states_.emplace_back(
        new CeresModel(obj_world.position_3d[0], obj_world.position_3d[1],
                       new_theta, 0.0, obj_world.size_3d.length,
                       obj_world.size_3d.width, 0, 0, obj_world.timestamp));
  }
}

bool MotionModelOptimize::Optimize(std::vector<SaveModel>& optimize_states) {
  // x y theta
  for (size_t start = 0; start < measure_states_.size(); start += step) {
    size_t end = start + window;
    ceres::Problem problem;
    ceres::LossFunction* size_loss_function = new ceres::HuberLoss(1.0);
    ceres::LossFunction* measure_loss_function =
        new ceres::HuberLoss(1.0);  // new ceres::HuberLoss(1.0); //nullptr;
    ceres::LossFunction* motion_loss_function = new ceres::HuberLoss(
        1.0);  // nullptr; //new ceres::HuberLoss(1.0); //nullptr;
    // 流形残差运算
    ceres::Manifold* angle_manifold = AngleManifold::Create();
    int valid_count = 0;
    for (size_t i = start; i < measure_states_.size() && i < end; ++i) {
      // size
      // 每一帧的size都贴近贴近均值
      // todo先定长宽 参考点与尺寸
      // invalid pos
      if (fabs(measure_states_[i][8] - uint8_t(ObjectTrajectory::invalid)) <
          0.001)
        continue;
      ++valid_count;
      problem.AddResidualBlock(
          SizeChangeCostFunction::CreateAutoDiffCostFunction(ave_length,
                                                             ave_width, 10.0),
          size_loss_function, inner_states_[i]->getLengthptr(),
          inner_states_[i]->getWidthptr());
      // measure cost : [ x, y, theta ]
      // 根据后轴模型,参考点贴近观测
      problem.AddResidualBlock(
          MeasureCostFunction::CreateAutoDiffCostFunction(
              static_cast<uint8_t>(measure_states_[i][6]),
              {measure_states_[i][0], measure_states_[i][1],
               measure_states_[i][2], measure_states_[i][4]},
              *inner_states_[i]->getLengthptr(), {1.0, 1.0, 10.0}),
          measure_loss_function, inner_states_[i]->getXYData(),
          inner_states_[i]->getThetaptr());

      // RearAxelMotion
      // 窗口内匀加速模型，每个函数同时优化两个位置，并共用同一个加速度，可以给窗口限长
      int next_idx = i + 1;
      while (next_idx < measure_states_.size()) {
        if (fabs(measure_states_[next_idx][8] -
                 uint8_t(ObjectTrajectory::invalid)) < 0.001) {
          ++next_idx;
        } else {
          break;
        }
      }
      if (next_idx < measure_states_.size()) {
        // invalid
        if (fabs(measure_states_[next_idx][8] -
                 uint8_t(ObjectTrajectory::invalid)) < 0.001)
          continue;
        double delta_time = *inner_states_[next_idx]->getTimeptr() -
                            *inner_states_[i]->getTimeptr();
        // 0.5s窗口共用一个acc
        problem.AddResidualBlock(
            RearAxelMotion::CreateAutoDiffCostFunction(
                static_cast<uint8_t>(measure_states_[i][6]),
                *inner_states_[i]->getLengthptr(),
                *inner_states_[next_idx]->getLengthptr(), delta_time,
                {1.0, 1.0, 10.0, 1.0}),
            motion_loss_function, inner_states_[i]->getXYData(),
            inner_states_[i]->getThetaptr(), inner_states_[i]->getVelptr(),
            inner_states_[next_idx]->getXYData(),
            inner_states_[next_idx]->getThetaptr(),
            inner_states_[next_idx]->getVelptr(),
            inner_states_[start]->getAccptr(),
            inner_states_[start]->getSteerAngleptr());
      }

      // theta manifold
      problem.SetManifold(inner_states_[i]->getThetaptr(), angle_manifold);

      // acc and steer angle
      ceres::LossFunction* acc_steer_angle_loss_function =
          nullptr;  // new ceres::SoftLOneLoss(1.0);
      problem.AddResidualBlock(
          EqualCostFusion::CreateAutoDiffCostFunction(0.0, 0.1),
          acc_steer_angle_loss_function, inner_states_[start]->getAccptr());

      problem.AddResidualBlock(
          EqualCostFusion::CreateAutoDiffCostFunction(0.0, 1.0),
          acc_steer_angle_loss_function,
          inner_states_[start]->getSteerAngleptr());
    }
    if (valid_count < 3) {
      continue;
    }
    ceres::Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    for (size_t i = start; i < measure_states_.size() && i < end; ++i) {
      *inner_states_[i]->getAccptr() = *inner_states_[start]->getAccptr();
      *inner_states_[i]->getSteerAngleptr() =
          *inner_states_[start]->getSteerAngleptr();
    }
  }

  optimize_states.resize(inner_states_.size());
  for (size_t i = 0; i < inner_states_.size(); ++i) {
    inner_states_[i]->computeOmega();
    optimize_states[i] = inner_states_[i]->convertToSaveModel();
  }
  return true;
}
void MotionModelOptimize::InitType(ObjectTrajectory& trajectory) {
  std::unordered_map<rally::ObjectType, int> type_num;
  int max_num = 0;
  rally::ObjectType max_type = rally::ObjectType::ERROR;
  for (auto& obj : trajectory.position_list) {
    auto cur_type = ObjectTypeNameToType(std::get<2>(obj).obj.type);
    if (cur_type == rally::ObjectType::ERROR) continue;
    type_num[cur_type]++;
    if (type_num[cur_type] >= max_num) {
      max_num = type_num[cur_type];
      max_type = cur_type;
    }
  }
  if (max_num != 0) {
    for (auto& obj : trajectory.position_list) {
      auto cur_type = ObjectTypeNameToType(std::get<2>(obj).obj.type);
      std::get<2>(obj).obj.type = ObjectTypeToLowerName(max_type);
      std::get<2>(obj).obj_world.type = ObjectTypeToLowerName(max_type);
      if (cur_type == max_type) continue;
      // no use to updata length
      std::get<2>(obj).change_type_ = true;
    }
  }
}
// only world
void MotionModelOptimize::InitYawInfo(ObjectTrajectory& trajectory,
                                      bool only_ped, bool check_move_dir_) {
  // 对于非机动车目标或仅行人目标，使用运动方向计算yaw角
  for (size_t i = 0; i < trajectory.position_list.size(); ++i) {
    auto& obj = trajectory.position_list[i];
    auto& obj_world = std::get<2>(obj).obj_world;
    if (only_ped && ref::ObjectTypeNameToType(obj_world.type) !=
                        rally::ObjectType::PEDESTRIAN)
      continue;
    if (!IsCar(obj_world.type) && !trajectory.is_statical_) {
      // 间隔三帧找到前进方向
      int gap_pos_idx = std::min(i + 3, trajectory.position_list.size() - 1);
      while (gap_pos_idx > i &&
             std::get<1>(trajectory.position_list[gap_pos_idx]) ==
                 ObjectTrajectory::invalid) {
        --gap_pos_idx;
      }
      // 校正最后一个位置
      if (gap_pos_idx == static_cast<int>(i)) {
        --gap_pos_idx;
        while (gap_pos_idx >= 0 &&
               std::get<1>(trajectory.position_list[gap_pos_idx]) ==
                   ObjectTrajectory::invalid) {
          --gap_pos_idx;
        }
        if (gap_pos_idx >= 0) {
          obj_world.orientation_3d.yaw =
              std::get<2>(trajectory.position_list[i - 1])
                  .obj_world.orientation_3d.yaw;
        }
        continue;
      }
      auto diff = std::get<2>(trajectory.position_list[gap_pos_idx])
                      .obj_world.position_3d -
                  obj_world.position_3d;
      // 处理静止
      if (diff.norm() < 0.03) {
        continue;
      } else if (diff[0] == 0 && diff[1] > 0) {
        obj_world.orientation_3d.yaw = M_PI_2;
      } else if (diff[0] == 0 && diff[1] < 0) {
        obj_world.orientation_3d.yaw = -M_PI_2;
      } else {
        obj_world.orientation_3d.yaw = std::atan(diff[1] / diff[0]);
        // 还需要校正方向，tan只能表示-pi/2 pi/2
        if (diff[0] < 0) {
          obj_world.orientation_3d.yaw =
              obj_world.orientation_3d.yaw >= 0
                  ? obj_world.orientation_3d.yaw - M_PI
                  : obj_world.orientation_3d.yaw + M_PI;
        }
      }
    }
  }
  // 非车辆目标不进行进一步yaw角统一
  if (!IsVehicle(std::get<2>(trajectory.position_list[0]).obj_world.type)) {
    trajectory.SetRefPointPos();
    return;
  }
  // 先利用速度或者运动方向把反方向yaw角校正，然后统计前10帧的象限
  std::vector<int> quadrant_statistics(4, 0);
  int max_idx = 0;
  for (size_t i = 0; i < trajectory.position_list.size() && i < 10; ++i) {
    Eigen::Vector2f motion_dir{
        std::get<2>(trajectory.position_list[i]).obj_world.velocity[0],
        std::get<2>(trajectory.position_list[i]).obj_world.velocity[1]};
    float vel = motion_dir.norm();
    motion_dir.normalize();
    // 速度大于一定阈值，不会发生反向运动
    if (vel > 2.7 * 2.7) {
      float vel_yaw = 0;
      if (vel == 0) {
        vel_yaw = 0;
      } else if (motion_dir[0] == 0 && motion_dir[1] > 0) {
        vel_yaw = M_PI_2;
      } else if (motion_dir[0] == 0 && motion_dir[1] < 0) {
        vel_yaw = M_PI_2;
      } else {
        vel_yaw = std::atan(motion_dir[1] / motion_dir[0]);
        if (motion_dir[0] < 0) {
          vel_yaw = vel_yaw >= 0 ? vel_yaw - M_PI : vel_yaw + M_PI;
        }
      }

      if (std::cos(
              CalYawDiff180(vel_yaw, std::get<2>(trajectory.position_list[i])
                                         .obj_world.orientation_3d.yaw)) < 0) {
        if (std::get<2>(trajectory.position_list[i])
                .obj_world.orientation_3d.yaw <= 0) {
          std::get<2>(trajectory.position_list[i])
              .obj_world.orientation_3d.yaw += M_PI;
        } else {
          std::get<2>(trajectory.position_list[i])
              .obj_world.orientation_3d.yaw -= M_PI;
        }
      }
    } else if (check_move_dir_ && !trajectory.is_statical_) {
      int right = i + 1;
      while (right < trajectory.position_list.size() &&
             std::get<1>(trajectory.position_list[right]) ==
                 ObjectTrajectory::invalid) {
        ++right;
      }
      if (right < trajectory.position_list.size()) {
        Eigen::Vector3d dir =
            std::get<2>(trajectory.position_list[right]).obj_world.position_3d -
            std::get<2>(trajectory.position_list[i]).obj_world.position_3d;
        float vel_yaw = 0;
        if (dir.norm() == 0) {
          vel_yaw = 0;
        } else if (dir[0] == 0 && dir[1] > 0) {
          vel_yaw = M_PI_2;
        } else if (dir[0] == 0 && dir[1] < 0) {
          vel_yaw = M_PI_2;
        } else {
          vel_yaw = std::atan(dir[1] / dir[0]);
          if (dir[0] < 0) {
            vel_yaw = vel_yaw >= 0 ? vel_yaw - M_PI : vel_yaw + M_PI;
          }
        }
        if (std::cos(CalYawDiff180(vel_yaw,
                                   std::get<2>(trajectory.position_list[i])
                                       .obj_world.orientation_3d.yaw)) < 0) {
          if (std::get<2>(trajectory.position_list[i])
                  .obj_world.orientation_3d.yaw <= 0) {
            std::get<2>(trajectory.position_list[i])
                .obj_world.orientation_3d.yaw += M_PI;
          } else {
            std::get<2>(trajectory.position_list[i])
                .obj_world.orientation_3d.yaw -= M_PI;
          }
        }
      }
    }
    // 统计象限
    int quadrant = GetYawQuadrant(
        std::get<2>(trajectory.position_list[i]).obj_world.orientation_3d.yaw);
    quadrant_statistics[quadrant]++;
    if (quadrant_statistics[quadrant] > quadrant_statistics[max_idx]) {
      max_idx = quadrant;
    }
  }
  // 利用统计的象限修正第一帧朝向
  double dir_yaw = kQuadrantYaw[max_idx];
  if (std::cos(CalYawDiff180(
          std::get<2>(trajectory.position_list[0]).obj_world.orientation_3d.yaw,
          dir_yaw)) < 0) {
    if (std::get<2>(trajectory.position_list[0]).obj_world.orientation_3d.yaw <=
        0) {
      std::get<2>(trajectory.position_list[0]).obj_world.orientation_3d.yaw +=
          M_PI;
    } else {
      std::get<2>(trajectory.position_list[0]).obj_world.orientation_3d.yaw -=
          M_PI;
    }
    if (std::get<3>(trajectory.position_list[0]) ==
        RefPointPos::REF_POINT_BACK) {
      std::get<3>(trajectory.position_list[0]) = RefPointPos::REF_POINT_FRONT;
    } else if (std::get<3>(trajectory.position_list[0]) ==
               RefPointPos::REF_POINT_FRONT) {
      std::get<3>(trajectory.position_list[0]) = RefPointPos::REF_POINT_BACK;
    }
  }
  std::get<3>(trajectory.position_list[0]) =
      computeRefPointStatus(std::get<2>(trajectory.position_list[0]));
  // 相邻两帧朝向方向统一
  for (size_t i = 1; i < trajectory.position_list.size(); ++i) {
    auto& obj = std::get<2>(trajectory.position_list[i]);
    obj.obj_world.orientation_3d.yaw =
        NormalizeAngle(obj.obj_world.orientation_3d.yaw);
    obj.obj.orientation_3d.yaw = NormalizeAngle(obj.obj.orientation_3d.yaw);
    if (std::cos(CalYawDiff180(obj.obj_world.orientation_3d.yaw,
                               std::get<2>(trajectory.position_list[i - 1])
                                   .obj_world.orientation_3d.yaw)) < 0) {
      if (obj.obj_world.orientation_3d.yaw <= 0) {
        obj.obj_world.orientation_3d.yaw += M_PI;
      } else {
        obj.obj_world.orientation_3d.yaw -= M_PI;
      }
      if (std::get<3>(trajectory.position_list[i]) ==
          RefPointPos::REF_POINT_BACK) {
        std::get<3>(trajectory.position_list[i]) = RefPointPos::REF_POINT_FRONT;
      } else if (std::get<3>(trajectory.position_list[i]) ==
                 RefPointPos::REF_POINT_FRONT) {
        std::get<3>(trajectory.position_list[i]) = RefPointPos::REF_POINT_BACK;
      }
    }
  }
  trajectory.SetRefPointPos();
}
// 求得加权平均,并统一yaw方向
void MotionModelOptimize::InitGeometryInfo(ObjectTrajectory& trajectory) {
  // size
  int observe_weight = 0;
  ave_length = 0;
  ave_width = 0;
  ave_height = 0;
  for (size_t i = 0; i < trajectory.position_list.size(); ++i) {
    if (std::get<1>(trajectory.position_list[i]) == ObjectTrajectory::invalid)
      continue;
    if (std::get<2>(trajectory.position_list[i]).change_type_) continue;
    auto& obj = std::get<2>(trajectory.position_list[i]);
    auto score = std::max(1, obj.observe_score);
    ave_length += obj.obj_world.size_3d.length * score;
    ave_width += obj.obj_world.size_3d.width * score;
    ave_height += obj.obj_world.size_3d.height * score;
    observe_weight += score;
  }
  // update size and init vel
  if (ave_length == 0 || ave_width == 0 || ave_height == 0) return;
  ave_length /= observe_weight;
  ave_width /= observe_weight;
  ave_height /= observe_weight;
  for (size_t i = 0; i < trajectory.position_list.size(); ++i) {
    // if(i!=0)
    // {
    //   Eigen::Vector3d pos_diff =
    //       std::get<2>(trajectory.position_list[i]).obj_world.position_3d -
    //       std::get<2>(trajectory.position_list[i - 1]).obj_world.position_3d;
    //   double time_dif = std::get<0>(trajectory.position_list[i]) -
    //                     std::get<0>(trajectory.position_list[i-1]);
    //   std::get<2>(trajectory.position_list[i]).obj_world.velocity =
    //       pos_diff / time_dif;
    // }
    // if(i==1)
    // {
    //   std::get<2>(trajectory.position_list[0]).obj_world.velocity =
    //       std::get<2>(trajectory.position_list[i]).obj_world.velocity;
    // }
    auto& obj = std::get<2>(trajectory.position_list[i]);
    // 不涉及时序优化，可以用真实参考点，提高准确性
    auto ref_point_state = std::get<3>(trajectory.position_list[i]);
    float ref_x = obj.obj_world.position_3d[0];
    float ref_y = obj.obj_world.position_3d[1];

    GetRefPointFromCenter(ref_point_state, obj.obj_world.orientation_3d.yaw,
                          obj.obj_world.size_3d.length,
                          obj.obj_world.position_3d[0],
                          obj.obj_world.position_3d[1], ref_x, ref_y);
    obj.obj_world.size_3d.length = ave_length;
    obj.obj_world.size_3d.width = ave_width;
    obj.obj_world.size_3d.height = ave_height;
    GetCenterFromRefPoint(ref_point_state, obj.obj_world.orientation_3d.yaw,
                          obj.obj_world.size_3d.length, ref_x, ref_y,
                          obj.obj_world.position_3d[0],
                          obj.obj_world.position_3d[1]);

    GetRefPointFromCenter(ref_point_state, obj.obj_world.orientation_3d.yaw,
                          obj.obj_world.size_3d.length,
                          obj.obj_world.position_3d[0],
                          obj.obj_world.position_3d[1], ref_x, ref_y);
  }
}
bool MotionModelOptimize::CheckStatic(ObjectTrajectory& trajectory,
                                      const bool use_speed) {
  if (trajectory.position_list.size() < 5) {
    return false;
  }
  if (use_speed) {
    int count = 0;
    for (auto& obj : trajectory.position_list) {
      if (std::get<2>(obj).obj_world.is_static ||
          std::get<2>(obj).obj_world.velocity.norm() < 0.1) {
        ++count;
      }
    }
    if (count > 0.9 * trajectory.position_list.size()) {
      trajectory.is_statical_ = true;
      return true;
    } else {
      trajectory.is_statical_ = false;
      return false;
    }
  }
  if (trajectory.is_statical_) return true;
  auto type =
      ObjectTypeNameToType(std::get<2>(trajectory.position_list[0]).obj.type);
  if (type == rally::ObjectType::TRAFFIC_CONE ||
      type == rally::ObjectType::WARN_TRIANGLE) {
    trajectory.is_statical_ = true;
    return true;
  }

  if (!IsVehicle(type)) {
    return false;
  }
  bool head_tail_move = true;
  bool dynamic_move = true;
  Eigen::Vector3d start_pos(0, 0, 0);
  Eigen::Vector3d end_pos(0, 0, 0);
  double start_time = 0, end_time = 0;
  // head tail overlay
  int static_label_num = 0;
  int all_count = 0;
  for (size_t i = 0; i < trajectory.position_list.size(); ++i) {
    if (std::get<1>(trajectory.position_list[i]) ==
        ObjectTrajectory::PostionState::invalid) {
      continue;
    }

    ++all_count;
    if (std::get<2>(trajectory.position_list[i]).obj.is_static) {
      ++static_label_num;
    }
    if (start_time < 0.1) {
      start_time = std::get<0>(trajectory.position_list[i]);
      start_pos =
          std::get<2>(trajectory.position_list[i]).obj_world.position_3d;
    }
    if (end_time != 0) {
      Eigen::Vector3d pos_diff =
          std::get<2>(trajectory.position_list[i]).obj_world.position_3d -
          end_pos;
      double time_diff = std::get<0>(trajectory.position_list[i]) - end_time;
      if (pos_diff.norm() > 2 * time_diff) {
        dynamic_move = false;
      }
    }
    end_time = std::get<0>(trajectory.position_list[i]);
    end_pos = std::get<2>(trajectory.position_list[i]).obj_world.position_3d;
  }

  Eigen::Vector3d pos_diff = end_pos - start_pos;

  pos_diff[2] = 0;
  if (type <= rally::ObjectType::CONSTRN_VEH) {
    if (pos_diff.norm() > 1 * (end_time - start_time) || pos_diff.norm() > 1) {
      head_tail_move = false;
    }
  } else {
    if (pos_diff.norm() > 0.5 * (end_time - start_time)) {
      head_tail_move = false;
    }
  }
  trajectory.is_statical_ = head_tail_move && dynamic_move;
  if (float(static_label_num) > float(all_count) * 0.95) {
    trajectory.is_statical_ = true;
  } else if (all_count > 5 &&
             float(static_label_num) < float(all_count) * 0.8) {
    trajectory.is_statical_ = false;
  }
  return trajectory.is_statical_;
}
void MotionModelOptimize::StaticOptimize(ObjectTrajectory& trajectory) {
  Eigen::Vector3d pos_ave(0, 0, 0);
  double yaw_ave{0.};
  int left_count = 0, right_count = 0;
  int weight{0};
  for (size_t i = 0; i < trajectory.position_list.size(); ++i) {
    auto& obj = trajectory.position_list[i];
    if (std::get<1>(obj) == ObjectTrajectory::invalid) {
      continue;
    }

    auto& obj_world = std::get<2>(obj).obj_world;
    weight += std::get<2>(obj).observe_score;
    pos_ave += obj_world.position_3d * std::get<2>(obj).observe_score;
    double cur_yaw = obj_world.orientation_3d.yaw;
    if (cur_yaw < 0) {
      ++left_count;
    } else {
      ++right_count;
    }
    yaw_ave += std::cos(cur_yaw) * std::get<2>(obj).observe_score;
  }
  if (weight == 0) {
    return;
  }
  pos_ave /= weight;
  yaw_ave /= weight;
  yaw_ave = std::max(std::min(1.0, yaw_ave), -1.0);

  for (size_t i = 0; i < trajectory.position_list.size(); ++i) {
    auto& obj = trajectory.position_list[i];
    auto& obj_world = std::get<2>(obj).obj_world;
    std::get<1>(obj) = ObjectTrajectory::valid;
    obj_world.timestamp = std::get<0>(obj);
    obj_world.yawrate = 0;
    obj_world.velocity = {0, 0, 0};
    obj_world.acceleration = {0, 0, 0};
    obj_world.position_3d = pos_ave;
    obj_world.orientation_3d.yaw =
        left_count > right_count ? -std::acos(yaw_ave) : std::acos(yaw_ave);
  }
}
// output for (int i = 0;i<inner_state) }
}  // namespace ref
