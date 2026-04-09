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

#include "robot/perception/uwb_postprocess/image_track/mask_track.h"

namespace robot {
namespace perception {

bool MaskTrack::Init(const YAML::Node& config, const SensorID sensor_id) {
  // 初始化初始主传感器
  trigger_sensor_ = sensor_id;
  sensor_id_ = sensor_id;
  next_sensor_ = sensor_id;
  camera_calib_ptr_ = std::dynamic_pointer_cast<CameraCalib>(
      CalibCenter::GetInstance().Get(static_cast<uint8_t>(sensor_id_)));
  if (camera_calib_ptr_ == nullptr) {
    AERROR << "sensor_calib is nullptr! " << kSensorIDToNameMap.at(sensor_id_);
    return false;
  } else {
    AINFO << " Init sensor id " << kSensorIDToNameMap.at(sensor_id_);
  }
  ori_width = camera_calib_ptr_->GetWidth();
  ori_height = camera_calib_ptr_->GetHeight();
  UWBServer::GetInstance().ori_image_width = ori_width;
  UWBServer::GetInstance().ori_image_height = ori_height;
  // 读取所有sensor的原始calib信息，尺寸必须和主传感器一致
  for (const auto& sensor : TrackCameraSensor) {
    auto cur_calib_ptr = std::dynamic_pointer_cast<CameraCalib>(
        CalibCenter::GetInstance().Get(static_cast<uint8_t>(sensor)));
    if (cur_calib_ptr == nullptr) {
      AERROR << "sensor_calib is nullptr! " << kSensorIDToNameMap.at(sensor);
    } else if (cur_calib_ptr->GetWidth() != ori_width ||
               cur_calib_ptr->GetHeight() != ori_height) {
      AERROR << "sensor_calib size not match! "
             << kSensorIDToNameMap.at(sensor);
    } else {
      track_calib_map_[sensor].camera_calib_ptr_ = cur_calib_ptr;
    }
  }

  YAML::Node image_track_config = config["MastTrack"];
  rally::yamlRead(image_track_config, "use_fast_sam", use_fast_sam_);
  // yolo
  YAML::Node detection_config = config["Detection"];
  yolov8_ptr_ = std::make_shared<Yolov8Detection>();
  if (!yolov8_ptr_->Init(detection_config)) {
    AERROR << "Yolov8Detection init failed!";
    return false;
  }
  object_list = std::make_shared<Yolov8ObjectList>();
  object_list->objs.reserve(100);
  // fastsam
  if (use_fast_sam_) {
    YAML::Node fastsam_config = config["FastSAM"];
    fastsam_ptr_ = std::make_shared<FastSAM>();
    if (!fastsam_ptr_->Init(fastsam_config)) {
      AERROR << "FastSAM init failed!";
      return false;
    }
  }

  // cutie
  cutie_config = config["Cutie"];
  cutie_ptr_ = std::make_shared<CutieCore>();
  if (!cutie_ptr_->Init(cutie_config)) {
    AERROR << "Cutie init failed!";
    return false;
  }
  // get config
  rally::yamlRead(image_track_config, "show_image", show_image_);
  rally::yamlRead(image_track_config, "show_depth_image", show_depth_image_);
  rally::yamlRead(image_track_config, "image_scale", image_scale_);
  rally::yamlRead(image_track_config, "track_mode", track_mode_);
  rally::yamlRead(image_track_config, "retrack_mode", retrack_mode_);
  rally::yamlRead(image_track_config, "allow_sensor_change",
                  allow_sensor_change_);
  rally::yamlRead(image_track_config, "h_w_ratio_thresh", h_w_ratio_thresh_);

  if (track_mode_ != 0) {
    UWBServer::GetInstance().input_data.command = COMMOND::START_TRACK;
  }
  // 生成remap
  for (auto& track_calib : track_calib_map_) {
    // 校验
    auto& cur_calib = track_calib.second;
    cur_calib.input_image_K =
        cur_calib.camera_calib_ptr_->GetInnerCvMat().clone();
    cur_calib.camera_calib_ptr_->ApplyScaleIntoK(image_scale_);
    bgr_image_size_ = cv::Size(cur_calib.camera_calib_ptr_->GetWidth(),
                               cur_calib.camera_calib_ptr_->GetHeight());
    if (cur_calib.camera_calib_ptr_->GetDistCoeffMatf().rows == 4) {
      cv::Mat zero_dist = cv::Mat::zeros(4, 1, CV_32F);
      cv::initUndistortRectifyMap(
          cur_calib.input_image_K, zero_dist, cv::noArray(),
          cur_calib.camera_calib_ptr_->GetInnerCvMat(), bgr_image_size_,
          CV_16SC2, cur_calib.map1, cur_calib.map2);
    } else {
      cv::initUndistortRectifyMap(
          cur_calib.input_image_K,
          cur_calib.camera_calib_ptr_->GetDistCoeffMatf(), cv::noArray(),
          cur_calib.camera_calib_ptr_->GetInnerCvMat(), bgr_image_size_,
          CV_16SC2, cur_calib.map1, cur_calib.map2);
    }
  }
  // 设置remap后画布
  bgr_image_ = cv::Mat(bgr_image_size_, CV_8UC3);

  // set task
  StartTrigger();
  inited_ = true;
  return true;
}
void MaskTrack::InitCameraSensor(
    const std::unordered_set<SensorID>& calib_sensor_list) {
  for (const auto& sensor : calib_sensor_list) {
    if (isCameraSensor(sensor)) {
      TrackCameraSensor.insert(sensor);
    }
  }
}
void MaskTrack::Perception(const std::shared_ptr<perception::Image>& msg) {
  if (busy_for_init_) {
    AWARN << " Is busy for init, return! ";
    return;
  }
  // PREPROCESS IMAGE
  busy_for_image_ = true;
  // 更新真实使用的标定
  sensor_id_ = next_sensor_;
  const auto& track_calib = track_calib_map_.at(sensor_id_);
  camera_calib_ptr_ = track_calib.camera_calib_ptr_;
  input_image_K = track_calib.input_image_K;
  map1 = track_calib.map1;
  map2 = track_calib.map2;
  UWBServer::GetInstance().calib_mutex.lock();
  UWBServer::GetInstance().ori_K = track_calib.input_image_K;
  UWBServer::GetInstance().ori_D =
      track_calib.camera_calib_ptr_->GetDistCoeffMatf();
  UWBServer::GetInstance().calib_mutex.unlock();
  rally::Time msg_time(msg->header.time);
  auto t_1 = apollo::cyber::Time::Now();
  cv::Mat yuv422_temp;
  yuv422_temp = cv::Mat(msg->height, msg->width, CV_8UC2, msg->data_ptr);
  cv::cvtColor(yuv422_temp, bgr_image_, cv::COLOR_YUV2BGR_YUY2);
  if (msg->height != ori_height || msg->width != ori_width) {
    cv::resize(bgr_image_, bgr_image_, cv::Size(ori_width, ori_height));
    AINFO << " CALIBRATION SIZE IS " << ori_width << " " << ori_height
          << " BUG INPUT " << msg->height << " " << msg->width;
  }
  cv::remap(bgr_image_, bgr_image_, map1, map2, cv::INTER_LINEAR);
  auto t_2 = apollo::cyber::Time::Now();
  busy_for_image_ = false;
  object_list->time_stamp = msg_time.toSecond();
  object_list->objs.clear();
  time_stamp_ = msg_time.toSecond();
  // yolo-fastsam-init
  if (!rect_init_) {
    if (track_state_ == TRACKER_STATE_CODE::ERROR_TRACK) {
      AERROR
          << " !!!!!!!!!!!!!!!!!!!!!!! ERROR TRACK "
             "!!!!!!!!!!!!!!!!!!!!!!! OUT IMAGE TRACK AND INIT !!!!!!!!!!!!!";
      if (track_mode_ == 0) {
        return;
      }
    }
    yolov8_ptr_->Perception(bgr_image_, object_list);
    // 利用图像宽高估计距离
    EstimateDist();
    // 利用点云估计距离
    FusionLidarPoints();
    if (track_mode_ == 0) {
      SetRectByOutside(bgr_image_);
    } else if (track_mode_ == 1) {
      SelectRect(bgr_image_);
    } else if (track_mode_ == 3) {
      ChooseOnlyPed();
    }
    if (rect_init_) {
      if (use_fast_sam_) {
        fastsam_ptr_->Perception(bgr_image_);
        cutie_ptr_->InitMask(bgr_image_, fastsam_ptr_->GetCombinedMask());
      } else {
        cv::Point ul(center_pt_.x - (size_.width - 1) / 2,
                     center_pt_.y - (size_.height - 1) / 2);
        cv::Rect roi(ul.x, ul.y, size_.width, size_.height);
        cv::Mat combined_mask = cv::Mat::zeros(bgr_image_.size(), CV_8UC1);
        combined_mask(roi).setTo(1);
        cutie_ptr_->InitMask(bgr_image_, combined_mask);
      }
      out_time_stamp_ = time_stamp_;
      out_sensor_id_ = sensor_id_;
      if (!cutie_ptr_->GetMaskInited()) {
        rect_init_ = false;
      }
    }
  } else {
    Track(bgr_image_);
  }

  AINFO << "convert cost time: " << (t_2 - t_1).ToSecond() * 1000 << " ms.";
}
void MaskTrack::Track(const cv::Mat& image) {
  object_list->objs.clear();
  cutie_ptr_->Perception(bgr_image_, need_flip_image_);
  // 生成一个目标，用于估计深度
  Yolov8Object::Ptr mask_obj = std::make_shared<Yolov8Object>();
  mask_obj->bbox = cutie_ptr_->GetOriRect();
  object_list->objs.emplace_back(mask_obj);
  // 利用图像宽高估计距离
  EstimateDist();
  // 利用点云估计距离
  if (!FusionDepthImage()) {
    FusionLidarPoints();
  }

  busy_for_process_ = true;
  UpdateState(object_list->objs[0]);
  ChooseNextSensor();
  out_time_stamp_ = time_stamp_;
  out_sensor_id_ = sensor_id_;
  busy_for_process_ = false;
}
// 根据跟踪的输出更新
void MaskTrack::UpdateState(const Yolov8Object::Ptr& det) {
  // 更新2D信息
  if (full_area < 1) {
    full_area = cutie_ptr_->GetMaskArea();
  }
  center_pt_ = cutie_ptr_->GetOriCenter();
  size_.width = cutie_ptr_->GetOriRect().width;
  size_.height = cutie_ptr_->GetOriRect().height;
  UpdateTrackState();
  // 更新3D信息 如果前一帧是uwb需要距离误差较小 如果不是uwb就直接更新
  if (IsTrack(track_state_) && uwb_match_time_ - time_stamp_ < 0 &&
      (dist_sensor_ != OBSERVE_SENSOR::UWB_OBSERVE ||
       std::abs(distance_ - det->distance) < 2)) {
    // 无遮挡的相机输入才有价值
    if (det->observe_sensor != OBSERVE_SENSOR::CAMERA_OBSERVE ||
        distance_ > 7 ||
        (cover_state == COVER_STATE::NO_COVER &&
         std::abs(distance_ - det->distance) < 5) ||
        (cover_state == COVER_STATE::PART_COVER &&
         std::abs(distance_ - det->distance) < 1)) {
      SetDistance(det->distance, time_stamp_, det->observe_sensor);
    }
  }
  // 历史
  if (history_.size() > 10) {
    history_.pop_front();
  }
  history_.emplace_back(MaskHisoty{cutie_ptr_->GetMaskArea(),
                                   cutie_ptr_->GetOriRect(), track_state_,
                                   cover_state});
}
bool MaskTrack::InitRect(const cv::Mat& ori_image, const cv::Point& left_up,
                         const cv::Point& right_down,
                         const Yolov8Object::Ptr& obj_ptr) {
  // 2D信息
  cv::Rect rect(left_up, right_down);
  rect = rect & cv::Rect(0, 0, ori_image.cols, ori_image.rows);
  size_ = cv::Size(rect.br().x - rect.tl().x, rect.br().y - rect.tl().y);
  center_pt_ = cv::Point2f(rect.tl().x + (rect.width - 1) / 2,
                           rect.tl().y + (rect.height - 1) / 2);
  // 3D信息
  if (obj_ptr != nullptr) {
    SetDistance(obj_ptr->distance, time_stamp_, obj_ptr->observe_sensor);
  }
  AINFO << "SIZE " << size_ << " CENTER " << center_pt_;
  // 设置fastsam prompt
  if (use_fast_sam_) {
    fastsam_ptr_->SetBoxPrompt({cv::Rect(left_up, right_down)});
  }
  rect_init_ = true;
  track_state_ = TRACKER_STATE_CODE::STABLE_TRACK;
  lost_count = 0;
  return true;
}
void MaskTrack::UpdateTrackState() {
  float area = cutie_ptr_->GetMaskArea();
  // 遮挡状态full_area
  if (area < 0.1 * full_area) {
    cover_state = COVER_STATE::FULL_COVER;
  } else if (area < 0.5 * full_area) {
    cover_state = COVER_STATE::PART_COVER;
  } else {
    cover_state = COVER_STATE::NO_COVER;
  }
  // 跟踪状态
  if (area < 0.1 * full_area) {
    cv::Rect pre_rect = history_.back().rect;
    if (pre_rect.tl().x < 5 || pre_rect.tl().y < 5 ||
        pre_rect.br().x > ori_width - 5 || pre_rect.br().y > ori_height - 5) {
      track_state_ = TRACKER_STATE_CODE::EDGE_LOST_TRACK;
    } else {
      track_state_ = TRACKER_STATE_CODE::LOW_SCORE_LOST_TRACK;
    }
    ++lost_count;
  } else {
    track_state_ = TRACKER_STATE_CODE::STABLE_TRACK;
    lost_count = 0;
  }
  if (lost_count > 100) {
    AERROR << " !!!!!!!!!!!!!!!!!!!!!!! ERROR TRACK "
              "!!!!!!!!!!!!!!!!!!!!!!! OUT IMAGE TRACK AND INIT !!!!!!!!!!!!!";
    if (retrack_mode_ != 0) {
      track_state_ = TRACKER_STATE_CODE::ERROR_TRACK;
      track_count = 0;
      //   if (retrack_mode_ == 1) {
      //     rect_init_ = false;
      //     cutie_ptr_ = std::make_shared<CutieCore>();
      //     cutie_ptr_->Init(cutie_config);
      //   }
    }
  }
  double time_gap = time_stamp_ - out_time_stamp_;
  float basic_thresh = vel_norm_ > 2.0 ? 0.3 : 0.2;
  float continue_thresh = (vel_norm_ > 2.0 && distance_ > 7.0) ? 500 : 3000;
  float area_thresh = basic_thresh * (time_gap * 10);
  area_thresh = clamp(area_thresh, 0.2f, 0.8f);
  if (full_area - area < area_thresh * full_area) {
    full_area = area;
  }
  // 往前5帧都是稳定的并且面积较大，认为是有效的，更新
  else if (history_.size() >= 10 &&
           std::abs(area - history_.back().area) < 0.2 * history_.back().area) {
    bool is_stable = true;
    for (int i = history_.size() - 1; i >= history_.size() - 5; --i) {
      if (std::abs(history_[i].area - history_[i - 1].area) >
              0.2 * history_[i].area ||
          history_[i].area < continue_thresh) {
        is_stable = false;
      }
    }
    if (is_stable) {
      full_area = area;
    }
  }
  AINFO << " TRACK STATE " << int(track_state_) << " cover " << int(cover_state)
        << " lost count " << lost_count << " full area " << full_area
        << " area " << area << " thresh " << area_thresh;
}
void MaskTrack::SetDistance(const float distance, const double uwb_time,
                            const OBSERVE_SENSOR observe_sensor) {
  // 获取当前矩形框
  float use_distance = distance < 0.7 ? 0.7 : distance;
  cv::Rect cur_rect = GetRectNoWait();

  // 声明中心点和尺寸变量
  Eigen::Vector3d center_base;
  float tall;
  float width;

  // 将矩形框转换为位姿
  Eigen::Vector3d camera_pose = camera_calib_ptr_->GetPos();
  camera_pose[2] = 0;
  RectToPose(cur_rect, camera_calib_ptr_, use_distance, center_base, tall,
             width);
  center_base_ = center_base.cast<float>();
  center_odom_ = base_to_rel_trans_ * center_base_;
  distance_ = use_distance;
  predict_rect_ = Predict2DRect();
  dist_sensor_ = observe_sensor;
  // 纯相机深度不更新尺寸
  if (observe_sensor == OBSERVE_SENSOR::CAMERA_OBSERVE) {
    return;
  }
  if (!template_init_ && observe_sensor == OBSERVE_SENSOR::UWB_OBSERVE) {
    template_3d_tall_ = tall;
    template_init_ = true;
    AINFO << " Init template tall " << tall;
  }
  // 激光和uwb深度都比较准确，更新尺寸及时间
  size_3d_.length = 1;
  size_3d_.width = width;
  size_3d_.height = tall;
  AINFO << " Change box info " << size_3d_.length << "," << size_3d_.width
        << "," << size_3d_.height;
  uwb_match_time_ = uwb_time;
}
void MaskTrack::SelectRect(const cv::Mat& ori_image) {
  busy_for_init_ = true;
  cv::Rect2d roi = cv::selectROI("Select ROI", ori_image, true, false);
  cv::destroyAllWindows();
  if (roi.width < 10 || roi.height < 10) {
    AERROR << "ROI is too small!";
  } else {
    Yolov8Object::Ptr match_obj;
    float best_iou = 0;
    for (const auto& obj : object_list->objs) {
      float iou = GetIoU(obj->bbox, roi);
      if (iou > best_iou) {
        match_obj = obj;
        best_iou = iou;
      }
    }
    if (match_obj != nullptr) {
      InitRect(ori_image, roi.tl(), roi.br(), match_obj);
    }
  }
  busy_for_init_ = false;
}
void MaskTrack::SetRectByOutside(const cv::Mat& ori_image) {
  busy_for_init_ = true;
  if (std::abs(UWBServer::GetInstance().input_data.time_stamp - time_stamp_) <
      UWBServer::GetInstance().time_thresh_) {
    AINFO << std::fixed << " UWB TIMESTAMP MATCH! Process Time: " << time_stamp_
          << " Server Time: " << UWBServer::GetInstance().input_data.time_stamp;
    cv::Rect roi = RemapRect(UWBServer::GetInstance().GetOriSizeInputRect(),
                             ori_width, ori_height, ori_width * image_scale_,
                             ori_height * image_scale_, input_image_K,
                             camera_calib_ptr_->GetDistCoeffMatf());

    Yolov8Object::Ptr match_obj;
    float best_iou = 0;
    for (const auto& obj : object_list->objs) {
      AINFO << " GET RECT " << roi << " obj iou " << obj->bbox;
      float iou = GetIoU(obj->bbox, roi);
      if (iou > best_iou) {
        match_obj = obj;
        best_iou = iou;
      }
    }
    if (match_obj != nullptr) {
      if (use_fast_sam_) {
        InitRect(ori_image, roi.tl(), roi.br(), match_obj);
      } else {
        float h_w_ratio = match_obj->bbox.height * 1.0 / match_obj->bbox.width;
        if (h_w_ratio < h_w_ratio_thresh_) {
          AERROR << "RATIO SMALL FAIL!"
                 << " ratio " << h_w_ratio;
        } else {
          InitRect(ori_image, match_obj->bbox.tl(), match_obj->bbox.br(),
                   match_obj);
        }
      }
    } else {
      AERROR << " CAN NOT MATCH OBJECT! INIT FAIL!";
    }
  } else {
    AWARN << std::fixed
          << " UWB TIMESTAMP NOT MATCH! Process Time: " << time_stamp_
          << " Server Time: " << UWBServer::GetInstance().input_data.time_stamp;
  }

  busy_for_init_ = false;
}
void MaskTrack::ChooseOnlyPed() {
  busy_for_init_ = true;
  if (object_list->objs.size() == 1) {
    if (track_count == 0) {
      track_rect = object_list->objs[0]->bbox;
    } else {
      float iou = GetIoU(object_list->objs[0]->bbox, track_rect);
      if (iou < 0.01) {
        track_count = 0;
        track_rect = cv::Rect();
      } else {
        track_rect = object_list->objs[0]->bbox;
      }
    }
    ++track_count;
  } else {
    track_count = 0;
    track_rect = cv::Rect();
  }
  if (track_count > 10) {
    InitRect(bgr_image_, track_rect.tl(), track_rect.br(),
             object_list->objs[0]);
  }
  busy_for_init_ = false;
}
void MaskTrack::ChoosePed(const float dist, const float angle_base) {
  WaitForProcess("chooseped");
  busy_for_init_ = true;
  int best_idx = -1;
  for (int i = 0; i < object_list->objs.size(); i++) {
    const auto& obj = object_list->objs[i];
    cv::Point center = (obj->bbox.tl() + obj->bbox.br()) * 0.5;
    Eigen::Vector3d camera_center =
        camera_calib_ptr_->UVToCameraPt(center.x, center.y, dist, false);
    Eigen::Vector3d veh_center =
        camera_calib_ptr_->CameraPtToVehPt(camera_center);
    float cur_angle = std::atan2(veh_center.y(), veh_center.x());
    if (std::abs(cur_angle - angle_base) < 10 * M_PI / 180) {
      // 存在临近目标，不使用
      if (best_idx != -1) {
        best_idx = -1;
        break;
      } else if (track_count != 0) {
        float iou = GetIoU(obj->bbox, track_rect);
        if (iou > 0.01) {
          best_idx = i;
        }
      } else {
        best_idx = i;
      }
    }
  }
  if (best_idx != -1) {
    ++track_count;
    track_rect = object_list->objs[best_idx]->bbox;
  } else {
    track_count = 0;
    track_rect = cv::Rect();
  }
  if (track_count > 10) {
    InitRect(bgr_image_, track_rect.tl(), track_rect.br(),
             object_list->objs[best_idx]);
  }
  if (rect_init_) {
    if (use_fast_sam_) {
      fastsam_ptr_->Perception(bgr_image_);
      cutie_ptr_->InitMask(bgr_image_, fastsam_ptr_->GetCombinedMask());
    } else {
      cv::Point ul(center_pt_.x - size_.width / 2,
                   center_pt_.y - size_.height / 2);
      cv::Rect roi(ul.x, ul.y, size_.width, size_.height);
      cv::Mat combined_mask = cv::Mat::zeros(bgr_image_.size(), CV_8UC1);
      combined_mask(roi).setTo(1);
      cutie_ptr_->InitMask(bgr_image_, combined_mask);
    }
    if (!cutie_ptr_->GetMaskInited()) {
      rect_init_ = false;
    }
  }
  busy_for_init_ = false;
}
cv::Rect MaskTrack::Predict2DRect() {
  cv::Rect rect;
  PoseToRect(center_base_.cast<double>(), size_3d_, camera_calib_ptr_, rect);
  return rect;
}
void MaskTrack::SetCenterBase(const Eigen::Vector3f& optimize_center,
                              const double uwb_time) {
  center_base_ = optimize_center;
}
void MaskTrack::SetCenterOdom(const Eigen::Vector3f& optimize_center) {}
bool MaskTrack::FusionDepthImage() {
  if (time_stamp_ < 1 || object_list->objs.size() != 1) {
    return false;
  }
  // 获取图像
  auto depth_image =
      DepthImageManager::GetInstance().GetDepthImage(sensor_id_, time_stamp_);
  if (depth_image == nullptr) {
    AINFO << "-----the depth image not finded "
          << kSensorIDToNameMap.at(sensor_id_)
          << ", timestamp = " << time_stamp_;
    return false;
  }
  cv::Mat depth_mat(depth_image->height, depth_image->width, CV_16S,
                    depth_image->data_ptr);
  AINFO << std::fixed << "-----the depth image finded "
        << kSensorIDToNameMap.at(sensor_id_) << ", timestamp = " << time_stamp_;

  // cutie输出迟钝的mask图和rect
  cv::Rect cutie_rect = cutie_ptr_->GetRect();
  cv::Mat cutie_mask = cutie_ptr_->GetMask();
  // mask腐蚀
  int erosion_size = 3;
  cv::Mat kernel = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1));
  cv::erode(cutie_mask, cutie_mask, kernel);
  // rect像素到原图像素的scale（先乘上输入resize再乘上网络前处理resize）
  float x_scale = image_scale_ * cutie_mask.cols * 1.0 / bgr_image_.cols;
  float y_scale = image_scale_ * cutie_mask.rows * 1.0 / bgr_image_.rows;
  // 获取从setro网络输入到原图的稀疏remap
  auto remap_info = DepthImageManager::GetInstance().GetRemapInfo(sensor_id_);
  cv::Mat inv_map_x = remap_info->inv_map_x;
  cv::Mat inv_map_y = remap_info->inv_map_y;
  // 实际深度图和输入图之间的scale（通常是0.5）
  float depth_scale = depth_mat.rows * 1.0 / remap_info->remap_size.height;
  // 遍历像素
  cv::Mat temp_image_2;
  if (show_depth_image_) {
    temp_image_2 = cv::Mat::zeros(depth_mat.size(), CV_8UC1);
  }
  // depth key | depth num | depth sum
  std::vector<std::tuple<float, int, float>> depth_cluster;
  int max_index = -1;
  for (int x = cutie_rect.tl().x; x < cutie_rect.br().x; ++x) {
    for (int y = cutie_rect.tl().y; y < cutie_rect.br().y; ++y) {
      if (cutie_mask.at<uint8_t>(y, x) == 255) {
        // 原图坐标
        float ori_x = x / x_scale;
        float ori_y = y / y_scale;
        if (ori_x < 0 || ori_x >= inv_map_x.cols || ori_y < 0 ||
            ori_y >= inv_map_x.rows) {
          continue;
        }
        // 双目输入坐标
        float remap_x = inv_map_x.at<float>(int(ori_y), int(ori_x));
        float remap_y = inv_map_y.at<float>(int(ori_y), int(ori_x));
        // 深度图坐标
        int depth_x = remap_x * depth_scale;
        int depth_y = remap_y * depth_scale;

        float depth = depth_mat.at<uint16_t>(depth_y, depth_x) * 25.0 / 65535.0;

        if (depth <= 0.1 || depth > 25) {
          continue;
        }

        bool have_cluster = false;
        for (int i = 0; i < depth_cluster.size(); ++i) {
          auto& cluster = depth_cluster[i];
          if (std::abs(std::get<0>(cluster) - depth) < 0.8) {
            std::get<1>(cluster) += 1;
            std::get<2>(cluster) += depth;
            have_cluster = true;
            if (std::get<1>(cluster) > std::get<1>(depth_cluster[max_index])) {
              max_index = i;
            }
            break;
          }
        }
        if (!have_cluster) {
          depth_cluster.emplace_back(std::make_tuple(depth, 1, depth));
          if (max_index == -1) {
            max_index = 0;
          }
        }
        if (show_depth_image_) {
          temp_image_2.at<uint8_t>(depth_y, depth_x) = 255;
        }
      }
    }
  }
  if (max_index != -1 && std::get<1>(depth_cluster[max_index]) > 10) {
    object_list->objs[0]->distance = std::get<2>(depth_cluster[max_index]) /
                                     std::get<1>(depth_cluster[max_index]);
    object_list->objs[0]->observe_sensor = OBSERVE_SENSOR::CAMERA_OBSERVE;
    AINFO << " DEPTH CLUSTER SIZE " << depth_cluster.size() << " MAX DIST "
          << object_list->objs[0]->distance << " num "
          << std::get<1>(depth_cluster[max_index]);
  }
  if (show_depth_image_) {
    cv::Mat normalized;
    depth_mat.convertTo(normalized, CV_32F, 25.0 / 65535.0);
    cv::normalize(normalized, normalized, 0, 255, cv::NORM_MINMAX);
    cv::Mat depth_gray;
    normalized.convertTo(depth_gray, CV_8U);
    cv::imshow("depth", depth_gray);
    cv::imshow("temp_image_2", temp_image_2);
    cv::waitKey(1);
  }
  return true;
}
void MaskTrack::FusionLidarPoints() {
  if (time_stamp_ < 1) {
    return;
  }
  rally::Time process_time(time_stamp_);
  int64_t time_micro = process_time.toMicrosecond();
  perception::RangeImage::Ptr find_range_image = nullptr;
  int64_t find_lidar_time = 0;
  perception::RangeImageCacheCenter::GetInstance().GetRangeImage(
      time_micro, find_range_image, find_lidar_time);
  if (find_range_image == nullptr) {
    AINFO << "-----the lidar depth image is not find"
          << ", timestamp = " << time_micro << " lastest lidar time "
          << find_lidar_time << " the lidar and image timestamp diff = "
          << std::abs(time_micro - find_lidar_time) / 1e3 << " ms";
    return;
  }
  int64_t lidar_image_diff = find_lidar_time - time_micro;
  AINFO << "-----the lidar depth image finded"
        << ", timestamp = " << time_micro << " find lidar time "
        << find_lidar_time << " the lidar and image timestamp diff = "
        << std::abs(time_micro - find_lidar_time) / 1e3 << " ms";
  if (!find_range_image->HaveLidar()) {
    AINFO << "the raneimage has no matched lidar points";
    return;
  }
  points_visited_.clear();
  points_visited_.resize(find_range_image->points_ptr_->points.size(), false);
  std::vector<std::vector<cv::Point>> points;
  std::vector<cv::Point> proj_points;
  //   std::vector<std::vector<cv::Point>> rects;

  for (const auto& obj : object_list->objs) {
    const auto& cur_rect = obj->bbox;
    // range抠框
    Eigen::Vector3d camera_pt_left = camera_calib_ptr_->UVToCameraPt(
        cur_rect.tl().x, cur_rect.tl().y, obj->distance, false);
    Eigen::Vector3d veh_pt_left =
        camera_calib_ptr_->CameraPtToVehPt(camera_pt_left);

    Eigen::Vector3d camera_pt_right = camera_calib_ptr_->UVToCameraPt(
        cur_rect.br().x, cur_rect.br().y, obj->distance, false);
    Eigen::Vector3d veh_pt_right =
        camera_calib_ptr_->CameraPtToVehPt(camera_pt_right);

    cv::Point left_range, right_range;
    find_range_image->GetVehCoordinate(veh_pt_left.cast<float>(), left_range.x,
                                       left_range.y, true);
    find_range_image->GetVehCoordinate(veh_pt_right.cast<float>(),
                                       right_range.x, right_range.y, true);
    std::vector<std::pair<float, std::vector<int>>> buckets;
    // rects.push_back(std::vector<cv::Point>{left_range, right_range});

    float range_width = right_range.x - left_range.x,
          range_height = right_range.y - left_range.y;
    for (int i = left_range.x + range_width * 0.2;
         i < left_range.x + range_width * 0.8; ++i) {
      for (int j = left_range.y + range_height * 0.2;
           j < left_range.y + range_height * 0.8; ++j) {
        std::vector<int> point_index;
        find_range_image->GetPointIndex(j, i, point_index);
        for (const auto& idx : point_index) {
          if (points_visited_[idx]) {
            continue;
          }
          points_visited_[idx] = true;
          const auto& p = find_range_image->points_ptr_->points[idx];
          float dist = std::hypotf(p.x, p.y);
          bool has_bucket = false;
          for (auto& bucket : buckets) {
            if (std::abs(dist - bucket.first) < 0.7) {
              bucket.second.emplace_back(idx);
              has_bucket = true;
              break;
            }
          }
          if (!has_bucket) {
            buckets.emplace_back(std::make_pair(dist, std::vector<int>{idx}));
          }
        }
      }
    }
    std::sort(buckets.begin(), buckets.end());
    float first_tall = 0;
    float second_tall = 0;
    float first_tall_num = 0;
    float first_tall_dist = 0;
    bool is_near_obj = false, is_far_obj = false;
    if (cur_rect.height >= bgr_image_.rows ||
        (cur_rect.height >= 0.8 * bgr_image_.rows && cur_rect.tl().x <= 0)) {
      is_near_obj = true;
    }
    if (1.0 * cur_rect.height / cur_rect.width > 2 &&
        cur_rect.height < 0.7 * bgr_image_.rows &&
        (cur_rect.tl().y > 10 || cur_rect.height < 0.4 * bgr_image_.rows)) {
      is_far_obj = true;
    }
    float distance_limit = cover_state == COVER_STATE::NO_COVER ? 2 : 1;
    for (auto& bucket : buckets) {
      if (is_far_obj && bucket.first < 1) {
        continue;
      }
      if (std::abs(bucket.first - obj->distance) >
          std::max(0.2f * obj->distance, distance_limit)) {
        continue;
      }
      if (bucket.second.size() > std::min(right_range.y - left_range.y, 70) &&
          bucket.second.size() > 10) {
        float z_min = std::numeric_limits<float>::max(),
              z_max = std::numeric_limits<float>::min();
        float dist_sum = 0;
        for (const auto& idx : bucket.second) {
          const auto& p = find_range_image->points_ptr_->points[idx];
          dist_sum += std::hypotf(p.x, p.y);
          z_min = std::min(p.z, z_min);
          z_max = std::max(p.z, z_max);
        }
        float z_diff = z_max - z_min;
        if (first_tall_num > 500 &&
            bucket.second.size() < 0.5 * first_tall_num) {
          continue;
        }
        if (z_diff > first_tall) {
          if (first_tall > 0.001) {
            second_tall = first_tall;
          }
          first_tall_num = bucket.second.size();
          first_tall = z_diff;
          first_tall_dist = dist_sum / bucket.second.size();
        }
        // 近距离目标直接选高度较大的最近目标
        if ((is_near_obj && z_diff > 0.5) ||
            (bucket.first < 2 && z_diff > 0.5 &&
             cover_state == COVER_STATE::NO_COVER)) {
          break;
        }
      }
    }
    if (first_tall - second_tall > 0.5) {
      obj->distance = first_tall_dist;
      obj->observe_sensor = OBSERVE_SENSOR::LIDAR_OBSERVE;
    }

    points.emplace_back(std::vector<cv::Point>{left_range, right_range});
  }
  //   find_range_image->DrawImage(
  //       "/apollo/mydata/range_image/" + std::to_string(find_lidar_time) +
  //       ".jpg", rects, proj_points);
}
void MaskTrack::EstimateDist() {
  for (const auto& obj : object_list->objs) {
    const auto& cur_rect = obj->bbox;
    obj->observe_sensor = OBSERVE_SENSOR::CAMERA_OBSERVE;
    // 利用宽高估算距离
    Eigen::Vector3d estimate_pose_tall, estimate_pose_width;
    RectWithTallToPose(cur_rect, camera_calib_ptr_, template_3d_tall_,
                       estimate_pose_tall);
    RectWithWidthToPose(cur_rect, camera_calib_ptr_, size_3d_.width,
                        estimate_pose_width);
    float estimate_pose_tall_dist = estimate_pose_tall.norm();
    float estimate_pose_width_dist = estimate_pose_width.norm();
    // 近距离比例不对，取平均
    if (1.0 * cur_rect.height / cur_rect.width < 1.7 &&
        estimate_pose_tall_dist < 3.0 && estimate_pose_width_dist < 3.0) {
      obj->distance =
          (estimate_pose_tall_dist + estimate_pose_width_dist) * 0.5;
      continue;
    }
    // 估算非常接近，直接使用其中一个
    float similar_thresh = std::max(
        1.0, 0.1 * std::max(estimate_pose_tall_dist, estimate_pose_width_dist));
    if (estimate_pose_tall_dist > 1 && estimate_pose_width_dist > 1 &&
        std::abs(estimate_pose_tall_dist - estimate_pose_width_dist) <
            similar_thresh) {
      obj->distance =
          std::min(estimate_pose_tall_dist, estimate_pose_width_dist);
      continue;
    }
    bool tall_invalid = false, width_invalid = false;
    // 高度不可信
    if (cur_rect.tl().y <= 0 || cur_rect.br().y >= bgr_image_.rows) {
      tall_invalid = true;
    }
    // 宽度不可信 占据上下边界宽度也不可信
    if (cur_rect.tl().y <= 0 && cur_rect.br().y >= bgr_image_.rows) {
      width_invalid = false;
    } else if (cur_rect.tl().x <= 5 || cur_rect.br().x >= bgr_image_.cols - 5) {
      width_invalid = true;
    }
    // 位于角落，均不可信，直接估算为1.5m
    if (tall_invalid && width_invalid) {
      obj->distance = 1.5;
    } else if (tall_invalid) {
      obj->distance = estimate_pose_width_dist;
    } else if (width_invalid) {
      obj->distance = estimate_pose_tall_dist;
    } else {
      // 均可信
      obj->distance = estimate_pose_tall_dist;
    }
    obj->distance = std::max(1.0f, obj->distance);
  }
  for (const auto& obj : object_list->objs) {
    // 再用接地点校正一次
    if (obj->distance < 2 &&
        std::abs(obj->bbox.x - camera_calib_ptr_->GetWidth() * 0.5) >
            0.2 * camera_calib_ptr_->GetWidth() &&
        obj->bbox.br().y < 0.75 * camera_calib_ptr_->GetHeight()) {
      // 中心点theta
      Eigen::Vector3d camera_pose = camera_calib_ptr_->GetPos();
      float ground_z = -0.3;
      float camera_gound = camera_pose[2] - ground_z;
      cv::Point cur_center = (obj->bbox.tl() + obj->bbox.br()) * 0.5;
      Eigen::Vector3d ground_pt = camera_calib_ptr_->UVToCameraPt(
          cur_center.x, obj->bbox.br().y, obj->distance, false);
      Eigen::Vector3d veh_ground_pt =
          camera_calib_ptr_->CameraPtToVehPt(ground_pt);

      // 到地面点的差值，如果是
      float z_diff = veh_ground_pt[2] - ground_z;
      float dist_diff = z_diff / camera_gound * obj->distance;
      if (veh_ground_pt.z() < camera_gound && obj->distance + dist_diff > 1) {
        AINFO << " tall refine " << obj->distance << " " << dist_diff;
        obj->distance += dist_diff;
      }
    }
  }
  std::sort(object_list->objs.begin(), object_list->objs.end(),
            [](const Yolov8Object::Ptr& a, const Yolov8Object::Ptr& b) {
              if (a->distance < 0.5 || b->distance < 0.5) {
                return a->bbox.br().y > b->bbox.br().y;
              } else {
                return a->distance < b->distance;
              }
            });
}
void MaskTrack::ReInit() {
  inited_ = false;
  // 初始化量清空
  track_count = 0;
  rect_init_ = false;
  template_init_ = false;
  template_3d_tall_ = 1.7;
  sensor_id_ = trigger_sensor_;
  next_sensor_ = trigger_sensor_;
  // 状态量清空
  time_stamp_ = 0;
  history_.clear();
  track_state_ = TRACKER_STATE_CODE::NO_INIT;
  cover_state = COVER_STATE::NO_COVER;
  dist_sensor_ = OBSERVE_SENSOR::NO_OBSERVE;
  size_3d_ = BoxSize{0.7, 0.7, 1.7};
  predict_rect_ = cv::Rect();
  uwb_match_time_ = 0;
  // 特殊量清空
  lost_count = 0;
  full_area = 0;
  cutie_ptr_ = std::make_shared<CutieCore>();
  cutie_ptr_->Init(cutie_config);
  lost_count = 0;
  inited_ = true;
}
void MaskTrack::ChooseNextSensor() {
  if (!allow_sensor_change_ || cover_state == COVER_STATE::PART_COVER) {
    next_sensor_ = sensor_id_;
    return;
  }
  SensorID best_sensor = next_sensor_;
  float min_dist = bgr_image_.cols;
  float angle = std::atan2(center_base_.y(), center_base_.x()) * 180 / M_PI;
  for (const auto& sensor : track_calib_map_) {
    // 角度初筛
    if (kSensorIDToNameMap.at(sensor.first).find("RIGHT") !=
            std::string::npos &&
        (angle > 0 && std::abs(angle) < 160)) {
      continue;
    } else if (kSensorIDToNameMap.at(sensor.first).find("LEFT") !=
                   std::string::npos &&
               (angle < 0 && std::abs(angle) < 160)) {
      continue;
    } else if (kSensorIDToNameMap.at(sensor.first).find("FRONT") ==
                   std::string::npos &&
               std::abs(angle) < 30) {
      continue;
    } else if (kSensorIDToNameMap.at(sensor.first).find("FRONT") !=
                   std::string::npos &&
               std::abs(angle) > 90) {
      continue;
    }
    cv::Point point_base;
    Eigen::Vector3d veh_pt(center_base_.x(), center_base_.y(),
                           center_base_.z());
    Eigen::Vector3d camera_pt =
        sensor.second.camera_calib_ptr_->VehPtToCameraPt(veh_pt);
    if (!sensor.second.camera_calib_ptr_->CameraPtToUV(
            camera_pt, point_base.x, point_base.y, false, true)) {
      continue;
    }

    float dist = std::abs(point_base.x - bgr_image_.cols * 0.5);
    if (sensor.first != sensor_id_ &&
        (sensor.first == SensorID::CAM_FISHEYE_LEFT1 ||
         sensor.first == SensorID::CAM_FISHEYE_RIGHT1)) {
      dist *= 1.5;
    }
    if (sensor.first == sensor_id_) {
      dist *= 0.8;
    }
    if (dist < min_dist) {
      best_sensor = sensor.first;
      min_dist = dist;
    }
  }
  if (best_sensor != sensor_id_) {
    // 切换相机时将area置为小值（不能是0，否则会导致漏检时出问题）
    full_area = 100;
    need_flip_image_ = (!need_flip_image_);
    AINFO << " CHANGE SENSOR to " << kSensorIDToNameMap.at(best_sensor)
          << " flip " << need_flip_image_;
  }
  next_sensor_ = best_sensor;
}
}  // namespace perception
}  // namespace robot