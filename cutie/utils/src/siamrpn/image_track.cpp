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

#include "robot/perception/uwb_postprocess/image_track/image_track.h"

namespace robot {
namespace perception {
bool ImageTrackNN::Init(const YAML::Node& config, const SensorID sensor_id) {
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

  // config
  YAML::Node image_track_config = config["ImageTrack"];
  YAML::Node detection_config = config["Detection"];
  yolov8_detection_ptr_ = std::make_shared<Yolov8Detection>();
  if (!yolov8_detection_ptr_->Init(detection_config)) {
    AERROR << "Yolov8Detection init failed!";
    return false;
  }
  object_list = std::make_shared<Yolov8ObjectList>();
  object_list->objs.reserve(100);
  // get config
  YAML::Node track_node;
  rally::yamlSubNode(image_track_config, "track", track_node);
  rally::yamlRead(image_track_config, "show_image", show_image_);
  rally::yamlRead(image_track_config, "image_scale", image_scale_);
  rally::yamlRead(image_track_config, "track_mode", track_mode_);
  rally::yamlRead(image_track_config, "retrack_mode", retrack_mode_);
  rally::yamlRead(track_node, "enlarge_ratio", enlarge_ratio_);
  rally::yamlRead(track_node, "template_size", template_size_);
  rally::yamlRead(track_node, "search_size", search_size_);
  rally::yamlRead(track_node, "base_size", base_size_);
  rally::yamlRead(track_node, "penalty_k", penalty_k_);
  rally::yamlRead(track_node, "window_influence", window_influence_);
  rally::yamlRead(track_node, "smooth_ratio", smooth_ratio_);
  AINFO << " TRACK CONFIG INFO : " << "enlarge_ratio: " << enlarge_ratio_
        << " template_size: " << template_size_
        << " search_size: " << search_size_ << " base_size: " << base_size_
        << " show image " << show_image_ << " scale " << image_scale_;
  template_temp_vec_.resize(3 * template_size_ * template_size_);
  search_temp_vec_.resize(3 * search_size_ * search_size_);
  if (track_mode_ != 0) {
    UWBServer::GetInstance().input_data.command = COMMOND::START_TRACK;
  }

  // 生成remap
  input_image_K = camera_calib_ptr_->GetInnerCvMat().clone();
  UWBServer::GetInstance().ori_K = input_image_K;
  UWBServer::GetInstance().ori_D = camera_calib_ptr_->GetDistCoeffMatf();
  input_image_width = camera_calib_ptr_->GetWidth();
  input_image_height = camera_calib_ptr_->GetHeight();
  camera_calib_ptr_->ApplyScaleIntoK(image_scale_);
  bgr_image_ = cv::Mat(camera_calib_ptr_->GetHeight(),
                       camera_calib_ptr_->GetWidth(), CV_8UC3);
  bgr_image_size_ = bgr_image_.size();
  if (camera_calib_ptr_->GetDistCoeffMatf().rows == 4) {
    cv::fisheye::initUndistortRectifyMap(
        input_image_K, camera_calib_ptr_->GetDistCoeffMatf(), cv::noArray(),
        camera_calib_ptr_->GetInnerCvMat(), bgr_image_.size(), CV_16SC2, map1,
        map2);
  } else {
    cv::initUndistortRectifyMap(
        input_image_K, camera_calib_ptr_->GetDistCoeffMatf(), cv::noArray(),
        camera_calib_ptr_->GetInnerCvMat(), bgr_image_.size(), CV_16SC2, map1,
        map2);
  }

  // anchor
  YAML::Node anchor_node;
  int stride = 8;
  std::vector<float> ratios;
  std::vector<float> scales;
  rally::yamlSubNode(image_track_config, "anchor", anchor_node);
  rally::yamlRead(anchor_node, "anchors_num_", anchors_num_);
  rally::yamlRead(anchor_node, "stride", stride);
  rally::yamlRead(anchor_node, "ratios", ratios);
  rally::yamlRead(anchor_node, "scales", scales);
  feature_size_ = (search_size_ - template_size_) / stride + 1 + base_size_;
  // 生成惩罚窗口，越靠近特征中心权重越大
  cv::Mat hanning1d(feature_size_, 1, CV_32F);
  for (int i = 0; i < feature_size_; ++i) {
    hanning1d.at<float>(i, 0) =
        0.5f - 0.5f * cos(2.0f * CV_PI * i / (feature_size_ - 1));
  }
  cv::Mat hanning2d = hanning1d * hanning1d.t();
  hanning2d = hanning2d.reshape(1, 1);
  // 为每个anchor生成权重
  for (int i = 0; i < feature_size_; ++i) {
    for (int j = 0; j < hanning2d.cols; ++j) {
      window_weight_.emplace_back(hanning2d.at<float>(0, j));
    }
  }

  AINFO << " ANCHOR INFO: " << " anchors_num: " << anchors_num_
        << " stride: " << stride << " ratios: [" << ratios << "] scales: ["
        << scales << "]" << " feature_size: " << feature_size_;
  anchor_generator_ptr_ =
      std::make_shared<AnchorGenerator>(stride, ratios, scales);
  anchor_generator_ptr_->GenerateGridAnchors(feature_size_, anchors_);

  // net
  engine_ptr_ = robosense::inference::createInferEngine(
      robosense::inference::EngineType::kTENSORRT);
  const auto& level = robosense::inference::DebugLevel::kDEBUG;
  engine_ptr_->setDebugLevel(level);
#ifdef __aarch64__
  std::string engine_path =
      image_track_config["arm_model_path"].as<std::string>();
  init_options_.use_Unified_Address = true;
#else
  std::string engine_path =
      image_track_config["x86_model_path"].as<std::string>();
  init_options_.use_Unified_Address = false;
#endif
  init_options_.save_path = engine_path;
  init_options_.model_format = robosense::inference::ModelFormat::kENGINE;
  init_options_.batch_size = 1;
  init_options_.encrypt = image_track_config["encyrpt"].as<bool>();  // false
  init_options_.use_cuda_graph =
      image_track_config["use_cuda_graph"].as<bool>();  // false
  init_options_.use_gpu_input =
      image_track_config["use_gpu_input"].as<bool>();  // true
  engine_ptr_->init(init_options_);
  AINFO << "Set Cuda Stream Priority is: " << engine_ptr_->getStreamPriority();
  printModelInfo(engine_ptr_);

  // set task
  StartTrigger();
  inited_ = true;
  return true;
}
void ImageTrackNN::InitCameraSensor(
    const std::unordered_set<SensorID>& calib_sensor_list) {}
void ImageTrackNN::Perception(const std::shared_ptr<perception::Image>& msg) {
  //   if (msg->encoding != "yuyv422"||) {
  //     AWARN << " only support yuyv422 encoding image! ";
  //     return;
  //   }
  if (busy_for_init_) {
    AWARN << " Is busy for init, return! ";
    return;
  }
  busy_for_process_ = true;
  busy_for_image_ = true;
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
  busy_for_image_ = false;
  auto t_2 = apollo::cyber::Time::Now();
  object_list->time_stamp = msg_time.toSecond();
  // 更新预测
  Eigen::Affine3d base_to_rel = Eigen::Affine3d::Identity();
  UwbPostProcessEgoPoseInfo::GetInstance().GetBaseToRel(object_list->time_stamp,
                                                        base_to_rel);
  if (base_to_rel.matrix() == Eigen::Matrix4d::Identity()) {
    AERROR << " Not find base to rel!!!!";
    return;
  }
  base_to_rel_trans_ = base_to_rel.cast<float>();
  center_base_ = base_to_rel_trans_.inverse() * center_odom_;
  cv::Mat temp_image = bgr_image_.clone();
  predict_rect_ = Predict2DRect();

  yolov8_detection_ptr_->Perception(bgr_image_, object_list);
  // 利用图像宽高估计距离
  EstimateDist();
  // 利用图像宽高和点云估计距离
  FusionLidarPoints();
  time_stamp_ = msg_time.toSecond();
  auto t_3 = apollo::cyber::Time::Now();
  if (!rect_init_) {
    if (track_state_ == TRACKER_STATE_CODE::ERROR_TRACK) {
      AERROR
          << " !!!!!!!!!!!!!!!!!!!!!!! ERROR TRACK "
             "!!!!!!!!!!!!!!!!!!!!!!! OUT IMAGE TRACK AND INIT !!!!!!!!!!!!!";
    }
    if (track_mode_ == 0) {
      SetRectByOutside(bgr_image_);
    } else if (track_mode_ == 1) {
      SetRect(bgr_image_);
    } else if (track_mode_ == 3) {
      ChoosePed();
    }
  } else if (retrack_mode_ == 2 &&
             track_state_ == TRACKER_STATE_CODE::ERROR_TRACK) {
    AERROR << " !!!!!!!!!!!!!!!!!!!!!!! ERROR TRACK "
              "!!!!!!!!!!!!!!!!!!!!!!! OUT IMAGE TRACK AND RE SERACH "
              "!!!!!!!!!!!!!";
    GlobalReSearch();
  } else {
    Track(bgr_image_);
    // GlobalReSearch();
  }
  out_time_stamp_ = time_stamp_;
  busy_for_process_ = false;
  auto t_4 = apollo::cyber::Time::Now();
  std::stringstream info_input;
  info_input << "convert cost time: " << (t_2 - t_1).ToSecond() * 1000
             << " ms.";
  info_input << "detection cost time: " << (t_3 - t_2).ToSecond() * 1000
             << " ms.";
  info_input << "track cost time: " << (t_4 - t_3).ToSecond() * 1000 << " ms.";
  AINFO << "IMAGE TRACK info: " << info_input.str();
}

void ImageTrackNN::SetRect(const cv::Mat& ori_image) {
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
void ImageTrackNN::SetRectByOutside(const cv::Mat& ori_image) {
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
      float iou = GetIoU(obj->bbox, roi);
      if (iou > best_iou) {
        match_obj = obj;
        best_iou = iou;
      }
    }
    if (match_obj != nullptr) {
      InitRect(ori_image, roi.tl(), roi.br(), match_obj);
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
void ImageTrackNN::ChoosePed(const float dist, const float angle_base) {
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
  busy_for_init_ = false;
}
void ImageTrackNN::ChoosePed() {
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
bool ImageTrackNN::InitRect(const cv::Mat& ori_image, const cv::Point& left_up,
                            const cv::Point& right_down,
                            const Yolov8Object::Ptr& obj_ptr) {
  // 2D信息
  size_ = cv::Size(right_down.x - left_up.x, right_down.y - left_up.y);
  center_pt_ = cv::Point2f(left_up.x + (size_.width - 1) / 2.0f,
                           left_up.y + (size_.height - 1) / 2.0f);
  fill_colr_ = cv::mean(ori_image);
  SetDistance(obj_ptr->distance, time_stamp_, obj_ptr->observe_sensor);
  AINFO << "SIZE " << size_ << " CENTER " << center_pt_ << " COLOR"
        << fill_colr_;
  // 生成初始化跟踪模板
  int enlarge_size = static_cast<int>(GetEnlargeSize(size_));
  GetSubImage(ori_image, center_pt_, enlarge_size, template_size_,
              template_region_);
  template_height_ = size_.height;
  cv::Rect real_roi(left_up, right_down);
  GeneratePureTemplate(GetRectNoWait(), ori_image, template_size_,
                       pure_template_);
  template_image_ = pure_template_.clone();
  if (show_image_) {
    ShowTwoImage(template_region_, pure_template_);
  }

  rect_init_ = true;
  track_state_ = TRACKER_STATE_CODE::STABLE_TRACK;
  return true;
}
void ImageTrackNN::Track(const cv::Mat& image) {
  if (engine_ptr_ == nullptr) {
    AERROR << "engine ptr is null";
    return;
  }
  auto t_1 = apollo::cyber::Time::Now();
  // 以上一帧计算得到的当前帧网络bbox尺寸
  float enlarge_template_size = GetEnlargeSize(size_);
  // 模板进入网络的放缩倍数
  float scale_z = template_size_ / enlarge_template_size;
  float real_search_size =
      enlarge_template_size * (1.0f * search_size_ / template_size_);
  cv::Mat search_crop;
  GetSubImage(image, center_pt_, static_cast<int>(real_search_size),
              search_size_, search_crop);

  void* template_input_memory = engine_ptr_->getTensorAddress("template");
  void* search_input_memory = engine_ptr_->getTensorAddress("search");

  cv::Mat input_region = cover_state == COVER_STATE::FULL_COVER
                             ? pure_template_
                             : template_region_;

#ifdef __aarch64__
  MatToCHW(input_region, static_cast<float*>(template_input_memory));
  MatToCHW(search_crop, static_cast<float*>(search_input_memory));
#else
  MatToCHW(input_region, template_temp_vec_.data());
  MatToCHW(search_crop, search_temp_vec_.data());
  cudaMemcpy(template_input_memory, template_temp_vec_.data(),
             sizeof(float) * 3 * template_size_ * template_size_,
             cudaMemcpyHostToDevice);
  cudaMemcpy(search_input_memory, search_temp_vec_.data(),
             sizeof(float) * 3 * search_size_ * search_size_,
             cudaMemcpyHostToDevice);
#endif
  auto t_2 = apollo::cyber::Time::Now();
  engine_ptr_->forward();
  auto t_3 = apollo::cyber::Time::Now();
  pre_net_score_ = GetOutPut(image, scale_z);
  FusionDetetionResult();
  UpdateCoverState();
  // 如果尺寸发生较大变化且非边缘，更新模板搜索区域
  float height_ratio = static_cast<float>(size_.height) / template_height_;
  bool size_change = (height_ratio < 0.7 || height_ratio > 1.3) &&
                     pre_net_score_ > 0.99 &&
                     track_state_ == TRACKER_STATE_CODE::STABLE_TRACK &&
                     cover_state == COVER_STATE::NO_COVER;
  if ((size_change || track_state_ == TRACKER_STATE_CODE::RECOVER_TRACK) &&
      std::abs(size_3d_.height - template_3d_tall_) < 0.3 &&
      std::abs(size_3d_.width - 0.75) < 0.25 &&
      cover_state == COVER_STATE::NO_COVER) {
    cv::Rect target_rect = GetRectNoWait();
    cv::Scalar mean_color = cv::mean(image(target_rect));
    float dist = cv::norm(mean_color - template_color_);
    AINFO << " Change Template for " << size_change << " " << int(track_state_)
          << " " << height_ratio << " " << size_3d_.height << " "
          << template_3d_tall_ << " " << dist;

    int enlarge_size = static_cast<int>(GetEnlargeSize(size_));
    int cover_box = 0;
    cv::Rect enlarge_rect =
        cv::Rect(center_pt_.x - enlarge_size / 2,
                 center_pt_.y - enlarge_size / 2, enlarge_size, enlarge_size);
    for (const auto& obj : object_list->objs) {
      cv::Rect rect = obj->bbox;
      float ratio = GetAreaRatio(enlarge_rect, rect);
      if (ratio > 0.5 && std::abs(obj->distance - distance_) < 5) {
        cover_box++;
      }
    }
    if (cover_box <= 1 && dist < 50) {
      GetSubImage(image, center_pt_, enlarge_size, template_size_,
                  template_region_);
      template_height_ = size_.height;
      GeneratePureTemplate(GetRectNoWait(), image, template_size_,
                           pure_template_);
      track_state_ = TRACKER_STATE_CODE::STABLE_TRACK;
      if (show_image_) {
        ShowTwoImage(template_region_, pure_template_);
        AINFO << " Update template region!";
      }
    }
  }
  AddHistoryInfo();
  if (track_state_ == TRACKER_STATE_CODE::EDGE_LOST_TRACK) {
    AERROR << " !!!!!!!!!!!!!!!!!!!!!!! EDGE LOST TRACK "
              "!!!!!!!!!!!!!!!!!!!!!!! PLEASE CHECK TARGET STATE !!!!!!!!!!!!!";
    bool all_egde_lost = true;
    for (int i = 0; i < history_track_.size(); ++i) {
      if (history_track_[i].track_state !=
          TRACKER_STATE_CODE::EDGE_LOST_TRACK) {
        all_egde_lost = false;
      }
    }
    if (all_egde_lost && retrack_mode_ != 0) {
      track_state_ = TRACKER_STATE_CODE::ERROR_TRACK;
      track_count = 0;
      if (retrack_mode_ == 1) {
        rect_init_ = false;
      }
    }
  } else if ((track_state_ == TRACKER_STATE_CODE::RECOVER_TRACK ||
              track_state_ == TRACKER_STATE_CODE::LOW_SCORE_LOST_TRACK) &&
             cover_state == COVER_STATE::NO_COVER) {
    int invalid_track_count = 0;
    int after_cover_invalid_track = 0;
    bool has_cover = false;
    for (int i = 0; i < history_track_.size(); ++i) {
      if (history_track_[i].cover_state != COVER_STATE::NO_COVER) {
        has_cover = true;
      }
      if (!IsTrack(history_track_[i].track_state) ||
          (history_track_[i].track_state == TRACKER_STATE_CODE::RECOVER_TRACK &&
           history_track_[i].net_score < 0.7)) {
        invalid_track_count++;
      }
      if (has_cover) {
        if (history_track_[i].track_state != TRACKER_STATE_CODE::STABLE_TRACK) {
          ++after_cover_invalid_track;
        } else {
          after_cover_invalid_track = 0;
        }
      }
    }
    if (invalid_track_count >= history_track_.size()) {
      AERROR
          << " !!!!!!!!!!!!!!!!!!!!!!! ERROR TRACK "
             "!!!!!!!!!!!!!!!!!!!!!!! OUT IMAGE TRACK AND INIT !!!!!!!!!!!!!";
      if (retrack_mode_ != 0) {
        track_state_ = TRACKER_STATE_CODE::ERROR_TRACK;
        track_count = 0;
        if (retrack_mode_ == 1) {
          rect_init_ = false;
        }
      }
    }
  }
  auto t_4 = apollo::cyber::Time::Now();
  std::stringstream info_input;
  info_input << "siam pre cost time: " << (t_2 - t_1).ToSecond() * 1000
             << " ms.";
  info_input << "siam infer cost time: " << (t_3 - t_2).ToSecond() * 1000
             << " ms.";
  info_input << "siam post cost time: " << (t_4 - t_3).ToSecond() * 1000
             << " ms.";
  //   AINFO << " SiamRPN++: " << info_input.str();
  //   if (show_image_) {
  //     cv::Rect new_rect = GetRectNoWait();
  //     cv::rectangle(image, new_rect, cv::Scalar(0, 255, 0), 2);
  //     cv::imshow("track_result", image);
  //     //   int key;
  //     //   do {
  //     cv::waitKey(1);
  //     //   } while (key != 27 && key != 32 && key != 13 && key != 10);
  //     //   cv::destroyAllWindows();
  //   }
}
float ImageTrackNN::GetOutPut(const cv::Mat& ori_image,
                              const float scale_template) {
  void* cls_out =
      engine_ptr_->getOutputPtr("cls", robosense::inference::DeviceType::kCPU);
  void* loc_out =
      engine_ptr_->getOutputPtr("loc", robosense::inference::DeviceType::kCPU);
  ConvertScore(static_cast<float*>(cls_out), feature_size_, feature_size_);
  ConvertBBox(static_cast<float*>(loc_out), feature_size_, feature_size_,
              anchor_boxes_);
  // 预测最优box
  int best_predict_idx = -1;
  float best_iou = 0.3;
  for (int i = 0; i < object_list->objs.size(); ++i) {
    float cur_iou = GetIoU(predict_rect_, object_list->objs[i]->bbox);
    if (cur_iou > best_iou &&
        std::abs(object_list->objs[i]->distance - distance_) < 1) {
      best_predict_idx = i;
      best_iou = cur_iou;
    }
  }
  // 选择最优评分
  float best_score = 0, best_penalty = 0, best_net_score = 0;
  int best_idx = -1;
  for (int i = 0; i < cls_score_vec_.size(); ++i) {
    const auto& score = cls_score_vec_[i];
    const auto& box = anchor_boxes_[i];
    // 丢失状态要关联必须和行人框有iou
    float center_x = box[0] / scale_template + center_pt_.x;
    float center_y = box[1] / scale_template + center_pt_.y;
    float width = box[2] / scale_template;
    float height = box[3] / scale_template;
    cv::Rect rect(center_x - width / 2, center_y - height / 2, width, height);
    if (IsTrack(track_state_) && best_predict_idx != -1) {
      if (GetIoU(rect, object_list->objs[best_predict_idx]->bbox) < 0.1) {
        continue;
      }
    }
    float iou = 0;
    float nearest_dist = 0;
    for (const auto& obj : object_list->objs) {
      iou = GetIoU(rect, obj->bbox);
      if (iou > 0.2) {
        nearest_dist =
            std::min(std::abs(obj->distance - distance_), nearest_dist);
      }
    }
    if (iou < 0.1 && !IsTrack(track_state_)) {
      continue;
    }
    // 关联到某个目标，但距离相差非常大
    if (IsTrack(track_state_) && nearest_dist > 5) {
      continue;
    }

    auto ref_size = cv::Size(size_.width, size_.height);
    if (uwb_match_time_ - time_stamp_ > -1 &&
        std::abs(size_3d_.height - template_3d_tall_) < 0.1) {
      ref_size = cv::Size(predict_rect_.width, predict_rect_.height);
    }

    // 有distance输入且尺寸合理时要进行严格的高度惩罚
    float h_penalty = 1;
    float predict_height = predict_rect_.height * scale_template;
    if (distance_ > 1.0 && predict_rect_.height < ori_image.rows * 0.9 &&
        1.5 < size_3d_.height && size_3d_.height < 2.2) {
      float h_ratio = box[3] / (ref_size.height * scale_template);
      float h_change = std::max(h_ratio, 1.0f / h_ratio);
      float h_k = !IsTrack(track_state_) ? 3 : 2;
      if (cover_state == COVER_STATE::PART_COVER ||
          cover_state == COVER_STATE::SIMILAR_PART_COVER) {
        h_k = 5;
      } else if (cover_state == COVER_STATE::FULL_COVER) {
        h_k = 10;
      }
      h_penalty =
          std::exp(-h_k * (h_change - 1.0f));  // 越偏离，惩罚越大 0.8->0.7
    }
    // 输入与输出的尺寸变化比
    float size_change = GetScoreSize(box[2], box[3], 0.5) /
                        GetScoreSize(ref_size.width * scale_template,
                                     ref_size.height * scale_template, 0.5);
    // 输入与输出的长宽形状变化比
    float ratio_change =
        (box[2] / box[3]) / (ref_size.width * 1.0f / ref_size.height);
    // 校正到>1
    size_change = std::max(size_change, 1.0f / size_change);
    ratio_change = std::max(ratio_change, 1.0f / ratio_change);
    float penalty =
        std::exp(-(ratio_change * size_change - 1) * penalty_k_) * h_penalty;
    float real_score = score * penalty;
    real_score = real_score * (1 - window_influence_) +
                 window_weight_[i] * window_influence_;
    if (real_score > best_score) {
      // 增加角度惩罚
      if (distance_ > 1.0 && !IsTrack(track_state_)) {
        float angle_penalty = 1;
        float recover_x = box[0] / scale_template + center_pt_.x;
        float recover_y = box[1] / scale_template + center_pt_.y;
        Eigen::Vector3d camera_center = camera_calib_ptr_->UVToCameraPt(
            recover_x, recover_y, distance_, false);
        Eigen::Vector3d veh_center =
            camera_calib_ptr_->CameraPtToVehPt(camera_center);
        float angle = std::atan2(veh_center[1], veh_center[0]) * 180 / M_PI;
        float pre_angle =
            std::atan2(center_base_[1], center_base_[0]) * 180 / M_PI;
        if (std::abs(angle - pre_angle) > 10) {
          continue;
        } else if (std::abs(angle - pre_angle) > 2) {
          angle_penalty = std::exp(-std::abs(angle - pre_angle) / 15);
          real_score *= angle_penalty;
        }
        penalty *= angle_penalty;
        real_score *= angle_penalty;
      }
      if (real_score > best_score) {
        best_idx = i;
        best_score = real_score;
        best_penalty = penalty;
        best_net_score = score;
      }
    }
  }
  // 没有关联，只跟踪大小不跟踪位置
  if (best_idx == -1) {
    float predict_ratio = uwb_match_time_ - time_stamp_ > -1 ? 0.2 : 0;
    size_.width =
        size_.width * (1 - predict_ratio) + predict_rect_.width * predict_ratio;
    size_.height = size_.height * (1 - predict_ratio) +
                   predict_rect_.height * predict_ratio;
    AINFO << " NO MATCH BOX USE SMOOTH " << size_ << " " << predict_rect_;
    return 0;
  }
  std::array<float, 4> best_box = anchor_boxes_[best_idx];
  // 恢复到原始尺度
  for (auto& param : best_box) {
    param /= scale_template;
  }
  AINFO << "BEST BOX " << best_idx << " " << best_box << " penalty "
        << best_penalty << " real score " << best_score << " net score "
        << best_net_score;
  // 判断状态
  auto pre_state = track_state_;
  if (best_net_score < 0.5 &&
      ((center_pt_.x - size_.width * 0.5 <= 5 && predict_rect_.x < 50) ||
       (center_pt_.x + size_.width * 0.5 >= ori_image.cols - 5 &&
        predict_rect_.x + predict_rect_.width >= ori_image.cols - 50))) {
    track_state_ = TRACKER_STATE_CODE::EDGE_LOST_TRACK;
  } else if (IsTrack(track_state_) &&
             (best_net_score < 0.3 || best_penalty < 0.5)) {
    track_state_ = TRACKER_STATE_CODE::LOW_SCORE_LOST_TRACK;
  }
  if (!IsTrack(track_state_) && best_net_score > 0.7 && best_penalty > 0.7) {
    track_state_ = TRACKER_STATE_CODE::RECOVER_TRACK;
  }
  float smooth_weight = best_penalty * best_net_score * smooth_ratio_;
  // 恢复跟踪时，不使用平滑权重，直接使用评分结果
  if (track_state_ == TRACKER_STATE_CODE::RECOVER_TRACK) {
    smooth_weight = best_penalty * best_net_score;
  }
  float center_smooth = 0;
  // 边缘丢失尽量不移动中心点，因为从哪里丢失更容易从哪里出现
  if (track_state_ == TRACKER_STATE_CODE::EDGE_LOST_TRACK ||
      best_net_score < 0.2) {
    center_smooth = (1 - best_net_score * smooth_ratio_);
    if (best_net_score < 0.5 && (best_box[0] > 0.2 * ori_image.cols ||
                                 best_box[0] < 0.8 * ori_image.cols)) {
      center_smooth = 1;
    }
  }
  float center_x = best_box[0] * (1 - center_smooth) + center_pt_.x;
  float center_y = best_box[1] * (1 - center_smooth) + center_pt_.y;
  float width = size_.width * (1 - smooth_weight) + best_box[2] * smooth_weight;
  float height =
      size_.height * (1 - smooth_weight) + best_box[3] * smooth_weight;
  if (track_state_ == TRACKER_STATE_CODE::EDGE_LOST_TRACK &&
      pre_state == TRACKER_STATE_CODE::EDGE_LOST_TRACK) {
    float predict_ratio = predict_rect_.height / height;
    if (predict_ratio < 0.8 || predict_ratio > 1.2) {
      width = width * 0.1 + predict_rect_.width * 0.9;
      height = height * 0.1 + predict_rect_.height * 0.9;
    }
  }

  // 生成最终结果
  center_x = clamp(center_x, 0.0f, static_cast<float>(ori_image.cols));
  center_y = clamp(center_y, 0.0f, static_cast<float>(ori_image.rows));
  width = clamp(width, 0.0f, static_cast<float>(ori_image.cols));
  height = clamp(height, 0.0f, static_cast<float>(ori_image.rows));
  AINFO << " OUT " << center_x << " " << center_y << " " << width << " "
        << height << " ori box " << best_box << " smooth weight "
        << smooth_weight << " " << center_smooth << " state "
        << int(track_state_);

  // 更新结果
  center_pt_.x = center_x;
  center_pt_.y = center_y;
  size_.width = width;
  size_.height = height;
  return best_net_score;
}
void ImageTrackNN::GetSubImage(const cv::Mat& ori_image,
                               const cv::Point2f& center_pt,
                               const int& ori_size, const int& target_size,
                               cv::Mat& image_patch) {
  // 原始图像角点坐标
  float offset = (ori_size + 1) / 2.0f;
  int region_x_min = std::floor(center_pt.x - offset + 0.5);
  int region_x_max = region_x_min + ori_size - 1;
  int region_y_min = std::floor(center_pt.y - offset + 0.5);
  int region_y_max = region_y_min + ori_size - 1;
  // 超出抠取图像边界的填充尺寸
  int left_pad = int(std::max(0, -region_x_min));
  int top_pad = int(std::max(0, -region_y_min));
  int right_pad = int(std::max(0, region_x_max - ori_image.cols + 1));
  int bottom_pad = int(std::max(0, region_y_max - ori_image.rows + 1));
  // 将原始图像放在中间填充边缘
  cv::Mat padded;
  if (top_pad > 0 || bottom_pad > 0 || left_pad > 0 || right_pad > 0) {
    cv::copyMakeBorder(ori_image, padded, top_pad, bottom_pad, left_pad,
                       right_pad, cv::BORDER_CONSTANT, fill_colr_);
  } else {
    padded = ori_image;
  }
  // 抠取固定尺寸
  region_x_min = region_x_min + left_pad;
  region_y_min = region_y_min + top_pad;
  cv::Rect roi(region_x_min, region_y_min, ori_size, ori_size);
  image_patch = padded(roi).clone();
  if (target_size != ori_size) {
    cv::resize(image_patch, image_patch, cv::Size(target_size, target_size));
  }
}
void ImageTrackNN::ConvertScore(const float* input, int height, int width) {
  // 返回最后一层特征图每个点，针对每个不同尺度anchor的前后景得分
  // siamrpn++使用了5个anchor，所以输出为(1,2*5,25,25)
  // 假设输入 input 是 [C, H, W] 排布，即 NCHW 格式，N=1
  // 不同尺度anchor数量
  // 每层特征图的像素数625
  int spatial_size = height * width;
  int total_anchors = anchors_num_ * spatial_size;

  // 最终输出是从[3125,2]中抽出的一维前景得分
  cls_score_vec_.resize(total_anchors);

  //
  for (int i = 0; i < total_anchors; ++i) {
    // 每625个anchor属于同一个尺度
    int anchor_idx = i / spatial_size;
    // 当前尺度中拉平后所在特征图索引
    int spatial_idx = i % spatial_size;
    // 背景评分的通道数
    int back_score_channel = anchor_idx;
    int front_score_channel = anchors_num_ + anchor_idx;
    // 真实索引
    float back_score = input[back_score_channel * spatial_size + spatial_idx];
    float front_score = input[front_score_channel * spatial_size + spatial_idx];

    // 数值稳定的 softmax
    float max_val = std::max(back_score, front_score);
    float exp0 = std::exp(back_score - max_val);
    float exp1 = std::exp(front_score - max_val);
    float sum = exp0 + exp1;
    cls_score_vec_[i] = exp1 / sum;  // 类别1的概率
  }

  return;
}
void ImageTrackNN::ConvertBBox(const float* delta_input, int height, int width,
                               AnchorArray& output_boxes) {
  // 输出为1*20*25*25,即5个尺度每个anchor有4个偏移量
  int spatial_size = height * width;
  // 共有的候选框数量
  int box_count = anchors_num_ * spatial_size;
  assert(anchors_.size() == box_count);

  output_boxes.resize(box_count);

  // 原始存储中每5个通道（anchor_num）代表了一个分数
  const float* dx = delta_input + 0 * box_count;
  const float* dy = delta_input + 1 * box_count;
  const float* dw = delta_input + 2 * box_count;
  const float* dh = delta_input + 3 * box_count;

  for (int i = 0; i < box_count; ++i) {
    float anchor_cx = anchors_[i][0];
    float anchor_cy = anchors_[i][1];
    float anchor_w = anchors_[i][2];
    float anchor_h = anchors_[i][3];

    float pred_cx = dx[i] * anchor_w + anchor_cx;
    float pred_cy = dy[i] * anchor_h + anchor_cy;
    float pred_w = std::exp(dw[i]) * anchor_w;
    float pred_h = std::exp(dh[i]) * anchor_h;

    output_boxes[i] = {pred_cx, pred_cy, pred_w, pred_h};
  }
}
// h w c to c h w
// 目前没有做rgb转换，待确认
void ImageTrackNN::MatToCHW(const cv::Mat& image, float* data) {
  CV_Assert(image.type() == CV_8UC3);
  int channels = 3;
  int height = image.rows;
  int width = image.cols;

  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data[c * height * width + h * width + w] =
            static_cast<float>(image.at<cv::Vec3b>(h, w)[c]);
      }
    }
  }
}
// 利用上一帧的box和距离预测3D位置
void ImageTrackNN::SetDistance(const float distance, const double uwb_time,
                               const OBSERVE_SENSOR observe_sensor) {
  cv::Rect cur_rect = GetRectNoWait();
  Eigen::Vector3d center_base;
  float tall;
  float width;
  RectToPose(cur_rect, camera_calib_ptr_, distance, center_base, tall, width);
  center_base_ = center_base.cast<float>();
  center_odom_ = base_to_rel_trans_ * center_base_;
  distance_ = distance;
  predict_rect_ = Predict2DRect();
  dist_sensor_ = observe_sensor;
  // 纯相机深度不更新尺寸
  if (observe_sensor == OBSERVE_SENSOR::CAMERA_OBSERVE) {
    return;
  }
  // 激光和uwb深度都比较准确，更新尺寸及时间
  if (uwb_match_time_ < 1 || (IsTrack(track_state_) && pre_net_score_ > 0.99)) {
    if (!template_init_ && observe_sensor == OBSERVE_SENSOR::UWB_OBSERVE) {
      template_3d_tall_ = tall;
      template_init_ = true;
      AINFO << " Init template tall " << tall;
    }
    size_3d_.length = 1;
    size_3d_.width = width;
    size_3d_.height = tall;
    AINFO << " Change box info " << size_3d_.length << "," << size_3d_.width
          << "," << size_3d_.height;
  }
  uwb_match_time_ = uwb_time;
}
// 利用3D位置预测
void ImageTrackNN::SetCenterBase(const Eigen::Vector3f& optimize_center,
                                 const double uwb_time) {
  if (track_state_ == TRACKER_STATE_CODE::EDGE_LOST_TRACK ||
      track_state_ == TRACKER_STATE_CODE::ERROR_TRACK) {
    return;
  }
  center_base_ = optimize_center;
  center_odom_ = base_to_rel_trans_ * center_base_;

  uwb_match_time_ = uwb_time;
  predict_rect_ = Predict2DRect();
}
void ImageTrackNN::SetCenterOdom(const Eigen::Vector3f& optimize_center) {
  if (track_state_ == TRACKER_STATE_CODE::EDGE_LOST_TRACK ||
      track_state_ == TRACKER_STATE_CODE::ERROR_TRACK) {
    return;
  }
  center_odom_ = optimize_center;
  center_base_ = base_to_rel_trans_.inverse() * center_odom_;
}
cv::Rect ImageTrackNN::Predict2DRect() {
  cv::Rect rect;
  PoseToRect(center_base_.cast<double>(), size_3d_, camera_calib_ptr_, rect);
  return rect;
  //   // 如果左右两边都不在成像范围了，预测宽度不变，
  //   if (!camera_calib_ptr_->IsInCameraDetectionRange(camera_pt_left[0],
  //                                                    camera_pt_left[2]) &&
  //       !camera_calib_ptr_->IsInCameraDetectionRange(camera_pt_right[0],
  //                                                    camera_pt_right[2])) {
  //     // 左边缘，将左边缘拉到端点，同时宽度保持原样
  //     if (p_left.x < 0 || p_right.x < 0) {
  //       p_left.x = p_right.x - size_.width;
  //       p_right.x += size_.width;
  //       p_left.x = 0;
  //     }
  //     // 右边缘，将右边缘拉到端点，同时宽度保持原样
  //     else {
  //       p_right.x = p_left.x + size_.width;
  //       p_left.x -= size_.width;
  //       p_right.x = camera_calib_ptr_->GetWidth();
  //     }
  //   }return cv::Rect(p_left, p_right);
}
void ImageTrackNN::AddHistoryInfo() {
  HistoryTrackInfo info = {time_stamp_, pre_net_score_, track_state_,
                           cover_state, distance_};
  if (history_track_.size() >= 10) {
    history_track_.pop_front();
  }
  history_track_.emplace_back(info);
}
void ImageTrackNN::GeneratePureTemplate(cv::Rect rect, const cv::Mat& ori_image,
                                        const float size, cv::Mat& out_image) {
  rect &= cv::Rect(0, 0, ori_image.cols, ori_image.rows);
  cv::Mat temp_image = ori_image(rect).clone();
  float scale = std::min(size * 1.0 / rect.height, size * 1.0 / rect.width);
  int new_w = static_cast<int>(rect.width * scale);
  int new_h = static_cast<int>(rect.height * scale);
  if (new_w < 0 || new_h < 0 || new_w > size || new_h > size) {
    return;
  }
  template_target_ = temp_image;
  template_color_ = cv::mean(temp_image);
  cv::resize(temp_image, temp_image, cv::Size(new_w, new_h));
  out_image = cv::Mat(size, size, CV_8UC3, fill_colr_);
  int x_offset = (size - new_w) / 2;
  int y_offset = (size - new_h) / 2;
  temp_image.copyTo(out_image(cv::Rect(x_offset, y_offset, new_w, new_h)));
}
void ImageTrackNN::ShowTwoImage(const cv::Mat& image_1,
                                const cv::Mat& image_2) {
  cv::namedWindow("image_1", cv::WINDOW_NORMAL);
  cv::namedWindow("image_2", cv::WINDOW_NORMAL);

  cv::imshow("image_1", image_1);
  cv::imshow("image_2", image_2);

  // 设置窗口大小（可选）
  cv::resizeWindow("image_1", 400, 400);
  cv::resizeWindow("image_2", 400, 400);

  // 设置窗口位置，假设屏幕左上角(0,0)
  cv::moveWindow("image_1", 100, 100);
  cv::moveWindow("image_2", 520, 100);

  //   int key;
  //   do {
  cv::waitKey(1);
  //   } while (key != 27 && key != 32 && key != 13 && key != 10);
  //   cv::destroyAllWindows();
}
void ImageTrackNN::FusionDetetionResult() {
  int best_idx = -1;
  const auto& cur_rect = GetRectNoWait();
  for (int i = 0; i < object_list->objs.size(); i++) {
    const auto& obj = object_list->objs[i];
    const auto& det_rect = obj->bbox;
    float iou = GetIoU(obj->bbox, cur_rect);
    if (iou > 0.01) {
      if (best_idx == -1) {
        best_idx = i;
      } else {
        return;
      }
    }
  }
  if (best_idx != -1) {
    const auto& det = object_list->objs[best_idx];
    if (uwb_match_time_ - time_stamp_ < 0 &&
        std::abs(distance_ - det->distance) < 2) {
      SetDistance(det->distance, time_stamp_, det->observe_sensor);
    }
    if (cover_state == COVER_STATE::NO_COVER &&
        std::abs(distance_ - det->distance) < 2) {
      if (distance_ > 10 || (size_3d_.height > template_3d_tall_ - 0.3 &&
                             size_3d_.height < template_3d_tall_ + 0.3)) {
        center_pt_ = (det->bbox.br() + det->bbox.tl()) * 0.5;
        size_ = cv::Size(det->bbox.width, det->bbox.height);
      }
    }
  } else {
    Eigen::Vector3d estimate_pose_tall;
    RectWithTallToPose(cur_rect, camera_calib_ptr_, size_3d_.height,
                       estimate_pose_tall);
    SetDistance(estimate_pose_tall.norm(), time_stamp_,
                OBSERVE_SENSOR::NO_OBSERVE);
  }
}
void ImageTrackNN::FusionLidarPoints() {
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
          bool has_bucket = false;
          for (auto& bucket : buckets) {
            if (std::abs(p.x - bucket.first) < 0.7) {
              bucket.second.emplace_back(idx);
              has_bucket = true;
              break;
            }
          }
          if (!has_bucket) {
            buckets.emplace_back(std::make_pair(p.x, std::vector<int>{idx}));
          }
        }
      }
    }
    float first_tall = 0;
    float second_tall = 0;
    float first_tall_dist = 0;
    for (auto& bucket : buckets) {
      if (std::abs(bucket.first - obj->distance) > 0.2 * obj->distance) {
        continue;
      }
      if (bucket.second.size() > right_range.y - left_range.y &&
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
        if (z_diff > first_tall) {
          if (first_tall > 0.001) {
            second_tall = first_tall;
          }
          first_tall = z_diff;
          first_tall_dist = dist_sum / bucket.second.size();
        }
      }
    }
    if (first_tall - second_tall > 0.5) {
      obj->distance = first_tall_dist;
      obj->observe_sensor = OBSERVE_SENSOR::LIDAR_OBSERVE;
    }

    points.emplace_back(std::vector<cv::Point>{left_range, right_range});
  }
  // std::string save_dir = "/apollo/mydata/range_image/";
  // rally::ensureDirectory(save_dir);
  // find_range_image->DrawImage(
  //     save_dir + std::to_string(find_lidar_time) + ".jpg", points,
  //     proj_points);
}
void ImageTrackNN::EstimateDist() {
  for (const auto& obj : object_list->objs) {
    const auto& cur_rect = obj->bbox;
    obj->observe_sensor = OBSERVE_SENSOR::CAMERA_OBSERVE;
    // 利用宽高估算距离
    Eigen::Vector3d estimate_pose_tall, estimate_pose_width;
    RectWithTallToPose(cur_rect, camera_calib_ptr_, template_3d_tall_,
                       estimate_pose_tall);
    RectWithWidthToPose(cur_rect, camera_calib_ptr_, size_3d_.width,
                        estimate_pose_width);
    // 估算非常接近，直接使用其中一个
    float estimate_pose_tall_dist = estimate_pose_tall.norm();
    float estimate_pose_width_dist = estimate_pose_width.norm();
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
    // 宽度不可信
    if (cur_rect.tl().x <= 5 || cur_rect.br().x >= bgr_image_.cols - 5) {
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
  std::sort(object_list->objs.begin(), object_list->objs.end(),
            [](const Yolov8Object::Ptr& a, const Yolov8Object::Ptr& b) {
              if (a->distance < 0.5 || b->distance < 0.5) {
                return a->bbox.br().y > b->bbox.br().y;
              } else {
                return a->distance < b->distance;
              }
            });
}
// 和原始模板判断是否匹配
float ImageTrackNN::MatchScore(const cv::Rect& rect) {
  cv::Mat temp_search;
  float enlarge_template_size = GetEnlargeSize(rect.size());
  float scale_template = template_size_ / enlarge_template_size;
  // 模板进入网络的放缩倍数
  float real_search_size =
      enlarge_template_size * (1.0f * search_size_ / template_size_);
  GetSubImage(bgr_image_, (rect.br() + rect.tl()) * 0.5,
              static_cast<int>(real_search_size), search_size_, temp_search);
  //   cv::imshow("second_search", temp_search);
  //   cv::waitKey(1e3);
  void* template_input_memory = engine_ptr_->getTensorAddress("template");
  void* search_input_memory = engine_ptr_->getTensorAddress("search");
#ifdef __aarch64__
  MatToCHW(pure_template_, static_cast<float*>(template_input_memory));
  MatToCHW(temp_search, static_cast<float*>(search_input_memory));
#else
  MatToCHW(pure_template_, template_temp_vec_.data());
  MatToCHW(temp_search, search_temp_vec_.data());
  cudaMemcpy(template_input_memory, template_temp_vec_.data(),
             sizeof(float) * 3 * template_size_ * template_size_,
             cudaMemcpyHostToDevice);
  cudaMemcpy(search_input_memory, search_temp_vec_.data(),
             sizeof(float) * 3 * search_size_ * search_size_,
             cudaMemcpyHostToDevice);
#endif

  engine_ptr_->forward();
  void* cls_out =
      engine_ptr_->getOutputPtr("cls", robosense::inference::DeviceType::kCPU);
  void* loc_out =
      engine_ptr_->getOutputPtr("loc", robosense::inference::DeviceType::kCPU);
  ConvertScore(static_cast<float*>(cls_out), feature_size_, feature_size_);
  ConvertBBox(static_cast<float*>(loc_out), feature_size_, feature_size_,
              anchor_boxes_);
  // 选择最优评分
  float best_score = 0;
  for (int i = 0; i < cls_score_vec_.size(); ++i) {
    const auto& score = cls_score_vec_[i];
    const auto& box = anchor_boxes_[i];
    float width = box[2] / scale_template;
    float height = box[3] / scale_template;
    if (std::abs(1 - width * 1.0f / rect.width) > 0.2 ||
        std::abs(1 - height * 1.0f / rect.height) > 0.2) {
      continue;
    }
    best_score = std::max(score, best_score);
  }
  return best_score;
}
void ImageTrackNN::GlobalReSearch() {
  float best_score = 0;
  float second_score = 0;
  int best_idx = -1;
  for (int i = 0; i < object_list->objs.size(); ++i) {
    const auto& obj = object_list->objs[i];
    const auto& cur_rect = obj->bbox;
    float score = MatchScore(cur_rect);
    if (best_score < score) {
      if (best_idx != -1) {
        second_score = best_score;
      }
      best_score = score;
      best_idx = i;
    }
    AINFO << " CUR RECT SCORE " << cur_rect << " score " << score;
  }
  if (track_count == 0) {
    if (best_idx != -1 && best_score - second_score > 0.5 && best_score > 0.9) {
      track_rect = object_list->objs[best_idx]->bbox;
      ++track_count;
    }
  } else {
    if (best_idx == -1 || best_score < 0.5 || best_score - second_score < 0.5) {
      track_count = 0;
    } else {
      cv::Rect cur_rect = object_list->objs[best_idx]->bbox;
      if (GetIoU(cur_rect, track_rect) > 0.5) {
        ++track_count;
      } else {
        track_count = 0;
      }
    }
  }
  if (track_count >= 3) {
    InitRect(bgr_image_, track_rect.tl(), track_rect.br(),
             object_list->objs[best_idx]);
  }
}
void ImageTrackNN::UpdateCoverState() {
  if (distance_ < 2) {
    cover_state = COVER_STATE::NO_COVER;
    return;
  }
  cv::Rect cur_rect = GetRectNoWait();
  int cover_count = 0;
  int similar_cover_count = 0;
  for (auto& obj : object_list->objs) {
    float iou = GetIoU(cur_rect, obj->bbox);
    float iou2 = GetAreaRatio(obj->bbox, cur_rect);
    if (iou > 0.1 || iou2 > 0.1) {
      ++cover_count;
      cv::Rect temp_rect =
          (obj->bbox & cv::Rect(0, 0, bgr_image_.cols, bgr_image_.rows));
      cv::Scalar color = cv::mean(bgr_image_(temp_rect));
      if ((cv::norm(color - template_color_) < 50 && iou2 > 0.3) ||
          cover_state == COVER_STATE::SIMILAR_PART_COVER) {
        ++similar_cover_count;
      }
    }
  }
  // 相似遮挡需要10帧才会恢复正常状态
  if (similar_cover_count >= 2) {
    cover_state = COVER_STATE::SIMILAR_PART_COVER;
    similar_cover_frame = 0;
    AINFO << " COVER STATE IS SIMILAR_PART_COVER";
    return;
  } else if (cover_count >= 2) {
    cover_state = COVER_STATE::PART_COVER;
    AINFO << " COVER STATE IS PART COVER";
    return;
  }
  if (cover_count == 1 && cover_state != COVER_STATE::NO_COVER) {
    if (cover_state == COVER_STATE::SIMILAR_PART_COVER &&
        similar_cover_frame <= 10) {
      ++similar_cover_frame;
      AINFO << " COVER STATE IS HOLD SIMILAR_PART_COVER" << " "
            << similar_cover_frame;
      return;
    }
    cv::Scalar cur_color = cv::mean(bgr_image_(cur_rect));
    if (cv::norm(cur_color - template_color_) > 50) {
      cover_state = COVER_STATE::FULL_COVER;
      AINFO << " COVER STATE IS FULL COVER";
      return;
    }
  }
  cover_state = COVER_STATE::NO_COVER;
}
void ImageTrackNN::ReInit() {
  // 初始化量清空
  track_count = 0;
  rect_init_ = false;
  template_init_ = false;
  template_3d_tall_ = 1.7;
  // 状态量清空
  time_stamp_ = 0;
  history_track_.clear();
  track_state_ = TRACKER_STATE_CODE::NO_INIT;
  cover_state = COVER_STATE::NO_COVER;
  dist_sensor_ = OBSERVE_SENSOR::NO_OBSERVE;
  size_3d_ = BoxSize{0.7, 0.7, 1.7};
  predict_rect_ = cv::Rect();
  uwb_match_time_ = 0;
  // 特殊量清空
  similar_cover_frame = 0;
}
}  // namespace perception
}  // namespace robot