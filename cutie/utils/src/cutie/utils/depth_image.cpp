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

#include "robot/perception/uwb_postprocess/common/depth_image.h"
namespace robot {
namespace perception {
bool DepthImage::Init(const SensorID sensor_id, const YAML::Node& config) {
  sensor_id_ = sensor_id;
  rally::yamlRead(config, "max_distance", max_distance_);
  rally::yamlRead(config, "invalid_value", invalid_value_);
  rally::yamlRead(config, "width", width_);
  rally::yamlRead(config, "height", height_);

  // 获取虚拟内参
  virtual_k_ = cv::Mat(3, 3, CV_32F);
  std::vector<float> virtual_k_vec = {254.647913291, 0, 400, 0, 254.647913291,
                                      400,           0, 0,   1};
  rally::yamlRead(config, "virtual_k", virtual_k_vec);
  for (int i = 0; i < virtual_k_vec.size(); i++) {
    virtual_k_.at<float>(i / 3, i % 3) = virtual_k_vec[i];
  }
  // calib
  remap_info_ptr = std::make_shared<RemapInfo>();
  if (!remap_info_ptr->Init(
          sensor_id,
          std::dynamic_pointer_cast<CameraCalib>(
              CalibCenter::GetInstance().Get(static_cast<uint8_t>(sensor_id))),
          virtual_k_, width_, height_, RemapModel::FISHEYE_ORIGIN)) {
    return false;
  }
  return true;
}
perception::Image ::Ptr DepthImage::GetDepthImage(const double timestamp) {
  // 查找时间戳最近的depth image
  if (depth_image_queue_.empty()) {
    return nullptr;
  }
  AINFO<<" CUR DEPTH INFO "<<depth_image_queue_.size();
  for (auto& depth_image : depth_image_queue_) {
    rally::Time depth_time(depth_image->header.time);
    AINFO << "CHECK DEPTH " << std::fixed << " " << depth_time.toSecond() << " "
          << timestamp << " diff "
          << std::abs(depth_time.toSecond() - timestamp);
    if (std::abs(depth_time.toSecond() - timestamp) < 0.1) {
      return depth_image;
    }
  }
  return nullptr;
}
bool DepthImage::SaveDepthImage(const perception::Image::Ptr depth_image) {
  if (depth_image_queue_.size() > 5) {
    depth_image_queue_.pop_front();
  }
  depth_image_queue_.emplace_back(depth_image);
  return true;
}

// 初始化一个depth image
bool DepthImageManager::AddSensor(const SensorID sensor_id,
                                  const YAML::Node& config) {
  DepthImage::Ptr new_depth_image = std::make_shared<DepthImage>();
  if (!new_depth_image->Init(sensor_id, config)) {
    AERROR << "DepthImage Init Failed! " << kSensorIDToNameMap.at(sensor_id);
    return false;
  }
  auto& pair = depth_image_map_[sensor_id];
  pair.second = new_depth_image;
  return true;
}
// 获取指定sensor对应时间的depth image
perception::Image::Ptr DepthImageManager::GetDepthImage(
    const SensorID sensor_id, const double timestamp) {
  if (depth_image_map_.find(sensor_id) == depth_image_map_.end()) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(depth_image_map_.at(sensor_id).first);
  return depth_image_map_.at(sensor_id).second->GetDepthImage(timestamp);
}
RemapInfo::Ptr DepthImageManager::GetRemapInfo(const SensorID sensor_id) {
  if (depth_image_map_.find(sensor_id) == depth_image_map_.end()) {
    return nullptr;
  }
  return depth_image_map_.at(sensor_id).second->GetRemapInfo();
}
// 天假指定depth image
bool DepthImageManager::SaveDepthImage(
    const SensorID sensor_id, const perception::Image::Ptr depth_image) {
  if (depth_image_map_.find(sensor_id) == depth_image_map_.end()) {
    return false;
  }
  std::lock_guard<std::mutex> lock(depth_image_map_.at(sensor_id).first);
  return depth_image_map_.at(sensor_id).second->SaveDepthImage(depth_image);
}

}  // namespace perception
}  // namespace robot