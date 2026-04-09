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

#include "robot/perception/uwb_postprocess/cutie/segment.h"
namespace robot {
namespace perception {
bool Segment::Init(const YAML::Node& config,
                   const MemoryManager::Ptr memory_manager, const int height,
                   const int width) {
  input_height_ = height;
  input_width_ = width;
  memory_manager_ = memory_manager;
  // memory net
  memory_engine_ptr_ = robosense::inference::createInferEngine(
      robosense::inference::EngineType::kTENSORRT);
  const auto& level = robosense::inference::DebugLevel::kDEBUG;
  memory_engine_ptr_->setDebugLevel(level);
#ifdef __aarch64__
  std::string memory_engine_path =
      config["memory_arm_model_path"].as<std::string>();
  memory_options_.use_Unified_Address = true;
#else
  std::string memory_engine_path =
      config["memory_x86_model_path"].as<std::string>();
  memory_options_.use_Unified_Address = false;
#endif
  memory_options_.save_path = memory_engine_path;
  memory_options_.model_format = robosense::inference::ModelFormat::kENGINE;
  memory_options_.batch_size = 1;
  memory_options_.encrypt = false;         // false
  memory_options_.use_cuda_graph = false;  // false
  memory_options_.use_gpu_input = true;    // true
  memory_engine_ptr_->init(memory_options_);
  AWARN << "Cutie Memory Fusion Model Info: ";
  printModelInfo(memory_engine_ptr_);

  // segment net
  segment_engine_ptr_ = robosense::inference::createInferEngine(
      robosense::inference::EngineType::kTENSORRT);
  segment_engine_ptr_->setDebugLevel(level);
#ifdef __aarch64__
  std::string segment_engine_path =
      config["segment_arm_model_path"].as<std::string>();
  segment_options_.use_Unified_Address = true;
#else
  std::string segment_engine_path =
      config["segment_x86_model_path"].as<std::string>();
  segment_options_.use_Unified_Address = false;
#endif
  segment_options_.save_path = segment_engine_path;
  segment_options_.model_format = robosense::inference::ModelFormat::kENGINE;
  segment_options_.batch_size = 1;
  segment_options_.encrypt = false;        // false
  segment_options_.use_cuda_graph = true;  // false
  segment_options_.use_gpu_input = true;   // true
  segment_engine_ptr_->init(segment_options_);
  AWARN << "Cutie Segment Model Info: ";
  printModelInfo(segment_engine_ptr_);
  return true;
}
void Segment::SetAddress(float* pix_feat, float* key, float* selection,
                         float* f8, float* f4) {
  /*memory*/
  mem_key_address_ =
      static_cast<float*>(memory_engine_ptr_->getTensorAddress("mem_key"));
  mem_shrinkage_address_ = static_cast<float*>(
      memory_engine_ptr_->getTensorAddress("mem_shrinkage"));
  mem_value_address_ =
      static_cast<float*>(memory_engine_ptr_->getTensorAddress("mem_value"));
  sensory_address_ =
      static_cast<float*>(memory_engine_ptr_->getTensorAddress("sensory"));
  obj_mem_address_ =
      static_cast<float*>(memory_engine_ptr_->getTensorAddress("obj_mem"));
  memory_engine_ptr_->setTensorAddress("pix_feat", pix_feat);
  memory_engine_ptr_->setTensorAddress("key", key);
  memory_engine_ptr_->setTensorAddress("selection", selection);
  readout_memory_address_ = static_cast<float*>(
      memory_engine_ptr_->getTensorAddress("readout_memory"));
  usage_addresss_ =
      static_cast<float*>(memory_engine_ptr_->getTensorAddress("usage"));
  /*segment*/
  segment_engine_ptr_->setTensorAddress("f8", f8);
  segment_engine_ptr_->setTensorAddress("f4", f4);
  segment_engine_ptr_->setTensorAddress("readout_memory",
                                        readout_memory_address_);
  // 和memory融合共享sensory_address_
  segment_engine_ptr_->setTensorAddress("sensory", sensory_address_);
  sensory_out_address_ =
      static_cast<float*>(segment_engine_ptr_->getTensorAddress("sensory_out"));
  net_mask_address_ =
      static_cast<float*>(segment_engine_ptr_->getTensorAddress("net_mask"));
  pred_prob_wight_bg_address_ = static_cast<float*>(
      segment_engine_ptr_->getTensorAddress("pred_prob_with_bg"));
  mask_address_ =
      static_cast<float*>(segment_engine_ptr_->getTensorAddress("mask"));
  //   net_mask_address直接作为memory_engine_ptr的输入
  memory_engine_ptr_->setTensorAddress("last_mask", net_mask_address_);
}
void Segment::Perception(bool save_sensory) {
  float* mem_key = memory_manager_->GetKey();
  float* mem_shrinkage = memory_manager_->GetShrinkage();
  float* mem_value = memory_manager_->GetValue();
  float* sensory = memory_manager_->GetSensoryMemory();
  float* obj_mem = memory_manager_->GetObjValueMemory();
  CopyToGpu(mem_key_address_, mem_key, memory_manager_->GetKeySize());
  CopyToGpu(mem_shrinkage_address_, mem_shrinkage,
            memory_manager_->GetShrinkageSize());
  CopyToGpu(mem_value_address_, mem_value, memory_manager_->GetValueSize());
  CopyToGpu(sensory_address_, sensory, memory_manager_->GetSensoryMemorySize());
  CopyToGpu(obj_mem_address_, obj_mem,
            memory_manager_->GetObjValueMemorySize());
  auto t_0 = apollo::cyber::Time::Now();
  memory_engine_ptr_->forward();
  float* out_usage = static_cast<float*>(memory_engine_ptr_->getOutputPtr(
      "usage", robosense::inference::DeviceType::kCPU));
  memory_manager_->UpdateUsage(out_usage);
  //   if(memory_manager_->GetLongMemory()->temp_slot_end_idx>0){
  //     SaveModelInfo(memory_engine_ptr_,
  //                   "/apollo/robot_dog_modules/perception/text_result");
  //   }
  auto t_1 = apollo::cyber::Time::Now();
  segment_engine_ptr_->forward();
  auto t_2 = apollo::cyber::Time::Now();
  AINFO << "memory fusion cost time: " << (t_1 - t_0).ToSecond() * 1000
        << " ms.";
  AINFO << "segment cost time: " << (t_2 - t_1).ToSecond() * 1000 << " ms.";
  if (save_sensory) {
    memory_manager_->UpdateSensory(sensory_out_address_);
  }
}
void Segment::GetMaskImage(cv::Mat& mask_image) {
  CV_Assert(mask_image.type() == CV_8UC1 &&
            mask_image.size() == cv::Size(input_width_, input_height_));
  int32_t* out_mask = static_cast<int32_t*>(segment_engine_ptr_->getOutputPtr(
      "mask", robosense::inference::DeviceType::kCPU));
  for (int i = 0; i < input_height_; ++i) {
    for (int j = 0; j < input_width_; ++j) {
      mask_image.at<uint8_t>(i, j) = out_mask[i * input_width_ + j];
    }
  }
}
}  // namespace perception
}  // namespace robot