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

#include "robot/perception/uwb_postprocess/cutie/image_encoder.h"
namespace robot {
namespace perception {
bool ImageEncoder::Init(const YAML::Node &config) {
  // net
  engine_ptr_ = robosense::inference::createInferEngine(
      robosense::inference::EngineType::kTENSORRT);
  const auto &level = robosense::inference::DebugLevel::kDEBUG;
  engine_ptr_->setDebugLevel(level);
#ifdef __aarch64__
  std::string engine_path = config["arm_model_path"].as<std::string>();
  init_options_.use_Unified_Address = true;
#else
  std::string engine_path = config["x86_model_path"].as<std::string>();
  init_options_.use_Unified_Address = false;
#endif
  init_options_.save_path = engine_path;
  init_options_.model_format = robosense::inference::ModelFormat::kENGINE;
  init_options_.batch_size = 1;
  init_options_.encrypt = false;         // false
  init_options_.use_cuda_graph = true;  // false
  init_options_.use_gpu_input = true;    // true
  engine_ptr_->init(init_options_);
  AWARN << "Cutie Image Encoder Model Info: ";
  printModelInfo(engine_ptr_);
  // 设置内存
  input_image_address_ =
      static_cast<float *>(engine_ptr_->getTensorAddress("input_image"));
  f16_address_ = static_cast<float *>(engine_ptr_->getTensorAddress("f16"));
  f8_address_ = static_cast<float *>(engine_ptr_->getTensorAddress("f8"));
  f4_address_ = static_cast<float *>(engine_ptr_->getTensorAddress("f4"));
  pix_feat_address_ =
      static_cast<float *>(engine_ptr_->getTensorAddress("pix_feat"));
  key_address_ = static_cast<float *>(engine_ptr_->getTensorAddress("key"));
  shrinkage_address_ =
      static_cast<float *>(engine_ptr_->getTensorAddress("shrinkage"));
  selection_address_ =
      static_cast<float *>(engine_ptr_->getTensorAddress("selection"));

  return true;
}
void ImageEncoder::Perception(const cv::Mat &chw_image) {
  CopyToGpu(input_image_address_, (float *)chw_image.data, chw_image.total());
  auto t_1 = apollo::cyber::Time::Now();
  engine_ptr_->forward();
  auto t_2 = apollo::cyber::Time::Now();
  AINFO << "encoder infer cost time: " << (t_2 - t_1).ToSecond() * 1000
        << " ms.";
}
}  // namespace perception
}  // namespace robot