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

#include "robot/perception/uwb_postprocess/cutie/cutie_core.h"
namespace robot {
namespace perception {
bool CutieCore::Init(const YAML::Node& config) {
  // param
  rally::yamlRead(config, "show_image", show_image);
  rally::yamlRead(config, "check_mem", check_mem_);
  rally::yamlRead(config, "image_width", input_image_width_);
  rally::yamlRead(config, "image_height", input_image_height_);
  YAML::Node memory_config, model_config;
  rally::yamlSubNode(config, "MemoryConfig", memory_config);
  rally::yamlSubNode(config, "ModelConfig", model_config);
  // image encoder
  YAML::Node image_encoder_node;
  rally::yamlSubNode(model_config, "ImageEncoder", image_encoder_node);
  image_encoder_ = std::make_shared<ImageEncoder>();
  if (!image_encoder_->Init(image_encoder_node)) {
    AERROR << "Cutie Image Encoder Init Failed!";
    return false;
  }
  // memory manager
  memory_manager_ = std::make_shared<MemoryManager>();
  if (!memory_manager_->Init(memory_config, input_image_width_,
                             input_image_height_)) {
    AERROR << "Cutie Memory Manager Init Failed!";
    return false;
  }
  // segment
  YAML::Node segment_node;
  rally::yamlSubNode(model_config, "Segment", segment_node);
  segment_ = std::make_shared<Segment>();
  if (!segment_->Init(segment_node, memory_manager_, input_image_height_,
                      input_image_width_)) {
    AERROR << "Cutie Segment Init Failed!";
    return false;
  }
  segment_->SetAddress(
      image_encoder_->GetPixFeatAddress(), image_encoder_->GetKeyAddress(),
      image_encoder_->GetSelectionAddress(), image_encoder_->GetF8Address(),
      image_encoder_->GetF4Address());
  memory_manager_->SetAddress(image_encoder_->GetInputImageAddress(),
                              image_encoder_->GetPixFeatAddress(),
                              segment_->GetNetMaskAddress());
  return true;
}
void CutieCore::InitMask(const cv::Mat& image, const cv::Mat& init_mask) {
  ori_image_widh_ = image.cols;
  ori_image_height_ = image.rows;
  CV_Assert(init_mask.type() == CV_8UC1);
  cv::Mat resize_mask;
  cv::resize(init_mask, resize_mask,
             cv::Size(input_image_width_, input_image_height_),
             cv::INTER_NEAREST);
  //   // 转为mask
  float* last_mask =
      new float[1 * 1 * input_image_width_ * input_image_height_];
  memset(last_mask, 0,
         1 * input_image_width_ * input_image_height_ * sizeof(float));
  int mask_num = 0;
  for (int row = 0; row < input_image_height_; ++row) {
    for (int col = 0; col < input_image_width_; ++col) {
      uint8_t gray = resize_mask.at<uint8_t>(row, col);
      if (gray != 0) {
        ++mask_num;
        last_mask[row * input_image_width_ + col] = 1.0;
      }
    }
  }
  if (mask_num < 50) {
    AWARN << " TO SAMLL MASK! REINIT MASK!";
    memset(last_mask, 0,
           1 * input_image_width_ * input_image_height_ * sizeof(float));
    return;
  } else {
    AINFO << " INIT MASK OK! MASK NUM: " << mask_num;
  }
  // 编码特征
  ++frame_count_;
  last_mem_frame_ = frame_count_;
  mask_ = cv::Mat(input_image_height_, input_image_width_, CV_8UC1);
  cv::Mat chw_image = PreProcess(image);
  image_encoder_->Perception(chw_image);
  CopyToGpu(segment_->GetNetMaskAddress(), last_mask,
            input_image_height_ * input_image_width_);
  memory_manager_->AddMmeory(image_encoder_->GetKeyAddress(),
                             image_encoder_->GetShrinkageAddress(),
                             image_encoder_->GetSelectionAddress());
  mask_inited_ = true;
  if (show_image) {
    cv::imshow("CutieInit", DrawMask(image, init_mask));
    cv::waitKey(1);
  }
}
void CutieCore::Perception(const cv::Mat& image, const bool flip) {
  flip_ = flip;
  auto t_0 = apollo::cyber::Time::Now();
  ++frame_count_;
  bool is_mem_frame =
      (frame_count_ - last_mem_frame_) >= memory_manager_->GetWorkMemoryGap();
  // 提取图像特征
  cv::Mat chw_image = PreProcess(image);
  image_encoder_->Perception(chw_image);
  // 注意力提取+记忆融合+分割
  segment_->Perception(!is_mem_frame);
  segment_->GetMaskImage(mask_);
  PostProcess();
  // 更新记忆
  bool can_be_mem = (!check_mem_) ||
                    (mask_area_ > 0 && rect_.height * 1.0 / rect_.width > 1.5);
  if (is_mem_frame && can_be_mem) {
    last_mem_frame_ = frame_count_;
    memory_manager_->AddMmeory(image_encoder_->GetKeyAddress(),
                               image_encoder_->GetShrinkageAddress(),
                               image_encoder_->GetSelectionAddress());
  }
  if (show_image) {
    cv::imshow("Cutie", DrawMask(image, GetOriMask()));
    cv::waitKey(1);
  }
  auto t_1 = apollo::cyber::Time::Now();
  AINFO << "Cutie cost time: " << (t_1 - t_0).ToSecond() * 1000 << " ms.";
}
cv::Mat CutieCore::PreProcess(const cv::Mat& image) {
  // 因为长宽比类似，直接resize
  ori_image_widh_ = image.cols;
  ori_image_height_ = image.rows;
  cv::Mat resize_img;
  cv::resize(image, resize_img,
             cv::Size(input_image_width_, input_image_height_),
             cv::INTER_NEAREST);
  if (flip_) {
    cv::flip(resize_img, resize_img, 1);
  }
  // 需要测试要不要swap
  return CustomBlobFromImage(resize_img, 1.0 / 255.0, true);
}
cv::Mat CutieCore::CustomBlobFromImage(const cv::Mat& input_image, float scale,
                                       bool swapRB) {
  // 1. 输入图像必须是 HxWx3, uchar
  CV_Assert(input_image.type() == CV_8UC3);

  int height = input_image.rows;
  int width = input_image.cols;

  // 2. 创建输出 blob：1 x 3 x H x W, float
  cv::Mat blob(1, 3 * height * width, CV_32F);
  float* blob_data = reinterpret_cast<float*>(blob.data);

  // 3. 遍历图像并填充 blob（NCHW）
  for (int y = 0; y < height; ++y) {
    const uchar* row_ptr = input_image.ptr<uchar>(y);
    for (int x = 0; x < width; ++x) {
      int pixel_index = y * width + x;
      uchar B = row_ptr[x * 3 + 0];
      uchar G = row_ptr[x * 3 + 1];
      uchar R = row_ptr[x * 3 + 2];

      if (swapRB) {
        blob_data[0 * height * width + pixel_index] = R * scale;
        blob_data[1 * height * width + pixel_index] = G * scale;
        blob_data[2 * height * width + pixel_index] = B * scale;
      } else {
        blob_data[0 * height * width + pixel_index] = B * scale;
        blob_data[1 * height * width + pixel_index] = G * scale;
        blob_data[2 * height * width + pixel_index] = R * scale;
      }
    }
  }
  // 4. reshape 成 OpenCV 的 "blob": 1x3xHxW
  return blob.reshape(1, {1, 3, height, width});
}
cv::Mat CutieCore::DrawMask(const cv::Mat& image, const cv::Mat& mask) {
  assert(image.size() == mask.size());
  cv::Rect draw_rect = rect_;
  cv::Point draw_center = center_;
  if (image.rows == ori_image_height_) {
    draw_rect = GetOriRect();
    draw_center = GetOriCenter();
  }
  cv::Mat color_mask;
  cv::cvtColor(mask, color_mask, cv::COLOR_GRAY2BGR);  // 变成3通道
  color_mask.setTo(cv::Scalar(255, 255, 0), mask);     // 仅对非0像素设为红色

  cv::Mat blended;
  cv::addWeighted(image, 1.0, color_mask, 0.5, 0,
                  blended);  // 0.5为红色透明度
  cv::rectangle(blended, draw_rect, cv::Scalar(255, 255, 0), 2);
  cv::circle(blended, draw_center, 5, cv::Scalar(255, 255, 0), -1);

  return blended;
}
void CutieCore::PostProcess() {
  cv::Mat labels, stats, centroids;
  if (flip_) {
    cv::flip(mask_, mask_, 1);
  }
  int n_labels =
      cv::connectedComponentsWithStats(mask_, labels, stats, centroids);
  float best_area = 0;
  int best_idx = 0;
  for (int i = 1; i < n_labels; ++i) {
    int area = stats.at<int>(i, cv::CC_STAT_AREA);
    if (area < 50) {
      continue;
    }
    if (area > best_area) {
      best_area = area;
      best_idx = i;
    }
  }
  if (best_idx != 0) {
    mask_ = (labels == best_idx);      // 得到一个 mask
    mask_.convertTo(mask_, CV_8U, 1);  // 转成 0 / 255 图像
    int x = stats.at<int>(best_idx, cv::CC_STAT_LEFT);
    int y = stats.at<int>(best_idx, cv::CC_STAT_TOP);
    int w = stats.at<int>(best_idx, cv::CC_STAT_WIDTH);
    int h = stats.at<int>(best_idx, cv::CC_STAT_HEIGHT);
    rect_ = cv::Rect(x, y, w, h);
    center_ = cv::Point(centroids.at<double>(best_idx, 0),
                        centroids.at<double>(best_idx, 1));
    mask_area_ = best_area;
  } else {
    rect_ = cv::Rect(0, 0, 0, 0);
    mask_area_ = 0;
  }
}
}  // namespace perception
}  // namespace robot