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

#include "robot/perception/uwb_postprocess/image_track/yolov8_detection.h"
namespace robot {
namespace perception {
bool Yolov8Detection::Init(const YAML::Node &config) {
  // param
  rally::yamlRead(config, "nms_threshold", nms_threshold);
  rally::yamlRead(config, "box_conf_threshold", box_conf_threshold);
  rally::yamlRead(config, "nn_input_width", nn_input_width);
  rally::yamlRead(config, "nn_input_height", nn_input_height);
  rally::yamlRead(config, "show_image", show_image);

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
  init_options_.encrypt = config["encyrpt"].as<bool>();                // false
  init_options_.use_cuda_graph = config["use_cuda_graph"].as<bool>();  // false
  init_options_.use_gpu_input = config["use_gpu_input"].as<bool>();    // true
  engine_ptr_->init(init_options_);
  AINFO << "Set Cuda Stream Priority is: " << engine_ptr_->getStreamPriority();
  printModelInfo(engine_ptr_);
  return true;
}
void Yolov8Detection::Perception(const cv::Mat &image,
                                 const Yolov8ObjectList::Ptr &obj_list) {
  obj_list->objs.clear();
  auto t_1 = apollo::cyber::Time::Now();
  cv::Mat blob = PreProcess(image);  //[1*3*H*W]
  CV_Assert(blob.isContinuous());

  void *input_memory = engine_ptr_->getTensorAddress("images");

#ifdef __aarch64__
  memcpy(input_memory, blob.data, blob.total() * sizeof(float));
#else
  cudaMemcpy(input_memory, blob.data, blob.total() * sizeof(float),
             cudaMemcpyHostToDevice);
#endif
  auto t_2 = apollo::cyber::Time::Now();
  engine_ptr_->forward();
  auto t_3 = apollo::cyber::Time::Now();

  float *output = static_cast<float *>(engine_ptr_->getOutputPtr(
      "output0", robosense::inference::DeviceType::kCPU));
  PostProcessNoDLF(output, obj_list, true);
  auto t_4 = apollo::cyber::Time::Now();
  std::stringstream info_input;
  info_input << "yolo pre cost time: " << (t_2 - t_1).ToSecond() * 1000
             << " ms.";
  info_input << "yolo infer cost time: " << (t_3 - t_2).ToSecond() * 1000
             << " ms.";
  info_input << "yolo post cost time: " << (t_4 - t_3).ToSecond() * 1000
             << " ms.";
  //   AINFO << info_input.str();
  if (show_image) {
    DrawResult(image, obj_list);
    cv::imshow("Detection", image);
    cv::waitKey(1);
  }
}
cv::Mat Yolov8Detection::PreProcess(const cv::Mat &image) {
  // 保持长宽比将长边缩放到输入尺寸
  float x_scale = std::min(1.0 * nn_input_height / image.rows,
                           1.0 * nn_input_width / image.cols);
  float y_scale = x_scale;
  if (x_scale <= 0 || y_scale <= 0) {
    throw std::runtime_error("Invalid scale factor.");
  }

  int new_w = image.cols * x_scale;
  int x_shift = (nn_input_width - new_w) / 2;
  int x_other = nn_input_width - new_w - x_shift;

  int new_h = image.rows * y_scale;
  int y_shift = (nn_input_height - new_h) / 2;
  int y_other = nn_input_height - new_h - y_shift;
  scale = x_scale;
  x_pad = x_shift;
  y_pad = y_shift;

  cv::Mat resize_img;
  cv::resize(image, resize_img, cv::Size(new_w, new_h), cv::INTER_NEAREST);
  // 便宜到图像中心
  cv::copyMakeBorder(resize_img, resize_img, y_shift, y_other, x_shift, x_other,
                     cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
  // 需要测试要不要swap
  return CustomBlobFromImage(resize_img, 1.0 / 255.0, true);
}
cv::Mat Yolov8Detection::CustomBlobFromImage(const cv::Mat &input_image,
                                             float scale, bool swapRB) {
  // 1. 输入图像必须是 HxWx3, uchar
  CV_Assert(input_image.type() == CV_8UC3);

  int height = input_image.rows;
  int width = input_image.cols;

  // 2. 创建输出 blob：1 x 3 x H x W, float
  cv::Mat blob(1, 3 * height * width, CV_32F);
  float *blob_data = reinterpret_cast<float *>(blob.data);

  // 3. 遍历图像并填充 blob（NCHW）
  for (int y = 0; y < height; ++y) {
    const uchar *row_ptr = input_image.ptr<uchar>(y);
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
void Yolov8Detection::PostProcessNoDLF(float *outputs,
                                       const Yolov8ObjectList::Ptr &obj_list,
                                       bool filter_no_peds) {
  for (int i = 0; i < 8400; ++i) {
    float cx = (outputs[0 * 8400 + i] * nn_input_width - x_pad) / scale;
    float cy = (outputs[1 * 8400 + i] * nn_input_height - y_pad) / scale;
    float w = outputs[2 * 8400 + i] * nn_input_width / scale;
    float h = outputs[3 * 8400 + i] * nn_input_height / scale;
    int best_classes = 0;
    float best_score = 0;
    for (int c = 0; c < 80; ++c) {
      float score = outputs[(c + 4) * 8400 + i];
      if (score > box_conf_threshold && score > best_score) {
        best_classes = c;
        best_score = score;
      }
    }
    if (best_score > box_conf_threshold) {
      cv::Point left = cv::Point(cx - w / 2, cy - h / 2);
      //   left.x = (int)(clamp(left.x, 0, nn_input_width));
      //   left.y = (int)(clamp(left.y, 0, nn_input_height));
      cv::Point right = cv::Point(cx + w / 2, cy + h / 2);
      //   right.x = (int)(clamp(right.x, 0, nn_input_width));
      //   right.y = (int)(clamp(right.y, 0, nn_input_height));
      Yolov8Object::Ptr obj = std::make_shared<Yolov8Object>();
      obj->bbox = cv::Rect(left, right);
      obj->score = best_score;
      auto type = static_cast<COCO_CLASSES>(best_classes);
      obj->type = COCOClassesToObjectType(type);
      if (filter_no_peds && obj->type != ObjectType::TYPE_PED) {
        continue;
      }
      obj_list->objs.emplace_back(obj);
    }
  }
  std::sort(obj_list->objs.begin(), obj_list->objs.end(),
            [](const Yolov8Object::Ptr &a, const Yolov8Object::Ptr &b) {
              return a->score > b->score;
            });
  Nms(obj_list);
}
void Yolov8Detection::Nms(const Yolov8ObjectList::Ptr &obj_list) {
  // 高分删除低分
  for (size_t i = 0; i + 1 < obj_list->objs.size(); ++i) {
    for (size_t j = obj_list->objs.size() - 1; j >= (i + 1); --j) {
      if (obj_list->objs[i]->type != obj_list->objs[j]->type) {
        continue;
      }
      float iou = GetIoU(obj_list->objs[i]->bbox, obj_list->objs[j]->bbox);
      if (iou > nms_threshold) {
        obj_list->objs.erase(obj_list->objs.begin() + j);
      }
    }
  }
}

void Yolov8Detection::DrawResult(const cv::Mat &image,
                                 const Yolov8ObjectList::Ptr &obj_list) {
  for (const auto &obj : obj_list->objs) {
    // 绘制矩形框
    cv::rectangle(image, obj->bbox, cv::Scalar(0, 0, 0), 2);

    // 组合文字内容：类别 + 分数
    std::string label = kObjectTypeToNameMap.at(obj->type).substr(5) + "|" +
                        cv::format("%.1f", obj->score) + "|" +
                        cv::format("%.1f", obj->distance);

    // 获取文字大小，设置文字框背景
    int baseLine = 0;
    cv::Size labelSize =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    int top = std::max(obj->bbox.y, labelSize.height);

    // 绘制背景框（黑色）
    cv::rectangle(image, cv::Point(obj->bbox.x, top - labelSize.height - 4),
                  cv::Point(obj->bbox.x + labelSize.width, top),
                  cv::Scalar(0, 0, 0), cv::FILLED);

    // 绘制文字（白色）
    cv::putText(image, label, cv::Point(obj->bbox.x, top - 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
  }
}
}  // namespace perception
}  // namespace robot