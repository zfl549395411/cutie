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
#include "robot/perception/uwb_postprocess/cutie/memory_manager.h"
namespace robot {
namespace perception {
Memory::Memory(float* key, float* value, float* shrinkage, float* selection,
               float* use_count, float* life_count, int all_cols,
               int start_pose, const PermanentType _permanent_type,
               int single_token_lens, int max_len, int channel_key,
               int channel_value) {
  permanent_type = _permanent_type;
  permanent_end_idx = 0;
  temp_slot_end_idx = 0;
  // 设置尺寸
  slot_size = single_token_lens;
  max_slot_size = slot_size * max_len;
  // 设置头指针
  key_rows_start.resize(channel_key);
  selection_rows_start.resize(channel_key);
  for (int i = 0; i < channel_key; ++i) {
    int offset = all_cols * i + start_pose;
    key_rows_start[i] = key + offset;
    selection_rows_start[i] = selection + offset;
  }
  value_rows_start.resize(channel_value);
  for (int i = 0; i < channel_value; ++i) {
    int offset = all_cols * i + start_pose;
    value_rows_start[i] = value + offset;
  }
  shrinkage_rows_start.resize(1);
  shrinkage_rows_start[0] = shrinkage + start_pose;
  use_count_rows_start.resize(1);
  use_count_rows_start[0] = use_count + start_pose;
  life_count_rows_start.resize(1);
  life_count_rows_start[0] = life_count + start_pose;
}
bool MemoryManager::Init(const YAML::Node& config, int nn_width,
                         int nn_height) {
  assert(nn_width % 16 == 0 && nn_height % 16 == 0);
  int H = nn_height / 16;
  int W = nn_width / 16;
  HW = H * W;
  // 读取配置
  work_memory_gap = config["work_memory_gap"].as<int>();
  max_work_memory_len = config["max_work_memory_len"].as<int>();
  min_work_memory_len = config["min_work_memory_len"].as<int>();
  max_long_memory_len = config["max_long_memory_len"].as<int>();
  long_memory_size = config["long_term_memory_size"].as<int>();
  channel_key = config["channel_key"].as<int>();
  channel_value = config["channel_value"].as<int>();

  // 目标特征存储 每帧累加
  int total_size = 1 * 1 * 16 * 257;
  obj_value_memory = new float[total_size];
  memset(obj_value_memory, 0, total_size * sizeof(float));
  AINFO << "Init obj_value_memory Size: " << total_size;
  // sensory存储，每帧重置
  total_size = 1 * 1 * channel_value * H * W;
  sensory_memory = new float[total_size];
  memset(sensory_memory, 0, total_size * sizeof(float));
  AINFO << "Init sensory_memory Size: " << total_size;

  // 开辟整块内存
  single_channel_cols =
      HW * max_work_memory_len + long_memory_size * max_long_memory_len;
  key_complete = new float[1 * channel_key * single_channel_cols];
  AINFO << " key size: " << 1 * channel_key * single_channel_cols;
  memset(key_complete, 0,
         1 * channel_key * single_channel_cols * sizeof(float));
  value_complete = new float[1 * channel_value * single_channel_cols];
  memset(value_complete, 0,
         1 * channel_value * single_channel_cols * sizeof(float));
  shrinkage_complete = new float[1 * 1 * single_channel_cols];
  memset(shrinkage_complete, 0, 1 * 1 * single_channel_cols * sizeof(float));
  selection_complete = new float[1 * channel_key * single_channel_cols];
  memset(selection_complete, 0,
         1 * channel_key * single_channel_cols * sizeof(float));
  use_count_complete = new float[1 * 1 * single_channel_cols];
  memset(use_count_complete, 0, 1 * 1 * single_channel_cols * sizeof(float));
  life_count_complete = new float[1 * 1 * single_channel_cols];
  memset(life_count_complete, 0, 1 * 1 * single_channel_cols * sizeof(float));

  // work memory 要额外加一帧永久存储
  work_memory = std::make_shared<Memory>(
      key_complete, value_complete, shrinkage_complete, selection_complete,
      use_count_complete, life_count_complete, single_channel_cols, 0,
      PermanentType::FIRST_PERMANENT, HW, max_work_memory_len, channel_key,
      channel_value);
  // long memory
  long_memory = std::make_shared<Memory>(
      key_complete, value_complete, shrinkage_complete, selection_complete,
      use_count_complete, life_count_complete, single_channel_cols,
      HW * max_work_memory_len, PermanentType::ALL_PERMANENT, long_memory_size,
      max_long_memory_len, channel_key, channel_value);

  // 初始化网络
  input_height_ = nn_height;
  input_width_ = nn_width;
  // encode net
  encode_engine_ptr_ = robosense::inference::createInferEngine(
      robosense::inference::EngineType::kTENSORRT);
  const auto& level = robosense::inference::DebugLevel::kDEBUG;
  encode_engine_ptr_->setDebugLevel(level);
#ifdef __aarch64__
  std::string memory_engine_path =
      config["encode_arm_model_path"].as<std::string>();
  encode_options_.use_Unified_Address = true;
#else
  std::string memory_engine_path =
      config["encode_x86_model_path"].as<std::string>();
  encode_options_.use_Unified_Address = false;
#endif
  encode_options_.save_path = memory_engine_path;
  encode_options_.model_format = robosense::inference::ModelFormat::kENGINE;
  encode_options_.batch_size = 1;
  encode_options_.encrypt = false;        // false
  encode_options_.use_cuda_graph = true;  // false
  encode_options_.use_gpu_input = true;   // true
  encode_engine_ptr_->init(encode_options_);
  AWARN << "Cutie Encode Fusion Model Info: ";
  printModelInfo(encode_engine_ptr_);

  // compress net
  compress_engine_ptr_ = robosense::inference::createInferEngine(
      robosense::inference::EngineType::kTENSORRT);
  compress_engine_ptr_->setDebugLevel(level);
#ifdef __aarch64__
  std::string segment_engine_path =
      config["compress_arm_model_path"].as<std::string>();
  compress_options_.use_Unified_Address = true;
#else
  std::string segment_engine_path =
      config["compress_x86_model_path"].as<std::string>();
  compress_options_.use_Unified_Address = false;
#endif
  compress_options_.save_path = segment_engine_path;
  compress_options_.model_format = robosense::inference::ModelFormat::kENGINE;
  compress_options_.batch_size = 1;
  compress_options_.encrypt = false;         // false
  compress_options_.use_cuda_graph = false;  // false
  compress_options_.use_gpu_input = true;    // true
  compress_engine_ptr_->init(compress_options_);
  AWARN << "Cutie Compress Model Info: ";
  printModelInfo(compress_engine_ptr_);
  return true;
}
void MemoryManager::SetAddress(float* image, float* pix_feat,
                               float* lask_mask) {
  encode_engine_ptr_->setTensorAddress("image", image);
  encode_engine_ptr_->setTensorAddress("pix_feat", pix_feat);
  encode_engine_ptr_->setTensorAddress("prob", lask_mask);
  sensory_address_ =
      static_cast<float*>(encode_engine_ptr_->getTensorAddress("sensory"));
  mask_value_address_ =
      static_cast<float*>(encode_engine_ptr_->getTensorAddress("msk_value"));
  sensory_out_address_ =
      static_cast<float*>(encode_engine_ptr_->getTensorAddress("sensory_out"));
  // compress
  history_key_address_ = static_cast<float*>(
      compress_engine_ptr_->getTensorAddress("history_key"));
  history_value_address_ = static_cast<float*>(
      compress_engine_ptr_->getTensorAddress("history_value"));
  history_shrinkage_address_ = static_cast<float*>(
      compress_engine_ptr_->getTensorAddress("history_shrinkage"));
  history_selection_address_ = static_cast<float*>(
      compress_engine_ptr_->getTensorAddress("history_selection"));
  usage_address_ =
      static_cast<float*>(compress_engine_ptr_->getTensorAddress("usage"));
  prototype_key_address_ = static_cast<float*>(
      compress_engine_ptr_->getTensorAddress("prototype_key"));
  prototype_value_address_ = static_cast<float*>(
      compress_engine_ptr_->getTensorAddress("prototype_value"));
  prototype_shrinkage_address_ = static_cast<float*>(
      compress_engine_ptr_->getTensorAddress("prototype_shrinkage"));
}
void MemoryManager::MaskEncoder() {
  float* sensory = GetSensoryMemory();
  CopyToGpu(sensory_address_, sensory, GetSensoryMemorySize());
  auto t_0 = apollo::cyber::Time::Now();
  encode_engine_ptr_->forward();
  auto t_1 = apollo::cyber::Time::Now();
  AINFO << "memory mask encoder cost time: " << (t_1 - t_0).ToSecond() * 1000
        << " ms.";
}
void MemoryManager::CompressFeature() {
  // 最前方填写的是最原始的memory
  int start_tokens = 1410;
  // 留下最新的4帧memmory
  int end_tokens = work_memory->max_slot_size -
                   (min_work_memory_len - 1) * work_memory->slot_size;
  // 实际上应该是固定的1410*5，前1后4，暂时不写死
  int copy_tokens = end_tokens - start_tokens;
  // key相关
  int key_copy_size = 0, key_value_copy_size = 0;
  for (int i = 0; i < channel_key; ++i) {
    // 拷贝起始内存
    float* key_start = history_key_address_ + i * copy_tokens;
    float* selection_start = history_selection_address_ + i * copy_tokens;
    // 行起始内存
    float* row_key_start = work_memory->key_rows_start[i] + start_tokens;
    float* row_selection_start =
        work_memory->selection_rows_start[i] + start_tokens;
    CopyToGpu(key_start, row_key_start, copy_tokens);
    CopyToGpu(selection_start, row_selection_start, copy_tokens);
    key_copy_size += (copy_tokens * sizeof(float));
  }
  // value相关
  for (int i = 0; i < channel_value; ++i) {
    float* value_start = history_value_address_ + i * copy_tokens;
    // 行起始内存
    float* row_value_start = work_memory->value_rows_start[i] + start_tokens;
    CopyToGpu(value_start, row_value_start, copy_tokens);
    key_value_copy_size += (copy_tokens * sizeof(float));
  }
  // 单channel的
  CopyToGpu(history_shrinkage_address_,
            work_memory->shrinkage_rows_start[0] + start_tokens, copy_tokens);
  CopyToGpu(usage_address_, work_memory->use_count_rows_start[0] + start_tokens,
            copy_tokens);
  auto t_0 = apollo::cyber::Time::Now();
  compress_engine_ptr_->forward();
  // SaveModelInfo(compress_engine_ptr_,
  //               "/apollo/robot_dog_modules/perception/text_result/compress");
  auto t_1 = apollo::cyber::Time::Now();
  AINFO << "memory compress feature cost time: "
        << (t_1 - t_0).ToSecond() * 1000 << " ms.";
}
void MemoryManager::AddMmeory(float* key, float* shrinkage, float* selection) {
  // 提取memory
  MaskEncoder();
  // 管理key-value字典
  work_memory->AddMmeory(key, mask_value_address_, shrinkage, selection);
  if (work_memory->temp_slot_end_idx >= work_memory->max_slot_size) {
    if (long_memory->temp_slot_end_idx >= long_memory->max_slot_size) {
      // 清除非topk的存储
      long_memory->RemoveMemoryByTopK(
          (max_long_memory_len - min_long_memory_len - 1) * long_memory_size);
    }
    CompressFeature();
    work_memory->RemoveMemoryByOld(max_work_memory_len, min_work_memory_len);
    long_memory->AddMmeory(prototype_key_address_, prototype_value_address_,
                           prototype_shrinkage_address_, nullptr);
  }
  // 管理obj_value
  float* obj_value_address_ =
      static_cast<float*>(encode_engine_ptr_->getOutputPtr(
          "obj_value", robosense::inference::DeviceType::kCPU));
  float max_obj_value = 0;
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 257; ++j) {
      obj_value_memory[i * 257 + j] += obj_value_address_[i * 257 + j];
      max_obj_value =
          std::max(std::abs(obj_value_memory[i * 257 + j]), max_obj_value);
    }
  }
  if (max_obj_value > 60000) {
    AINFO << " CLEAR OBJ VALUE max value " << max_obj_value;
    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 257; ++j) {
        obj_value_memory[i * 257 + j] /= max_obj_value;
      }
    }
  }
  UpdateSensory(sensory_out_address_);
}
void MemoryManager::UpdateSensory(float* sensory) {
  // 直接复制拷贝
  // 将感官数据复制到CPU中
  CopyToCpu(sensory_memory, sensory, 1 * 1 * channel_value * HW);
}
// usage是和整内存等大的，不是单slot的
void Memory::UpdateUsage(float* usage, int single_channel_cols) {
  for (int i = 0; i < use_count_rows_start.size(); ++i) {
    // 跳过整行
    float* row_start = usage + single_channel_cols * i;
    // 永久存储更新
    for (int j = 0; j < permanent_end_idx; ++j) {
      use_count_rows_start[i][j] += row_start[j];
      ++life_count_rows_start[i][j];
    }
    // 临时存储更新
    for (int j = permanent_end_idx; j < temp_slot_end_idx; ++j) {
      use_count_rows_start[i][j] += row_start[j];
      ++life_count_rows_start[i][j];
    }
  }
}
void MemoryManager::UpdateUsage(float* usage) {
  work_memory->UpdateUsage(usage, single_channel_cols);
  long_memory->UpdateUsage(usage + max_work_memory_len * HW,
                           single_channel_cols);
}
// 单帧slot 这里不管理usage因为默认就是全0，所以这里不更新
void Memory::AddMmeory(float* key, float* value, float* shrinkage,
                       float* selection) {
  // 更新永久存储结束位置
  if (permanent_type == PermanentType::FIRST_PERMANENT) {
    permanent_end_idx = slot_size;
  } else {
    permanent_end_idx += slot_size;
  }
  // key存储
  for (int i = 0; i < key_rows_start.size(); ++i) {
    float* key_row_start = key + i * slot_size;
    float* selection_row_start = selection + i * slot_size;
    CopyToCpu(key_rows_start[i] + temp_slot_end_idx, key_row_start, slot_size);
    if (selection == nullptr) {
      // longtenr不存储，不存储就填全1，全0的话就都不会考虑
      std::fill(selection_rows_start[i] + temp_slot_end_idx,
                selection_rows_start[i] + temp_slot_end_idx + slot_size, 0.0f);
    } else {
      CopyToCpu(selection_rows_start[i] + temp_slot_end_idx,
                selection_row_start, slot_size);
    }
  }
  // value存储
  for (int i = 0; i < value_rows_start.size(); ++i) {
    float* value_row_start = value + i * slot_size;
    CopyToCpu(value_rows_start[i] + temp_slot_end_idx, value_row_start,
              slot_size);
  }
  // shinkage存储
  CopyToCpu(shrinkage_rows_start[0] + temp_slot_end_idx, shrinkage, slot_size);
  // 结束位置向后偏移
  temp_slot_end_idx += slot_size;
}
// 用于long_term，所以不需要考虑永久存储
void Memory::RemoveMemoryByTopK(int topk) {
  if (temp_slot_end_idx <= topk) {
    return;
  }
  std::vector<int> help_idx(temp_slot_end_idx);
  // 记录距离起始点的真实偏移
  for (int i = 0; i < help_idx.size(); ++i) {
    help_idx[i] = i;
  }
  std::partial_sort(help_idx.begin(), help_idx.begin() + topk, help_idx.end(),
                    [this](int a, int b) {
                      return use_count_rows_start[0][a] >
                             use_count_rows_start[0][b];
                    });
  std::sort(help_idx.begin(), help_idx.begin() + topk);
  // topk前移动到最前，topk后填充0
  for (int c = 0; c < key_rows_start.size(); ++c) {
    for (int i = 0; i < topk; ++i) {
      key_rows_start[c][i] = key_rows_start[c][help_idx[i]];
      selection_rows_start[c][i] = selection_rows_start[c][help_idx[i]];
    }
    std::fill(key_rows_start[c] + topk, key_rows_start[c] + temp_slot_end_idx,
              0.f);
    std::fill(selection_rows_start[c] + topk,
              selection_rows_start[c] + temp_slot_end_idx, 0.f);
  }
  for (int c = 0; c < value_rows_start.size(); ++c) {
    for (int i = 0; i < topk; ++i) {
      value_rows_start[c][i] = value_rows_start[c][help_idx[i]];
    }
    std::fill(value_rows_start[c] + topk,
              value_rows_start[c] + temp_slot_end_idx, 0.f);
  }
  for (int i = 0; i < topk; ++i) {
    use_count_rows_start[0][i] = use_count_rows_start[0][help_idx[i]];
    life_count_rows_start[0][i] = life_count_rows_start[0][help_idx[i]];
    shrinkage_rows_start[0][i] = shrinkage_rows_start[0][help_idx[i]];
  }
  std::fill(shrinkage_rows_start[0] + topk,
            shrinkage_rows_start[0] + temp_slot_end_idx, 0.f);
  std::fill(use_count_rows_start[0] + topk,
            use_count_rows_start[0] + temp_slot_end_idx, 0.f);
  std::fill(life_count_rows_start[0] + topk,
            life_count_rows_start[0] + temp_slot_end_idx, 0.f);
  temp_slot_end_idx = topk;
  permanent_end_idx = topk;
}
// 输入存储中包含永久存储，保留最新的min_size个
void Memory::RemoveMemoryByOld(int max_size, int min_size) {
  // 临时存储的上下限
  int max_temp_size = max_size * slot_size - permanent_end_idx;
  int min_temp_size = min_size * slot_size - permanent_end_idx;
  if (max_temp_size < 0) {
    AERROR << " TEMP SIZE IS < 0!";
    return;
  }
  if (GetTempSlotSize() <= min_temp_size) {
    return;
  }
  // 需要转存的内存起始
  int save_end = temp_slot_end_idx;
  int save_start = save_end - min_temp_size;
  // 实际使用的内存起始
  int memory_start = permanent_end_idx;
  int memory_end = permanent_end_idx + min_temp_size;
  // memory_end到save_end内存清空
  for (int c = 0; c < key_rows_start.size(); ++c) {
    memcpy(key_rows_start[c] + memory_start, key_rows_start[c] + save_start,
           min_temp_size * sizeof(float));
    memcpy(selection_rows_start[c] + memory_start,
           selection_rows_start[c] + save_start, min_temp_size * sizeof(float));
    std::fill(key_rows_start[c] + memory_end, key_rows_start[c] + save_end,
              0.f);
    std::fill(selection_rows_start[c] + memory_end,
              selection_rows_start[c] + save_end, 0.f);
  }
  for (int c = 0; c < value_rows_start.size(); ++c) {
    memcpy(value_rows_start[c] + memory_start, value_rows_start[c] + save_start,
           min_temp_size * sizeof(float));
    std::fill(value_rows_start[c] + memory_end, value_rows_start[c] + save_end,
              0.f);
  }
  memcpy(shrinkage_rows_start[0] + memory_start,
         shrinkage_rows_start[0] + save_start, min_temp_size * sizeof(float));
  std::fill(shrinkage_rows_start[0] + memory_end,
            shrinkage_rows_start[0] + save_end, 0.f);
  memcpy(use_count_rows_start[0] + memory_start,
         use_count_rows_start[0] + save_start, min_temp_size * sizeof(float));
  std::fill(use_count_rows_start[0] + memory_end,
            use_count_rows_start[0] + save_end, 0.f);
  memcpy(life_count_rows_start[0] + memory_start,
         life_count_rows_start[0] + save_start, min_temp_size * sizeof(float));
  std::fill(life_count_rows_start[0] + memory_end,
            life_count_rows_start[0] + save_end, 0.f);
  temp_slot_end_idx = memory_end;
}
}  // namespace perception
}  // namespace robot
