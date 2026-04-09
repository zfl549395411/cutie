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
#include "robot/perception/uwb_postprocess/cutie/inference.h"

#include <string.h>

namespace robot {
namespace inference {

#define ALIGN(value, alignment) (((value) + ((alignment)-1)) & ~((alignment)-1))
#define ALIGN_32(value) ALIGN(value, 32)

// ── 私有辅助：名称→index查找 ────────────────────────────────────────────────

int32_t createInferHbm::findInputIdx(const std::string& name) const {
  auto it = input_name_to_idx_.find(name);
  if (it == input_name_to_idx_.end()) {
    INFER_ERROR << "[createInferHbm] input tensor not found: " << name;
    return -1;
  }
  return it->second;
}

int32_t createInferHbm::findOutputIdx(const std::string& name) const {
  auto it = output_name_to_idx_.find(name);
  if (it == output_name_to_idx_.end()) {
    INFER_ERROR << "[createInferHbm] output tensor not found: " << name;
    return -1;
  }
  return it->second;
}

// ── prepare_tensor: 分配内存并建立 name→index 映射 ─────────────────────────

int8_t createInferHbm::prepare_tensor(hbDNNTensor* input_tensor,
                                      hbDNNTensor* output_tensor,
                                      hbDNNHandle_t dnn_handle) {
  hbDNNGetInputCount(&input_count, dnn_handle);
  hbDNNGetOutputCount(&output_count, dnn_handle);

  hbDNNTensor* input = input_tensor;
  input_owned_sysMem_.resize(input_count);
  input_valid_byte_size_.resize(input_count);
  // 设置输入tensor
  for (int i = 0; i < input_count; i++) {
    hbDNNGetInputTensorProperties(&input[i].properties, dnn_handle, i);
    auto dim_len = input[i].properties.validShape.numDimensions;
    // stride[dim_len-1] 应由 API 设定（基础步长/element_size），此处不做修改
    for (int32_t d = dim_len - 2; d >= 0; --d) {
      if (input[i].properties.stride[d] == -1) {
        auto cur = input[i].properties.stride[d + 1] *
                   input[i].properties.validShape.dimensionSize[d + 1];
        input[i].properties.stride[d] = ALIGN_32(cur);
      }
    }
    int mem_size = input[i].properties.stride[0] *
                   input[i].properties.validShape.dimensionSize[0];
    hbUCPMallocCached(&input[i].sysMem, mem_size, 0);
    // 保留原始分配，reset时用于释放（zero-copy后sysMem会被替换）
    input_owned_sysMem_[i] = input[i].sysMem;
    // 有效字节数 = 各维度元素数之积 × stride[last]（element_size by API）
    // 用于 copyDataToTensor 的 memcpy，避免过读 CPU 缓冲区
    uint32_t valid_elems = 1;
    for (int d = 0; d < dim_len; ++d) {
      valid_elems *= input[i].properties.validShape.dimensionSize[d];
    }
    input_valid_byte_size_[i] =
        valid_elems *
        static_cast<uint32_t>(input[i].properties.stride[dim_len - 1]);

    // 建立 name→index 映射
    const char* input_name = nullptr;
    hbDNNGetInputName(&input_name, dnn_handle, i);
    input_name_to_idx_[input_name] = i;
  }

  hbDNNTensor* output = output_tensor;
  for (int i = 0; i < output_count; i++) {
    hbDNNGetOutputTensorProperties(&output[i].properties, dnn_handle, i);
    int mem_size = output[i].properties.alignedByteSize;
    hbUCPMallocCached(&output[i].sysMem, mem_size, 0);

    // 建立 name→index 映射
    const char* output_name = nullptr;
    hbDNNGetOutputName(&output_name, dnn_handle, i);
    output_name_to_idx_[output_name] = i;
  }
  return 0;
}

// ── init ────────────────────────────────────────────────────────────────────

int8_t createInferHbm::init(const char* hbm_path) {
  const char* model_path[] = {hbm_path};
  hbDNNInitializeFromFiles(&packed_dnn_handle, model_path, 1);
  hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
  hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);
  hbDNNGetInputCount(&input_count, dnn_handle);
  hbDNNGetOutputCount(&output_count, dnn_handle);
  input_tensors.resize(input_count);
  output_tensors.resize(output_count);
  prepare_tensor(input_tensors.data(), output_tensors.data(), dnn_handle);
  return 0;
}

// ── forward ─────────────────────────────────────────────────────────────────

int8_t createInferHbm::forward() {
  // 释放上一次推理的 task（否则每次 forward 泄漏一个 task，最终耗尽 task pool）
  if (task_handle != nullptr) {
    hbUCPReleaseTask(task_handle);
    task_handle = nullptr;
  }
  hbDNNInferV2(&task_handle, output_tensors.data(), input_tensors.data(),
               dnn_handle);
  HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
  ctrl_param.backend = HB_UCP_BPU_CORE_ANY;
  hbUCPSubmitTask(task_handle, &ctrl_param);
  hbUCPWaitTaskDone(task_handle, 0);
  for (int i = 0; i < output_count; ++i) {
    hbUCPMemFlush(&output_tensors[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
  }
  return 0;
}

// ── 公开 name-based 接口 ────────────────────────────────────────────────────

void* createInferHbm::getInputTensorAddress(const std::string& name) {
  int32_t idx = findInputIdx(name);
  if (idx < 0) return nullptr;
  return input_tensors[idx].sysMem.virAddr;
}

void* createInferHbm::getOutputTensorAddress(const std::string& name) {
  int32_t idx = findOutputIdx(name);
  if (idx < 0) return nullptr;
  return output_tensors[idx].sysMem.virAddr;
}

uint32_t createInferHbm::getInputTensorSize(const std::string& name) {
  int32_t idx = findInputIdx(name);
  if (idx < 0) return 0;
  return input_tensors[idx].sysMem.memSize;
}

uint32_t createInferHbm::getOutputTensorSize(const std::string& name) {
  int32_t idx = findOutputIdx(name);
  if (idx < 0) return 0;
  return output_tensors[idx].sysMem.memSize;
}

hbSysMem createInferHbm::getOutputSysMem(const std::string& name) {
  int32_t idx = findOutputIdx(name);
  if (idx < 0) return hbSysMem{};
  return output_tensors[idx].sysMem;
}

hbSysMem createInferHbm::getInputSysMem(const std::string& name) {
  int32_t idx = findInputIdx(name);
  if (idx < 0) return hbSysMem{};
  return input_tensors[idx].sysMem;
}
// 将一块内存的地址直接切换掉
void createInferHbm::setTensorAddress(hbSysMem src_mem,
                                      const std::string& name) {
  int32_t idx = findInputIdx(name);
  if (idx < 0) return;
  // Zero-copy: 替换整个sysMem，BPU DMA直接使用src的phyAddr
  input_tensors[idx].sysMem = src_mem;
}
// 将cpu数据拷贝到ddr中
void createInferHbm::copyDataToTensor(void* cpu_data, const std::string& name) {
  int32_t idx = findInputIdx(name);
  if (idx < 0) return;
  // 使用有效字节数而非对齐后的 memSize，避免从 CPU 缓冲区过读
  memcpy(input_tensors[idx].sysMem.virAddr, cpu_data,
         input_valid_byte_size_[idx]);
  hbUCPMemFlush(&input_tensors[idx].sysMem, HB_SYS_MEM_CACHE_CLEAN);
}

// ── printModelInfo ──────────────────────────────────────────────────────────

void createInferHbm::printModelInfo() noexcept {
  INFER_INFO << "[createInferHbm] model: " << model_name_list[0];
  for (const auto& kv : input_name_to_idx_) {
    int32_t i = kv.second;
    const auto& p = input_tensors[i].properties;
    std::string dims;
    for (int d = 0; d < p.validShape.numDimensions; ++d)
      dims += std::to_string(p.validShape.dimensionSize[d]) +
              (d + 1 < p.validShape.numDimensions ? "x" : "");
    std::string strides;
    for (int d = 0; d < p.validShape.numDimensions; ++d)
      strides += std::to_string(p.stride[d]) +
                 (d + 1 < p.validShape.numDimensions ? "," : "");
    INFER_INFO << "  input  [" << i << "] name=" << kv.first
               << "  type=" << p.tensorType << "  shape=[" << dims << "]"
               << "  stride=[" << strides << "]"
               << "  valid_bytes=" << input_valid_byte_size_[i]
               << "  alloc_bytes=" << input_tensors[i].sysMem.memSize;
  }
  for (const auto& kv : output_name_to_idx_) {
    int32_t i = kv.second;
    const auto& p = output_tensors[i].properties;
    std::string dims;
    for (int d = 0; d < p.validShape.numDimensions; ++d)
      dims += std::to_string(p.validShape.dimensionSize[d]) +
              (d + 1 < p.validShape.numDimensions ? "x" : "");
    std::string strides;
    for (int d = 0; d < p.validShape.numDimensions; ++d)
      strides += std::to_string(p.stride[d]) +
                 (d + 1 < p.validShape.numDimensions ? "," : "");
    INFER_INFO << "  output [" << i << "] name=" << kv.first
               << "  type=" << p.tensorType << "  shape=[" << dims << "]"
               << "  stride=[" << strides << "]"
               << "  alloc_bytes=" << p.alignedByteSize;
  }
}

// ── reset ───────────────────────────────────────────────────────────────────

int8_t createInferHbm::reset() {
  if (task_handle != nullptr) {
    hbUCPReleaseTask(task_handle);
    task_handle = nullptr;
  }
  // 释放init时分配的原始输入缓冲区（即使发生了zero-copy替换也能正确释放）
  for (auto& mem : input_owned_sysMem_) {
    hbUCPFree(&mem);
  }
  input_owned_sysMem_.clear();
  for (int i = 0; i < output_count; i++) {
    hbUCPFree(&output_tensors[i].sysMem);
  }
  hbDNNRelease(packed_dnn_handle);
  return 0;
}

void createInferHbm::setDebugLevel(DebugLevel level) noexcept {}

}  // namespace inference
}  // namespace robot
