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
#ifndef ROBOT_UWB_POSTPROCESS_PROCESS_CUTIE_INFERENCE_H
#define ROBOT_UWB_POSTPROCESS_PROCESS_CUTIE_INFERENCE_H
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "robot/perception/uwb_postprocess/cutie/log.h"
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

namespace robot {
namespace inference {

enum class DebugLevel : uint8_t { kTRACE, kDEBUG, kINFO, kERROR };
constexpr int8_t kSUCCESS{0};
constexpr int8_t kFAILURE{-1};

class createInferHbm {
 public:
  using Ptr = std::shared_ptr<createInferHbm>;
  createInferHbm() = default;
  ~createInferHbm() { reset(); }

  int8_t init(const char* hbm_path);
  int8_t forward();

  // 通过tensor名查找，名称不存在时打印错误并返回nullptr/0，便于debug
  void* getInputTensorAddress(const std::string& name);
  void* getOutputTensorAddress(const std::string& name);
  uint32_t getInputTensorSize(const std::string& name);
  uint32_t getOutputTensorSize(const std::string& name);

  // 返回完整sysMem（virAddr+phyAddr），用于BPU→BPU zero-copy绑定
  hbUCPSysMem getOutputSysMem(const std::string& name);
  hbUCPSysMem getInputSysMem(const std::string& name);

  // Zero-copy: 替换整个sysMem，BPU DMA直接使用src的物理地址（初始化时调用一次）
  void setTensorAddress(hbUCPSysMem src_mem, const std::string& name);

  // CPU→BPU: 将CPU侧数据memcpy进BPU缓冲区并刷缓存（每帧调用）
  void copyDataToTensor(void* cpu_data, const std::string& name);

  // 返回输出tensor第dim维的字节步长（含对齐padding）
  int64_t getOutputTensorStride(const std::string& name, int dim) const;
  // 返回输出tensor第dim维的逻辑尺寸
  int32_t getOutputTensorDim(const std::string& name, int dim) const;
  // 返回输出tensor有效字节数（不含对齐padding，= prod(dims) * elem_size）
  uint32_t getOutputValidBytes(const std::string& name) const;

  // BPU→CPU stride-aware 拷贝：自动检测W维padding，逐行正确提取
  void copyDataFromTensor(void* cpu_data, const std::string& name);
  // CPU→BPU output tensor stride-aware 写入（用于初始化BPU输出缓冲区，如InitMask）
  void copyDataToOutputTensor(void* cpu_data, const std::string& name);

  // 打印所有输入输出tensor名称、index和内存大小，用于调试
  void printModelInfo() noexcept;

  void setDebugLevel(DebugLevel level) noexcept;
  int8_t reset();

 private:
  int8_t prepare_tensor(hbDNNTensor* input_tensor, hbDNNTensor* output_tensor,
                        hbDNNHandle_t dnn_handle);
  // 名称→index查找，找不到时打印错误返回-1
  int32_t findInputIdx(const std::string& name) const;
  int32_t findOutputIdx(const std::string& name) const;

  int32_t input_count = 0;
  int32_t output_count = 0;
  int32_t model_count = 0;
  hbDNNPackedHandle_t packed_dnn_handle;
  hbDNNHandle_t dnn_handle;
  hbUCPSchedParam ctrl_param;
  const char** model_name_list;
  hbUCPTaskHandle_t task_handle{nullptr};
  std::vector<hbDNNTensor> input_tensors;
  std::vector<hbDNNTensor> output_tensors;
  // tensor名称→index映射，init时从模型中自动构建
  std::unordered_map<std::string, int32_t> input_name_to_idx_;
  std::unordered_map<std::string, int32_t> output_name_to_idx_;
  // 记录init时hbUCPMallocCached分配的原始sysMem，reset时安全释放
  std::vector<hbUCPSysMem> input_owned_sysMem_;
  // 每个输入/输出tensor的有效字节数（不含对齐padding = prod(dims) * elem_size），init时预计算
  std::vector<uint32_t> input_valid_byte_size_;
  std::vector<uint32_t> output_valid_byte_size_;
};

}  // namespace inference
}  // namespace robot
#endif  // ROBOT_UWB_POSTPROCESS_PROCESS_CUTIE_INFERENCE_H
