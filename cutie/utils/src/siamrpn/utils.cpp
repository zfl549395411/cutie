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

#include "robot/perception/uwb_postprocess/image_track/utils.h"
namespace robot {
namespace perception {
void AnchorGenerator::GenerateGridAnchors(int feature_size,
                                          AnchorArray& out_anchors) {
  int anchor_num = model_anchor_center_.size();
  int total_anchors = feature_size * feature_size * anchor_num;

  out_anchors.resize(total_anchors);

  // 框中心点的偏移量
  int ori = -(feature_size / 2) * stride_;

  // 遍历特征图,x->列,y->行
  for (int a = 0; a < anchor_num; ++a) {
    float w = model_anchor_center_[a][2];
    float h = model_anchor_center_[a][3];
    AINFO << " W H " << w << " " << h;

    for (int y = 0; y < feature_size; ++y) {
      for (int x = 0; x < feature_size; ++x) {
        int grid_id = y * feature_size + x;
        int idx = a * feature_size * feature_size + grid_id;

        float center_x = ori + x * stride_;
        float center_y = ori + y * stride_;

        out_anchors[idx] = {center_x, center_y, w, h};
      }
    }
  }
}
std::string formatDims(robosense::inference::Dims dims) {
  std::stringstream ss;
  int i;
  for (i = 0; i < dims.nbDims - 1; i++) {
    ss << dims.d[i] << "x";
  }
  ss << dims.d[i];
  return ss.str();
}
void printModelInfo(robosense::inference::InferEngine::Ptr infer_ptr) {
  for (auto i = 0; i < infer_ptr->getNumInputs(); i++) {
    AINFO << "Input " << i << " " << infer_ptr->getInputName(i) << " "
          << formatDims(infer_ptr->getInputDims(i)) << " "
          << infer_ptr->getInputSize(i) << std::endl;
  }

  for (auto i = 0; i < infer_ptr->getNumOutputs(); i++) {
    AINFO << "Output " << i << " " << infer_ptr->getOutputName(i) << " "
          << formatDims(infer_ptr->getOutputDims(i)) << " "
          << infer_ptr->getOutputSize(i) << std::endl;
  }
}

}  // namespace perception
}  // namespace robot