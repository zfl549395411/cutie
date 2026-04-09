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

#include "robot/perception/uwb_postprocess/process/kalman_filter.h"

namespace robot {
namespace perception {
cv::Mat KalmanFilter::UpdateObserve(const ObserveArray& observe,
                                    const ObserveArray& error,
                                    const double cur_time) {
  ++frame_count;
  double time_gap = cur_time - pre_update_time;
  pre_update_time = cur_time;
  // 更新状态转移
  *A.ptr<float>(0, 3) = time_gap;
  *A.ptr<float>(1, 4) = time_gap;
  // 更新输入
  for (int i = 0; i < NUM_OBSERVE; ++i) {
    *observe_mat.ptr<float>(i, 0) = observe[i];
    *C.ptr<float>(i, i) = error[i];
  }
  //   AINFO << " OBSERVE " << observe_mat.t();
  //   AINFO << " C " << C;
  float x = pre_predict_res.at<float>(0), y = pre_predict_res.at<float>(1),
        z = pre_predict_res.at<float>(2);
  float d = std::sqrt(x * x + y * y + z * z);
  float xy_sq = x * x + y * y;
  H = cv::Mat::zeros(5, 5, CV_32F);
  H.at<float>(0, 0) = x / d;
  H.at<float>(0, 1) = y / d;
  H.at<float>(0, 2) = z / d;
  H.at<float>(1, 0) = -y / xy_sq;
  H.at<float>(1, 1) = x / xy_sq;
  H.at<float>(2, 2) = 1;
  H.at<float>(3, 3) = 1;
  H.at<float>(4, 4) = 1;
  //   AINFO << "H " << H;
  // 预测
  pre_predict_res = A * pre_predict_res;
  //   AINFO << " PREDICT " << pre_predict_res.t();
  // 更新方差
  M = A.t() * M * A + Q;
  //   AINFO << " M " << M;
  // 更新增益
  cv::Mat tmp = (H * M * H.t() + C);
  K = M * H.t() * tmp.inv();
  //   AINFO << " K" << " " << K;
  // 更新预测
  // 残差
  cv::Mat predict_observe = H * pre_predict_res;
  predict_observe.at<float>(0, 0) = d;
  predict_observe.at<float>(1, 0) = std::atan2(y, x);

  float delta = std::fmod(
      (observe_mat.at<float>(1, 0) - predict_observe.at<float>(1, 0)) +
          static_cast<float>(M_PI),
      2.0f * static_cast<float>(M_PI));
  if (delta < 0) delta += 2.0f * static_cast<float>(M_PI);
  delta = delta - static_cast<float>(M_PI);
  // 残差过大，直接使用观测值
  if (std::abs(delta) > 0.25) {
    // 恢复x y
    // 不能用d，而是用xy距离，因为实际上是bev下的极坐标，这里描述用的更像是柱面坐标
    float observe_x = std::cos(observe_mat.at<float>(1, 0)) * std::sqrt(xy_sq);
    float observe_y = std::sin(observe_mat.at<float>(1, 0)) * std::sqrt(xy_sq);

    pre_predict_res.at<float>(0, 0) = observe_x;
    pre_predict_res.at<float>(0, 1) = observe_y;
    predict_observe.at<float>(0, 0) = observe_mat.at<float>(0, 0);
    predict_observe.at<float>(1, 0) = observe_mat.at<float>(1, 0);
    AINFO << " TOO BIG DELTA , USE OBSERVE " << observe_mat.at<float>(1, 0)
          << " delta " << delta << " REALY " << predict_observe.at<float>(1, 0)
          << " x y " << pre_predict_res.t();
    delta = 0;
  }
  // 更新预测(卡尔曼增益*残差
  cv::Mat residual = observe_mat - predict_observe;
  residual.at<float>(1, 0) = delta;
  //   AINFO << " predict_observe " << predict_observe.t() << " residula "
  //         << residual.t() << " delta " << delta;
  pre_predict_res = pre_predict_res + K * (residual);
  //   AINFO << " UPDATE " << pre_predict_res.t();
  // 更新方差
  //   M = (cv::Mat::eye(M.rows, M.cols, CV_32F) - K * H) * M;
  cv::Mat I_KH = cv::Mat::eye(M.size(), CV_32F) - K * H;
  M = I_KH * M * I_KH.t() + K * C * K.t();  // Joseph form，确保对称正定
  return pre_predict_res;
}
}  // namespace perception
}  // namespace robot