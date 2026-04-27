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

 * @brief  gauss & eigen
 * @ref    https://robosense.feishu.cn/docx/OhErdwv03oI7Quxm0kXc7OVnnDe
 *****************************************************************************/

#include "hyper_vision/perception/occupancy_objects/utils/gauss.h"

namespace robosense {
namespace perception {

void calEigenValues(const Eigen::Matrix3d &cov_mat,
                    Eigen::Vector3d &eigen_vals) {
  // λ^3 - c2*λ^2 + c1*λ - c0 = 0
  const auto &a = cov_mat(0, 0);
  const auto &b = cov_mat(0, 1);
  const auto &c = cov_mat(0, 2);
  const auto &d = cov_mat(1, 1);
  const auto &e = cov_mat(1, 2);
  const auto &f = cov_mat(2, 2);
  double c0 = a * e * e + d * c * c + f * b * b - a * d * f - 2.0 * b * c * e;
  double c1 = a * d + a * f + d * f - b * b - c * c - e * e;
  double c2 = a + d + f;
  double p = std::max(0.0, (c2 * c2 - 3.0 * c1) / 9.0);
  double q = (-2.0 * c2 * c2 * c2 + 9.0 * c2 * c1 + 27.0 * c0) / 54.0;
  double p3_q2 = std::max(0.0, p * p * p - q * q);
  double theta = std::atan2(std::sqrt(p3_q2), -q) / 3.0;
  double p_cos_theta = std::sqrt(p) * std::cos(theta);
  double p_sin_theta = std::sqrt(3 * p) * std::sin(theta);
  eigen_vals[0] = std::max(0.0, c2 / 3.0 + 2.0 * p_cos_theta);
  eigen_vals[1] = std::max(0.0, c2 / 3.0 - p_cos_theta - p_sin_theta);
  eigen_vals[2] = std::max(0.0, c2 / 3.0 - p_cos_theta + p_sin_theta);
  // 对特征值从小到大排序
  if (eigen_vals[0] > eigen_vals[1]) {
    std::swap(eigen_vals[0], eigen_vals[1]);
  }
  if (eigen_vals[1] > eigen_vals[2]) {
    std::swap(eigen_vals[1], eigen_vals[2]);
    if (eigen_vals[0] > eigen_vals[1]) {
      std::swap(eigen_vals[0], eigen_vals[1]);
    }
  }
}

void calEigenMatrix(const Eigen::Matrix3d &cov_mat,
                    const Eigen::Vector3d &eigen_vals,
                    Eigen::Matrix3d &eigen_mat) {
  // 输入的 eigen_vals 已经从小到大排序
  if (isEqual(eigen_vals[0], eigen_vals[2])) {
    eigen_mat = Eigen::Matrix3d::Identity();
    return;
  }
  Eigen::Vector3d v0, v1, v2;
  bool is_success_0 = calEigenVector(cov_mat, eigen_vals[0], v0);
  bool is_success_2 = calEigenVector(cov_mat, eigen_vals[2], v2);
  if (!is_success_0 || !is_success_2) {
    eigen_mat = Eigen::Matrix3d::Identity();
    return;
  }
  v1 = v2.cross(v0);
  v1.normalize();
  if (isEqual(eigen_vals[1], eigen_vals[2])) {
    v2 = v0.cross(v1);
    v2.normalize();
  } else if (isEqual(eigen_vals[1], eigen_vals[0])) {
    v0 = v1.cross(v2);
    v0.normalize();
  }
  eigen_mat.col(0) = v0;
  eigen_mat.col(1) = v1;
  eigen_mat.col(2) = v2;
}

bool calEigenVector(const Eigen::Matrix3d &cov_mat, const double &eigen_val,
                    Eigen::Vector3d &eigen_vec) {
  const auto &a = cov_mat(0, 0);
  const auto &b = cov_mat(0, 1);
  const auto &c = cov_mat(0, 2);
  const auto &d = cov_mat(1, 1);
  const auto &e = cov_mat(1, 2);
  const auto &f = cov_mat(2, 2);
  eigen_vec[0] = c * (eigen_val - d) + b * e;
  eigen_vec[1] = e * (eigen_val - a) + b * c;
  eigen_vec[2] = (eigen_val - a) * (eigen_val - d) - b * b;
  if (isNonZero(eigen_vec.norm())) {
    eigen_vec.normalize();
    return true;
  }
  Eigen::Vector3d sigma1{a - eigen_val, b, c};
  Eigen::Vector3d sigma2{b, d - eigen_val, e};
  double dot11 = sigma1.transpose() * sigma1;
  double dot12 = sigma1.transpose() * sigma2;
  double dot22 = sigma2.transpose() * sigma2;
  if (isNonZero(dot11)) {
    eigen_vec = Eigen::Vector3d(dot12, -dot11, 0.0);
    eigen_vec.normalize();
    return true;
  }
  if (isNonZero(dot22)) {
    eigen_vec = Eigen::Vector3d(-dot22, dot12, 0.0);
    eigen_vec.normalize();
    return true;
  }
  eigen_vec = Eigen::Vector3d(1.0, 0.0, 0.0);
  return false;
}

bool CommonGrid::check() {
  if (count < 500000) return true;
  count /= 2;
  sum_x *= 0.5;
  sum_y *= 0.5;
  sum_z *= 0.5;
  return false;
}

void GaussGrid::half() {
  sum_xx *= 0.5;
  sum_yy *= 0.5;
  sum_zz *= 0.5;
  sum_xy *= 0.5;
  sum_xz *= 0.5;
  sum_yz *= 0.5;
}

void CommonGrid::init(const Eigen::Vector3d &pt) {
  count = 1;
  sum_x = pt.x();
  sum_y = pt.y();
  sum_z = pt.z();
  min_z = pt.z();
  max_z = pt.z();
}

void GaussGrid::init(const Eigen::Vector3d &pt) {
  CommonGrid::init(pt);
  sum_xx = pt.x() * pt.x();
  sum_yy = pt.y() * pt.y();
  sum_zz = pt.z() * pt.z();
  sum_xy = pt.x() * pt.y();
  sum_xz = pt.x() * pt.z();
  sum_yz = pt.y() * pt.z();
}

bool CommonGrid::update(const Eigen::Vector3d &pt) {
  count += 1;
  sum_x += pt.x();
  sum_y += pt.y();
  sum_z += pt.z();
  min_z = std::min(min_z, pt.z());
  max_z = std::max(max_z, pt.z());
  return check();
}

bool GaussGrid::update(const Eigen::Vector3d &pt) {
  sum_xx += pt.x() * pt.x();
  sum_yy += pt.y() * pt.y();
  sum_zz += pt.z() * pt.z();
  sum_xy += pt.x() * pt.y();
  sum_xz += pt.x() * pt.z();
  sum_yz += pt.y() * pt.z();
  bool is_valid = CommonGrid::update(pt);
  if (!is_valid) half();
  return is_valid;
}

bool CommonGrid::merge(const CommonGrid &grid_info) {
  if (grid_info.empty()) return true;
  count += grid_info.count;
  sum_x += grid_info.sum_x;
  sum_y += grid_info.sum_y;
  sum_z += grid_info.sum_z;
  min_z = std::min(min_z, grid_info.min_z);
  max_z = std::max(max_z, grid_info.max_z);
  return check();
}

bool GaussGrid::merge(const GaussGrid &grid_info) {
  if (grid_info.empty()) return true;
  sum_xx += grid_info.sum_xx;
  sum_yy += grid_info.sum_yy;
  sum_zz += grid_info.sum_zz;
  sum_xy += grid_info.sum_xy;
  sum_xz += grid_info.sum_xz;
  sum_yz += grid_info.sum_yz;
  bool is_valid = CommonGrid::merge(grid_info);
  if (!is_valid) half();
  return is_valid;
}

void GaussGrid::moveFrom(const GaussGrid &grid_info, const Eigen::Matrix3d &rot,
                         const Eigen::Vector3d &centroid) {
  // 获取旋转后协方差
  Eigen::Matrix3d cov = grid_info.getCovMat();
  rotateCovMat(cov, rot);
  // 最大最小值
  min_z = centroid.z() - grid_info.getMinDiffZ();
  max_z = centroid.z() + grid_info.getMaxDiffZ();
  // 根据质心更新一次项
  count = grid_info.count;
  sum_x = count * centroid.x();
  sum_y = count * centroid.y();
  sum_z = count * centroid.z();
  // 计算二阶累积量矩阵并更新二次项
  Eigen::Matrix3d sigma_mat = count * (cov + centroid * centroid.transpose());
  sum_xx = sigma_mat(0, 0);
  sum_yy = sigma_mat(1, 1);
  sum_zz = sigma_mat(2, 2);
  sum_xy = sigma_mat(0, 1);
  sum_xz = sigma_mat(0, 2);
  sum_yz = sigma_mat(1, 2);
}

bool GaussGrid::mergeFrom(const GaussGrid &grid_info,
                          const Eigen::Matrix3d &rot,
                          const Eigen::Vector3d &centroid) {
  // 获取旋转后协方差
  Eigen::Matrix3d cov = grid_info.getCovMat();
  rotateCovMat(cov, rot);
  // 最大最小值
  double grid_mean_z = grid_info.getMeanZ();
  min_z = std::min(min_z, centroid.z() - grid_info.getMinDiffZ());
  max_z = std::max(max_z, centroid.z() + grid_info.getMaxDiffZ());
  // 根据质心更新一次项
  count += grid_info.count;
  sum_x += grid_info.count * centroid.x();
  sum_y += grid_info.count * centroid.y();
  sum_z += grid_info.count * centroid.z();
  // 计算二阶累积量矩阵并更新二次项
  Eigen::Matrix3d sigma_mat =
      grid_info.count * (cov + centroid * centroid.transpose());
  sum_xx += sigma_mat(0, 0);
  sum_yy += sigma_mat(1, 1);
  sum_zz += sigma_mat(2, 2);
  sum_xy += sigma_mat(0, 1);
  sum_xz += sigma_mat(0, 2);
  sum_yz += sigma_mat(1, 2);
  // check count
  bool is_valid = check();
  if (!is_valid) half();
  return is_valid;
}

}  // namespace perception
}  // namespace robosense
