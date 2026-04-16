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
#ifndef HYPER_VISIONPERCEPTION_SEGFORMER_VOXEL_MANAGER_H
#define HYPER_VISIONPERCEPTION_SEGFORMER_VOXEL_MANAGER_H
#include "hyper_vision/perception/common/common.h"
#include "hyper_vision/perception/segformer/common/common.h"
namespace robosense {
namespace perception {
struct VoxelData {
  /*
  栅格行优先，实际上是, 也就是都是序号增长，对应的 真实坐标x越大,
  row = ix 越大 转到图像坐标对应y越小 真实坐标y越大,
  col = iy 越大 转到图像坐标对应x越大
  图像坐标系下的真实图像原点对应了栅格的row = nx,col = 0
        ↑x
        ↑
        ↑
        ↑
        ↑
        ↑
  origin↑ → → → → → → → y
  */
  static constexpr size_t Z_NUM = 256;
  Eigen::Vector3i voxel_origin;          // 当前自车所在栅格
  float voxel_resolution{0.1};           // voxel分辨率 m/voxel
  float z_resolution{0.1};               //
  Vec3D voxel_size{128.0, 128.0, 12.8};  // voxel真实 m z/voxel必须为64的整数倍
  Eigen::Vector3i voxel_num{1280, 1280, Z_NUM};         // voxel个数
  std::vector<std::vector<std::bitset<Z_NUM>>> voxels;  // 每个voxel的z占用状态
  cv::Mat debug_img;

  bool Init(const YAML::Node& config);
  bool ReInit(const SegformerCommand& command);
  cv::Mat DrawVoxels();
  // 转换成img可视化的点
  inline cv::Point voxel_to_img(int voxel_row, int voxel_col) {
    return cv::Point(voxel_col, voxel_num.x() - 1 - voxel_row);
  }
  inline void Reset(int x, int y) {
    voxels[x][y].reset();  // 重置voxels[x][y]的所有bit为0
  }
  inline void Set(int x, int y, int z) {
    voxels[x][y].set(z);  // 设置voxels[x][y]的第z个bit为1
  }
  inline void Reset(int x, int y, int z) {
    voxels[x][y].reset(z);  // 重置voxels[x][y]的第z个bit为0
  }
  inline bool Check(int x, int y, int z) {
    return voxels[x][y].test(z);  // 检查voxels[x][y]的第z个bit是否为1
  }
};
struct ROI {
  std::vector<float> x_limit{-20, 128};
  std::vector<float> y_limit{-20, 20};
  std::vector<float> z_limit{-2, 10};
  inline bool Init(const YAML::Node& config) {
    rally::yamlRead(config, "x", x_limit);
    rally::yamlRead(config, "y", y_limit);
    rally::yamlRead(config, "z", z_limit);
    return true;
  }
  inline bool InROI(const Eigen::Vector3d& point) {
    return (point.x() >= x_limit[0] && point.x() <= x_limit[1]) &&
           (point.y() >= y_limit[0] && point.y() <= y_limit[1]) &&
           (point.z() >= z_limit[0] && point.z() <= z_limit[1]);
  }
};
class VoxelManager {
 public:
  using Ptr = std::shared_ptr<VoxelManager>;
  static VoxelManager& GetInstance() {
    static VoxelManager voxel_manager;
    return voxel_manager;
  }
  bool Init(const YAML::Node& config);
  inline bool ReInit(const SegformerCommand command) {
    return voxel_data_.ReInit(command);
  }
  void UpdateVoxels(int64_t time_stamp, const Eigen::Affine3d& new_veh_to_rel);
  bool SaveVoxel(const Eigen::Vector3d& point_base,
                 Eigen::Vector3i& point_voxel);
  Eigen::Vector3d VoxelToOdom(const Eigen::Vector3i& point_voxel);
  Eigen::Vector3d VoxelToBase(const Eigen::Vector3i& point_voxel);
  inline Eigen::Affine3d& GetVehToRel() { return veh_to_rel; }
  Eigen::Vector3i BaseToVoxel(const Eigen::Vector3d& point_base);

 private:
  bool inited_{false};
  ROI roi_;
  VoxelData voxel_data_;
  bool show_image_{false};
  // 栅格参考坐标系，以fist_frame_to_rel_pose作为参考系
  // 栅格坐标系，栅格参考坐标系离散化并且存在offset，以voxel_ref_center作为中心点（0.5nx
  // 0.5ny 0.5nz）
  Eigen::Vector3d voxel_ref_center{64, 0, 0};  // 中心base坐标栅格参考坐标系原点

  int64_t time_stamp_{0};  // ns
  Eigen::Affine3d veh_to_rel;
  // 首帧参考点位姿，也是栅格参考坐标系的过渡系（差一个voxel_ref_center offset）
  Eigen::Affine3d fist_frame_to_rel_pose;
  // 当前滑窗参考点位置，位于栅格参考坐标系下，用于差值计算
  Eigen::Vector3d center_rel_pose;

    Eigen::Vector3i OdomToVoxel(const Eigen::Vector3d& point_odom);

  inline int wrap(int idx, int max_num) {
    idx %= max_num;
    if (idx < 0) idx += max_num;
    return idx;
  }
  cv::Mat DrawRoi();
};
}  // namespace perception
}  // namespace robosense
#endif