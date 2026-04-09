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

#include "robot/perception/uwb_postprocess/common/utils.h"
namespace robot {
namespace perception {
void addOdomInfo(const Eigen::Affine3f& base_to_rel_trans,
                 ObjectFusion& obj_fusion) {
  float vel =
      std::hypotf(obj_fusion.velocity_base.x, obj_fusion.velocity_base.y);
  float acc = std::hypotf(obj_fusion.acceleration_base.x,
                          obj_fusion.acceleration_base.y);
  vel = isSameDirection(obj_fusion.velocity_base, obj_fusion.yaw_base) ? vel
                                                                       : -vel;
  acc = isSameDirection(obj_fusion.acceleration_base, obj_fusion.yaw_base)
            ? acc
            : -acc;
  Eigen::Vector3f pos_base(obj_fusion.box_center_base.x,
                           obj_fusion.box_center_base.y,
                           obj_fusion.box_center_base.z);
  Eigen::Vector3f pos_odom;
  pos_odom = transformTo(base_to_rel_trans, pos_base);

  obj_fusion.box_center_odom.x = pos_odom[0];
  obj_fusion.box_center_odom.y = pos_odom[1];
  obj_fusion.box_center_odom.z = pos_odom[2];

  Eigen::Vector3f middle_center_edge_pt = Eigen::Vector3f(
      obj_fusion.box_center_base.x +
          std::cos(obj_fusion.yaw_base) * obj_fusion.box_size.length * 0.5,
      obj_fusion.box_center_base.y +
          std::sin(obj_fusion.yaw_base) * obj_fusion.box_size.length * 0.5,
      obj_fusion.box_center_base.z);

  Eigen::Vector3f pos_odom_middle;
  pos_odom_middle = transformTo(base_to_rel_trans, middle_center_edge_pt);

  obj_fusion.yaw_odom = std::atan2(pos_odom_middle[1] - pos_odom[1],
                                   pos_odom_middle[0] - pos_odom[0]);
  obj_fusion.velocity_odom.x = vel * std::cos(obj_fusion.yaw_odom);
  obj_fusion.velocity_odom.y = vel * std::sin(obj_fusion.yaw_odom);
  obj_fusion.acceleration_odom.x = acc * std::cos(obj_fusion.yaw_odom);
  obj_fusion.acceleration_odom.y = acc * std::sin(obj_fusion.yaw_odom);

  obj_fusion.polygon_odom.clear();
  obj_fusion.polygon_odom.reserve(obj_fusion.polygon_base.size());
  for (const auto& pt : obj_fusion.polygon_base) {
    pos_base = Eigen::Vector3f(pt.x, pt.y, 0.0);
    pos_odom = transformTo(base_to_rel_trans, pos_base);
    obj_fusion.polygon_odom.emplace_back(pos_odom[0], pos_odom[1]);
  }
}
void MatToImage(const cv::Mat& mat, const Header& header,
                perception::Image::Ptr& image) {
  image->header = header;
  image->height = mat.rows;
  image->width = mat.cols;
  image->is_bigendian = 0;
  image->encoding = "bgr8";
  image->step = image->width * 3;
  image->data_ptr = reinterpret_cast<char*>(const_cast<uint8_t*>(mat.data));
  image->ptr = std::shared_ptr<void>(
      new cv::Mat(mat),  // 新分配了一个 Mat 对象，其内容是 mat 的浅拷贝
      [](void* p) {
        delete static_cast<cv::Mat*>(p);  // 用 shared_ptr 管理这份拷贝
      });
}
void MatToCompressedImage(cv::Mat& mat, const Header& header,
                          perception::CompressedImage::Ptr& image) {
  image->header = header;
  std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 95};
  if (!cv::imencode(".jpg", mat, image->data, params)) {
    AERROR << " imencode error!";
  }
  image->format = "rgb8; jpeg compressed rgb8";  // 实际上读取的好像就是bgr
  //   cv::Mat decoded = cv::imdecode(image->data, cv::IMREAD_COLOR);
  //   cv::imshow("Decoded Image", decoded);
  //   cv::waitKey(1);
}
float GetIoU(const cv::Rect& rect1, const cv::Rect& rect2) {
  int x1 = std::max(rect1.x, rect2.x);
  int y1 = std::max(rect1.y, rect2.y);
  int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
  int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

  int interWidth = x2 - x1;
  int interHeight = y2 - y1;

  if (interWidth <= 0 || interHeight <= 0) {
    return 0.0f;  // 无交集
  }

  int interArea = interWidth * interHeight;
  int unionArea = rect1.area() + rect2.area() - interArea;

  return static_cast<float>(interArea) / unionArea;
}
// 相交面积占第二个box的比例
float GetAreaRatio(const cv::Rect& rect1, const cv::Rect& rect2) {
  int x1 = std::max(rect1.x, rect2.x);
  int y1 = std::max(rect1.y, rect2.y);
  int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
  int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

  int interWidth = x2 - x1;
  int interHeight = y2 - y1;

  if (interWidth <= 0 || interHeight <= 0) {
    return 0.0f;  // 无交集
  }
  int interArea = interWidth * interHeight;

  return static_cast<float>(interArea) / rect2.area();
}
void RectToPose(const cv::Rect& rect, const CameraCalib::Ptr& cam_calib,
                const float& dist, Eigen::Vector3d& pose, float& tall,
                float& width) {
  Eigen::Vector3d camera_pt_left =
      cam_calib->UVToCameraPt(rect.x, rect.y, dist, false);
  Eigen::Vector3d camera_pt_right = cam_calib->UVToCameraPt(
      rect.x + rect.width, rect.y + rect.height, dist, false);
  Eigen::Vector3d veh_pt_left = cam_calib->CameraPtToVehPt(camera_pt_left);
  Eigen::Vector3d veh_pt_right = cam_calib->CameraPtToVehPt(camera_pt_right);
  pose = (veh_pt_left + veh_pt_right) / 2;
  tall = std::abs(veh_pt_left.z() - veh_pt_right.z());
  width = std::max(std::abs(veh_pt_left.y() - veh_pt_right.y()),
                   std::abs(veh_pt_left.x() - veh_pt_right.x()));
}
void PoseToRect(const Eigen::Vector3d& pose, const BoxSize& size,
                const CameraCalib::Ptr& cam_calib, cv::Rect& rect) {
  cv::Point p_left, p_right;
  Eigen::Vector3d veh_pt_left(pose.x(), pose.y() + size.width / 2,
                              pose.z() + size.height / 2);
  Eigen::Vector3d veh_pt_right(pose.x(), pose.y() - size.width / 2,
                               pose.z() - size.height / 2);
  Eigen::Vector3d camera_pt_left = cam_calib->VehPtToCameraPt(veh_pt_left);
  cam_calib->CameraPtToUV(camera_pt_left, p_left.x, p_left.y, false, true);
  Eigen::Vector3d camera_pt_right = cam_calib->VehPtToCameraPt(veh_pt_right);
  cam_calib->CameraPtToUV(camera_pt_right, p_right.x, p_right.y, false, true);
  rect = cv::Rect(p_left, p_right);
}
// 这里不对，纯方向向量不加平移，平移对应的是射线方程中的原点
void PointToDirection(const cv::Point& p, const CameraCalib::Ptr& cam_calib,
                      Eigen::Vector3d& direction) {
  const auto& inner_mat = cam_calib->GetInnerMat();

  Eigen::Vector3d camera_direction((p.x - inner_mat(0, 2)) / inner_mat(0, 0),
                                   (p.y - inner_mat(1, 2)) / inner_mat(1, 1),
                                   1);
  camera_direction.normalize();
  direction = cam_calib->CameraPtToVehPt(camera_direction);
}
// 用2D框和真实高度计算3D中心点位置
void RectWithTallToPose(const cv::Rect& rect, const CameraCalib::Ptr& cam_calib,
                        const float& tall, Eigen::Vector3d& pose) {
  float fy = cam_calib->GetInnerMat()(1, 1);
  float pixel_depth = tall * fy / rect.height;
  Eigen::Vector3d camera_pt =
      cam_calib->UVToCameraPt(rect.x + 0.5 * rect.width,
                              rect.y + 0.5 * rect.height, pixel_depth, false);
  pose = cam_calib->CameraPtToVehPt(camera_pt);
}
void RectWithWidthToPose(const cv::Rect& rect,
                         const CameraCalib::Ptr& cam_calib, const float& width,
                         Eigen::Vector3d& pose) {
  float fx = cam_calib->GetInnerMat()(0, 0);
  float pixel_depth = width * fx / rect.width;
  Eigen::Vector3d camera_pt =
      cam_calib->UVToCameraPt(rect.x + 0.5 * rect.width,
                              rect.y + 0.5 * rect.height, pixel_depth, false);
  pose = cam_calib->CameraPtToVehPt(camera_pt);
}

// CPU拷贝到NPU 
void CopyToNpu(float* npu_address, float* cpu_address, int copy_size){
  memcpy(npu_address, static_cast<void*>(cpu_address),
         copy_size * sizeof(float));
  // npu_address = static_cast<float*>(cpu_address);
}
// NPU拷贝到CPU
void CopyToCpu(float* cpu_address, float* npu_address, int copy_size) {
  memcpy(cpu_address, static_cast<void*>(npu_address),
         copy_size * sizeof(float));
//    cpu_address = static_cast<float*>(npu_address);

}

void SaveFloatToTxt(float* data, int N, const std::string& filename) {
  std::ofstream fout(filename);
  if (!fout) {
    std::cerr << "Cannot open file: " << filename << std::endl;
    return;
  }

  for (int i = 0; i < N; ++i) {
    fout << data[i] << "\n";  // 每行一个 float
  }

  fout.close();
}
void SaveGpuFloatToTxt(float* data, int N, const std::string& filename) {
  float* cpu_data = new float[N];
  CopyToCpu(cpu_data, data, N);
  SaveFloatToTxt(cpu_data, N, filename);
}
// 将一个remap后的rect恢复到原图带畸变中
cv::Rect RecoverRect(const cv::Rect& rect, int resize_w, int resize_h,
                     int ori_w, int ori_h, cv::Mat ori_k, cv::Mat ori_dist) {
  std::vector<cv::Point> corners = {
      rect.tl(), cv::Point(rect.x + rect.width - 1, rect.y),
      cv::Point(rect.x, rect.y + rect.height - 1),
      cv::Point(rect.x + rect.width - 1, rect.y + rect.height - 1)};
  std::vector<cv::Point> remap_corners;
  for (auto& p : corners) {
    remap_corners.emplace_back(
        RecoverPoints(p, resize_w, resize_h, ori_w, ori_h, ori_k, ori_dist));
  }
  return cv::boundingRect(remap_corners);
}
// 将一个remap后的点恢复到原图带畸变中
cv::Point RecoverPoints(const cv::Point& resize_p, int resize_w, int resize_h,
                        int ori_w, int ori_h, cv::Mat ori_k, cv::Mat ori_dist) {
  cv::Point out_p = resize_p;
  out_p.x = resize_p.x * ori_w / resize_w;
  out_p.y = resize_p.y * ori_h / resize_h;
  float fx = ori_k.at<float>(0, 0);
  float fy = ori_k.at<float>(1, 1);
  float cx = ori_k.at<float>(0, 2);
  float cy = ori_k.at<float>(1, 2);

  // 归一化坐标（相机坐标系下单位深度）
  float x = (out_p.x - cx) / fx;
  float y = (out_p.y - cy) / fy;

  std::vector<cv::Point3f> object_pts = {cv::Point3f(x, y, 1.0f)};
  std::vector<cv::Point2f> distorted_pixels;

  // 鱼眼不去畸变等价于直接resize
  if (ori_dist.rows == 4 || ori_dist.cols == 4) {
    // cv::fisheye::projectPoints(object_pts, distorted_pixels,
    //                            cv::Mat::zeros(3, 1, CV_64F),
    //                            cv::Mat::zeros(3, 1, CV_64F), ori_k,
    //                            ori_dist);
    return out_p;
  } else {
    cv::projectPoints(object_pts, cv::Mat::zeros(3, 1, CV_64F),  // 无旋转
                      cv::Mat::zeros(3, 1, CV_64F),              // 无平移
                      ori_k, ori_dist, distorted_pixels);
  }
  out_p.x = distorted_pixels[0].x;
  out_p.y = distorted_pixels[0].y;
  return out_p;
}
// 将一个原图点remap到去畸变图
cv::Point RemapPoints(const cv::Point& ori_p, int ori_w, int ori_h,
                      int resize_w, int resize_h, const cv::Mat& ori_k,
                      const cv::Mat& ori_dist) {
  std::vector<cv::Point2f> distorted_points = {
      cv::Point2f(static_cast<float>(ori_p.x), static_cast<float>(ori_p.y))};
  std::vector<cv::Point2f> undistorted_points;

  // 3. 判断是否为 fisheye 模型
  if (ori_dist.rows == 4 || ori_dist.cols == 4) {
    // fisheye 去畸变
    // cv::fisheye::undistortPoints(distorted_points, undistorted_points, ori_k,
    //                              ori_dist, cv::noArray(), ori_k);
    undistorted_points = distorted_points;
  } else {
    // 普通畸变模型
    cv::undistortPoints(distorted_points, undistorted_points, ori_k, ori_dist,
                        cv::noArray(), ori_k);
  }
  // 4. 像素坐标输出
  cv::Point out_p;
  out_p.x = static_cast<int>(undistorted_points[0].x * resize_w / ori_w);
  out_p.y = static_cast<int>(undistorted_points[0].y * resize_h / ori_h);
  return out_p;
}
// 将一个rect remap到去畸变图
cv::Rect RemapRect(const cv::Rect& rect, int ori_w, int ori_h, int resize_w,
                   int resize_h, const cv::Mat& ori_k,
                   const cv::Mat& ori_dist) {
  std::vector<cv::Point> corners = {
      rect.tl(), cv::Point(rect.x + rect.width - 1, rect.y),
      cv::Point(rect.x, rect.y + rect.height - 1),
      cv::Point(rect.x + rect.width - 1, rect.y + rect.height - 1)};
  std::vector<cv::Point> remap_corners;
  for (auto& p : corners) {
    // AINFO << " ORI P " << p << " " << "remap " << RemapPoints(p, map1, map2);
    remap_corners.emplace_back(
        RemapPoints(p, ori_w, ori_h, resize_w, resize_h, ori_k, ori_dist));
  }
  return cv::boundingRect(remap_corners);
}
}  // namespace perception
}  // namespace robot