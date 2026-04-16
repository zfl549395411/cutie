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

#include "hyper_vision/perception/occupancy_objects/occupancy_detection/polygon.h"

namespace robosense {
namespace perception {

PolygonDivide::PolygonDivide(const YAML::Node &cfg_node,
                             Projector::Ptr proj_ptr)
    : ogm(OccGridMap::GetInstance()),
      rd(RangeDetection::GetInstance()),
      gd(GroundDetection::GetInstance()) {
  rally::yamlRead(cfg_node, "filter_road_studs", filter_road_studs_);
  rally::yamlRead(cfg_node, "filter_flying_obj", filter_flying_obj_);

  proj_ptr_ = proj_ptr;
  local_img = cv::Mat(ogm.base_rows_, ogm.base_cols_, CV_8U, cv::Scalar(0));
  Ogm<double> tmp_info(ogm.base_rows_, ogm.base_cols_);
  for (int i = 0; i < INFO_NUM; i++) {
    local_info.push_back(tmp_info);
  }
  erase_voxel_confidence_.resize(ogm.base_rows_, ogm.base_cols_);
}

void PolygonDivide::getLocalImg(const Ogm<GaussGrid> &nonground_info_map,
                                const Ogm<uint8_t> &intensity_map) {
  local_visited_id.clear();
  local_img.setTo(cv::Scalar(0));
  for (int i = 0; i < 2; i++) {
    local_info[i].setZero();
  }
  erase_voxel_confidence_.setZero();

  Eigen::Matrix3d rot_grid2ego = ogm.tf_grid2ego_.linear();
  ogm.forInSubMap([&](int sub_x, int sub_y, int map_x, int map_y,
                      double grid_px, double grid_py) {
    const auto &track_cnt = ogm.at(kTrackCntMap)(map_x, map_y);
    if (track_cnt < ogm.pub_track_cnt_) {
      return;
    }
    const auto &grid_type = ogm.at(kTypeMap)(map_x, map_y);
    const auto &confidence = ogm.at(kConfMap)(map_x, map_y);
    const auto &det_obj_dis = ogm.at(kDetDistMap)(map_x, map_y);
    const auto &nonground_grid = nonground_info_map(sub_x, sub_y);
    const auto &intensity = intensity_map(sub_x, sub_y);
    const auto &gauss_grid = ogm.gauss(map_x, map_y);
    Eigen::Vector3d grid_pt{grid_px, grid_py, 0.0};
    grid_pt = grid_pt + gauss_grid.getCentroid();
    Eigen::Vector3d ego_pt = ogm.tf_grid2ego_ * grid_pt;
    int vi = ogm.xBaseToGrid(ego_pt.x());
    int vj = ogm.yBaseToGrid(ego_pt.y());
    if (vi < 0 || vi >= ogm.base_rows_ - 1 || vj < 0 ||
        vj >= ogm.base_cols_ - 1) {
      // 此处控制占据栅格比整个图少一个维度
      return;
    }
    Eigen::Matrix3d cov = gauss_grid.getCovMat();
    rotateCovMat(cov, rot_grid2ego);
    int pts_cnt = gauss_grid.size();
    cov = pts_cnt * (cov + ego_pt * ego_pt.transpose());
    // 普通占据栅格
    if (int(local_info[1](vi, vj)) == 0) {
      local_info[0](vi, vj) = ogm.at(kTrackIdMap)(map_x, map_y);
      local_info[1](vi, vj) = pts_cnt;
      local_info[2](vi, vj) = pts_cnt * ego_pt.x();
      local_info[3](vi, vj) = pts_cnt * ego_pt.y();
      local_info[4](vi, vj) = pts_cnt * ego_pt.z();
      local_info[5](vi, vj) = cov(0, 0);
      local_info[6](vi, vj) = cov(1, 1);
      local_info[7](vi, vj) = cov(2, 2);
      local_info[8](vi, vj) = cov(0, 1);
      local_info[9](vi, vj) = cov(0, 2);
      local_info[10](vi, vj) = cov(1, 2);
      local_info[11](vi, vj) = sub_x;
      local_info[12](vi, vj) = sub_y;
      local_info[13](vi, vj) = gd.getGndHeight(ego_pt.x(), ego_pt.y());
      local_info[15](vi, vj) = confidence;
      local_info[16](vi, vj) = det_obj_dis;
      local_info[17](vi, vj) = nonground_grid.getDiffZ();
      local_info[18](vi, vj) = intensity;
      if (grid_type == kDynamicGrid) {
        local_img.ptr<uchar>(vi)[vj] = kDynamicPolygon;
      } else {
        local_img.ptr<uchar>(vi)[vj] = kStaticPolygon;
      }
    } else {
      local_info[1](vi, vj) += pts_cnt;
      local_info[2](vi, vj) += pts_cnt * ego_pt.x();
      local_info[3](vi, vj) += pts_cnt * ego_pt.y();
      local_info[4](vi, vj) += pts_cnt * ego_pt.z();
      local_info[5](vi, vj) += cov(0, 0);
      local_info[6](vi, vj) += cov(1, 1);
      local_info[7](vi, vj) += cov(2, 2);
      local_info[8](vi, vj) += cov(0, 1);
      local_info[9](vi, vj) += cov(0, 2);
      local_info[10](vi, vj) += cov(1, 2);
      local_info[17](vi, vj) =
          std::max(local_info[17](vi, vj), nonground_grid.getDiffZ());
      local_info[18](vi, vj) =
          std::max(local_info[18](vi, vj), (double)intensity);
    }
  });
}

void PolygonDivide::updateConvex() {
  std::unordered_map<int, std::vector<cv::Point>> cluster_points;
  std::unordered_set<int> valid_id;
  for (int i = 0; i < local_img.rows; ++i) {
    for (int j = 0; j < local_img.cols; ++j) {
      // valid and classified
      int tid = int(local_info[0](i, j));
      int grid_type = local_img.ptr<uchar>(i)[j];
      int track_id = -1;
      if (grid_type == kDynamicPolygon) {
        track_id = ogm.subObjId(local_info[11](i, j), local_info[12](i, j));
      }

      if (grid_type != 0 && tid != 0) {
        if (!polygon_map_.count(tid) || id_manager_.searchIdCount(tid) == -6 ||
            grid_type != polygon_map_[tid]->polygon_type ||
            (track_id != -1 && polygon_map_[tid]->object_track_id != -1 &&
             polygon_map_[tid]->object_track_id != track_id)) {
          local_info[0](i, j) = 0;
          // 类型不同或id不同重新聚类时要记录上次分割的置信度信息
          if (polygon_map_.find(tid) != polygon_map_.end()) {
            erase_voxel_confidence_(i, j) = polygon_map_[tid]->confidence;
          }
          continue;
        }
        if (polygon_map_[tid]->polygon_type == kDynamicPolygon &&
            polygon_map_[tid]->object_track_id == -1 && track_id != -1) {
          polygon_map_[tid]->object_track_id = track_id;
        }
        valid_id.insert(tid);
        cluster_points[tid].emplace_back(cv::Point(j, i));
      }
    }
  }
  // 利用计数器消除小目标
  auto erase_id_vec = id_manager_.updateId(valid_id);
  for (auto &id : erase_id_vec) {
    erase_cluster(id);
  }
  // 先更新每个聚类,对小目标的合并一定要基于当前的信息进行，所以要先更新了当前的信息
  for (auto &cluster : cluster_points) {
    std::vector<cv::Point> cur_convex;
    std::vector<cv::Point2f> ellipse_convex;
    cv_convex(cluster.second, cur_convex, true);
    // cv::convexHull(cv::Mat(cluster.second), cur_convex, true);
    float convex_area = convex_hull_area(cur_convex);
    // voxels of hole more than valid voxels && the size of hole bigger than
    // merge_thres
    // 横跨自车polgyon判断阈值要减小
    float area_thresh_weight = 1;
    if (cur_convex.size() > 0 && ogm.xBaseToPos(cur_convex[0].y) < 30) {
      bool is_left = ogm.yBaseToPos(cur_convex[0].x) > 0;
      for (const auto &p : cur_convex) {
        if (is_left ^ (ogm.yBaseToPos(p.x) > 0)) {
          area_thresh_weight = 0.5;
          break;
        }
      }
    }

    if (std::abs(convex_area) > 2 * cluster.second.size() &&
        std::abs(convex_area) - cluster.second.size() >
            area_th_ * area_thresh_weight) {  // merge_thresh
      for (auto p : cluster.second) {
        local_info[0](p.y, p.x) = 0;
        erase_voxel_confidence_(p.y, p.x) =
            polygon_map_[cluster.first]->confidence;
      }
      erase_cluster(cluster.first);

    } else {
      local_visited_id.insert(cluster.first);
      polygon_map_[cluster.first]->voxel_polygon_cv = cur_convex;
      std::vector<cv::Point2f> voxel_gauss_points;
      gauss_convex(cluster.second, ellipse_convex, voxel_gauss_points);
      // if(voxel_gauss_points.size() <= 0){
      //   erase_cluster(cluster.first);
      //   for (auto p : cluster.second) {
      //     local_info[0](p.y, p.x) = 0;
      //   }
      //    continue;
      // }
      polygon_map_[cluster.first]->gauss_polygon_base = ellipse_convex;
      float area = convex_hull_area(cur_convex);
      polygon_map_[cluster.first]->area = area;
      polygon_map_[cluster.first]->valid_count = cluster.second.size();
      if (polygon_map_[cluster.first]->valid_count > small_target_thresh)
        id_manager_.setId(cluster.first, 1);
    }
  }
  for (auto &cluster : cluster_points) {
    // 合并小目标
    if (id_manager_.searchIdCount(cluster.first) == 0) {
      int cur_id = cluster.first;
      if (polygon_map_[cur_id]->gauss_polygon_base.size() <= 0) {
        continue;
      }
      std::vector<cv::Point> merge_convex;
      std::vector<cv::Point2f> merge_convex_res;
      int merge_id =
          enlarge_small_cluster(cluster.second, cur_id, merge_convex);
      // 无法合并
      if (merge_id == cur_id || !cluster_points.count(merge_id)) {
        continue;
      }
      merge_polygon(polygon_map_[cur_id]->gauss_polygon_base, merge_id,
                    merge_convex_res);
      erase_cluster(cur_id);
      // 更新合并后的id对应信息
      if (local_visited_id.count(cur_id))
        local_visited_id.erase(local_visited_id.find(cur_id));
      polygon_map_[merge_id]->voxel_polygon_cv = merge_convex;

      polygon_map_[merge_id]->gauss_polygon_base = merge_convex_res;
      for (auto p : cluster.second) {
        cluster_points[merge_id].emplace_back(p);
        local_info[0](p.y, p.x) = merge_id;
      }
      polygon_map_[merge_id]->valid_count = cluster_points[merge_id].size();
      float area = convex_hull_area(merge_convex);
      polygon_map_[merge_id]->area = area;
      if (polygon_map_[merge_id]->valid_count > small_target_thresh)
        id_manager_.setId(merge_id, 1);
    }
  }
}

void PolygonDivide::clusterPolygon(const cv::Point &start) {
  int valid_count = 0, row_count = 0;
  int cur_id = id_manager_.getId();
  int start_grid_type = local_img.ptr<uchar>(start.y)[start.x];
  int track_id = -1;
  if (start_grid_type == kDynamicPolygon) {
    track_id = ogm.subObjId(local_info[11](start.y, start.x),
                            local_info[12](start.y, start.x));
  }
  polygon_map_[cur_id] = std::make_shared<Polygon>(
      cur_id, static_cast<PolygonType>(start_grid_type), track_id);

  // all convex corner
  std::vector<cv::Point> res;
  std::vector<cv::Point> convex_left;
  std::vector<cv::Point> convex_right;

  // valid points of the cluster
  std::vector<cv::Point> cluster_points;
  //
  std::vector<cv::Point> gauss_edge_points;

  // id -> number of connection
  std::unordered_map<int, int> connect_id;

  // corner voxel of pre row
  cv::Point pre_left_voxel(-1, -1), pre_right_voxel(-1, -1);
  float accumulate_area = 0;
  int cur_row = start.y;
  int cur_min_x = start.x, cur_max_x = start.x;
  int pre_min_x = start.x, pre_max_x = start.x;
  // 聚类中最高的历史置信度
  OccConfidence cluster_erase_voxel_confidence = kLowConfOcc;
  // 当前行聚类过程访问到了已经删除的栅格置信度，默认为低
  OccConfidence cur_row_erase_voxel_confidence = kLowConfOcc;

  // y small-top
  std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>,
                      std::greater<std::pair<int, int>>>
      bfs;
  bfs.emplace(std::make_pair(start.y, start.x));
  std::vector<bool> visited(local_img.cols, false);
  std::vector<bool> next_visited(local_img.cols, false);
  visited[start.x] = true;
  bool end = false;
  while (!bfs.empty() && !end) {
    auto point = bfs.top();
    bfs.pop();
    if (point.first != cur_row) {
      // first row
      cv::Point cur_left_voxel(cur_min_x, cur_row);
      cv::Point cur_right_voxel(cur_max_x, cur_row);
      cv::Point left_up(cur_min_x, cur_row);
      cv::Point left_down(cur_min_x, cur_row + 1);
      cv::Point right_up(cur_max_x + 1, cur_row);
      cv::Point right_down(cur_max_x + 1, cur_row + 1);
      if (pre_left_voxel.x == -1) {
        convex_left.emplace_back(left_up);
        convex_left.emplace_back(left_down);
        convex_right.emplace_back(right_up);
        convex_right.emplace_back(right_down);

        // init accumulate_area
        accumulate_area = triangle_area(REF, convex_right[0], convex_left[0]);
        accumulate_area += triangle_area(REF, convex_left[0], convex_left[1]);
        accumulate_area += triangle_area(REF, convex_right[1], convex_right[0]);

        for (int c = cur_min_x; c <= cur_max_x; ++c) {
          int tid = local_info[0](cur_row, c);
          if (int(local_img.ptr<uchar>(cur_row)[c]) == start_grid_type &&
              tid == 0) {
            gauss_edge_points.emplace_back(cv::Point(c, cur_row));
          }
        }
      } else {
        // convex hull algorithm
        std::queue<cv::Point> log_left, log_right;
        std::vector<cv::Point> log_convex_left = convex_left;
        std::vector<cv::Point> log_convex_right = convex_right;
        float temp_area = accumulate_area;
        // try to put left_up
        pop_left_convex_corner_cv(left_up, convex_left, temp_area, log_left);
        temp_area += triangle_area(REF, convex_left.back(), left_up);
        convex_left.emplace_back(left_up);
        // try to put left_down
        pop_left_convex_corner_cv(left_down, convex_left, temp_area, log_left);
        temp_area += triangle_area(REF, convex_left.back(), left_down);
        convex_left.emplace_back(left_down);
        // try to put right_up
        pop_right_convex_corner_cv(right_up, convex_right, temp_area,
                                   log_right);
        temp_area += triangle_area(REF, right_up, convex_right.back());
        convex_right.emplace_back(right_up);
        // try to put right_down
        pop_right_convex_corner_cv(right_down, convex_right, temp_area,
                                   log_right);
        temp_area += triangle_area(REF, right_down, convex_right.back());
        convex_right.emplace_back(right_down);

        // complete convex hull area
        float convex_area =
            temp_area + triangle_area(REF, left_down, right_down);

        if (std::fabs(convex_area) - valid_count - row_count > area_th_) {
          end = true;
          // recover old corner
          convex_left = log_convex_left;
          convex_right = log_convex_right;
          // end row
          break;
        }
        valid_count += row_count;
        row_count = 0;
        accumulate_area = temp_area;
      }

      for (int c = cur_min_x; c <= cur_max_x; ++c) {
        int tid = local_info[0](cur_row, c);
        if (int(local_img.ptr<uchar>(cur_row)[c]) == start_grid_type &&
            tid == 0) {
          cluster_points.emplace_back(cv::Point(c, cur_row));
          local_info[0](cur_row, c) = cur_id;
        }
      }
      // 非第一行，更新一下高斯栅格的边缘
      if (pre_left_voxel.x != -1) {
        // for(int c = pre_min_x; c<=cur_min_x;++c){
        //     int tid = local_info[0](cur_row-1,c);
        //     if(int(local_img.ptr<uchar>(cur_row-1)[c])!=0 && tid==cur_id){
        //             gauss_edge_points.emplace_back(cv::Point(c,cur_row-1));
        //     }
        // }

        // for(int c = pre_min_x;c>=cur_min_x;--c){
        //     int tid = local_info[0](cur_row,c);
        //     if(int(local_img.ptr<uchar>(cur_row)[c])!=0 && tid==cur_id){
        //             gauss_edge_points.emplace_back(cv::Point(c,cur_row));
        //     }
        // }

        // for(int c = cur_max_x; c<=pre_max_x;++c){
        //     int tid = local_info[0](cur_row-1,c);
        //     if(int(local_img.ptr<uchar>(cur_row-1)[c])!=0 && tid==cur_id){
        //             gauss_edge_points.emplace_back(cv::Point(c,cur_row-1));
        //     }
        // }

        // for(int c = cur_max_x;c>=pre_max_x;--c){
        //     int tid = local_info[0](cur_row,c);
        //     if(int(local_img.ptr<uchar>(cur_row)[c])!=0 && tid==cur_id){
        //             gauss_edge_points.emplace_back(cv::Point(c,cur_row));
        //     }
        // }

        if (pre_min_x <= cur_min_x) {
          for (int c = pre_min_x; c <= cur_min_x; ++c) {
            int tid = local_info[0](cur_row - 1, c);
            if (int(local_img.ptr<uchar>(cur_row - 1)[c]) == start_grid_type &&
                tid == cur_id) {
              gauss_edge_points.emplace_back(cv::Point(c, cur_row - 1));
            }
          }
          if (int(local_img.ptr<uchar>(cur_row)[cur_min_x]) ==
                  start_grid_type &&
              int(local_info[0](cur_row, cur_min_x)) == cur_id) {
            gauss_edge_points.emplace_back(cv::Point(cur_min_x, cur_row));
          }
        } else {
          if (int(local_img.ptr<uchar>(cur_row - 1)[pre_min_x]) ==
                  start_grid_type &&
              int(local_info[0](cur_row - 1, pre_min_x)) == cur_id) {
            gauss_edge_points.emplace_back(cv::Point(pre_min_x, cur_row - 1));
          }
          for (int c = cur_min_x; c <= pre_min_x; ++c) {
            int tid = local_info[0](cur_row, c);
            if (int(local_img.ptr<uchar>(cur_row)[c]) == start_grid_type &&
                tid == cur_id) {
              gauss_edge_points.emplace_back(cv::Point(c, cur_row));
            }
          }
        }

        if (cur_max_x <= pre_max_x) {
          for (int c = cur_max_x; c <= pre_max_x; ++c) {
            int tid = local_info[0](cur_row - 1, c);
            if (int(local_img.ptr<uchar>(cur_row - 1)[c]) == start_grid_type &&
                tid == cur_id) {
              gauss_edge_points.emplace_back(cv::Point(c, cur_row - 1));
            }
          }
          if (int(local_img.ptr<uchar>(cur_row)[cur_max_x]) ==
                  start_grid_type &&
              int(local_info[0](cur_row, cur_max_x)) == cur_id) {
            gauss_edge_points.emplace_back(cv::Point(cur_max_x, cur_row));
          }
        } else {
          for (int c = pre_max_x; c <= cur_max_x; ++c) {
            int tid = local_info[0](cur_row, c);
            if (int(local_img.ptr<uchar>(cur_row)[c]) == start_grid_type &&
                tid == cur_id) {
              gauss_edge_points.emplace_back(cv::Point(c, cur_row));
            }
          }
          if (int(local_img.ptr<uchar>(cur_row - 1)[pre_max_x]) ==
                  start_grid_type &&
              int(local_info[0](cur_row - 1, pre_max_x)) == cur_id) {
            gauss_edge_points.emplace_back(cv::Point(pre_max_x, cur_row - 1));
          }
        }
      }
      cluster_erase_voxel_confidence = std::max(cur_row_erase_voxel_confidence,
                                                cluster_erase_voxel_confidence);
      cur_row_erase_voxel_confidence = kLowConfOcc;
      pre_left_voxel = cur_left_voxel;
      pre_right_voxel = cur_right_voxel;
      pre_min_x = cur_min_x;
      pre_max_x = cur_max_x;
      cur_row = point.first;
      cur_min_x = point.second;
      cur_max_x = point.second;
      visited = next_visited;
      std::fill(next_visited.begin(), next_visited.end(), 0);
      if (convex_left.size() + convex_right.size() > 10) {
        end = true;
        break;
      }
    }
    // update info of current row
    cur_row_erase_voxel_confidence =
        std::max(cur_row_erase_voxel_confidence,
                 erase_voxel_confidence_(point.first, point.second));
    ++row_count;
    cur_min_x = std::min(cur_min_x, point.second);
    cur_max_x = std::max(cur_max_x, point.second);
    // 连通的cluster
    for (auto off : BFS_CONNECT_CV) {
      cv::Point tmp(point.second + off.x, point.first + off.y);
      if (ogm.inBaseRange(tmp.y, tmp.x)) {
        int tmp_tid = local_info[0](tmp.y, tmp.x);
        int tmp_grid_type = local_img.ptr<uchar>(tmp.y)[tmp.x];
        if (tmp_tid > 0 && tmp_tid != cur_id &&
            tmp_grid_type == start_grid_type) {
          ++connect_id[tmp_tid];
        }
      }
    }
    // 连通的点
    for (auto off : BFS_OFFSET_CV) {
      cv::Point tmp(point.second + off.x, point.first + off.y);
      if (ogm.inBaseRange(tmp.y, tmp.x)) {
        int tmp_tid = local_info[0](tmp.y, tmp.x);
        int tmp_grid_type = local_img.ptr<uchar>(tmp.y)[tmp.x];
        int tmp_track_id = -1;
        if (tmp_grid_type == kDynamicPolygon) {
          tmp_track_id = ogm.subObjId(local_info[11](tmp.y, tmp.x),
                                      local_info[12](tmp.y, tmp.x));
        }
        if (tmp.y == cur_row + 1 && !next_visited[tmp.x] && tmp_tid == 0 &&
            tmp_grid_type == start_grid_type &&
            (tmp_track_id == -1 || track_id == -1 ||
             tmp_track_id == track_id)) {
          bfs.emplace(std::make_pair(tmp.y, tmp.x));
          next_visited[tmp.x] = true;
        } else if (tmp.y == cur_row && !visited[tmp.x] && tmp_tid == 0 &&
                   tmp_grid_type == start_grid_type &&
                   (tmp_track_id == -1 || track_id == -1 ||
                    tmp_track_id == track_id)) {
          bfs.emplace(std::make_pair(tmp.y, tmp.x));
          visited[tmp.x] = true;
        }

        if (tmp_tid > 0 && tmp_grid_type == start_grid_type) {
          ++connect_id[tmp_tid];
        }
      }
    }
  }

  // 非主动退出的最后一行
  if (!end) {
    cv::Point cur_left_voxel(cur_min_x, cur_row);
    cv::Point cur_right_voxel(cur_max_x, cur_row);
    cv::Point left_up(cur_min_x, cur_row);
    cv::Point left_down(cur_min_x, cur_row + 1);
    cv::Point right_up(cur_max_x + 1, cur_row);
    cv::Point right_down(cur_max_x + 1, cur_row + 1);
    if (convex_left.size() == 0) {
      // only one row
      convex_left.emplace_back(left_up);
      convex_left.emplace_back(left_down);
      convex_right.emplace_back(right_up);
      convex_right.emplace_back(right_down);
      cluster_erase_voxel_confidence = std::max(cur_row_erase_voxel_confidence,
                                                cluster_erase_voxel_confidence);
      cur_row_erase_voxel_confidence = kLowConfOcc;

      for (int c = cur_min_x; c <= cur_max_x; ++c) {
        int tid = local_info[0](cur_row, c);
        if (int(local_img.ptr<uchar>(cur_row)[c]) == start_grid_type &&
            tid == 0) {
          cluster_points.emplace_back(cv::Point(c, cur_row));
          local_info[0](cur_row, c) = cur_id;
          gauss_edge_points.emplace_back(cv::Point(c, cur_row));
        }
      }
    } else {
      std::queue<cv::Point> log_left, log_right;
      std::vector<cv::Point> log_convex_left = convex_left;
      std::vector<cv::Point> log_convex_right = convex_right;
      float temp_area = accumulate_area;

      pop_left_convex_corner_cv(left_up, convex_left, temp_area, log_left);
      temp_area += triangle_area(REF, convex_left.back(), left_up);
      convex_left.emplace_back(left_up);

      pop_left_convex_corner_cv(left_down, convex_left, temp_area, log_left);
      temp_area += triangle_area(REF, convex_left.back(), left_down);
      convex_left.emplace_back(left_down);

      pop_right_convex_corner_cv(right_up, convex_right, temp_area, log_right);
      temp_area += triangle_area(REF, right_up, convex_right.back());
      convex_right.emplace_back(right_up);

      pop_right_convex_corner_cv(right_down, convex_right, temp_area,
                                 log_right);
      temp_area += triangle_area(REF, right_down, convex_right.back());
      convex_right.emplace_back(right_down);

      // complete convex hull area
      float convex_area = temp_area + triangle_area(REF, left_down, right_down);

      if (std::fabs(convex_area) - valid_count - row_count > area_th_) {
        end = true;
        // recover old corner
        convex_left = log_convex_left;
        convex_right = log_convex_right;
        // end row
      } else {
        cluster_erase_voxel_confidence = std::max(
            cur_row_erase_voxel_confidence, cluster_erase_voxel_confidence);
        cur_row_erase_voxel_confidence = kLowConfOcc;
        valid_count += row_count;
        for (int c = cur_min_x; c <= cur_max_x; ++c) {
          int tid = local_info[0](cur_row, c);
          if (int(local_img.ptr<uchar>(cur_row)[c]) == start_grid_type &&
              tid == 0) {
            cluster_points.emplace_back(cv::Point(c, cur_row));
            local_info[0](cur_row, c) = cur_id;
            gauss_edge_points.emplace_back(cv::Point(c, cur_row));
          }
        }
        if (pre_min_x < cur_min_x) {
          for (int c = pre_min_x; c <= cur_min_x; ++c) {
            int tid = local_info[0](cur_row - 1, c);
            if (int(local_img.ptr<uchar>(cur_row - 1)[c]) == start_grid_type &&
                tid == cur_id) {
              gauss_edge_points.emplace_back(cv::Point(c, cur_row - 1));
            }
          }
        }
        if (cur_max_x < pre_max_x) {
          for (int c = cur_max_x; c <= pre_max_x; ++c) {
            int tid = local_info[0](cur_row - 1, c);
            if (int(local_img.ptr<uchar>(cur_row - 1)[c]) == start_grid_type &&
                tid == cur_id) {
              gauss_edge_points.emplace_back(cv::Point(c, cur_row - 1));
            }
          }
        }
      }
    }
  }
  res = convex_left;
  for (int i = convex_right.size() - 1; i >= 0; --i) {
    res.emplace_back(convex_right[i]);
  }
  std::vector<cv::Point> cur_convex;
  cv_convex(res, cur_convex, false);

  std::vector<cv::Point2f> ellipse_convex;
  std::vector<cv::Point2f> ellipse_convex_res;
  std::vector<cv::Point2f> voxel_gauss_points;
  gauss_convex(gauss_edge_points, ellipse_convex, voxel_gauss_points);

  // 包括曾经断开栅格的点，直接记录为对应置信度
  if (cluster_erase_voxel_confidence != kLowConfOcc) {
    polygon_map_[cur_id]->confidence = cluster_erase_voxel_confidence;
  }

  std::vector<std::pair<int, int>> id_sort(connect_id.begin(),
                                           connect_id.end());
  std::sort(id_sort.begin(), id_sort.end(),
            [](const std::pair<int, int> &p1, const std::pair<int, int> &p2) {
              return p1.second > p2.second;
            });
  for (auto id_pair : id_sort) {
    int id = id_pair.first;
    if (polygon_map_[id]->object_track_id != -1 &&
        polygon_map_[cur_id]->object_track_id != -1 &&
        polygon_map_[cur_id]->object_track_id !=
            polygon_map_[id]->object_track_id) {
      continue;
    }
    if (check_merge(cur_id, valid_count, cur_convex, id, res, area_th_)) {
      for (auto p : cluster_points) {
        local_info[0](p.y, p.x) = id;
      }
      merge_polygon(ellipse_convex, id, ellipse_convex_res);
      polygon_map_[id]->voxel_polygon_cv = res;
      polygon_map_[id]->gauss_polygon_base = ellipse_convex_res;
      polygon_map_[id]->valid_count += valid_count;
      float area = convex_hull_area(res);
      polygon_map_[id]->area = area;
      erase_cluster(cur_id);
      // 如果够大了，就设置为1
      if (polygon_map_[id]->valid_count > small_target_thresh)
        id_manager_.setId(id, 1);
      return;
    }
  }

  polygon_map_[cur_id]->voxel_polygon_cv = cur_convex;
  polygon_map_[cur_id]->gauss_polygon_base = ellipse_convex;
  polygon_map_[cur_id]->valid_count = valid_count;
  float area = convex_hull_area(cur_convex);
  polygon_map_[cur_id]->area = area;
  if (valid_count < small_target_thresh)
    id_manager_.setId(cur_id, 0);  // 设置为小目标
  local_visited_id.insert(cur_id);
}

void PolygonDivide::updatePolygonInfo(const Ogm<uint8_t> &ghost_map) {
  for (auto &polygon : polygon_map_) {
    polygon.second->static_info.clear();
  }
  for (int i = 0; i < local_img.rows; ++i) {
    for (int j = 0; j < local_img.cols; ++j) {
      // 无效栅格
      if (local_img.ptr<uchar>(i)[j] == 0) {
        continue;
      }
      int tid = int(local_info[0](i, j));
      if (int(local_info[1](i, j)) == 0 || tid == 0 ||
          polygon_map_.find(tid) == polygon_map_.end()) {
        continue;
      }
      auto &polygon_info = polygon_map_[tid]->static_info;
      // 基础信息
      polygon_info.update(i, j, local_info);
      if (ghost_map(i, j) != 0) {
        polygon_info.ghost_num += 1;
      }
    }
  }
}

void PolygonDivide::generatePubPolygon() {
  output_id_.clear();
  current_traffic_cone.clear();
  for (const auto &id : local_visited_id) {
    auto &polygon_ptr = polygon_map_[id];
    auto &polygon_info = polygon_ptr->static_info;
    // 不存在于output_id的目标认为不应该继续被跟踪，计数和polygon都会在后续直接清除
    if (!polygon_ptr->generatePubInfo(filter_road_studs_, filter_flying_obj_,
                                      ogm.obj_infos_)) {
      continue;
    }
    // 当前帧置信度
    double all_num = polygon_info.high_conf_num + polygon_info.mid_conf_num +
                     polygon_info.low_conf_num;
    double hig_confi_ratio = polygon_info.high_conf_num / all_num;
    double middle_confi_ratio = polygon_info.mid_conf_num / all_num;
    double low_confi_ratio = polygon_info.low_conf_num / all_num;
    OccConfidence confidence = kLowConfOcc;
    if (hig_confi_ratio >= 0.1) {
      confidence = kHighConfOcc;
    } else if (middle_confi_ratio >= 0.2) {
      confidence = kMidConfOcc;
    }

    OccConfidence pre_confidence = kLowConfOcc;
    if (polygon_ptr->publish_state == kNoPub &&
        polygon_ptr->confidence == kLowConfOcc) {
      polygon_ptr->confidence = confidence;
    } else {
      pre_confidence = polygon_ptr->confidence;
      polygon_ptr->confidence = std::max(polygon_ptr->confidence, confidence);
    }

    // 30m自车道目标，置低置信度
    if (pre_confidence == kLowConfOcc) {
      float check_x = polygon_ptr->publish_state == kNoPub
                          ? polygon_ptr->first_det_distance.x()
                          : polygon_ptr->rect_base.center_x;
      float check_y = polygon_ptr->publish_state == kNoPub
                          ? polygon_ptr->first_det_distance.y()
                          : polygon_ptr->rect_base.center_y;
      if (check_x < ogm.near_dist_ && std::abs(check_y) < 3) {
        polygon_ptr->confidence = kLowConfOcc;
      }
    }
    polygon_ptr->publish_state = kPublished;
    // 候选锥桶
    float height_above_ground =
        polygon_info.height_from_gnd / (polygon_info.point_num + 1e-10);
    if (std::abs(polygon_ptr->area) <= 6.6 && height_above_ground <= 0.8 &&
        polygon_info.max_intensity > 150) {
      if (!polygon_ptr->is_traffic_cone) {
        current_traffic_cone.insert(id);
      }
    } else if (std::abs(polygon_ptr->area) > 6.6 || height_above_ground > 1.2) {
      if (polygon_ptr->is_traffic_cone) {
        polygon_ptr->is_traffic_cone = false;
      }
    }
    // 路牌下的polygon, 打上疑似鬼影的type
    if ((all_num < 10 && polygon_info.ghost_num > 0) ||
        polygon_info.ghost_num / all_num >= 0.5) {
      polygon_ptr->is_ghost = true;
    }
    if (polygon_ptr->gauss_polygon_base.size() > 2) {
      output_id_.insert(id);
    }
  }
}

void PolygonDivide::detectRoadCone(
    const ObjectInnerArray::Ptr &fm_narrow_object_ptr) {
  // detect potential roadcone
  // proj and detect roadcone
  std::unordered_map<int, Eigen::ArrayXd> gauss_map_;
  Eigen::ArrayXd voxels_info;
  for (int i = 0; i < local_img.rows; i++) {
    for (int j = 0; j < local_img.cols; j++) {
      int tid = int(local_info[0](i, j));
      if (int(local_info[1](i, j)) == 0 || tid == 0) {
        continue;
      }
      if (!current_traffic_cone.count(tid)) {
        continue;
      }
      if (!gauss_map_.count(tid)) {
        voxels_info = Eigen::ArrayXd::Zero(12);
      } else {
        voxels_info = gauss_map_[tid];
      }
      for (int k = 0; k < 10; k++) {
        voxels_info[k] += local_info[k + 1](i, j);
      }
      gauss_map_[tid] = voxels_info;
    }
  }
  // TRAFFIC_CONE by pv_fm_narrow_object
  double pv_narrow_stamp =
      (fm_narrow_object_ptr == nullptr)
          ? cloud_ptr->time_sec
          : (double)fm_narrow_object_ptr->header.time / 1e9;
  Eigen::Affine3d tf_sync = Eigen::Affine3d::Identity();
  Eigen::Affine3d tf_narrow2global = Eigen::Affine3d::Identity();
  if (getEgoPose(pv_narrow_stamp, tf_narrow2global)) {
    tf_sync = tf_narrow2global.inverse() * ogm.tf_ego2global_;
  }
  std::vector<Box2dInfo> pv_narrow_traffic_cone;
  if (fm_narrow_object_ptr != nullptr) {
    for (const auto &obj : fm_narrow_object_ptr->object_list) {
      if (obj.type != ObjectType::TYPE_TRAFFIC_CONE) {
        continue;
      }
      Box2dInfo tmp_box;
      tmp_box.x = obj.box_full.x;
      tmp_box.y = obj.box_full.y;
      tmp_box.height = obj.box_full.height;
      tmp_box.width = obj.box_full.width;
      pv_narrow_traffic_cone.push_back(tmp_box);
    }
  }
  std::unordered_map<int, ProjBoxInfo> proj_map;
  Eigen::Vector3d eigen_vals;
  Eigen::Matrix3d eigen_mat;
  for (const auto &key_value : gauss_map_) {
    voxels_info = key_value.second;
    double alpha = 1.0 / voxels_info[0];
    Eigen::Vector3d mu{voxels_info[1] * alpha, voxels_info[2] * alpha,
                       voxels_info[3] * alpha};
    Eigen::Matrix3d cov;
    cal_covariance(voxels_info, cov);
    calEigenInfo(cov, eigen_vals, eigen_mat);
    double x_offset = std::sqrt(2 * std::log(2.5) * eigen_vals(0));
    double y_offset = std::sqrt(2 * std::log(2.5) * eigen_vals(1));
    double z_offset = std::sqrt(2 * std::log(2.5) * eigen_vals(2));
    std::vector<Eigen::Vector3d> gauss_points(8,
                                              Eigen::Vector3d{0.0, 0.0, 0.0});
    ProjBoxInfo proj_obj_3d;
    for (int j = 0; j < cof_size; ++j) {
      Eigen::Vector3d lidar_stamp_point =
          mu + eigen_mat * Eigen::Vector3d(cof_x[j] * x_offset,
                                           cof_y[j] * y_offset,
                                           cof_z[j] * z_offset);
      gauss_points[j] = tf_sync * lidar_stamp_point;
    }
    proj_ptr_->projectCorners(gauss_points, proj_obj_3d, true);
    proj_map[key_value.first] = proj_obj_3d;
  }

  for (const auto &key_value : proj_map) {
    int id = key_value.first;
    auto proj_box_3d = key_value.second;
    for (const auto &pv_narrow_box : pv_narrow_traffic_cone) {
      auto iou_value =
          proj_ptr_->calIouStandard(proj_box_3d.box_2d, pv_narrow_box);
      if (iou_value >= 0.1) {
        polygon_map_[id]->is_traffic_cone = true;
      }
    }
  }
}

// voxel(x,y) has four corners -> (x,y) (x+1,y) (x,y+1) (x+1,y+1)
void PolygonDivide::dividePolygon(
    const ObjectInnerArray::Ptr &fm_narrow_object_ptr,
    const Ogm<uint8_t> &ghost_map, const Ogm<GaussGrid> &nonground_info_map,
    const Ogm<uint8_t> &intensity_map) {
  getLocalImg(nonground_info_map, intensity_map);
  updateConvex();
  for (int i = 0; i < local_img.rows; ++i) {
    for (int j = 0; j < local_img.cols; ++j) {
      int cur_tid = local_info[0](i, j);
      if (int(local_img.ptr<uchar>(i)[j]) > 0 && cur_tid == 0) {
        clusterPolygon(cv::Point(j, i));
      }
    }
  }
  updatePolygonInfo(ghost_map);
  generatePubPolygon();
  detectRoadCone(fm_narrow_object_ptr);
  uploadLocalInfo();  // 同步世界图层, 保留占据栅格
}

void PolygonDivide::uploadLocalInfo() {
  auto &track_cnt_ogm = ogm.at(kTrackCntMap);
  auto &track_id_ogm = ogm.at(kTrackIdMap);
  for (int i = 0; i < local_img.rows; i++) {
    for (int j = 0; j < local_img.cols; j++) {
      if (int(local_info[1](i, j)) == 0) {
        continue;
      }
      int map_x = ogm.xSubToMap(int(local_info[11](i, j)));
      int map_y = ogm.ySubToMap(int(local_info[12](i, j)));
      int polygon_id = int(local_info[0](i, j));
      if (output_id_.count(polygon_id)) {
        track_id_ogm(map_x, map_y) = polygon_id;
      } else {
        track_cnt_ogm(map_x, map_y) = 0;
        erase_cluster(polygon_id);
      }
    }
  }
}

// 以下是 utils
// tool functions
void PolygonDivide::cal_gauss_distribution(
    int i, int j, std::vector<Eigen::Vector3d> &_ellipse_points) {
  _ellipse_points.clear();
  int point_num = local_info[1](i, j);
  auto &sum_x = local_info[2](i, j);
  auto &sum_y = local_info[3](i, j);
  auto &sum_z = local_info[4](i, j);
  auto &sum_xx = local_info[5](i, j);
  auto &sum_yy = local_info[6](i, j);
  auto &sum_zz = local_info[7](i, j);
  auto &sum_xy = local_info[8](i, j);
  auto &sum_xz = local_info[9](i, j);
  auto &sum_yz = local_info[10](i, j);
  if (point_num == 0) {
    return;
  }
  double alpha = 1.0 / point_num;
  Eigen::Vector3d mu{sum_x * alpha, sum_y * alpha, sum_z * alpha};
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  cov(0, 0) = sum_xx - alpha * sum_x * sum_x;
  cov(0, 1) = sum_xy - alpha * sum_x * sum_y;
  cov(0, 2) = sum_xz - alpha * sum_x * sum_z;
  cov(1, 0) = cov(0, 1);
  cov(1, 1) = sum_yy - alpha * sum_y * sum_y;
  cov(1, 2) = sum_yz - alpha * sum_y * sum_z;
  cov(2, 0) = cov(0, 2);
  cov(2, 1) = cov(1, 2);
  cov(2, 2) = sum_zz - alpha * sum_z * sum_z;
  cov = cov * alpha;
  Eigen::Vector3d eigen_vals;
  Eigen::Matrix3d eigen_mat;
  calEigenInfo(cov, eigen_vals, eigen_mat);
  double x_offset = std::sqrt(2 * std::log(2) * eigen_vals(0));
  double y_offset = std::sqrt(2 * std::log(2) * eigen_vals(1));
  double z_offset = std::sqrt(2 * std::log(2) * eigen_vals(2));
  for (int j = 0; j < cof_size; ++j) {
    Eigen::Vector3d p = mu + eigen_mat * Eigen::Vector3d(cof_x[j] * x_offset,
                                                         cof_y[j] * y_offset,
                                                         cof_z[j] * z_offset);
    _ellipse_points.emplace_back(p);
  }
}

bool PolygonDivide::check_merge(const int id_ori, const int valid_count_ori,
                                const std::vector<cv::Point> &polygon_ori,
                                const int id_target,
                                std::vector<cv::Point> &polygon_merge,
                                float merge_th) {
  // 已存在置信度目标置信度更高，则不合并
  // 如果都是静止，其中一个为近距离目标，且原始为低置信度，不允许向更高合并
  if (polygon_map_.find(id_ori) != polygon_map_.end()) {
    if (polygon_map_.find(id_target) == polygon_map_.end()) {
      return false;
    } else if (polygon_map_[id_target]->confidence <
               polygon_map_[id_ori]->confidence) {
      return false;
    }
  }

  std::vector<cv::Point> polygon_all(polygon_ori);
  for (int i = 0; i < polygon_map_[id_target]->voxel_polygon_cv.size(); ++i) {
    polygon_all.emplace_back(polygon_map_[id_target]->voxel_polygon_cv[i]);
  }
  cv_convex(polygon_all, polygon_merge, false);
  // cv::convexHull(cv::Mat(polygon_all), polygon_merge, true);
  auto area = convex_hull_area(polygon_merge);
  if (std::abs(area) - std::abs(polygon_map_[id_target]->area) > 0.5 &&
      polygon_map_.find(id_ori) != polygon_map_.end() &&
      polygon_map_.find(id_target) != polygon_map_.end() &&
      polygon_map_[id_target]->confidence > polygon_map_[id_ori]->confidence &&
      polygon_map_[id_ori]->confidence == kLowConfOcc) {
    if (polygon_map_[id_target]->publish_state != kNoPub &&
        polygon_map_[id_target]->rect_base.center_x <
            std::min(ogm.near_dist_, 15.0)) {
      return false;
    } else if (polygon_map_[id_ori]->publish_state != kNoPub &&
               polygon_map_[id_ori]->rect_base.center_x <
                   std::min(ogm.near_dist_, 15.0)) {
      return false;
    }
  }
  bool merge_flag =
      std::abs(area) - valid_count_ori - polygon_map_[id_target]->valid_count <
          merge_th ||
      std::abs(area) - std::abs(polygon_map_[id_target]->area) < 3;
  return merge_flag;
}

void PolygonDivide::merge_polygon(const std::vector<cv::Point2f> &polygon_ori,
                                  const int id_target,
                                  std::vector<cv::Point2f> &polygon_merge) {
  std::vector<cv::Point2f> polygon_all(polygon_ori);
  for (int i = 0; i < polygon_map_[id_target]->gauss_polygon_base.size(); ++i) {
    polygon_all.emplace_back(polygon_map_[id_target]->gauss_polygon_base[i]);
  }
  cv::convexHull(polygon_all, polygon_merge, true);
}
void PolygonDivide::pop_left_convex_corner_cv(
    const cv::Point &p, std::vector<cv::Point> &convex_left,
    float &accumulate_area, std::queue<cv::Point> &log_left) {
  while (p == convex_left.back()) {
    log_left.emplace(convex_left.back());
    accumulate_area -= triangle_area(REF, convex_left[convex_left.size() - 2],
                                     convex_left[convex_left.size() - 1]);
    convex_left.pop_back();
  }
  while (convex_left.size() > 2 &&
         !check_cross_left_cv(convex_left[convex_left.size() - 2],
                              convex_left[convex_left.size() - 1], p)) {
    log_left.emplace(convex_left.back());
    accumulate_area -= triangle_area(REF, convex_left[convex_left.size() - 2],
                                     convex_left[convex_left.size() - 1]);
    convex_left.pop_back();
  }
}
void PolygonDivide::pop_right_convex_corner_cv(
    const cv::Point &p, std::vector<cv::Point> &convex_right,
    float &accumulate_area, std::queue<cv::Point> &log_right) {
  // 对于角点，要先把相同的点pop掉（出现相同的点，一定会大于2，因为代表把上下角点加入了队列）
  while (p == convex_right.back()) {
    log_right.emplace(convex_right.back());
    accumulate_area -= triangle_area(REF, convex_right[convex_right.size() - 1],
                                     convex_right[convex_right.size() - 2]);
    convex_right.pop_back();
  }
  while (convex_right.size() > 2 &&
         !check_cross_right_cv(convex_right[convex_right.size() - 2],
                               convex_right[convex_right.size() - 1], p)) {
    log_right.emplace(convex_right.back());
    accumulate_area -= triangle_area(REF, convex_right[convex_right.size() - 1],
                                     convex_right[convex_right.size() - 2]);
    convex_right.pop_back();
  }
}

int PolygonDivide::enlarge_small_cluster(
    const std::vector<cv::Point> &cur_points, const int cur_id,
    std::vector<cv::Point> &merge_convex) {
  std::unordered_map<int, int> connect_id;
  int grid_type = local_img.ptr<uchar>(cur_points[0].y)[cur_points[0].x];
  // 找连通
  for (auto point : cur_points) {
    for (auto off : BFS_CONNECT_CV) {
      cv::Point tmp(point.x + off.x, point.y + off.y);
      if (ogm.inBaseRange(tmp.y, tmp.x)) {
        int tmp_tid = local_info[0](tmp.y, tmp.x);
        int tmp_grid_type = local_img.ptr<uchar>(tmp.y)[tmp.x];
        if (tmp_tid > 0 && tmp_tid != cur_id && tmp_grid_type == grid_type &&
            (polygon_map_[tmp_tid]->object_track_id ==
                 polygon_map_[cur_id]->object_track_id ||
             polygon_map_[tmp_tid]->object_track_id == -1 ||
             polygon_map_[cur_id]->object_track_id == -1)) {
          ++connect_id[tmp_tid];
        }
      }
    }
    for (auto off : BFS_OFFSET_CV) {
      cv::Point tmp(point.x + off.x, point.y + off.y);
      if (ogm.inBaseRange(tmp.y, tmp.x)) {
        int tmp_tid = local_info[0](tmp.y, tmp.x);
        int tmp_grid_type = local_img.ptr<uchar>(tmp.y)[tmp.x];
        if (tmp_tid > 0 && tmp_tid != cur_id && tmp_grid_type == grid_type &&
            (polygon_map_[tmp_tid]->object_track_id ==
                 polygon_map_[cur_id]->object_track_id ||
             polygon_map_[tmp_tid]->object_track_id == -1 ||
             polygon_map_[cur_id]->object_track_id == -1)) {
          ++connect_id[tmp_tid];
        }
      }
    }
  }
  std::vector<std::pair<int, int>> id_sort(connect_id.begin(),
                                           connect_id.end());
  std::sort(id_sort.begin(), id_sort.end(),
            [](const std::pair<int, int> &p1, const std::pair<int, int> &p2) {
              return p1.second > p2.second;
            });
  for (auto id_pair : id_sort) {
    int id = id_pair.first;
    if (check_merge(cur_id, polygon_map_[cur_id]->valid_count,
                    polygon_map_[cur_id]->voxel_polygon_cv, id, merge_convex,
                    area_th_)) {
      return id;
    }
  }
  return cur_id;
}
void PolygonDivide::erase_cluster(const int id) {
  id_manager_.eraseId(id);
  if (polygon_map_.count(id)) polygon_map_.erase(polygon_map_.find(id));
}

void PolygonDivide::gauss_convex(std::vector<cv::Point> gauss_edge_points,
                                 std::vector<cv::Point2f> &ellipse_convex,
                                 std::vector<cv::Point2f> &voxel_gauss_points) {
  int points_num = gauss_edge_points.size();
  std::vector<Eigen::Vector3d> ellipse_points;
  voxel_gauss_points.clear();
  for (const auto &p : gauss_edge_points) {
    // 加入四个栅格角点对应的虚拟base点
    if (int(local_info[1](p.y, p.x)) == 0) {
      float px_0 = ogm.yBaseToPos(p.x);
      float py_0 = ogm.xBaseToPos(p.y);
      float px_1 = ogm.yBaseToPos(p.x + 1);
      float py_1 = ogm.yBaseToPos(p.y + 1);
      voxel_gauss_points.emplace_back(cv::Point2f(px_0, py_0));
      voxel_gauss_points.emplace_back(cv::Point2f(px_0, py_1));
      voxel_gauss_points.emplace_back(cv::Point2f(px_1, py_1));
      voxel_gauss_points.emplace_back(cv::Point2f(px_1, py_0));
      continue;
    };
    // 栅格内高斯椭圆的八个点
    cal_gauss_distribution(p.y, p.x, ellipse_points);
    for (const auto &ep : ellipse_points) {
      voxel_gauss_points.emplace_back(cv::Point2f(ep[0], ep[1]));
    }
  }
  if (voxel_gauss_points.size() == 0) {
    return;
  }
  // 避免只有一两个点的情况
  if (voxel_gauss_points.size() == 1) {
    auto first_point = voxel_gauss_points[0];
    voxel_gauss_points.emplace_back(
        cv::Point2f(first_point.x + 0.1, first_point.y + 0.1));
    voxel_gauss_points.emplace_back(
        cv::Point2f(first_point.x - 0.1, first_point.y + 0.1));
    voxel_gauss_points.emplace_back(
        cv::Point2f(first_point.x - 0.1, first_point.y - 0.1));
    voxel_gauss_points.emplace_back(
        cv::Point2f(first_point.x + 0.1, first_point.y - 0.1));
  } else if (voxel_gauss_points.size() == 2) {
    auto first_point = voxel_gauss_points[0];
    auto second_point = voxel_gauss_points[1];
    voxel_gauss_points.emplace_back(
        cv::Point2f(first_point.x + 0.1, first_point.y + 0.1));
    voxel_gauss_points.emplace_back(
        cv::Point2f(first_point.x - 0.1, first_point.y + 0.1));
    voxel_gauss_points.emplace_back(
        cv::Point2f(first_point.x - 0.1, first_point.y - 0.1));
    voxel_gauss_points.emplace_back(
        cv::Point2f(first_point.x + 0.1, first_point.y - 0.1));
    voxel_gauss_points.emplace_back(
        cv::Point2f(second_point.x + 0.1, second_point.y + 0.1));
    voxel_gauss_points.emplace_back(
        cv::Point2f(second_point.x - 0.1, second_point.y + 0.1));
    voxel_gauss_points.emplace_back(
        cv::Point2f(second_point.x - 0.1, second_point.y - 0.1));
    voxel_gauss_points.emplace_back(
        cv::Point2f(second_point.x + 0.1, second_point.y - 0.1));
  }
  cv::convexHull(voxel_gauss_points, ellipse_convex, true);
}

void PolygonDivide::cv_convex(const std::vector<cv::Point> &points,
                              std::vector<cv::Point> &convex, bool is_voxel) {
  if (is_voxel) {
    std::vector<cv::Point> corners;
    for (int i = 0; i < points.size(); ++i) {
      corners.emplace_back(cv::Point(points[i].x, points[i].y));
      corners.emplace_back(cv::Point(points[i].x + 1, points[i].y));
      corners.emplace_back(cv::Point(points[i].x, points[i].y + 1));
      corners.emplace_back(cv::Point(points[i].x + 1, points[i].y + 1));
    }
    cv::convexHull(cv::Mat(corners), convex, true);
  } else {
    cv::convexHull(points, convex, true);
  }
}
bool PolygonDivide::checkDynamicConnect(const int polygon_id,
                                        const int obj_id) {
  if (polygon_map_.find(polygon_id) == polygon_map_.end()) {
    return true;
  } else {
    const auto &polygon = polygon_map_[polygon_id];
    if (polygon->no_connecti_id_.find(obj_id) !=
        polygon->no_connecti_id_.end()) {
      return false;
    } else {
      return true;
    }
  }
}
// 生成rect和一些需要输出的信息，返回false目标会被直接清除
bool Polygon::generatePubInfo(
    const bool filter_road_studs, const bool filter_flying_obj,
    const std::unordered_map<int, Box3dInfo> &obj_infos) {
  // 清空部分属性信息
  is_ghost = false;
  // 生成rect信息
  auto rect = cv::minAreaRect(gauss_polygon_base);
  cv::Mat box;
  cv::boxPoints(rect, box);
  Eigen::Vector2f p{0, 0};
  std::vector<Eigen::Vector2f> ps;
  for (int i = 0; i < box.rows; i++) {
    for (int j = 0; j < box.cols; j++) {
      p[j] = box.at<float>(i, j);
    }
    ps.emplace_back(p);
  }
  Eigen::Vector2f l01 = ps[1] - ps[0];
  Eigen::Vector2f l02 = ps[3] - ps[0];
  float l01_norm = l01.norm();
  float l02_norm = l02.norm();
  float l = l02_norm, w = l01_norm;
  Eigen::Vector2f dir = l02 / (l02_norm + 1e-6);
  if (l01_norm >= l02_norm) {
    dir = l01 / (l01_norm + 1e-6);
    l = l01_norm;
    w = l02_norm;
  }
  float yaw = atan2(dir[1], dir[0]);
  rect_base.center_x = rect.center.x;
  rect_base.center_y = rect.center.y;
  rect_base.length = l;
  rect_base.width = w;
  rect_base.yaw = yaw;
  // 保证输出逆时针
  if (check_closckwise_car(gauss_polygon_base)) {
    std::reverse(gauss_polygon_base.begin(), gauss_polygon_base.end());
  }
  // 过滤远处低矮高反小目标(道钉)
  if (filter_road_studs && std::abs(area) < 2.1 &&
      static_info.height_diff < 0.4 && static_info.max_intensity > 30 &&
      static_info.det_obj_dis > 40) {
    AINFO << "Filter RoadStud: " << id;
    return false;
  }
  // 滤除近处凌空小目标
  float dis2 = rect_base.center_x * rect_base.center_x +
               rect_base.center_y * rect_base.center_y;
  if (filter_flying_obj && std::abs(area) < 3.1 && dis2 < 100 &&
      static_info.max_intensity < 30 && static_info.point_num > 0) {
    float h_from_gnd = static_info.height_from_gnd / static_info.point_num;
    if (h_from_gnd > 1.0) {
      AINFO << "Filter FlyingNearSmallObj: " << id;
      return false;
    }
  }
  // 更新首次检出距离
  if (publish_state == kNoPub) {
    first_det_distance.x() = static_info.det_obj_dis;
    first_det_distance.y() = rect_base.center_y;
  }
  // 更新no connect id
  for (auto it = no_connecti_id_.begin(); it != no_connecti_id_.end();) {
    int cur_id = *it;
    if (obj_infos.find(cur_id) == obj_infos.end()) {
      it = no_connecti_id_.erase(it);
    } else {
      ++it;
    }
  }
  for (const auto &obj : obj_infos) {
    // 已经连通的目标不计入无法联通
    if (obj.first == object_track_id) {
      continue;
    }
    if (obj.second.center.x() > 50 || obj.second.center.x() < -20) {
      continue;
    }
    if (no_connecti_id_.find(obj.first) != no_connecti_id_.end()) {
      continue;
    }
    float dis_x = obj.second.center.x() - rect_base.center_x;
    float dis_y = obj.second.center.y() - rect_base.center_y;
    double dist_thresh = obj.second.size[0] + rect_base.length;
    if (IsVRU(obj.second.type)) {
      dist_thresh = std::max(obj.second.size[0] + 3.0, dist_thresh);
    } else {
      dist_thresh = std::max(0.5 * obj.second.size[0] + 5, dist_thresh);
    }
    if (dis_x * dis_x + dis_y * dis_y > dist_thresh * dist_thresh) {
      no_connecti_id_.insert(obj.first);
    }
  }
  return true;
}

ObjectInner Polygon::toObjectInner() {
  ObjectInner obj;
  obj.object_id = uint32_t(id);
  obj.box_center_base.x = rect_base.center_x;
  obj.box_center_base.y = rect_base.center_y;
  obj.box_center_base.z = 0.0;
  obj.box_size.length = rect_base.length;
  obj.box_size.width = rect_base.width;
  obj.box_size.height = 0.0;
  obj.yaw_base = rect_base.yaw;
  obj.type_confidence = kOccConfToPubConf.at(confidence);
  obj.type = is_traffic_cone ? ObjectType::TYPE_TRAFFIC_CONE
                             : ObjectType::TYPE_STATIC_UNKNOWN;
  obj.motion_type = polygon_type == kDynamicPolygon ? MotionType::Stoped
                                                    : MotionType::Stationary;
  // polygon
  obj.polygon_base.resize(gauss_polygon_base.size());
  for (size_t i = 0; i < gauss_polygon_base.size(); ++i) {
    obj.polygon_base[i].x = gauss_polygon_base[i].x;
    obj.polygon_base[i].y = gauss_polygon_base[i].y;
  }
  // suspected ghost
  obj.extended_key.emplace_back(static_cast<uint32_t>(
      perception_framework::EXTENDED_KEY::OCCUPANCY_GHOST_OBJECT_KEY));
  obj.extended_value.emplace_back(is_ghost ? 1.0f : 0);
  // object track id
  obj.extended_key.emplace_back(static_cast<uint32_t>(
      perception_framework::EXTENDED_KEY::OCCUPANCY_POLYGON_TRACK_ID_KEY));
  obj.extended_value.emplace_back(
      polygon_type == kDynamicPolygon ? object_track_id : -1);

  return obj;
}

}  // namespace perception
}  // namespace robosense
