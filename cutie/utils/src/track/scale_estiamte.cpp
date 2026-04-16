#include "hyper_vision/perception/pv_post_process/scale_estimate.hpp"

#include "cyber/cyber.h"
double _all_time = 0;
double max_time = 0;
int frame_count = 0;
std::pair<float, float> ScaleEstimate::RunEstimate(
    cv::Mat& img_gray_1, cv::Mat& img_gray_2, cv::Mat& cover_mask1,
    cv::Mat& cover_mask2, std::pair<float, float>& pixel_move,
    const float cover_ratio1, const float cover_ratio2) {
  // cv::setNumThreads(0);
  // 过大的图像尺寸变化
  ++save_id;
  if (float(img_gray_1.cols) / img_gray_2.rows >= 2 ||
      float(img_gray_1.cols) / img_gray_2.rows < 0.5)
    return {1, 0};
  else if (float(img_gray_1.cols) / img_gray_2.cols >= 2 ||
           float(img_gray_1.cols) / img_gray_2.cols < 0.5)
    return {1, 0};

  double log_time = 0;
  timeval t1, t2;
  gettimeofday(&t1, NULL);
  if (!cover_mask1.data || cover_mask1.size() != img_gray_1.size()) {
    cover_mask1 = cv::Mat::zeros(img_gray_1.rows, img_gray_1.cols, CV_32F) + 1;
  }
  if (!cover_mask2.data || cover_mask2.size() != img_gray_2.size()) {
    cover_mask2 = cv::Mat::zeros(img_gray_2.rows, img_gray_2.cols, CV_32F) + 1;
  }
  if (img_gray_1.channels() == 3) {
    cv::cvtColor(img_gray_1, img_gray_1, cv::COLOR_BGR2GRAY);
  }
  if (img_gray_2.channels() == 3) {
    cv::cvtColor(img_gray_2, img_gray_2, cv::COLOR_BGR2GRAY);
  }
  cv::Mat img1, img2, mask1, mask2;

  //   AINFO << "input size:" << img_gray_1.size() << " " << img_gray_2.size()
  //         << cover_mask1.size() << " " << cover_mask2.size() << " " <<
  //         std::endl;
  float img_scale = 1;
  if (std::max(img_gray_1.rows, img_gray_1.cols) > max_size &&
      std::max(img_gray_2.rows, img_gray_2.cols) > max_size) {
    float img_scale_1 =
        std::min(std::max(img_gray_1.rows, img_gray_1.cols) / float(max_size),
                 std::max(img_gray_2.rows, img_gray_2.cols) / float(max_size));
    img_scale_1 = std::max(img_scale_1, 1.0f);
    float img_scale_2 =
        std::min(std::min(img_gray_1.rows, img_gray_1.cols) / float(min_size),
                 std::min(img_gray_2.rows, img_gray_2.cols) / float(min_size));
    img_scale_2 = std::max(img_scale_2, 1.0f);
    img_scale = std::min(img_scale_1, img_scale_2);
    cv::resize(img_gray_1, img1, cv::Size(), 1.0 / img_scale, 1.0 / img_scale,
               cv::INTER_LINEAR);
    cv::resize(img_gray_2, img2, cv::Size(), 1.0 / img_scale, 1.0 / img_scale,
               cv::INTER_LINEAR);
    if (cover_ratio1 > 0.1)
      cv::resize(cover_mask1, mask1, cv::Size(), 1.0 / img_scale,
                 1.0 / img_scale, cv::INTER_LINEAR_EXACT);
    if (cover_ratio2 > 0.1)
      cv::resize(cover_mask2, mask2, cv::Size(), 1.0 / img_scale,
                 1.0 / img_scale, cv::INTER_LINEAR_EXACT);
  } else {
    img_gray_1.copyTo(img1);
    img_gray_2.copyTo(img2);
    if (cover_ratio1 > 0.1) cover_mask1.copyTo(mask1);
    if (cover_ratio2 > 0.1) cover_mask2.copyTo(mask2);
  }
  if (debug_mode_) {
    // AINFO << img1.size() << " " << img1.size() << std::endl;
    // cv::namedWindow("img_gray1", cv::WINDOW_NORMAL);
    // cv::imshow("img_gray1", img1);
    // cv::namedWindow("img_gray2", cv::WINDOW_NORMAL);
    // cv::imshow("img_gray2", img2);
    // cv::waitKey(-1);
  }
  gettimeofday(&t2, NULL);
  double t_filter_ground =
      (t2.tv_sec - t1.tv_sec) * 1000 * 1000 + (t2.tv_usec - t1.tv_usec);
  log_time += t_filter_ground / 1e3;
  if (debug_time_mode_)
    AINFO << "resize time: " << t_filter_ground << std::endl;
  // gray equalize
  gettimeofday(&t1, NULL);
  cv::equalizeHist(img1, img1);
  cv::equalizeHist(img2, img2);
  if (cover_ratio1 > 0.1) img1 = img1.mul(mask1);
  if (cover_ratio2 > 0.1) img2 = img2.mul(mask2);
  gettimeofday(&t2, NULL);
  t_filter_ground =
      (t2.tv_sec - t1.tv_sec) * 1000 * 1000 + (t2.tv_usec - t1.tv_usec);
  log_time += t_filter_ground / 1e3;
  if (debug_time_mode_)
    AINFO << "equalize time: " << t_filter_ground << std::endl;
  gettimeofday(&t1, NULL);
  cv::Mat gx1, gx2, gy1, gy2, edge_label1, edge_label2, mag1, mag2;
  std::vector<cv::Point2d> normal1, normal2;
  std::vector<cv::Point2d> edge_points1, edge_points2;
  // get grid
  GradCal(mask1, img1, gx1, gy1, mag1, cover_ratio1 > 0.1);
  GradCal(mask2, img2, gx2, gy2, mag2, cover_ratio2 > 0.1);

  gettimeofday(&t2, NULL);
  t_filter_ground =
      (t2.tv_sec - t1.tv_sec) * 1000 * 1000 + (t2.tv_usec - t1.tv_usec);
  log_time += t_filter_ground / 1e3;
  if (debug_time_mode_)
    AINFO << "fearture time: " << t_filter_ground << std::endl;

  gettimeofday(&t1, NULL);
  cv::Point2d offset;
  // float response = BaseTranslation(mag1, mag2, offset);
  float res2 = TemplateMatch(mag1, mag2, offset);
  pixel_move.first = offset.x * img_scale;
  pixel_move.second = offset.y * img_scale;
  if (res2 <= 0.2) return {1, 0};
  gettimeofday(&t2, NULL);
  t_filter_ground =
      (t2.tv_sec - t1.tv_sec) * 1000 * 1000 + (t2.tv_usec - t1.tv_usec);
  log_time += t_filter_ground / 1e3;
  if (debug_time_mode_)
    AINFO << "base translation time: " << t_filter_ground << std::endl;

  // get edge
  gettimeofday(&t1, NULL);

  int row_count1, row_count2;
  EdgePreprocess(gx1, gy1, mag1, normal1, edge_label1, edge_points1);
  EdgePreprocess(gx2, gy2, mag2, normal2, edge_label2, edge_points2);
  int left_x_1 = 0, right_x_1 = img1.cols - 1;

  gettimeofday(&t2, NULL);
  t_filter_ground =
      (t2.tv_sec - t1.tv_sec) * 1000 * 1000 + (t2.tv_usec - t1.tv_usec);
  log_time += t_filter_ground / 1e3;
  if (debug_time_mode_)
    AINFO << "get edge time: " << t_filter_ground << std::endl;
  // get match
  gettimeofday(&t1, NULL);
  std::vector<cv::Point2d> src_points, target_points, src_normal, target_normal;
  // std::vector<cv::Point2d> src_col_points, target_col_points, src_row_points,
  // target_row_points,row_normal,col_normal;
  cv::Mat rgb_src, rgb_target;
  if (debug_mode_) {
    cv::cvtColor(img1, rgb_src, cv::COLOR_GRAY2RGB);
    cv::cvtColor(img2, rgb_target, cv::COLOR_GRAY2RGB);
  }

  for (int i = 0; i < edge_points1.size(); ++i) {
    auto p = edge_points1[i];

    auto normal = normal1[i];
    bool matched = false;
    float min_mag_diff = 10000;
    cv::Point2d match_p, match_normal;
    for (auto off : offset55) {
      if (matched) break;
      cv::Point2d new_point_double = p + off + offset;
      cv::Point2i new_point(round(new_point_double.x),
                            round(new_point_double.y));
      if (new_point.x < 0 || new_point.y < 0 || new_point.x >= img2.cols ||
          new_point.y >= img2.rows)
        continue;
      cv::Point2d new_grid;

      float new_gx = *gx2.ptr<float>(new_point.y, new_point.x);
      float new_gy = *gy2.ptr<float>(new_point.y, new_point.x);
      float new_mag = sqrt(*mag2.ptr<float>(new_point.y, new_point.x));
      if (new_mag < 1) continue;
      cv::Point2d new_normal(new_gx / new_mag, new_gy / new_mag);

      // row
      if (fabs(normal.y) > edge_normal_thresh) {
        // 1. is edge 2. small angel

        if (*edge_label2.ptr<int>(new_point.y, new_point.x) != 0 &&
            normal.x * new_normal.x + normal.y * new_normal.y >
                edge_normal_thresh) {
          // if(min_mag_diff>fabs(new_mag-LinearInter(mag1,p)))
          // {
          //     if(new_point.y!=0&&new_point.y!=gy2.rows-1)
          //     {
          //         new_point.y = EdgeInter({new_point.y - 1, new_point.y,
          //         new_point.y+ 1},
          //             {*gy2.ptr<float>(new_point.y-1, new_point.x),
          //             *gy2.ptr<float>(new_point.y, new_point.x),
          //             *gy2.ptr<float>(new_point.y+1, new_point.x)});
          //     }
          //     matched = true;
          //     match_p = new_point;
          //     match_normal = new_normal;
          // }
          // else
          //     continue;
          new_point_double.x = new_point.x;
          new_point_double.y = new_point.y;
          if (new_point.y != 0 && new_point.y != gy2.rows - 1) {
            new_point_double.y =
                EdgeInter({new_point.y - 1, new_point.y, new_point.y + 1},
                          {*gy2.ptr<float>(new_point.y - 1, new_point.x),
                           *gy2.ptr<float>(new_point.y, new_point.x),
                           *gy2.ptr<float>(new_point.y + 1, new_point.x)});
          }

          src_points.emplace_back(p);
          target_points.emplace_back(new_point_double);
          src_normal.emplace_back(normal);
          // new_normal = NormalInter(gx2, gy2, new_point, new_grid);
          target_normal.emplace_back(new_normal);
          if (debug_mode_) {
            int b = rand() % 255;
            int g = rand() % 255;
            int r = rand() % 255;
            cv::circle(
                rgb_src,
                {static_cast<int>(round(p.x)), static_cast<int>(round(p.y))}, 1,
                cv::Scalar(r, g, b), -1);
            cv::circle(rgb_target, {new_point.x, new_point.y}, 1,
                       cv::Scalar(r, g, b), -1);
          }
          matched = true;
          // src_row_points.emplace_back(cv::Point2d{p.y, p.x});
          // target_row_points.emplace_back(cv::Point2d{new_point.y,
          // new_point.x}); row_normal.emplace_back(new_normal);
        }
      }
      // col
      else if (fabs(normal.x) > edge_normal_thresh) {
        // 1. is edge 2. small angel
        if (*edge_label2.ptr<int>(new_point.y, new_point.x) != 0 &&
            normal.x * new_normal.x + normal.y * new_normal.y >
                edge_normal_thresh) {
          // if(min_mag_diff>fabs(new_mag-LinearInter(mag1,p)))
          // {

          //      if (new_point.x != 0 && new_point.x != gx2.cols - 1)
          //     {
          //         new_point.x = EdgeInter({new_point.x - 1, new_point.x,
          //         new_point.x + 1},
          //                                     {*gx2.ptr<float>(new_point.x,
          //                                     new_point.x - 1),
          //                                     *gx2.ptr<float>(new_point.y,
          //                                     new_point.x),
          //                                     *gx2.ptr<float>(new_point.y,
          //                                     new_point.x + 1)});
          //     }
          //     matched = true;
          //     match_p = new_point;
          //     match_normal = new_normal;
          // }
          // else
          //     continue;
          new_point_double.x = new_point.x;
          new_point_double.y = new_point.y;
          if (new_point.x != 0 && new_point.x != gx2.cols - 1) {
            new_point_double.x =
                EdgeInter({new_point.x - 1, new_point.x, new_point.x + 1},
                          {*gx2.ptr<float>(new_point.y, new_point.x - 1),
                           *gx2.ptr<float>(new_point.y, new_point.x),
                           *gx2.ptr<float>(new_point.y, new_point.x + 1)});
          }
          //    outfile <<new_point << std::endl;
          src_points.emplace_back(p);
          target_points.emplace_back(new_point_double);
          src_normal.emplace_back(normal);
          // new_normal = NormalInter(gx2, gy2, new_point, new_grid);
          target_normal.emplace_back(new_normal);
          matched = true;
          if (debug_mode_) {
            int b = static_cast<int>(rand() % 255);
            int g = static_cast<int>(rand() % 255);
            int r = static_cast<int>(rand() % 255);
            cv::circle(rgb_src, {round(p.x), round(p.y)}, 1,
                       cv::Scalar(r, g, b), -1);
            cv::circle(rgb_target, {new_point.x, new_point.y}, 1,
                       cv::Scalar(r, g, b), -1);
          }
          // src_col_points.emplace_back(cv::Point2d{p.y, p.x});
          // target_col_points.emplace_back(cv::Point2d{new_point.y,
          // new_point.x}); col_normal.emplace_back(new_normal);
        }
      }
      // if(matched)
      // {
      //     src_points.emplace_back(p);
      //     target_points.emplace_back(match_p);
      //     src_normal.emplace_back(normal);
      //     target_normal.emplace_back(match_normal);
      //     if(debug_mode_)
      //     {
      //         int b =  rand() % 255;
      //         int g =  rand() % 255;
      //         int r =  rand() % 255;
      //         cv::circle(rgb_src, {round(p.x),round(p.y)}, 1, cv::Scalar(r,
      //         g, b), -1); cv::circle(rgb_target,
      //         {round(match_p.x),round(match_p.y)}, 1, cv::Scalar(r, g, b),
      //         -1);
      //     }
      // }
    }
  }
  //   AINFO << "match points num:" << src_points.size() << std::endl;
  if (src_points.size() < 50) {
    res2 *= (src_points.size() / 50);
  }
  if (debug_mode_) {
    // cv::namedWindow("match_src", cv::WINDOW_NORMAL);
    // cv::imshow("match_src", rgb_src);
    // cv::namedWindow("match_target", cv::WINDOW_NORMAL);
    // cv::imshow("match_target",rgb_target);
    cv::imwrite(out_dir + "match_src" + std::to_string(save_id) + ".jpg",
                rgb_src);
    cv::imwrite(out_dir + "match_target" + std::to_string(save_id) + ".jpg",
                rgb_target);
    // cv::waitKey(-1);
  }

  float scale = ScaleCal(src_points, target_points, src_normal, target_normal);
  // auto t = T_cal(src_points, target_points, src_normal,target_normal,scale);
  gettimeofday(&t2, NULL);
  t_filter_ground =
      (t2.tv_sec - t1.tv_sec) * 1000 * 1000 + (t2.tv_usec - t1.tv_usec);
  log_time += t_filter_ground / 1e3;
  if (debug_time_mode_) {
    AINFO << "match and cal: " << t_filter_ground << std::endl;
    _all_time += log_time;
    max_time = std::max(log_time, max_time);
    AINFO << "ave:" << _all_time / frame_count << "max:" << max_time;
  }

  // std::ofstream stime_txt("../scale_time.txt");
  // stime_txt<<""<<std::endl;
  // float speed = SingleSpeedEstimate(53.75, 0.5, scale);
  // AINFO << "speed: " << speed << std::endl;
  return {scale, res2};
}
float ScaleEstimate::EdgeInter(const std::vector<int>& coordinate,
                               const std::vector<float>& grid) {
  float grid0 = fabs(grid[0]);
  float grid1 = fabs(grid[1]);
  float grid2 = fabs(grid[2]);
  float new_coor = coordinate[1];
  if (grid0 == grid1 && grid0 == grid2 && grid0 + grid2 == 2 * grid1)
    return new_coor;
  else if (grid1 < grid0 || grid1 < grid2)
    return new_coor;
  else {
    new_coor =
        coordinate[1] + (grid0 - grid2) / (grid0 + grid2 - 2 * grid1) * 0.5;
    // if(debug_mode_)
    //     AINFO << grid0 << " "<< grid1 << " " << grid2 << " "
    //     <<coordinate[1]<<" " <<new_coor << std::endl;
    return new_coor;
  }
}

float ScaleEstimate::LinearInter(const cv::Mat& img, const cv::Point2d& pos) {
  float row = pos.y, col = pos.x;
  float row1 = int(row), row2 = int(row + 1), col1 = int(col),
        col2 = int(col + 1);
  if (row1 == img.rows - 1 || col1 == img.cols - 1 || row < 0 || col < 0)
    return *img.ptr<float>(row, col);
  float row_inter1 = (row2 - row) / (row2 - row1) * img.at<float>(row1, col1) +
                     (row - row1) / (row2 - row1) * img.at<float>(row2, col1);
  float row_inter2 = (row2 - row) / (row2 - row1) * img.at<float>(row1, col2) +
                     (row - row1) / (row2 - row1) * img.at<float>(row2, col2);
  return (col2 - col) / (col2 - col1) * row_inter1 +
         (col - col1) / (col2 - col1) * row_inter2;
}

cv::Point2d ScaleEstimate::NormalInter(const cv::Mat& gx, const cv::Mat& gy,
                                       const cv::Point2d& pos,
                                       cv::Point2d& grid) {
  float inter_gx = LinearInter(gx, pos);
  float inter_gy = LinearInter(gy, pos);
  grid = {inter_gy, inter_gy};
  cv::Point2d normal_direction(inter_gx, inter_gy);
  if (inter_gx == 0 && inter_gy == 0) return normal_direction;
  return normal_direction / sqrt(inter_gx * inter_gx + inter_gy * inter_gy);
}

void ScaleEstimate::GradCal(cv::Mat& mask, cv::Mat& img, cv::Mat& gx,
                            cv::Mat& gy, cv::Mat& mag_sqr, bool need_mask) {
  cv::Sobel(img, gx, CV_32FC1, 1, 0);
  cv::Sobel(img, gy, CV_32FC1, 0, 1);
  // 单独处理mask的边缘点
  // x方向跳变点，按行正反两个方向寻找所有跳变点
  if (need_mask) {
    for (int row = 0; row < mask.rows; ++row) {
      uint8_t* line = mask.ptr<uint8_t>(row);

      for (int col = 0; col < mask.cols - 1; ++col) {
        if (bool(line[col]) == 0 && bool(line[col + 1]) == 1) {
          *gx.ptr<float>(row, col + 1) = 0;
          *gx.ptr<float>(row, col) = 0;
        } else if (bool(line[col]) == 1 && bool(line[col + 1]) == 0) {
          *gx.ptr<float>(row, col + 1) = 0;
          *gx.ptr<float>(row, col) = 0;
        }
      }
    }

    // y方向跳变点
    for (int row = 0; row < mask.rows - 1; ++row) {
      uint8_t* line_0 = mask.ptr<uint8_t>(row);
      uint8_t* line_1 = mask.ptr<uint8_t>(row + 1);
      for (int col = 0; col < mask.cols; ++col) {
        if (line_0[col] == 0 && line_1[col] == 1) {
          *gy.ptr<float>(row, col) = 0;
          *gy.ptr<float>(row + 1, col) = 0;
        } else if (line_0[col] == 1 && line_1[col] == 0) {
          *gy.ptr<float>(row, col) = 0;
          *gy.ptr<float>(row + 1, col) = 0;
        }
      }
    }
  }

  cv::Mat gx_ = gx.mul(gx);
  cv::Mat gy_ = gy.mul(gy);
  // 记录为平方
  mag_sqr = gx_ + gy_;

  // 拓展，减少if
  gx.adjustROI(2, 2, 2, 2);
  gy.adjustROI(2, 2, 2, 2);
  // get magg
  if (debug_mode_) {
    cv::Mat tmp1, tmp2, tmp3;
    gx.convertTo(tmp1, CV_16S);
    gy.convertTo(tmp2, CV_16S);
    mag_sqr.convertTo(tmp3, CV_16S);
    cv::convertScaleAbs(tmp2, tmp2);
    cv::convertScaleAbs(tmp1, tmp1);
    cv::convertScaleAbs(tmp3, tmp3);
    std::string mask_bool = need_mask ? "mask" : "no";
    cv::imwrite(
        out_dir + "/mag_" + std::to_string(save_id) + mask_bool + ".jpg", tmp1);
    // cv::namedWindow("img_sobel_x", cv::WINDOW_NORMAL);
    // cv::imshow("img_sobel_x", tmp1);
    // cv::namedWindow("img_sobel_y", cv::WINDOW_NORMAL);
    // cv::imshow("img_sobel_y", tmp2);
    // cv::namedWindow("img_mag", cv::WINDOW_NORMAL);
    // cv::imshow("img_mag", tmp3);
    // cv::waitKey(-1);
  }
}
float ScaleEstimate::BaseTranslation(const cv::Mat& img1, const cv::Mat& img2,
                                     cv::Point2d& offset) {
  int h = std::min(img1.rows, img2.rows), w = std::min(img1.cols, img2.cols);
  cv::Mat img1_part = img1(cv::Range(0, h), cv::Range(0, w));
  cv::Mat img2_part = img2(cv::Range(0, h), cv::Range(0, w));
  img1_part.convertTo(img1_part, CV_32F);
  img2_part.convertTo(img2_part, CV_32F);
  double response = 0;
  offset = cv::phaseCorrelate(img1_part, img2_part, cv::noArray(), &response);
  if (response < 0.5) {
    h = int(h / 2);
    w = int(w / 2);
    img1_part = img1(cv::Range(0, h), cv::Range(0, w));
    img2_part = img2(cv::Range(0, h), cv::Range(0, w));
    offset = cv::phaseCorrelate(img1_part, img2_part, cv::noArray(), &response);
  }
  if (response < 0.5) {
    img1_part = img1(cv::Range(0, h), cv::Range(w, 2 * w));
    img2_part = img2(cv::Range(0, h), cv::Range(w, 2 * w));
    offset = cv::phaseCorrelate(img1_part, img2_part, cv::noArray(), &response);
  }
  if (debug_mode_) {
    AINFO << img1_part.rows << " " << img1_part.cols << "response: " << response
          << " offset: " << offset << std::endl;
  }
  return response;
}
float ScaleEstimate::TemplateMatch(const cv::Mat& src, const cv::Mat& templ,
                                   cv::Point2d& offset) {
  // template must larger than src
  // cut image to keep enough translation space
  // todo: add resize
  int h =
      std::min(src.rows - template_cut_size, templ.rows - template_cut_size);
  int w =
      std::min(src.cols - template_cut_size, templ.cols - template_cut_size);
  if (h <= 0 || w <= 0) return 0;
  // 所以配准图像的0 0 实际上就是原始图像的
  // (template_cut_size,template_cut_size) 那么如果0 0
  // 评分最大，对应了将原始图像的(template_cut_size,template_cut_size)
  // 移动到模板的0 0 所以结果要再减一个(template_cut_size,template_cut_size)
  cv::Mat part_img = src(cv::Range(template_cut_size, template_cut_size + h),
                         cv::Range(template_cut_size, template_cut_size + w));
  cv::Mat response;
  cv::matchTemplate(part_img, templ, response, cv::TM_CCOEFF_NORMED);
  double min_val, max_val = 0;
  cv::Point min_loc, max_loc;
  cv::minMaxLoc(response, &min_val, &max_val, &min_loc, &max_loc);
  offset = {static_cast<double>(max_loc.x - template_cut_size),
            static_cast<double>(max_loc.y - template_cut_size)};
  if (debug_mode_) {
    // cv::Mat tmp1;
    // part_img.convertTo(tmp1, CV_16S);
    // cv::convertScaleAbs(tmp1, tmp1);
    // cv::namedWindow("TemplateMatch_img", cv::WINDOW_NORMAL);
    // cv::imshow("TemplateMatch_img", tmp1);
  }
  //   AINFO << "max_loc:" << max_val << "offset:" << offset << std::endl;
  return max_val;
}
void ScaleEstimate::EdgePreprocess(const cv::Mat& gx, const cv::Mat& gy,
                                   const cv::Mat& mag,
                                   std::vector<cv::Point2d>& normal,
                                   cv::Mat& edge_label,
                                   std::vector<cv::Point2d>& edge_points) {
  edge_label = cv::Mat::zeros(gx.rows, gx.cols, CV_32S);

  // for relative threshhold
  double mag_max;
  cv::minMaxLoc(mag, 0, &mag_max, 0, 0);
  double gx_max, gx_min;
  cv::minMaxLoc(gx, &gx_min, &gx_max, 0, 0);
  double gy_max, gy_min;
  cv::minMaxLoc(gy, &gy_min, &gy_max, 0, 0);
  // 边缘相对阈值
  float mag_th = float(mag_max * 0.5);
  float gx_th = float(std::max(fabs(gx_max), fabs(gx_min)) * 0.8);
  float gy_th = float(std::max(fabs(gy_max), fabs(gy_min)) * 0.8);

  for (int row = 2; row < gx.rows - 2; ++row) {
    const float* mag_line = mag.ptr<float>(row);

    // 同时取出五行，用于非极大值抑制
    const float* gx2_line_0 = gx.ptr<float>(row - 2);
    const float* gx2_line_1 = gx.ptr<float>(row - 1);
    const float* gx2_line_2 = gx.ptr<float>(row);
    const float* gx2_line_3 = gx.ptr<float>(row + 1);
    const float* gx2_line_4 = gx.ptr<float>(row + 2);

    const float* gy2_line_0 = gy.ptr<float>(row - 2);
    const float* gy2_line_1 = gy.ptr<float>(row - 1);
    const float* gy2_line_2 = gy.ptr<float>(row);
    const float* gy2_line_3 = gy.ptr<float>(row + 1);
    const float* gy2_line_4 = gy.ptr<float>(row + 2);

    // float edge_row_thresh_sqr = edge_row_thresh * edge_row_thresh;
    // float edge_col_thresh_sqr = edge_col_thresh * edge_col_thresh;
    // float edge_row_thresh_sqr = gx_th;
    // float edge_col_thresh_sqr = gy_th;
    // 平方阈值
    float edge_normal_thresh_sqr = edge_normal_thresh * edge_normal_thresh;

    int* edge_label_ptr = edge_label.ptr<int>(row);

    for (int col = 2; col < gx.cols - 2; ++col) {
      float mag_local = mag_line[col];
      // if(mag_local < mag_th)
      if (mag_local < 10000) {
        continue;
      }

      // row points
      float gx_local = gx2_line_2[col];
      float gy_local = gy2_line_2[col];
      float gy_local_fabs = fabs(gy_local);
      float gx_local_fabs = fabs(gx_local);

      if (gy_local_fabs > edge_row_thresh &&
          gy_local * gy_local > edge_normal_thresh_sqr * mag_local) {
        // 非极大值抑制
        if (gy_local_fabs < fabs(gy2_line_1[col]) ||
            gy_local_fabs < fabs(gy2_line_0[col]) ||
            gy_local_fabs < fabs(gy2_line_3[col]) ||
            gy_local_fabs < fabs(gy2_line_4[col])) {
          continue;
        }

        edge_label_ptr[col] = 1;
        cv::Point2d cur_point(col, row);
        if (row != 2 && row != gx.rows - 3) {
          cur_point.y = EdgeInter({row - 1, row, row + 1},
                                  {gy2_line_1[col], gy_local, gy2_line_3[col]});
        }
        edge_points.emplace_back(cur_point);
        // normal inter?
        // 这个normal并不会用于实际计算，没必要插值
        cv::Point2d normal_direction(gx2_line_2[col], gy2_line_2[col]);
        normal_direction = normal_direction / sqrt(mag_local);
        normal.emplace_back(normal_direction);
      }
      // col points
      else if (gx_local_fabs > edge_col_thresh &&
               gx_local * gx_local > edge_normal_thresh_sqr * mag_local) {
        // 非极大值抑制
        if (gx_local_fabs < fabs(gx2_line_2[col - 2]) ||
            gx_local_fabs < fabs(gx2_line_2[col - 1]) ||
            gx_local_fabs < fabs(gx2_line_2[col + 1]) ||
            gx_local_fabs < fabs(gx2_line_2[col + 2])) {
          continue;
        }
        *edge_label.ptr<int>(row, col) = 1;
        cv::Point2d cur_point(col, row);
        if (col != 2 && col != gx.cols - 3) {
          cur_point.x =
              EdgeInter({col - 1, col, col + 1},
                        {gx2_line_2[col - 1], gx_local, gx2_line_2[col + 1]});
        }
        edge_points.emplace_back(cur_point);
        // normal inter?
        cv::Point2d normal_direction(gx2_line_2[col], gy2_line_2[col]);
        normal_direction = normal_direction / sqrt(mag_local);
        normal.emplace_back(normal_direction);
      }
    }
  }

  if (debug_mode_) {
    cv::Mat vis_edge;
    edge_label.convertTo(vis_edge, CV_8U);
    // cv::namedWindow("edge", cv::WINDOW_NORMAL);
    // cv::imshow("edge", vis_edge*255);
    cv::imwrite(out_dir + "/edge_label" + std::to_string(save_id) + ".jpg",
                vis_edge * 255);
    // cv::waitKey(-1);
  }
  return;
}

float ScaleEstimate::ScaleCal(const std::vector<cv::Point2d>& src_points,
                              const std::vector<cv::Point2d>& target_points,
                              const std::vector<cv::Point2d>& src_normal,
                              const std::vector<cv::Point2d>& target_normal) {
  if (src_points.size() < 5) return 0;
  int num_x = 0, num_y = 0;
  double AY = 0, BY = 0, CY = 0, DY = 0, EY = 0, y_sum1 = 0, y_sum2 = 0,
         y2_sum1 = 0, y2_sum2 = 0, yy_sum = 0;
  double AX = 0, BX = 0, CX = 0, DX = 0, EX = 0, x_sum1 = 0, x_sum2 = 0,
         x2_sum1 = 0, x2_sum2 = 0, xx_sum = 0;
  for (int i = 0; i < src_points.size(); ++i) {
    // col point -> cal x
    if (fabs(src_normal[i].x) > edge_normal_thresh) {
      ++num_x;
      double px = src_points[i].x;
      double qx = target_points[i].x;
      double nqx = target_normal[i].x;
      // AINFO << px << " " << qx << " " << nqx << std::endl;
      x_sum1 = x_sum1 + px;
      x_sum2 = x_sum2 + qx;
      x2_sum1 = x2_sum1 + px * px;
      x2_sum2 = x2_sum2 + qx * qx;
      xx_sum = xx_sum + px * qx;
      AX = AX + nqx * nqx * px * qx;
      BX = BX + nqx * nqx * px;
      CX = CX + nqx * nqx * qx;
      DX = DX + nqx * nqx;
      EX = EX + nqx * nqx * px * px;
    }

    // row point -> cal y
    else if (fabs(src_normal[i].y) > edge_normal_thresh) {
      ++num_y;
      double py = src_points[i].y;
      double qy = target_points[i].y;
      double nqy = target_normal[i].y;
      y_sum1 = y_sum1 + py;
      y_sum2 = y_sum2 + qy;
      y2_sum1 = y2_sum1 + py * py;
      y2_sum2 = y2_sum2 + qy * qy;
      yy_sum = yy_sum + py * qy;
      AY = AY + nqy * nqy * py * qy;
      BY = BY + nqy * nqy * py;
      CY = CY + nqy * nqy * qy;
      DY = DY + nqy * nqy;
      EY = EY + nqy * nqy * py * py;
    }
  }
  // AINFO<< AY <<" "<< BY <<" "<< CY<<" "<<  DY<<" "<<  EY<<" "<<y_sum1<<"
  // "<<  y_sum2<<" "<<  y2_sum1<<" "<<y2_sum2<<" "<<yy_sum<<" "<<std::endl;
  // AINFO<< AX <<" "<< BX <<" "<< CX<<" "<<  DX<<" "<<  EX<<" "<< x_sum1<<"
  // "<<  x_sum2<<" "<<  x2_sum1<<" "<<x2_sum2<<" "<<xx_sum<<" "<<std::endl;

  if (DY != 0 && DX != 0) {
    double numerator = AX - BX * CX / DX + AY - BY * CY / DY;
    double denominator = EX - BX * BX / DX + EY - BY * BY / DY;
    double scale = numerator / denominator;
    if (debug_mode_) {
      AINFO << "scale with n: " << scale << std::endl;
    }
    return scale;
  }
  if (num_x != 0 && num_y != 0) {
    double numerator = yy_sum / num_y - (y_sum1 / num_y) * (y_sum2 / num_y) +
                       xx_sum / num_x - x_sum1 / num_x * x_sum2 / num_x;
    double denominator = y2_sum1 / num_y - (y_sum1 / num_y) * (y_sum1 / num_y) +
                         x2_sum1 / num_x - (x_sum1 / num_x) * (x_sum1 / num_x);
    if (denominator != 0) {
      double scale = numerator / denominator;
      if (debug_mode_) {
        AINFO << "scale with x-y: " << scale << std::endl;
      }
      return scale;
    }
  }
  if (num_x < 5 && num_y < 5) return 0;
  if (num_y != 0) {
    double y_ave1 = y_sum1 / num_y;
    double y_ave2 = y_sum2 / num_y;
    double numerator = 0, denominator = 0;
    for (int i = 0; i < src_points.size(); ++i) {
      numerator += (src_points[i].y - y_ave1) * (target_points[i].y - y_ave2);
      denominator += (src_points[i].y - y_ave1) * (src_points[i].y - y_ave1);
    }
    if (denominator != 0) {
      double scale = numerator / denominator;
      if (debug_mode_) {
        AINFO << "scale with y: " << scale << std::endl;
      }
      return scale;
    }
  }
  if (num_x != 0) {
    double x_ave1 = x_sum1 / num_x;
    double x_ave2 = x_sum2 / num_x;
    double numerator = 0, denominator = 0;
    for (int i = 0; i < src_points.size(); ++i) {
      numerator += (src_points[i].x - x_ave1) * (target_points[i].x - x_ave2);
      denominator += (src_points[i].x - x_ave1) * (src_points[i].x - x_ave1);
    }
    if (denominator != 0) {
      double scale = numerator / denominator;
      if (debug_mode_) {
        AINFO << "scale with x: " << scale << std::endl;
      }
      return scale;
    }
  }
  return 0;
}
// m/s
float ScaleEstimate::SingleSpeedEstimate(const float pre_d, const float t,
                                         const float scale) {
  return (pre_d / scale - pre_d) / t;
}

cv::Point2d ScaleEstimate::T_cal(const std::vector<cv::Point2d>& src_points,
                                 const std::vector<cv::Point2d>& target_points,
                                 const std::vector<cv::Point2d>& src_normal,
                                 const std::vector<cv::Point2d>& target_normal,
                                 const float scale) {
  if (scale == 0) return {0, 0};
  cv::Mat A = cv::Mat::zeros(2, 2, CV_32F);
  cv::Mat B = cv::Mat::zeros(2, 1, CV_32F);
  for (int i = 0; i < src_points.size(); ++i) {
    cv::Mat nq = cv::Mat::zeros(2, 1, CV_32F);
    cv::Mat p = cv::Mat::zeros(2, 1, CV_32F);
    cv::Mat q = cv::Mat::zeros(2, 1, CV_32F);
    *nq.ptr<float>(0, 0) = target_normal[i].x;
    *nq.ptr<float>(1, 0) = target_normal[i].y;
    *p.ptr<float>(0, 0) = src_points[i].x;
    *p.ptr<float>(1, 0) = src_points[i].y;
    *q.ptr<float>(0, 0) = target_points[i].x;
    *q.ptr<float>(1, 0) = target_points[i].y;
    A = A + nq * nq.t();
    B = B + nq * nq.t() * (q - scale * p);
  }
  cv::Mat t = A.inv() * B;
  AINFO << "translation: " << t << std::endl;
  return {*t.ptr<float>(0, 0), *t.ptr<float>(1, 0)};
}
