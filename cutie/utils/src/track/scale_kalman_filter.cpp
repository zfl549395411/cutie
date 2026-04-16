#include "hyper_vision/perception/pv_post_process/scale_kalman_filter.hpp"

#include "cyber/cyber.h"

std::pair<float, float> KalmanFilter::update_observe_and_predict(
    const float cur_deta_t, const float observe_y) {
  cv::Mat current_measurment(1, 1, CV_32F);
  *current_measurment.ptr<float>(0, 0) = observe_y;
  *F.ptr<float>(0, 1) = cur_deta_t;
  cv::Mat x_predict = F * last_prediction;
  P = F * P * F.t() + Q;
  cv::Mat s = H * P * H.t() + R;
  cv::Mat K = P * H.t() * s.inv();
  x_predict = x_predict + K * (current_measurment - H * x_predict);
  P = P - K * H * P;

  last_prediction = x_predict;
  last_measurment = current_measurment;
  if (debug_mode_) {
    AINFO << x_predict << std::endl;
  }
  return {*x_predict.ptr<float>(0, 0), *x_predict.ptr<float>(1, 0)};
}

// update all data
void CombineOptimizeFilter::update_data(const float weight_d, const float d_o,
                                        const float deta_t,
                                        const float weight_v, const float v_o) {
  if (weight_di.size() == window) {
    weight_di.pop_front();
    weight_di.push_back(weight_d);
    weight_vi.pop_front();
    weight_vi.push_back(weight_v);

    for (auto &t : deta_ti) {
      t -= deta_t;
    }
    deta_ti.pop_front();
    deta_ti.push_back(0);

    d_oi.pop_front();
    d_oi.push_back(d_o);
    v_oi.pop_front();
    v_oi.push_back(v_o);
  } else {
    weight_di.push_back(weight_d);
    weight_vi.push_back(weight_v);

    for (auto &t : deta_ti) {
      t -= deta_t;
    }
    deta_ti.push_back(0);

    d_oi.push_back(d_o);
    v_oi.push_back(v_o);
  }
}
void CombineOptimizeFilter::delete_data() {
  if (weight_di.size() > 0) {
    weight_di.pop_front();
    weight_vi.pop_front();
    deta_ti.pop_front();
    d_oi.pop_front();
    v_oi.pop_front();
  }
}
std::pair<float, float> CombineOptimizeFilter::predict(const float sn) {
  if (s_i.size() == window) {
    s_i.pop_front();
  }
  s_i.push_back(sn);
  float A1 = 0, A2 = 0, A3 = 0, A4 = 0, B1 = 0, B2 = 0;
  for (int i = 0; i < weight_di.size(); ++i) {
    // AINFO << weight_di[i] << " " << weight_vi[i] << " " << deta_ti[i] <<
    // " " << d_oi[i] << " " << v_oi[i] <<" "<< s_i[i] << std::endl;
    A1 = A1 + weight_di[i] + (s_i[i] - sn) * (s_i[i] - sn);
    A2 = A2 + (weight_di[i] + s_i[i] * (s_i[i] - sn)) * deta_ti[i];
    A3 = A2;
    A4 = A4 + weight_vi[i] +
         (s_i[i] * s_i[i] + weight_di[i]) * deta_ti[i] * deta_ti[i];
    B1 = B1 + weight_di[i] * d_oi[i];
    B2 = B2 + weight_di[i] * d_oi[i] * deta_ti[i] + weight_vi[i] * v_oi[i];
  }
  cv::Mat A = cv::Mat::zeros(2, 2, CV_32F);
  cv::Mat B = cv::Mat::zeros(2, 1, CV_32F);
  *A.ptr<float>(0, 0) = A1;
  *A.ptr<float>(0, 1) = A2;
  *A.ptr<float>(1, 0) = A3;
  *A.ptr<float>(1, 1) = A4 + 1;
  *B.ptr<float>(0, 0) = B1;
  *B.ptr<float>(1, 0) = B2 + pre_v;

  cv::Mat res = A.inv() * B;

  // AINFO << "A: " << A << " B:" << B <<" res: "<<res<< std::endl;
  pre_v = *res.ptr<float>(1, 0);
  weight_di.back() = std::max(float(0.5), float(weight_di.back()));
  d_oi.back() = *res.ptr<float>(0, 0);
  weight_vi.back() = 1;
  v_oi.back() = *res.ptr<float>(1, 0);
  return {*res.ptr<float>(0, 0), *res.ptr<float>(1, 0)};
}

void KalmanFilterExtend::update_v(const float mono3d_v) {
  *pre_predict_res.ptr<float>(1, 0) = mono3d_v;
}

// 仅用于update之后额外添加，因为并不会清空d_o_n和d_c_n
void KalmanFilterExtend::AddDepthObserve(
    const std::vector<std::pair<float, double>> &d_obs,
    const std::vector<float> &w_d, const bool is_radar, const bool is_lidar) {
  if (is_radar) {
    pre_match_radar = 0;
  }
  if (is_lidar) {
    pre_match_lidar = 0;
  }
  cv::Mat add_d = cv::Mat::zeros(d_obs.size(), 1, CV_32F);
  cv::Mat tmp_d_c = cv::Mat::zeros(d_c_n.rows + d_obs.size(),
                                   d_c_n.rows + d_obs.size(), CV_32F);
  for (int i = 0; i < d_c_n.rows; ++i) {
    for (int j = 0; j < d_c_n.cols; ++j) {
      *tmp_d_c.ptr<float>(i, j) = *d_c_n.ptr<float>(i, j);
    }
  }
  cv::Mat add_H = cv::Mat::zeros(d_obs.size(), 2, CV_32F);
  for (int i = 0; i < d_obs.size(); ++i) {
    *add_d.ptr<float>(i, 0) =
        d_obs[i].first + *pre_predict_res.ptr<float>(1, 0) * d_obs[i].second;
    *tmp_d_c.ptr<float>(d_c_n.rows + i, d_c_n.rows + i) = fabs(w_d[i]);
    *add_H.ptr<float>(i, 0) = 1;
    // *H1.ptr<float>(i, 1) = d_obs[i].second;
  }
  cv::vconcat(d_o_n, add_d, d_o_n);
  d_c_n = tmp_d_c;
  cv::vconcat(H1, add_H, H1);
}
void KalmanFilterExtend::AddVelObserve(const std::vector<float> &v_obs,
                                       const std::vector<float> &w_v) {
  cv::Mat add_v = cv::Mat::zeros(v_obs.size(), 1, CV_32F);
  cv::Mat tmp_v_c = cv::Mat::zeros(v_c_n.rows + v_obs.size(),
                                   v_c_n.rows + v_obs.size(), CV_32F);
  for (int i = 0; i < v_c_n.rows; ++i) {
    for (int j = 0; j < v_c_n.cols; ++j) {
      *tmp_v_c.ptr<float>(i, j) = *v_c_n.ptr<float>(i, j);
    }
  }
  tmp_v_c(cv::Rect(0, 0, v_c_n.rows, v_c_n.cols)) = v_c_n;
  cv::Mat add_H = cv::Mat::zeros(v_obs.size(), 2, CV_32F);
  for (int i = 0; i < v_obs.size(); ++i) {
    *add_v.ptr<float>(i, 0) = v_obs[i];
    *tmp_v_c.ptr<float>(v_c_n.rows + i, v_c_n.rows + i) = fabs(w_v[i]);
    *add_H.ptr<float>(i, 1) = 1;
  }

  cv::vconcat(v_o_n, add_v, v_o_n);
  v_c_n = tmp_v_c;
  cv::vconcat(H2, add_H, H2);
}
// 距离输入要包括时间戳插值
void KalmanFilterExtend::update_observe(
    const std::vector<std::pair<float, double>> &d_obs,
    const std::vector<float> &w_d, const std::vector<float> &v_obs,
    const std::vector<float> &w_v, const float deta_t_n, const float sn,
    const float w_s, const int64_t cur_time) {
  pre_match_lidar = pre_match_lidar == 5 ? 5 : pre_match_lidar + 1;
  pre_match_radar = pre_match_radar == 5 ? 5 : pre_match_radar + 1;
  ++frame_count;
  time_gap = float(cur_time - pre_update_time) / 1e6;
  time_gap = std::min(std::max(0.1f, time_gap), 0.5f);
  pre_update_time = cur_time;

  // 这个deta_t_n必须是连续的，因为里面的递推依赖于均匀关系
  if (motionless) {
    if (frame_count < 3) {
      *Q.ptr<float>(0, 0) = 1;
      *Q.ptr<float>(1, 1) = 5;
    } else {
      *Q.ptr<float>(0, 0) = 0.1;
      *Q.ptr<float>(1, 1) = 5;
    }
  } else {
    *Q.ptr<float>(0, 0) = float(std::min(pre_match_lidar, pre_match_radar)) / 5;
    *Q.ptr<float>(1, 1) = pre_match_radar;
  }

  // AINFO << "update" << std::endl;
  // AINFO << sn << " " << deta_t_n << std::endl;
  // 优化间隔,这个必须和实际对上，因为要用这个初始化当前量的结果
  *A.ptr<float>(0, 1) = time_gap;

  // *Q.ptr<float>(1, 0) = speed_weight ==0? 0:speed_weight--;
  // dont use scale

  if (sn < 0.1) {
    update_s = false;
    for (auto &t : deta_t_i) {
      t += time_gap;
    }
    if (!deta_t_i.empty() && deta_t_i[0] > 0.5) {
      deta_t_i.pop_front();
      s_c_i.pop_front();
      s_i.pop_front();
    }
  } else {
    update_s = true;
    if (deta_t_i.size() == window) {
      deta_t_i.pop_front();
      for (auto &t : deta_t_i) {
        t += time_gap;
      }

      deta_t_i.push_back(deta_t_n);
      s_i.pop_front();
      for (auto &s : s_i) {
        s *= sn;
      }
      s_i.push_back(sn);
      s_c_i.pop_front();
      s_c_i.push_back(fabs(w_s));
    } else {
      for (auto &t : deta_t_i) {
        t += time_gap;
      }

      deta_t_i.push_back(deta_t_n);
      for (auto &s : s_i) {
        s *= sn;
      }
      s_i.push_back(sn);
      s_c_i.push_back(fabs(w_s));
    }
  }
  if (fabs(*pre_predict_res.ptr<float>(0, 0)) < 5) {
    deta_t_i.clear();
    s_c_i.clear();
    s_i.clear();
  }

  d_o_n = cv::Mat::zeros(d_obs.size(), 1, CV_32F);
  d_c_n = cv::Mat::zeros(w_d.size(), w_d.size(), CV_32F);
  H1 = cv::Mat::zeros(d_obs.size(), 2, CV_32F);
  for (int i = 0; i < d_obs.size(); ++i) {
    *d_o_n.ptr<float>(i, 0) =
        d_obs[i].first + *pre_predict_res.ptr<float>(1, 0) * d_obs[i].second;
    *d_c_n.ptr<float>(i, i) = fabs(w_d[i]);
    *H1.ptr<float>(i, 0) = 1;
    // *H1.ptr<float>(i, 1) = d_obs[i].second;
  }

  v_o_n = cv::Mat::zeros(v_obs.size(), 1, CV_32F);
  v_c_n = cv::Mat::zeros(w_v.size(), w_v.size(), CV_32F);
  H2 = cv::Mat::zeros(v_obs.size(), 2, CV_32F);
  for (int i = 0; i < v_obs.size(); ++i) {
    *v_o_n.ptr<float>(i, 0) = v_obs[i];
    *v_c_n.ptr<float>(i, i) = fabs(w_v[i]);
    *H2.ptr<float>(i, 1) = 1;
  }
}
std::pair<float, float> KalmanFilterExtend::update_and_predict() {
  if (update_s == false) return update_and_predict_without_s();
  pre_predict_res = A * pre_predict_res;
  //   AINFO << "predict" << pre_predict_res;
  M = A.t() * M * A + Q;
  H3 = cv::Mat::zeros(deta_t_i.size(), 2, CV_32F);
  float tmp_d = *pre_predict_res.ptr<float>(0, 0);
  float tmp_v = *pre_predict_res.ptr<float>(1, 0);
  for (int i = 0; i < deta_t_i.size(); ++i) {
    *H3.ptr<float>(i, 0) = tmp_v / (tmp_d * tmp_d) * deta_t_i[i];
    *H3.ptr<float>(i, 1) = -1 / tmp_d * deta_t_i[i];
  }
  cv::Mat mid_H;
  cv::vconcat(H1, H2, mid_H);
  cv::vconcat(mid_H, H3, H);

  cv::Mat tmp_s_i = cv::Mat::zeros(s_i.size(), 1, CV_32F);
  cv::Mat tmp_s_c = cv::Mat::zeros(s_i.size(), s_i.size(), CV_32F);
  for (int i = 0; i < s_i.size(); ++i) {
    *tmp_s_i.ptr<float>(i, 0) = s_i[i] - (1 - tmp_v / tmp_d * deta_t_i[i]);
    *tmp_s_c.ptr<float>(i, i) = s_c_i[i];
  }

  int mat_size = d_o_n.rows + v_o_n.rows + tmp_s_c.rows;
  cv::Mat C = cv::Mat::zeros(mat_size, mat_size, CV_32F);
  for (int i = 0; i < C.rows; ++i) {
    if (i < d_o_n.rows) {
      *C.ptr<float>(i, i) = *d_c_n.ptr<float>(i, i);
    } else if (i < d_o_n.rows + v_c_n.rows) {
      *C.ptr<float>(i, i) = *v_c_n.ptr<float>(i - d_o_n.rows, i - d_o_n.rows);
    } else {
      *C.ptr<float>(i, i) = *tmp_s_c.ptr<float>(i - d_o_n.rows - v_o_n.rows,
                                                i - d_o_n.rows - v_o_n.rows);
    }
  }
  // AINFO << "w_d_n" << d_c_n << std::endl;
  // AINFO << "s_d_n" << tmp_s_c << std::endl;
  cv::Mat tmp = (H * M * H.t() + C);
  //   AINFO << "A" << A << std::endl;
  //   AINFO << "Q" << Q << std::endl;
  //   AINFO << "H" << H << std::endl;
  //   AINFO << "C" << C << std::endl;
  //   AINFO << "PRE_M" << M << std::endl;
  //   AINFO << " input x y vx vy" << d_o_n << " " << v_o_n << " " << tmp_s_i <<
  //   " "
  //         << d_c_n << " " << v_c_n;
  K = M * H.t() * tmp.inv();
  //   AINFO << "K" << K << std::endl;
  if (d_o_n.rows != 0) d_o_n = d_o_n - tmp_d;
  if (v_o_n.rows != 0) v_o_n = v_o_n - tmp_v;
  cv::Mat mid_k, tmp_k;
  cv::vconcat(d_o_n, v_o_n, mid_k);
  cv::vconcat(mid_k, tmp_s_i, tmp_k);
  // AINFO << "tmp_k" << tmp_k << std::endl;
  pre_predict_res = pre_predict_res + K * tmp_k;
  //   AINFO << "pred" << pre_predict_res << std::endl;
  cv::Mat tmp_m = cv::Mat::eye(M.rows, M.cols, CV_32F);
  M = (tmp_m - K * H) * M;

  if (smooth_window_ != 0) {
    float ret = 0;
    if (smooth_window_ == v_predict_history.size()) {
      v_predict_sum -= v_predict_history.front();
      v_predict_history.pop_front();
    }
    v_predict_history.push_back(*pre_predict_res.ptr<float>(1, 0));
    ret = (v_predict_sum + *pre_predict_res.ptr<float>(1, 0) * smooth_weight) /
          (v_predict_history.size() + smooth_weight - 1);
    v_predict_sum += *pre_predict_res.ptr<float>(1, 0);
    if (frame_count < 2 * smooth_window_)
      return {*pre_predict_res.ptr<float>(0, 0),
              *pre_predict_res.ptr<float>(1, 0)};
    else
      return {*pre_predict_res.ptr<float>(0, 0), ret};
  }
  // AINFO << "pred" << pre_predict_res << std::endl;
  return {*pre_predict_res.ptr<float>(0, 0), *pre_predict_res.ptr<float>(1, 0)};
}
std::pair<float, float> KalmanFilterExtend::update_and_predict_without_s() {
  pre_predict_res = A * pre_predict_res;
  M = A.t() * M * A + Q;
  float tmp_d = *pre_predict_res.ptr<float>(0, 0);
  float tmp_v = *pre_predict_res.ptr<float>(1, 0);

  cv::vconcat(H1, H2, H);

  int mat_size = d_o_n.rows + v_o_n.rows;
  cv::Mat C = cv::Mat::zeros(mat_size, mat_size, CV_32F);
  for (int i = 0; i < C.rows; ++i) {
    if (i < d_o_n.rows) {
      *C.ptr<float>(i, i) = *d_c_n.ptr<float>(i, i);
    } else if (i < d_o_n.rows + v_c_n.rows) {
      *C.ptr<float>(i, i) = *v_c_n.ptr<float>(i - d_o_n.rows, i - d_o_n.rows);
    }
  }
  // AINFO << "w_d_n" << d_c_n << std::endl;
  cv::Mat tmp = (H * M * H.t() + C);
  //   AINFO << "A" << A << std::endl;
  //   AINFO << "Q" << Q << std::endl;
  //   AINFO << "H" << H << std::endl;
  //   AINFO << "C" << C << std::endl;
  //   AINFO << " input x y vx vy" << d_o_n << " " << v_o_n << " " << d_c_n << "
  //   "
  //         << v_c_n;
  //   AINFO << "PRE_M" << M << std::endl;
  K = M * H.t() * tmp.inv();
  //   AINFO << "K" << K << std::endl;
  if (d_o_n.rows != 0) d_o_n = d_o_n - tmp_d;
  if (v_o_n.rows != 0) v_o_n = v_o_n - tmp_v;
  cv::Mat tmp_k;
  cv::vconcat(d_o_n, v_o_n, tmp_k);
  //   AINFO << "tmp_k" << tmp_k << std::endl;
  pre_predict_res = pre_predict_res + K * tmp_k;
  cv::Mat tmp_m = cv::Mat::eye(M.rows, M.cols, CV_32F);
  M = (tmp_m - K * H) * M;
  //   AINFO << "pred" << pre_predict_res << std::endl;
  if (smooth_window_ != 0) {
    float ret = 0;
    if (smooth_window_ == v_predict_history.size()) {
      v_predict_sum -= v_predict_history.front();
      v_predict_history.pop_front();
    }
    v_predict_history.push_back(*pre_predict_res.ptr<float>(1, 0));
    ret = (v_predict_sum + *pre_predict_res.ptr<float>(1, 0) * smooth_weight) /
          (v_predict_history.size() + smooth_weight - 1);
    v_predict_sum += *pre_predict_res.ptr<float>(1, 0);
    if (frame_count < 2 * smooth_window_)
      return {*pre_predict_res.ptr<float>(0, 0),
              *pre_predict_res.ptr<float>(1, 0)};
    else
      return {*pre_predict_res.ptr<float>(0, 0), ret};
  }

  return {*pre_predict_res.ptr<float>(0, 0), *pre_predict_res.ptr<float>(1, 0)};
}
void KalmanFilterExtend::delete_data() {
  deta_t_i.pop_back();
  s_i.pop_back();
  s_c_i.pop_back();
}

void KalmanFilterExtend::set_speed(const float ori_speed) {
  *pre_predict_res.ptr<float>(1, 0) = ori_speed;
}

void KalmanFilterExtendForOdom::update_v(const float mono3d_v_x,
                                         const float mono3d_v_y) {
  *pre_predict_res.ptr<float>(2, 0) = mono3d_v_x;
  *pre_predict_res.ptr<float>(3, 0) = mono3d_v_y;
}

void KalmanFilterExtendForOdom::update_observe(
    const std::vector<std::pair<float, double>> &x_obs,
    const std::vector<std::pair<float, double>> &y_obs,
    const std::vector<float> &w_d, const std::vector<float> &v_x_obs,
    const std::vector<float> &v_y_obs, const std::vector<float> &w_v_x,
    const std::vector<float> &w_v_y, const int64_t cur_time) {
  ++frame_count;
  time_gap = float(cur_time - pre_update_time) / 1e6;
  time_gap = std::min(std::max(0.1f, time_gap), 0.5f);
  pre_update_time = cur_time;

  if (frame_count < 10) {
    *Q.ptr<float>(0, 0) = 50;
    *Q.ptr<float>(1, 1) = 50;
    *Q.ptr<float>(2, 2) = 5;
    *Q.ptr<float>(3, 3) = 5;
  } else {
    *Q.ptr<float>(0, 0) = 5;
    *Q.ptr<float>(1, 1) = 5;
    *Q.ptr<float>(2, 2) = 1;
    *Q.ptr<float>(3, 3) = 1;
  }

  *A.ptr<float>(0, 2) = time_gap;
  *A.ptr<float>(1, 3) = time_gap;

  H1 = cv::Mat::zeros(x_obs.size() + y_obs.size(), 4, CV_32F);
  H2 = cv::Mat::zeros(v_x_obs.size() + v_y_obs.size(), 4, CV_32F);

  x_o_n = cv::Mat::zeros(x_obs.size(), 1, CV_32F);
  x_c_n = cv::Mat::zeros(x_obs.size(), x_obs.size(), CV_32F);
  for (int i = 0; i < x_obs.size(); ++i) {
    *x_o_n.ptr<float>(i, 0) =
        x_obs[i].first + *pre_predict_res.ptr<float>(2, 0) * x_obs[i].second;
    *x_c_n.ptr<float>(i, i) = fabs(w_d[i]);
    *H1.ptr<float>(i, 0) = 1;
  }
  y_o_n = cv::Mat::zeros(y_obs.size(), 1, CV_32F);
  y_c_n = cv::Mat::zeros(y_obs.size(), y_obs.size(), CV_32F);
  for (int i = 0; i < y_obs.size(); ++i) {
    *y_o_n.ptr<float>(i, 0) =
        y_obs[i].first + *pre_predict_res.ptr<float>(3, 0) * y_obs[i].second;
    *y_c_n.ptr<float>(i, i) = fabs(w_d[i]);
    *H1.ptr<float>(i + x_obs.size(), 1) = 1;
  }

  v_x_o_n = cv::Mat::zeros(v_x_obs.size(), 1, CV_32F);
  v_x_c_n = cv::Mat::zeros(v_x_obs.size(), v_x_obs.size(), CV_32F);
  for (int i = 0; i < v_x_obs.size(); ++i) {
    *v_x_o_n.ptr<float>(i, 0) = v_x_obs[i];
    *v_x_c_n.ptr<float>(i, i) = fabs(w_v_x[i]);
    *H2.ptr<float>(i, 2) = 1;
  }
  v_y_o_n = cv::Mat::zeros(v_y_obs.size(), 1, CV_32F);
  v_y_c_n = cv::Mat::zeros(v_y_obs.size(), v_y_obs.size(), CV_32F);
  for (int i = 0; i < v_y_obs.size(); ++i) {
    *v_y_o_n.ptr<float>(i, 0) = v_y_obs[i];
    *v_y_c_n.ptr<float>(i, i) = fabs(w_v_y[i]);
    *H2.ptr<float>(i + v_x_obs.size(), 3) = 1;
  }
}
std::array<float, 4> KalmanFilterExtendForOdom::update_and_predict() {
  //   AINFO << "prdict " << pre_predict_res;
  pre_predict_res = A * pre_predict_res;

  M = A.t() * M * A + Q;
  float tmp_x = *pre_predict_res.ptr<float>(0, 0);
  float tmp_y = *pre_predict_res.ptr<float>(1, 0);
  float tmp_v_x = *pre_predict_res.ptr<float>(2, 0);
  float tmp_v_y = *pre_predict_res.ptr<float>(3, 0);

  cv::vconcat(H1, H2, H);

  int tmp_count = 0;
  int mat_size = x_o_n.rows + y_o_n.rows + v_x_o_n.rows + v_y_o_n.rows;
  cv::Mat C = cv::Mat::zeros(mat_size, mat_size, CV_32F);
  for (int i = 0; i < x_o_n.rows; ++i) {
    *C.ptr<float>(tmp_count, tmp_count) = *x_c_n.ptr<float>(i, i);
    ++tmp_count;
  }
  for (int i = 0; i < y_o_n.rows; ++i) {
    *C.ptr<float>(tmp_count, tmp_count) = *y_c_n.ptr<float>(i, i);
    ++tmp_count;
  }
  for (int i = 0; i < v_x_o_n.rows; ++i) {
    *C.ptr<float>(tmp_count, tmp_count) = *v_x_c_n.ptr<float>(i, i);
    ++tmp_count;
  }
  for (int i = 0; i < v_y_o_n.rows; ++i) {
    *C.ptr<float>(tmp_count, tmp_count) = *v_y_c_n.ptr<float>(i, i);
    ++tmp_count;
  }
  //   AINFO << "C" << C;
  cv::Mat tmp = (H * M * H.t() + C);

  K = M * H.t() * tmp.inv();
  //   AINFO << " input x y vx vy" << x_o_n << " " << y_o_n << " " << v_x_o_n <<
  //   " "
  //         << v_y_o_n;
  if (x_o_n.rows != 0) x_o_n = x_o_n - tmp_x;
  if (y_o_n.rows != 0) y_o_n = y_o_n - tmp_y;
  if (v_x_o_n.rows != 0) v_x_o_n = v_x_o_n - tmp_v_x;
  if (v_y_o_n.rows != 0) v_y_o_n = v_y_o_n - tmp_v_y;

  cv::Mat tmp_k1, tmp_k2, tmp_k;
  cv::vconcat(x_o_n, y_o_n, tmp_k1);
  cv::vconcat(v_x_o_n, v_y_o_n, tmp_k2);
  cv::vconcat(tmp_k1, tmp_k2, tmp_k);
  pre_predict_res = pre_predict_res + K * tmp_k;
  //   AINFO << "prdict " << pre_predict_res;
  cv::Mat tmp_m = cv::Mat::eye(M.rows, M.cols, CV_32F);
  M = (tmp_m - K * H) * M;
  // AINFO << "pred" << pre_predict_res << std::endl;
  if (smooth_window_ != 0) {
    float ret_x = 0;
    float ret_y = 0;
    float acc = 0;
    if (smooth_window_ == v_predict_history.size()) {
      float first_v = std::hypot(v_predict_history.front().first,
                                 v_predict_history.front().second);
      float back_v = std::hypot(v_predict_history.back().first,
                                v_predict_history.back().second);
      float acc = std::abs(back_v - first_v) / (v_predict_history.size() * 0.1);
      if (acc > 1) {
        smooth_weight = smooth_window_ * acc;
      } else {
        smooth_weight = 2;
      }
      v_predict_sum.first -= v_predict_history.front().first;
      v_predict_sum.second -= v_predict_history.front().second;
      v_predict_history.pop_front();
    }
    v_predict_history.push_back(
        {*pre_predict_res.ptr<float>(2, 0), *pre_predict_res.ptr<float>(3, 0)});
    ret_x = (v_predict_sum.first +
             *pre_predict_res.ptr<float>(2, 0) * smooth_weight) /
            (v_predict_history.size() + smooth_weight - 1);
    ret_y = (v_predict_sum.second +
             *pre_predict_res.ptr<float>(3, 0) * smooth_weight) /
            (v_predict_history.size() + smooth_weight - 1);
    v_predict_sum.first += *pre_predict_res.ptr<float>(2, 0);
    v_predict_sum.second += *pre_predict_res.ptr<float>(3, 0);
    if (frame_count < 2 * smooth_window_)
      return {
          *pre_predict_res.ptr<float>(0, 0), *pre_predict_res.ptr<float>(1, 0),
          *pre_predict_res.ptr<float>(2, 0), *pre_predict_res.ptr<float>(3, 0)};
    else
      return {*pre_predict_res.ptr<float>(0, 0),
              *pre_predict_res.ptr<float>(1, 0), ret_x, ret_y};
  }

  return {*pre_predict_res.ptr<float>(0, 0), *pre_predict_res.ptr<float>(1, 0),
          *pre_predict_res.ptr<float>(2, 0), *pre_predict_res.ptr<float>(3, 0)};
}
// return : x_odom y_odom height x_veh
// lidar_refine_type 0: no lidar 1: lidar_refine 2: low confidence lidar_refine
std::array<float, 4> KalmanFilterExtendForMotionless::update_and_predict(
    const std::array<float, 3> &input_observe,
    const std::array<float, 3> &observe_weight, const cv::Mat &cur_pose,
    const uint8_t lidar_refine_type) {
  // #define MOTIONLESS_DEBUG
  ++frame_count;
  if (!lidar_refined_ && lidar_refine_type != 0) {
    *Q.ptr<float>(0, 0) = 1;
    *Q.ptr<float>(1, 1) = 1;
    *Q.ptr<float>(2, 2) = 0.01;
    *Q.ptr<float>(0, 2) = 0;
    *Q.ptr<float>(1, 2) = 0;
    if (lidar_refine_type == 1) {
      lidar_refined_ = true;
    }
    lidar_refined_low_confidence_ = true;
  } else if (lidar_refined_) {
    *Q.ptr<float>(0, 0) = 0.1;
    *Q.ptr<float>(1, 1) = 0.1;
    *Q.ptr<float>(2, 2) = 0.01;
    *Q.ptr<float>(0, 2) = 0;
    *Q.ptr<float>(1, 2) = 0;
  } else if (fabs(input_observe[0]) > 50) {
    if (frame_count < 5 && !lidar_refined_low_confidence_) {
      *Q.ptr<float>(0, 0) = 1;
      *Q.ptr<float>(1, 1) = 1;
      *Q.ptr<float>(2, 2) = 0;
      *Q.ptr<float>(0, 2) = 0;
      *Q.ptr<float>(1, 2) = 0;
    } else {
      *Q.ptr<float>(0, 0) = 0.5;
      *Q.ptr<float>(1, 1) = 0.5;
      *Q.ptr<float>(2, 2) = 0.0001;
      *Q.ptr<float>(0, 2) = 0;
      *Q.ptr<float>(1, 2) = 0;
    }

  } else if (fabs(input_observe[0]) > 20) {
    *Q.ptr<float>(0, 0) = 0.3;
    *Q.ptr<float>(1, 1) = 0.3;
    *Q.ptr<float>(2, 2) = 0.0001;
    *Q.ptr<float>(0, 2) = 0;
    *Q.ptr<float>(1, 2) = 0;
  } else {
    *Q.ptr<float>(0, 0) = 0.1;
    *Q.ptr<float>(1, 1) = 0.1;
    *Q.ptr<float>(2, 2) = 0.001;
    *Q.ptr<float>(0, 2) = 0;
    *Q.ptr<float>(1, 2) = 0;
  }
#ifdef MOTIONLESS_DEBUG
  AINFO << "cur pose " << cur_pose;
#endif
  pre_predict_res = A * pre_predict_res;
#ifdef MOTIONLESS_DEBUG
  AINFO << "prdict " << pre_predict_res;
#endif
  float cur_h = *pre_predict_res.ptr<float>(2, 0);

  M = A.t() * M * A + Q;
#ifdef MOTIONLESS_DEBUG
  AINFO << "M" << M;
#endif

  cv::Mat veh2world;
  cur_pose.convertTo(veh2world, CV_32F);
  cv::Mat tmp;
  pre_predict_res.copyTo(tmp);
  tmp = tmp - veh2world(cv::Range(0, 3), cv::Range(3, 4));

  *H.ptr<float>(0, 0) = *veh2world.ptr<float>(0, 0);
  *H.ptr<float>(0, 1) = *veh2world.ptr<float>(1, 0);
  *H.ptr<float>(1, 0) = *veh2world.ptr<float>(0, 1);
  *H.ptr<float>(1, 1) = *veh2world.ptr<float>(1, 1);

  cv::Mat world2sensor = (veh2world * sensor2veh_).inv();
  cv::Mat veh2sensor = sensor2veh_.inv();

  float H_down = *world2sensor.ptr<float>(2, 0) * *tmp.ptr<float>(0, 0) +
                 *world2sensor.ptr<float>(2, 1) * *tmp.ptr<float>(1, 0) +
                 *veh2sensor.ptr<float>(2, 3);

#ifdef MOTIONLESS_DEBUG
  float H_down_with_z =
      *world2sensor.ptr<float>(2, 0) * *tmp.ptr<float>(0, 0) +
      *world2sensor.ptr<float>(2, 1) * *tmp.ptr<float>(1, 0) +
      *world2sensor.ptr<float>(2, 2) * (5.18531 - *veh2world.ptr<float>(2, 3)) +
      *veh2sensor.ptr<float>(2, 3);
  AINFO << "H F " << cur_h << " " << f_ref_;
  AINFO << "H_DOWN " << H_down << " hdown2 " << H_down_with_z;
  AINFO << "estimate tall " << cur_h * f_ref_ / H_down << " estimate tall 2 "
        << cur_h * f_ref_ / H_down_with_z;
#endif

  *H.ptr<float>(2, 0) =
      -(f_ref_ * cur_h * *world2sensor.ptr<float>(2, 0) / (H_down * H_down));
  *H.ptr<float>(2, 1) =
      -(f_ref_ * cur_h * *world2sensor.ptr<float>(2, 1) / (H_down * H_down));
  *H.ptr<float>(2, 2) = f_ref_ / H_down;
#ifdef MOTIONLESS_DEBUG
  AINFO << "H" << H;
#endif

  *C.ptr<float>(0, 0) = fabs(observe_weight[0]);
  *C.ptr<float>(1, 1) = fabs(observe_weight[1]);
  *C.ptr<float>(2, 2) = fabs(observe_weight[2]);
#ifdef MOTIONLESS_DEBUG
  AINFO << "C" << C;
#endif
  K = H * M * H.t() + C;
  K = M * H.t() * K.inv();
#ifdef MOTIONLESS_DEBUG
  AINFO << "K" << K;
#endif
  *observe_mat.ptr<float>(0, 0) =
      input_observe[0] - *veh2world.ptr<float>(0, 0) * *tmp.ptr<float>(0, 0) -
      *veh2world.ptr<float>(1, 0) * *tmp.ptr<float>(1, 0);
  *observe_mat.ptr<float>(1, 0) =
      input_observe[1] - *veh2world.ptr<float>(0, 1) * *tmp.ptr<float>(0, 0) -
      *veh2world.ptr<float>(1, 1) * *tmp.ptr<float>(1, 0);
  *observe_mat.ptr<float>(2, 0) = input_observe[2] - cur_h * f_ref_ / H_down;
#ifdef MOTIONLESS_DEBUG
  AINFO << input_observe[0] << " " << input_observe[1] << " "
        << input_observe[2];
  AINFO << "input " << observe_mat;
#endif

  pre_predict_res = pre_predict_res + K * observe_mat;
#ifdef MOTIONLESS_DEBUG
  AINFO << "prdict2 " << pre_predict_res;
#endif
  cv::Mat tmp_m = cv::Mat::eye(M.rows, M.cols, CV_32F);
  M = (tmp_m - K * H) * M;

  pre_predict_res.copyTo(tmp);
  tmp = tmp - veh2world(cv::Range(0, 3), cv::Range(3, 4));
  cv::Mat tmp_R;
  veh2world.copyTo(tmp_R);
  tmp_R = tmp_R(cv::Range(0, 2), cv::Range(0, 3));
  *tmp_R.ptr<float>(0, 2) = 0;
  *tmp_R.ptr<float>(1, 2) = 0;
  std::swap(*tmp_R.ptr<float>(0, 1), *tmp_R.ptr<float>(1, 0));
  tmp = tmp_R * tmp;

  return {*pre_predict_res.ptr<float>(0, 0), *pre_predict_res.ptr<float>(1, 0),
          *pre_predict_res.ptr<float>(2, 0), *tmp.ptr<float>(0, 0)};
}
