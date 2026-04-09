/******************************************************************************
 * Copyright 2017 RoboSense All rights reserved.
 * Suteng Innovation Technology Co., Ltd. www.robot.ai

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

#ifndef ROBOT_PERCEPTION_COMMON_UTILS_MSG_TRIGGER_H
#define ROBOT_PERCEPTION_COMMON_UTILS_MSG_TRIGGER_H

#include <condition_variable>
#include <stack>

#include <boost/shared_ptr.hpp>

#include "rally/utils/any.h"
#include "rally/utils/details/macros.h"
#include "rally/utils/logger/log.h"

namespace robot {
namespace perception {

class MsgTrigger {
 public:
  MsgTrigger() {}
  ~MsgTrigger() {}

  void Start() {
    if (run_flag_.exchange(true)) {
      AINFO << "MsgTrigger is running";
    }

    return;
  }

  void Stop() {
    while (!msgs_stack_.empty()) {
      msgs_stack_.pop();
    }
    if (!run_flag_.exchange(false)) {
      AINFO << "MsgTrigger is stopped";
      return;
    }

    stack_full_cv_.notify_all();
    return;
  }

  void TriggerMsgArrive(const rally::Any& msg) {
    // stop
    if (!run_flag_.load()) {
      return;
    }

    std::unique_lock<std::mutex> lck(stack_mutex_);
    // protect
    if (msgs_stack_.size() > 3) {
      msgs_stack_.pop();
    }

    msgs_stack_.push(msg);
    stack_full_cv_.notify_all();
    return;
  }

  const rally::Any WaitTriggerMsg() {
    // stop
    if (!run_flag_.load()) {
      return rally::Any();
    }
    std::unique_lock<std::mutex> lck(stack_mutex_);
    if(msgs_stack_.empty()){
      stack_full_cv_.wait(lck);
    }

    if (msgs_stack_.empty() or !run_flag_.load()) {
      return rally::Any();
    }

    auto data = msgs_stack_.top();
    while (!msgs_stack_.empty()) {
      msgs_stack_.pop();
    }
    return data;
  }

 private:
  RALLY_DISALLOW_COPY_AND_ASSIGN(MsgTrigger)
  std::atomic_bool run_flag_ = {false};
  std::mutex stack_mutex_;
  std::condition_variable stack_full_cv_;
  std::stack<rally::Any> msgs_stack_;
};

}  // namespace perception
}  // namespace robot

#endif  // ROBOT_PERCEPTION_COMMON_UTILS_MSG_TRIGGER_H
