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

#include "robot/perception/uwb_postprocess/image_track/base_track.h"

namespace robot {
namespace perception {
BaseTracker::~BaseTracker() = default;
void BaseTracker::Core() {
  while (run_flag_.load()) {
    DoTask();
  }
}
void BaseTracker::DoTask() {
  auto any = msg_trigger_.WaitTriggerMsg();
  auto any_ptr = any.AnyCast<const std::shared_ptr<perception::Image>>();
  if (any_ptr == nullptr) {
    return;
  }
  auto msg = *(any_ptr);
  Perception(msg);
}
void BaseTracker::ProcessYUV422Image(
    const std::shared_ptr<perception::Image>& msg_ptr) {
  //   if (msg_ptr->encoding != "yuyv422") {
  //     AWARN << " only support yuyv422 encoding image! ";
  //     return;
  //   }
  if (busy_for_init_) {
    AWARN << " Is busy for init, return! ";
    return;
  } else {
    msg_trigger_.TriggerMsgArrive(rally::Any(msg_ptr));
  }
}
void BaseTracker::StartTrigger() {
  if (run_flag_.exchange(true)) {
    AINFO << "The task is running in " << __PRETTY_FUNCTION__;
    return;
  }
  if (do_task_thread_ptr_ == nullptr) {
    do_task_thread_ptr_.reset(new std::thread(&BaseTracker::Core, this));
  }
  msg_trigger_.Start();
}
}  // namespace perception
}  // namespace robot