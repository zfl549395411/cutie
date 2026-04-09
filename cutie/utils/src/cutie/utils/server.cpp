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

#include "robot/perception/uwb_postprocess/common/server.h"
namespace robot {
namespace perception {
void UWBServer::InputCommand(const perception::CommonCommand& command) {
  rally::Time timer(command.header.time);
  input_data.time_stamp = timer.toSecond();
  if (command.follow_info.height > 10) {
    input_data.image_height = command.follow_info.height;
    input_data.image_width = command.follow_info.width;
    input_data.undistort = command.follow_info.remove_distortion;
    input_data.rect = command.follow_info.rect;
    input_data.sensor_id = command.follow_info.sensor_id;
    AINFO << " GET FOLLOW INFO " << input_data.image_height << " " << " "
          << input_data.image_width << " " << input_data.rect << " sensor id "
          << kSensorIDToNameMap.at(input_data.sensor_id);
  }
  // 发出开关指令 stoptrack会在完成停止后变为notrack
  if (input_data.command == COMMOND::NO_TRACK && command.enable) {
    input_data.command = COMMOND::START_TRACK;
    AINFO << " START TRACK INFO " << input_data.image_height << " "
          << input_data.image_width;
  } else if (input_data.command == COMMOND::START_TRACK && !command.enable) {
    input_data.command = COMMOND::STOP_TRACK;
    AINFO << " STOP TRACK";
  }
}
FollowResponse UWBServer::OutputFollowResponse() {
  FollowResponse follow_response;
  rally::Time timer(output_data.time_stamp);
  follow_response.header.time = timer.toNanosecond();
  follow_response.follow_info.width = output_data.image_width;
  follow_response.follow_info.height = output_data.image_height;
  follow_response.follow_info.rect = output_data.rect;
  follow_response.follow_info.remove_distortion = input_data.undistort;
  follow_response.follow_info.track_state =
      kTrackStateToMessageTrackState.at(output_data.track_state);
  follow_response.follow_info.dist_base = output_data.dist_base;
  follow_response.follow_info.angle_base = output_data.angle_base;
  follow_response.follow_info.sensor_id = output_data.sensor_id;
  return follow_response;
}
}  // namespace perception
}  // namespace robot