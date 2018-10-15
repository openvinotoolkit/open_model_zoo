/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "logger.hpp"
#include <string>

DetectionsLogger::DetectionsLogger(std::ostream& stream, bool enabled) : log_stream_(stream) {
    write_logs_ = enabled;
}

void DetectionsLogger::CreateNextFrameRecord(const std::string& path, const int frame_idx,
                                             const size_t width, const size_t height) {
    if (write_logs_)
        log_stream_ << "Frame_name: " << path << "@" << frame_idx << " width: "
                    << width << " height: " << height << std::endl;
}

void DetectionsLogger::AddFaceToFrame(const cv::Rect& rect, const std::string& id) {
    if (write_logs_)
        log_stream_ << "Object type: face. Box: " << rect << " id: " << id << std::endl;
}

void DetectionsLogger::AddPersonToFrame(const cv::Rect& rect, const std::string& action) {
    if (write_logs_)
        log_stream_ << "Object type: person. Box: " << rect << " action: " << action << std::endl;
}

void DetectionsLogger::FinalizeFrameRecord() {
    if (write_logs_)
        log_stream_ << std::endl;
}

DetectionsLogger::~DetectionsLogger() {}
