/*
// Copyright (C) 2018-2021 Intel Corporation
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

#pragma once

#include <string>

#include <gst/gst.h>
#include <glib-object.h>
#include <gst/app/gstappsink.h>
#include <gst/allocators/allocators.h>
#include <gst/gststructure.h>
#include <gst/gstquery.h>
#include <gst/video/video.h>
#include "vaapi_context.h"

#include "vaapi_images.h"
#include "utils/performance_metrics.hpp"

class GstVaApiDecoder
{
public:
    GstVaApiDecoder();
    ~GstVaApiDecoder();

public:
    void open(const std::string& filename, bool sync = false);
    void play();
    bool read(std::shared_ptr<VaApiImage>& image);
    void close();
    double getFPS(){ return fps;}
    PerformanceMetrics getMetrics() { return readerMetrics;}

private:
    std::shared_ptr<VaApiImage>  CreateImage(GstSample* sampleRead, GstMapFlags map_flags);
    std::unique_ptr<VaApiImage> bufferToImage(GstBuffer *buffer);

    VaApiContext::Ptr vaContext;
    std::string filename_;

    GstElement* pipeline_;
    GstElement* file_source_;
    GstElement* demux_;
    GstElement* parser_;
    GstElement* dec_;
    GstElement* capsfilter_;
    GstElement* queue_;
    GstElement* app_sink_;

    GstVideoInfo* video_info_;
    double fps;
    PerformanceMetrics readerMetrics;
};
