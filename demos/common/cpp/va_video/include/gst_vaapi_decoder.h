#pragma once

#include <string>

#include <gst/gst.h>
#include <glib-object.h>
#include <gst/app/gstappsink.h>
#include <gst/allocators/allocators.h>
#include <gst/gststructure.h>
#include <gst/gstquery.h>
#include <gst/video/video.h>

#include "vaapi_images.h"

using namespace InferenceBackend;

namespace pz
{

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

private:
    std::shared_ptr<InferenceBackend::VaApiImage>  CreateImage(GstSample* sampleRead, GstMapFlags map_flags);
    std::unique_ptr<VaApiImage> bufferToImage(GstBuffer *buffer);

    VaApiContext::Ptr vaContext;;
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
};

}
