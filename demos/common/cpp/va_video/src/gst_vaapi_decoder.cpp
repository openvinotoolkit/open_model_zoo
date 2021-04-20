#include "gst_vaapi_decoder.h"

#include <cassert>
#include <iostream>
#include <sstream>

#if not defined(UNUSED)
#define UNUSED(x) (void)(x)
#endif

namespace pz
{

using GstMemoryUniquePtr = std::unique_ptr<GstMemory, decltype(&gst_memory_unref)>;

static void on_demux_new_pad (GstElement *element, GstPad *pad, gpointer data)
{
    GstPad* parserSinkPad;
    GstElement *parser = (GstElement*)data;

    std::cout << "Going to link Demux and Parser" << std::endl;

    parserSinkPad = gst_element_get_static_pad(parser, "sink");
    gst_pad_link(pad, parserSinkPad);

    gst_object_unref(parserSinkPad);
}

inline int gstFormatToFourCC(int format) {
    switch (format) {
    case GST_VIDEO_FORMAT_NV12:
        GST_DEBUG("GST_VIDEO_FORMAT_NV12");
        return FourCC::FOURCC_NV12;
    case GST_VIDEO_FORMAT_BGR:
        GST_DEBUG("GST_VIDEO_FORMAT_BGR");
        return FourCC::FOURCC_BGR;
    case GST_VIDEO_FORMAT_BGRx:
        GST_DEBUG("GST_VIDEO_FORMAT_BGRx");
        return FourCC::FOURCC_BGRX;
    case GST_VIDEO_FORMAT_BGRA:
        GST_DEBUG("GST_VIDEO_FORMAT_BGRA");
        return FourCC::FOURCC_BGRA;
    case GST_VIDEO_FORMAT_RGBA:
        GST_DEBUG("GST_VIDEO_FORMAT_RGBA");
        return FourCC::FOURCC_RGBA;
    case GST_VIDEO_FORMAT_I420:
        GST_DEBUG("GST_VIDEO_FORMAT_I420");
        return FourCC::FOURCC_I420;
    }

    GST_WARNING("Unsupported GST Format: %d.", format);
    return 0;
}

std::unique_ptr<VaApiImage> GstVaApiDecoder::bufferToImage(GstBuffer *buffer)
{
    try
    {
        if (!video_info_)
            throw std::invalid_argument("GstVideoInfo is absent during GstBuffer mapping");

        auto bufDisplay = gst_mini_object_get_qdata(&buffer->mini_object, g_quark_from_static_string("VADisplay"));
        if(!vaContext || vaContext->display()!= bufDisplay)
        {
            vaContext = std::make_shared<VaApiContext>(bufDisplay);
        }

        auto image = std::unique_ptr<VaApiImage>(new VaApiImage(
            vaContext, // TODO: check this for consistancy
            static_cast<uint32_t>(GST_VIDEO_INFO_WIDTH(video_info_)),
            static_cast<uint32_t>(GST_VIDEO_INFO_HEIGHT(video_info_)),
            static_cast<FourCC>(gstFormatToFourCC(GST_VIDEO_INFO_FORMAT(video_info_))),
            reinterpret_cast<uint64_t>(gst_mini_object_get_qdata(&buffer->mini_object, g_quark_from_static_string("VASurfaceID"))),
            false
            ));

//!!!            std::cout<<"GS img:"<<reinterpret_cast<uint64_t>(gst_mini_object_get_qdata(&buffer->mini_object, g_quark_from_static_string("VASurfaceID")))<<std::endl;

        // getting data from VA_API
        if (!image->context->display()) {
            std::ostringstream os;
            os << "Failed to get VADisplay=" << image->context->display();
            throw std::runtime_error(os.str());
        }
        if ((int)image->va_surface_id < 0) {
            std::ostringstream os;
            os << "Failed to get VASurfaceID=" << image->va_surface_id;
            throw std::runtime_error(os.str());
        }

        return image;

    } catch (const std::exception &e) {
        std::throw_with_nested(std::runtime_error("Failed to make VaApiImage from GstBuffer"));
    }
}

std::shared_ptr<VaApiImage> GstVaApiDecoder::CreateImage(GstSample* sample, GstMapFlags map_flags)
{
    try
    {
        GstCaps * frame_caps = gst_sample_get_caps(sample);

        if (video_info_ == nullptr)
        {
            video_info_  = gst_video_info_new();
            gst_video_info_init(video_info_);
            gst_video_info_from_caps(video_info_, frame_caps);
            fps = ((double)GST_VIDEO_INFO_FPS_N(video_info_)) / GST_VIDEO_INFO_FPS_D(video_info_);
        }

        GstBuffer* buffer = gst_sample_get_buffer(sample);

        auto unique_image =  bufferToImage(buffer);

        auto image_deleter = [sample, buffer](InferenceBackend::VaApiImage *image)
        {
            gst_sample_unref(sample);
            delete image;
        };

        return std::shared_ptr<InferenceBackend::VaApiImage>(unique_image.release(), image_deleter);
    }
    catch (const std::exception &e)
    {
        std::throw_with_nested(std::runtime_error("Failed to create image from GstBuffer"));
    }
}


GstVaApiDecoder::GstVaApiDecoder()
    : pipeline_(nullptr), file_source_(nullptr), demux_(nullptr)
    , parser_(nullptr), dec_(nullptr), capsfilter_(nullptr)
    , queue_(nullptr), app_sink_(nullptr), video_info_(nullptr), fps(0)
{

}

GstVaApiDecoder::~GstVaApiDecoder()
{
    close();
}

void GstVaApiDecoder::open(const std::string &filename, bool sync)
{
    filename_ = filename;

    gst_init(0, NULL);

    GstElement* pipeline = gst_pipeline_new("pipeline");
    GstElement* file_source = gst_element_factory_make("filesrc", "file_source");
    GstElement* demux = gst_element_factory_make("qtdemux", "demux");
    GstElement* parser = gst_element_factory_make("h264parse", "parser");
    GstElement* dec = gst_element_factory_make("vaapih264dec", "dec");
    GstElement* capsfilter = gst_element_factory_make("capsfilter", "capsfilter");
    GstElement* queue = gst_element_factory_make("queue", "queue");
    GstElement* app_sink = gst_element_factory_make("appsink", "appsink");

    if(!pipeline || !file_source || !demux || !parser || !dec || !capsfilter || !app_sink || !queue)
    {
        throw std::runtime_error("Fail to make one of the gstreamer plugins!");
    }

    GstCaps* caps = gst_caps_from_string("video/x-raw(memory:VASurface), format=(string)NV12");
    if (!caps)
    {
        throw std::runtime_error("Fail to make gst caps");
    }

    g_object_set(capsfilter, "caps", caps, NULL);
    gst_caps_unref (caps);

    g_object_set(file_source, "location", filename.c_str(), NULL);
    g_object_set(app_sink, "sync", sync, NULL);

    gst_bin_add_many(GST_BIN(pipeline), file_source, demux, parser, dec, capsfilter, queue, app_sink, NULL);

    if(!gst_element_link_many(file_source, demux, NULL))
    {
        throw std::runtime_error("Fail to link file src to demux");
    }
    if(!gst_element_link_many(parser, dec, capsfilter, queue, app_sink, NULL))
    {
        throw std::runtime_error("Fail to link gst plugins");
    }

    g_signal_connect(demux, "pad-added", G_CALLBACK(on_demux_new_pad), parser);

    pipeline_    = pipeline;
    file_source_ = file_source;
    demux_       = demux;
    parser_      = parser;
    dec_         = dec;
    capsfilter_  = capsfilter;
    queue_       = queue;
    app_sink_    = app_sink;
}

void GstVaApiDecoder::play()
{
    if (pipeline_)
    {
        /// Start!!!
        gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    }
}

bool GstVaApiDecoder::read(std::shared_ptr<VaApiImage>& src_image)
{
    auto startTime = std::chrono::steady_clock::now();
    if (gst_app_sink_is_eos(GST_APP_SINK(app_sink_)))
    {
        std::cout << "EOS received" << std::endl;
        return false;
    }

    GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(app_sink_));
    if(!sample)
    {
        std::cerr << "Fail to read gstSample from appsink!" << std::endl;
        return false;
    }

    src_image = CreateImage(sample, GstMapFlags(GST_MAP_READ | GST_VIDEO_FRAME_MAP_FLAG_NO_REF));
    readerMetrics.update(startTime);
    return true;
}

void GstVaApiDecoder::close()
{
    if (video_info_ != nullptr)
    {
        gst_video_info_free(video_info_);
        video_info_ = nullptr;
    }

    if (pipeline_ != nullptr)
    {
        gst_object_unref (pipeline_);
        pipeline_ = nullptr;
    }
}

}
