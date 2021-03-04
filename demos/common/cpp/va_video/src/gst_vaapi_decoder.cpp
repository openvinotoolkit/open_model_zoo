#include "gst_vaapi_decoder.h"

#include <cassert>
#include <iostream>
#include <sstream>

#define UNUSED(x) (void)(x)

namespace pz
{

using GstMemoryUniquePtr = std::unique_ptr<GstMemory, decltype(&gst_memory_unref)>;

struct BufferMapContext {
    GstVideoFrame frame;
};

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

void gva_buffer_map(GstBuffer *buffer, Image &image, BufferMapContext &map_context, GstVideoInfo *info,
                    MemoryType memory_type, GstMapFlags map_flags, unsigned int vpu_device_id)
{

    try
    {
        if (not info)
            throw std::invalid_argument("GstVideoInfo is absent during GstBuffer mapping");

        image = Image();
        map_context = BufferMapContext();
        map_context.frame.buffer = nullptr;
        guint n_planes = GST_VIDEO_INFO_N_PLANES(info);
        if (n_planes == 0 or n_planes > Image::MAX_PLANES_NUMBER)
            throw std::logic_error("Image planes number " + std::to_string(n_planes) + " isn't supported");

        image.format = gstFormatToFourCC(GST_VIDEO_INFO_FORMAT(info));
        image.width = static_cast<uint32_t>(GST_VIDEO_INFO_WIDTH(info));
        image.height = static_cast<uint32_t>(GST_VIDEO_INFO_HEIGHT(info));
        image.size = GST_VIDEO_INFO_SIZE(info);
        image.type = memory_type;
        for (guint i = 0; i < n_planes; ++i)
        {
            image.stride[i] = GST_VIDEO_INFO_PLANE_STRIDE(info, i);
            image.offsets[i] = GST_VIDEO_INFO_PLANE_OFFSET(info, i);
        }

        switch (memory_type)
        {
        case MemoryType::SYSTEM:
        {
            if (not gst_video_frame_map(&map_context.frame, info, buffer, map_flags))
            {
                throw std::runtime_error("Failed to map GstBuffer to system memory");
            }
            for (guint i = 0; i < n_planes; ++i)
            {
                image.planes[i] = static_cast<uint8_t *>(GST_VIDEO_FRAME_PLANE_DATA(&map_context.frame, i));
            }
            for (guint i = 0; i < n_planes; ++i)
            {
                image.stride[i] = GST_VIDEO_FRAME_PLANE_STRIDE(&map_context.frame, i);
            }

            UNUSED(vpu_device_id);

            break;
        }
        case MemoryType::DMA_BUFFER:
        {
            GstMemory *mem = gst_buffer_peek_memory(buffer, 0);
            if (not mem)
                throw std::runtime_error("Failed to get GstBuffer memory");
            image.dma_fd = gst_dmabuf_memory_get_fd(mem);
            if (image.dma_fd < 0)
                throw std::runtime_error("Failed to import DMA buffer FD");
            break;
        }
        case MemoryType::VAAPI:
        {
            image.va_display = gst_mini_object_get_qdata(&buffer->mini_object, g_quark_from_static_string("VADisplay"));
            image.va_surface_id =
                (uint64_t)gst_mini_object_get_qdata(&buffer->mini_object, g_quark_from_static_string("VASurfaceID"));
            if (not image.va_display) {
                std::ostringstream os;
                os << "Failed to get VADisplay=" << image.va_display;
                throw std::runtime_error(os.str());
            }
            if ((int)image.va_surface_id < 0) {
                std::ostringstream os;
                os << "Failed to get VASurfaceID=" << image.va_surface_id;
                throw std::runtime_error(os.str());
            }
            break;
        }
        default:
            throw std::logic_error("Unsupported destination memory type");
        }
    } catch (const std::exception &e) {
        image = Image();
        map_context.frame.buffer = nullptr;
        std::throw_with_nested(std::runtime_error("Failed to map GstBuffer to specific memory type"));
    }
}

void gva_buffer_unmap(GstBuffer *buffer, Image &, BufferMapContext &map_context, unsigned int vpu_device_id)
{
    if (map_context.frame.buffer)
    {
        UNUSED(buffer);
        UNUSED(vpu_device_id);
        gst_video_frame_unmap(&map_context.frame);
    }
}

std::shared_ptr<Image> GstVaApiDecoder::CreateImage(GstSample* sample, MemoryType mem_type, GstMapFlags map_flags)
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

        std::unique_ptr<InferenceBackend::Image> unique_image = std::unique_ptr<InferenceBackend::Image>(new Image);
        assert(unique_image.get() != nullptr);

        std::shared_ptr<BufferMapContext> map_context = std::make_shared<BufferMapContext>();
        assert(map_context.get() != nullptr);

        gva_buffer_map(buffer, *unique_image, *map_context, video_info_, mem_type, map_flags, -1);

        auto image_deleter = [sample, buffer, map_context](InferenceBackend::Image *image)
        {
            gva_buffer_unmap(buffer, *image, *map_context, -1);
            gst_sample_unref(sample);
            delete image;
        };

        // only support full frame
        unique_image->rect.x      = 0;
        unique_image->rect.y      = 0;
        unique_image->rect.width  = unique_image->width;
        unique_image->rect.height = unique_image->height;

        return std::shared_ptr<InferenceBackend::Image>(unique_image.release(), image_deleter);
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

bool GstVaApiDecoder::read(std::shared_ptr<Image>& src_image)
{
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

    src_image = CreateImage(sample, MemoryType::VAAPI, GstMapFlags(GST_MAP_READ | GST_VIDEO_FRAME_MAP_FLAG_NO_REF));

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
