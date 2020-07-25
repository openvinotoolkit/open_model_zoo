// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <numeric>
#include <queue>
#include <string>
#include <utility>

#include <samples/args_helper.hpp>
#include <samples/images_capture.h>

#include "perf_timer.hpp"

#include "decoder.hpp"
#include "threading.hpp"

#ifdef USE_NATIVE_CAMERA_API
#include "multicam/camera.hpp"
#include "multicam/utils.hpp"
#endif

#ifdef USE_TBB
#include <tbb/concurrent_queue.h>
#endif

#ifdef USE_LIBVA
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

class VideoSource {
public:
    virtual bool isRunning() const = 0;

    virtual void start() = 0;

    virtual bool read(VideoFrame& frame) = 0;

    virtual float getAvgReadTime() const = 0;

    virtual ~VideoSource();
};

VideoSource::~VideoSource() {}

#ifdef USE_LIBVA

struct VideoStream {
    struct frame_t {
        void* ptr;
        size_t offset;
        size_t length;
        int width;
        int height;
    } frame;

    std::unique_ptr<void, std::function<void(void*)>> ptr;
    size_t length;

    size_t frame_offset;
    size_t next_frame_offset;

    mcam::file_descriptor fd;

    using stream_t = unsigned char;

    explicit VideoStream(const std::string& filepath)
        : ptr(0, [](void*){}), fd(open(filepath.c_str(), O_RDONLY)) {
        struct stat sb;
        if (!fd.valid())
            throw std::runtime_error(std::string("Cannot open input file: ") + std::string(strerror(errno)));
        if (fstat(fd.get(), &sb))
            throw std::runtime_error(std::string("Cannot stat input file: ") + std::string(strerror(errno)));
        length = sb.st_size;
        void* p = mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd.get(), 0);
        if (MAP_FAILED == p)
            throw std::runtime_error(std::string("Cannot map input file: ") + std::string(strerror(errno)));

        auto l = sb.st_size;
        ptr = std::unique_ptr<void, std::function<void(void*)>>(p, [l](void* _p) { munmap(_p, l); });

        advance_frame();
    }

    VideoStream (const VideoStream&) = delete;
    VideoStream (VideoStream&&) = delete;

    void update_frame_dims() {
        auto p = static_cast<stream_t*>(frame.ptr);
        size_t header_offset = 0;
        bool is_marker_found = false;
        while (header_offset < frame.length && !is_marker_found) {
            if (p[header_offset] == static_cast<stream_t>(0xFF) &&
                p[header_offset + 1] == static_cast<stream_t>(0xC0))
                is_marker_found = true;
            else
                header_offset++;
        }
        if (is_marker_found) {
            size_t offset = header_offset;
            offset += 2;  // skip header marker
            offset += 2;  // skip header size
            offset += 1;  // skip precision
            frame.height = p[offset + 0] * 256 + p[offset + 1];
            frame.width = p[offset + 2] * 256 + p[offset + 3];
        } else {
            assert(false);
        }
    }

    void advance_frame() {
        // Loop
        if (next_frame_offset >= length) {
            next_frame_offset = 0;
        }

        frame_offset = next_frame_offset;
        auto p = static_cast<stream_t*>(ptr.get());
        while (next_frame_offset < length) {
            if (p[next_frame_offset] == static_cast<stream_t>(0xFF) &&
                p[next_frame_offset + 1] == static_cast<stream_t>(0xD9) )
                break;
            else
                next_frame_offset++;
        }
        next_frame_offset += 2;
        frame.length = next_frame_offset - frame_offset;
        frame.ptr = &(p[frame_offset]);
        update_frame_dims();
    }
};

class VideoSourceStreamFile : public VideoSource {
    using queue_elem_t = std::pair<bool, cv::Mat>;
    using queue_t = std::queue<queue_elem_t>;

    VideoSources& parent;

    VideoStream stream;

    std::atomic_bool running = {false};
    std::atomic_bool is_decoding = {false};

    std::mutex mutex;
    std::thread workThread;
    std::condition_variable condVar;
    std::condition_variable hasFrame;

    queue_t frameQueue;
    const std::size_t queueSize;

    using clock = std::chrono::high_resolution_clock;
    clock::time_point lastFrameTime;
    PerfTimer perfTimer;

public:
    VideoSourceStreamFile(VideoSources& p,
                          bool async,
                          bool collectStats_,
                          const std::string& name,
                          size_t queueSize_,
                          size_t pollingTimeMSec_,
                          bool realFps_):
        parent(p),
        stream(name),
        queueSize(queueSize_),
        perfTimer(collectStats_ ? PerfTimer::DefaultIterationsCount : 0) { }

    bool isRunning() const override {
        return running;
    }

    void start() {
        running = true;
        workThread = std::thread([&]() {
            while (running) {
                {
                    cv::Mat frame;
                    {
                        is_decoding = true;
                        std::unique_lock<std::mutex> lock(parent.decode_mutex);

                        parent.decoder.decode(stream.frame.ptr, stream.frame.length, stream.frame.width, stream.frame.height,
                            [this](cv::Mat&& img) mutable {
                            bool success = !img.empty();
                            frameQueue.push({success, std::move(img)});
                            if (perfTimer.enabled()) {
                                auto prev = lastFrameTime;
                                auto current = clock::now();
                                using dur = decltype (prev.time_since_epoch());
                                if (dur::zero() != prev.time_since_epoch()) {
                                    perfTimer.addValue(current - prev);
                                }
                                lastFrameTime = current;
                            }
                            is_decoding = false;
                            condVar.notify_one();
                        });
                        stream.advance_frame();
                    }

                    std::unique_lock<std::mutex> lock(mutex);
                    condVar.wait(lock, [&]() {
                        return !is_decoding && (frameQueue.size() < queueSize || !running);
                    });
                }
                hasFrame.notify_one();
            }
        });
    }

    void stop() {
        running = false;
        condVar.notify_one();
        if (workThread.joinable()) {
            workThread.join();
        }
    }

    bool read(VideoFrame& frame)  {
        queue_elem_t elem;

        if (!running)
            return false;

        {
            std::unique_lock<std::mutex> lock(mutex);
            hasFrame.wait(lock, [&]() {
                return !frameQueue.empty() || !running;
            });
            elem = std::move(frameQueue.front());
            frameQueue.pop();
        }
        condVar.notify_one();
        frame.frame = std::move(elem.second);

        return elem.first && running;
    }

    float getAvgReadTime() const {
        return perfTimer.getValue();
    }
};

#endif

class GeneralCaptureSource : public VideoSource {
    PerfTimer perfTimer;
    std::thread workThread;
    const bool isAsync;
    std::atomic_bool running = {true};

    std::mutex mutex;
    std::condition_variable condVar;
    std::condition_variable hasFrame;
    std::queue<std::pair<bool, cv::Mat>> queue;

    std::unique_ptr<ImagesCapture> cap;

    bool realFps;

    const size_t queueSize;
    const size_t pollingTimeMSec;

    template<bool CollectStats>
    cv::Mat readFrame();

    template<bool CollectStats>
    void startImpl();

public:
    GeneralCaptureSource(bool async, bool collectStats_, const std::string& name, bool loopVideo,
                size_t queueSize_, size_t pollingTimeMSec_, bool realFps_);

    ~GeneralCaptureSource();

    void start();

    bool isRunning() const override;

    void stop();

    bool read(cv::Mat& frame);
    bool read(VideoFrame& frame);

    float getAvgReadTime() const {
        return perfTimer.getValue();
    }

private:
    template<bool CollectStats>
    static void thread_fn(GeneralCaptureSource*);
};

#ifdef USE_NATIVE_CAMERA_API
class VideoSourceNative : public VideoSource {
    VideoSources& parent;
    using queue_elem_t = std::pair<bool, cv::Mat>;
#ifdef USE_TBB
    using queue_t = tbb::concurrent_bounded_queue<queue_elem_t>;
#else
    using queue_t = std::queue<queue_elem_t>;
#endif
    const int queueSize = 0;
    const bool realFps = false;
    cv::Mat dummyFrame;
    std::size_t frameIdx = 0;
    queue_t frameQueue;
    mcam::camera camera;
    PerfTimer perfTimer;

    using clock = std::chrono::high_resolution_clock;
    clock::time_point lastFrameTime;

    void frameHandler(mcam::camera::frame_status status,
                      const mcam::camera::settings& settings,
                      mcam::camera::frame frame);

public:
    VideoSourceNative(VideoSources& p, mcam::controller& ctrl,
           const std::string& source, const mcam::camera::settings& settings,
           size_t queueSize, bool realFps, bool collectStats);

    ~VideoSourceNative();

    void start();

    bool isRunning() const override;

    bool read(VideoFrame& frame);

    float getAvgReadTime() const {
        return perfTimer.getValue();
    }
};


VideoSourceNative::VideoSourceNative(VideoSources& p, mcam::controller& ctrl,
       const std::string& source, const mcam::camera::settings& settings,
       size_t queueSize, bool realFps, bool collectStats):
    parent(p),
    queueSize(static_cast<int>(queueSize)),
    realFps(realFps),
    camera(ctrl, source, [this](
           mcam::camera::frame_status status,
           const mcam::camera::settings& settings,
           mcam::camera::frame frame) {
    frameHandler(status, settings, std::move(frame));
}, settings),
    perfTimer(collectStats ? PerfTimer::DefaultIterationsCount : 0) {
}

VideoSourceNative::~VideoSourceNative() {
    // nothing
}

void VideoSourceNative::start() {
    // nothing
}

bool VideoSourceNative::isRunning() const override {
    return true;
}

void VideoSourceNative::frameHandler(mcam::camera::frame_status status,
                  const mcam::camera::settings& settings,
                  mcam::camera::frame frame) {
    if (status == mcam::camera::frame_status::ok) {
        if (frameQueue.size() < queueSize) {
            (void)settings;
            assert(mcam::make_4cc('M', 'J', 'P', 'G') ==
                   settings.format4cc);
            assert(frame.valid());
            auto data = frame.data();
            auto size = frame.size();

            std::unique_lock<std::mutex> lock(parent.decode_mutex);

            parent.decoder.decode(
                        data, size, settings.width, settings.height,
            [this, fr = std::move(frame)](cv::Mat&& img) mutable {
                fr = {};
                bool success = !img.empty();
                frameQueue.push({success, std::move(img)});
                if (perfTimer.enabled()) {
                    auto prev = lastFrameTime;
                    auto current = clock::now();
                    using dur = decltype (prev.time_since_epoch());
                    if (dur::zero() != prev.time_since_epoch()) {
                        perfTimer.addValue(current - prev);
                    }

                    lastFrameTime = current;
                }
            });
        }
    }
}

bool VideoSourceNative::read(VideoFrame& frame) {
    queue_elem_t elem;
    if (realFps) {
#ifdef USE_TBB
        frameQueue.pop(elem);
#else
        elem = std::move(frameQueue.front());
        frameQueue.pop();
#endif
    } else {
#ifdef USE_TBB
        if (frameQueue.try_pop(elem)) {
#else
        elem = std::move(frameQueue.front());
        frameQueue.pop();
        if (elem.first) {
#endif
            if (elem.first) {
                dummyFrame = elem.second;
            }
        } else {
            elem.first = (!dummyFrame.empty());
            elem.second = dummyFrame;
        }
    }
    frame.frame = std::move(elem.second);
    return elem.first;
}
#endif  // USE_NATIVE_CAMERA_API

namespace {
bool isNumeric(const std::string& str) {
    return std::strspn(str.c_str(), "0123456789") == str.length();
}
}  // namespace

template<bool CollectStats>
cv::Mat GeneralCaptureSource::readFrame() {
    if (CollectStats) {
        ScopedTimer st(perfTimer);
        return cap->read();
    } else {
        return cap->read();
    }
}

GeneralCaptureSource::GeneralCaptureSource(bool async, bool collectStats_,
                         const std::string& name, bool loopVideo, size_t queueSize_,
                         size_t pollingTimeMSec_, bool realFps_):
    perfTimer(collectStats_ ? PerfTimer::DefaultIterationsCount : 0),
    isAsync(async),
    cap(openImagesCapture(name, loopVideo)),
    realFps(realFps_),
    queueSize(queueSize_),
    pollingTimeMSec(pollingTimeMSec_) {}

GeneralCaptureSource::~GeneralCaptureSource() {
    stop();
}

bool GeneralCaptureSource::isRunning() const {
    return running;
}

template<bool CollectStats>
void GeneralCaptureSource::thread_fn(GeneralCaptureSource *vs) {
    while (vs->running) {
        cv::Mat frame = vs->readFrame<CollectStats>();
        const bool result = frame.data;
        if (!result) {
            vs->running = false; // stop() also affects running, so override it only when out of frames
        }
        std::unique_lock<std::mutex> lock(vs->mutex);
        vs->condVar.wait(lock, [&]() {
            return vs->queue.size() < vs->queueSize || !vs->running; // queue has space or source ran out of frames
        });
        vs->queue.push({result, frame});
        vs->hasFrame.notify_one();
    }
}

template<bool CollectStats>
void GeneralCaptureSource::startImpl() {
    if (isAsync) {
        running = true;
        workThread = std::thread(&GeneralCaptureSource::thread_fn<CollectStats>, this);
    }
}

void GeneralCaptureSource::start() {
    if (perfTimer.enabled()) {
        startImpl<true>();
    } else {
        startImpl<false>();
    }
}

void GeneralCaptureSource::stop() {
    if (isAsync) {
        running = false;
        condVar.notify_one();
        if (workThread.joinable()) {
            workThread.join();
        }
    }
}

bool GeneralCaptureSource::read(cv::Mat& frame) {
    if (isAsync) {
        bool res;
        {
            std::unique_lock<std::mutex> lock(mutex);
            hasFrame.wait(lock, [&]() {
                return !queue.empty() || !running;
            });
            res = queue.front().first;
            frame = queue.front().second;
            if (realFps || queue.size() > 1 || queueSize == 1) {
                queue.pop();
            }
        }
        condVar.notify_one();
        return res;
    } else {
        frame = cap->read();
        return frame.data;
    }
}

bool GeneralCaptureSource::read(VideoFrame& frame) {
    return read(frame.frame);
}

namespace {
Decoder::Settings makeDecoderSettings(bool collectStats, std::size_t queueSize,
                                      unsigned width, unsigned height) {
    Decoder::Settings ret = {};
#if defined(USE_LIBVA)
    ret.mode = Decoder::Mode::Hw;
    ret.num_buffers = static_cast<unsigned>(queueSize);
    ret.output_width = width;
    ret.output_height = height;
#elif defined(USE_TBB)
    ret.mode = Decoder::Mode::Async;
#else
    ret.mode = Decoder::Mode::Immediate;
#endif
    ret.collect_stats = collectStats;
    return ret;
}
}  // namespace

VideoSources::VideoSources(const InitParams& p):
    decoder(makeDecoderSettings(p.collectStats, p.queueSize, p.expectedWidth,
                                p.expectedHeight)),
    isAsync(p.isAsync),
    collectStats(p.collectStats),
    realFps(p.realFps),
    queueSize(p.queueSize),
    pollingTimeMSec(p.pollingTimeMSec) {
        for (const std::string& input : split(p.inputs, ','))
            openVideo(input, isNumeric(input), p.loop);
    }

VideoSources::~VideoSources() {
    // nothing
}

bool VideoSources::isRunning() const {
    // when one of VideoSources will be out of frames, it will stop IEGraph,
    // so this isRunning() requires that all inputs were running
    return std::all_of(inputs.begin(), inputs.end(),
        [](const std::unique_ptr<VideoSource>& input){return input->isRunning();});
}

void VideoSources::openVideo(const std::string& source, bool native, bool loopVideo) {
#ifdef USE_NATIVE_CAMERA_API
    if (native) {
        std::string dev;
        if (isNumeric(source)) {
            dev = "/dev/video" + source;
        } else {
            dev = source;
        }
        mcam::camera::settings camSettings;
        camSettings.format4cc = mcam::make_4cc('M', 'J', 'P', 'G');
        camSettings.width = 640;
        camSettings.height = 480;
        camSettings.num_buffers = static_cast<unsigned>(queueSize);

        std::unique_ptr<VideoSource> newSrc(new VideoSourceNative(*this, controller, dev, camSettings,
                                                                     queueSize, realFps, collectStats));
        inputs.emplace_back(std::move(newSrc));
    } else {
#else
    {
#endif
#if defined(USE_LIBVA)
        const std::string extension = ".mjpeg";
        std::unique_ptr<VideoSource> newSrc;
        if (source.size() > extension.size() && std::equal(extension.rbegin(), extension.rend(), source.rbegin()))
            if (loopVideo)
                throw std::runtime_error("Looping video is not supported for .mjpeg when built with USE_LIBVA");
            newSrc.reset(new VideoSourceStreamFile(*this, isAsync, collectStats, source,
                                            queueSize, pollingTimeMSec, realFps));
        else
            newSrc.reset(new GeneralCaptureSource(isAsync, collectStats, source, loopVideo,
                                            queueSize, pollingTimeMSec, realFps));
#else
        std::unique_ptr<VideoSource> newSrc(new GeneralCaptureSource(isAsync, collectStats, source, loopVideo,
                                            queueSize, pollingTimeMSec, realFps));
#endif
        inputs.emplace_back(std::move(newSrc));
    }
}

void VideoSources::start() {
    for (auto& input : inputs) {
        input->start();
    }
}

bool VideoSources::getFrame(size_t index, VideoFrame& frame) {
    if (inputs.size() > 0) {
        if (index < inputs.size()) {
            return inputs[index]->read(frame);
        }
    }
    return false;
}

VideoSources::Stats VideoSources::getStats() const {
    Stats ret;
    if (collectStats) {
        ret.readTimes.reserve(inputs.size());
        for (auto& input : inputs) {
            ret.readTimes.push_back(input->getAvgReadTime());
        }
        ret.decodingLatency = decoder.getStats().decoding_latency;
    }
    return ret;
}
