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
    virtual bool init() = 0;

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

    std::atomic_bool terminate = {false};
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

    bool init() { return true; }

    void start() {
        terminate = false;
        workThread = std::thread([&]() {
            while (!terminate) {
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
                        return !is_decoding && (frameQueue.size() < queueSize || terminate);
                    });
                }
                hasFrame.notify_one();
            }
        });
    }

    void stop() {
        terminate = true;
        condVar.notify_one();
        if (workThread.joinable()) {
            workThread.join();
        }
    }

    bool read(VideoFrame& frame)  {
        queue_elem_t elem;

        if (terminate)
            return false;

        {
            std::unique_lock<std::mutex> lock(mutex);
            hasFrame.wait(lock, [&]() {
                return !frameQueue.empty() || terminate;
            });
            elem = std::move(frameQueue.front());
            frameQueue.pop();
        }
        condVar.notify_one();
        frame.frame = std::move(elem.second);

        return elem.first && !terminate;
    }

    float getAvgReadTime() const {
        return perfTimer.getValue();
    }
};

#endif

class VideoSourceOCV : public VideoSource {
    PerfTimer perfTimer;
    std::thread workThread;
    const bool isAsync = false;
    std::atomic_bool terminate = {false};
    std::string videoName;

    std::mutex mutex;
    std::condition_variable condVar;
    std::condition_variable hasFrame;
    std::queue<std::pair<bool, cv::Mat>> queue;

    cv::VideoCapture source;

    bool realFps;

    const size_t queueSize = 1;
    const size_t pollingTimeMSec = 1000;

    template<bool CollectStats>
    bool readFrame(cv::Mat& frame);

    template<bool CollectStats>
    bool readFrameImpl(cv::Mat& frame);

    template<bool CollectStats>
    void startImpl();

public:
    VideoSourceOCV(bool async, bool collectStats_, const std::string& name,
                size_t queueSize_, size_t pollingTimeMSec_, bool realFps_);

    ~VideoSourceOCV();

    void start();

    bool init();

    void stop();

    bool read(cv::Mat& frame);
    bool read(VideoFrame& frame);

    float getAvgReadTime() const {
        return perfTimer.getValue();
    }

private:
    template<bool CollectStats>
    static void thread_fn(VideoSourceOCV*);
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

    bool init();

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

bool VideoSourceNative::init() {
    // nothing
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

bool VideoSourceOCV::init() {
    static std::mutex initMutex;  // HACK: opencv camera init is not thread-safe
    std::unique_lock<std::mutex> lock(initMutex);
    bool res = false;
    if (isNumeric(videoName)) {
#ifdef __linux__
        res = source.open("/dev/video" + videoName);
#else
        res = source.open(std::stoi(videoName));
#endif
    } else {
        res = source.open(videoName);
    }
    if (res) {
        source.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    }
    return res;
}

template<bool CollectStats>
bool VideoSourceOCV::readFrame(cv::Mat& frame) {
    if (!source.isOpened() && !init()) {
        return false;
    }
    if (!readFrameImpl<CollectStats>(frame)) {
        return init() && readFrameImpl<CollectStats>(frame);
    }
    return true;
}

template<bool CollectStats>
bool VideoSourceOCV::readFrameImpl(cv::Mat& frame) {
    if (CollectStats) {
        ScopedTimer st(perfTimer);
        return source.read(frame);
    } else {
        return source.read(frame);
    }
}

VideoSourceOCV::VideoSourceOCV(bool async, bool collectStats_,
                         const std::string& name, size_t queueSize_,
                         size_t pollingTimeMSec_, bool realFps_):
    perfTimer(collectStats_ ? PerfTimer::DefaultIterationsCount : 0),
    isAsync(async), videoName(name),
    realFps(realFps_),
    queueSize(queueSize_),
    pollingTimeMSec(pollingTimeMSec_) {}

VideoSourceOCV::~VideoSourceOCV() {
    stop();
}

template<bool CollectStats>
void VideoSourceOCV::thread_fn(VideoSourceOCV *vs) {
    while (!vs->terminate) {
        cv::Mat frame;
        bool result = false;
        while (!((result = vs->readFrame<CollectStats>(frame)) || vs->terminate)) {
            std::unique_lock<std::mutex> lock(vs->mutex);
            if (vs->queue.empty() || vs->queue.back().first) {
                vs->queue.push({false, frame});
                lock.unlock();
                vs->hasFrame.notify_one();
                lock.lock();
            }
            std::chrono::milliseconds timeout(vs->pollingTimeMSec);
            vs->condVar.wait_for(lock,
                             timeout,
                             [&]() {
                                 return vs->terminate.load();
                             });
        }

        if (vs->queue.size() < vs->queueSize) {
            std::unique_lock<std::mutex> lock(vs->mutex);
            vs->queue.push({result, frame});
        }
        vs->hasFrame.notify_one();
    }
}

template<bool CollectStats>
void VideoSourceOCV::startImpl() {
    if (isAsync) {
        terminate = false;
        workThread = std::thread(&VideoSourceOCV::thread_fn<CollectStats>, this);
    }
}

void VideoSourceOCV::start() {
    if (perfTimer.enabled()) {
        startImpl<true>();
    } else {
        startImpl<false>();
    }
}

void VideoSourceOCV::stop() {
    if (isAsync) {
        terminate = true;
        condVar.notify_one();
        if (workThread.joinable()) {
            workThread.join();
        }
    }
}

bool VideoSourceOCV::read(cv::Mat& frame) {
    if (isAsync) {
        size_t count = 0;
        bool res = false;
        {
            std::unique_lock<std::mutex> lock(mutex);
            hasFrame.wait(lock, [&]() {
                return !queue.empty() || terminate;
            });
            res = queue.front().first;
            frame = queue.front().second;
            if (realFps || queue.size() > 1 || queueSize == 1) {
                queue.pop();
            }
            count = queue.size();
            (void)count;
        }
        condVar.notify_one();
        return res;
    } else {
        return source.read(frame);
    }
}

bool VideoSourceOCV::read(VideoFrame& frame) {
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
    pollingTimeMSec(p.pollingTimeMSec) {}

VideoSources::~VideoSources() {
    // nothing
}

void VideoSources::openVideo(const std::string& source, bool native) {
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
#else
    if (false) {
#endif
    } else {
#if defined(USE_LIBVA)
        const std::string extension = ".mjpeg";
        std::unique_ptr<VideoSource> newSrc;
        if (source.size() > extension.size() && std::equal(extension.rbegin(), extension.rend(), source.rbegin()))
            newSrc.reset(new VideoSourceStreamFile(*this, isAsync, collectStats, source,
                                            queueSize, pollingTimeMSec, realFps));
        else
            newSrc.reset(new VideoSourceOCV(isAsync, collectStats, source,
                                            queueSize, pollingTimeMSec, realFps));
#else
        std::unique_ptr<VideoSource> newSrc(new VideoSourceOCV(isAsync, collectStats, source,
                                            queueSize, pollingTimeMSec, realFps));
#endif
        if (newSrc->init()) {
            inputs.emplace_back(std::move(newSrc));
        } else {
            throw std::runtime_error("Cannot open cv::VideoCapture");
        }
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
