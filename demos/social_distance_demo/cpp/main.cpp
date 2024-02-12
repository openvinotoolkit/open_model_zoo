// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <list>
#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <set>
#include <stdexcept>
#include <string>

#include "openvino/openvino.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"

#include "monitors/presenter.h"
#include "utils/args_helper.hpp"
#include "utils/grid_mat.hpp"
#include "utils/input_wrappers.hpp"
#include "utils/ocv_common.hpp"
#include "utils/slog.hpp"
#include "utils/threads_common.hpp"

#include "geodist.hpp"
#include "net_wrappers.hpp"
#include "person_trackers.hpp"
#include "social_distance_demo.hpp"

typedef std::chrono::duration<float, std::chrono::seconds::period> Sec;

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    if (FLAGS_m_det.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }
    if (FLAGS_nc == 0 && FLAGS_i.empty()) {
        throw std::logic_error("Please specify at least one video source(web cam or video file)");
    }
    if (FLAGS_display_resolution.find("x") == std::string::npos) {
        throw std::logic_error("Incorrect format of -displayresolution parameter. Correct format is  \"width\"x\"height\". For example \"1920x1080\"");
    }
    if (FLAGS_n_wt == 0) {
        throw std::logic_error("-n_wt can not be zero");
    }
    return true;
}

struct InferRequestsContainer {
    InferRequestsContainer() = default;
    InferRequestsContainer(const InferRequestsContainer&) = delete;
    InferRequestsContainer& operator=(const InferRequestsContainer&) = delete;

    void assign(const std::vector<ov::InferRequest>& inferRequests) {
        actualInferRequests = inferRequests;
        this->inferRequests.container.clear();

        for (auto& ir : this->actualInferRequests) {
            this->inferRequests.container.push_back(ir);
        }
    }

    std::vector<ov::InferRequest> getActualInferRequests() {
        return actualInferRequests;
    }
    ConcurrentContainer<std::vector<std::reference_wrapper<ov::InferRequest>>> inferRequests;

private:
    std::vector<ov::InferRequest> actualInferRequests;
};

struct Context {  // stores all global data for tasks
    Context(const std::vector<std::shared_ptr<InputChannel>>& inputChannels,
            const PersonDetector& detector,
            const ReId& reid,
            int pause, const std::vector<cv::Size>& gridParam, cv::Size displayResolution, std::chrono::steady_clock::duration showPeriod,
            const std::string& monitorsStr,
            uint64_t lastFrameId,
            uint64_t nireq,
            bool isVideo,
            std::size_t nreidireq):
        readersContext{inputChannels, std::vector<int64_t>(inputChannels.size(), -1), std::vector<std::mutex>(inputChannels.size())},
        inferTasksContext{detector},
        detectionsProcessorsContext{reid},
        drawersContext{pause, gridParam, displayResolution, showPeriod, monitorsStr},
        videoFramesContext{std::vector<uint64_t>(inputChannels.size(), lastFrameId), std::vector<std::mutex>(inputChannels.size())},
        trackersContext{std::vector<PersonTrackers>(inputChannels.size()), std::vector<int>(inputChannels.size(), 9999),
            std::vector<int>(inputChannels.size(), 1)},
        nireq{nireq},
        isVideo{isVideo},
        freeDetectionInfersCount{0},
        frameCounter{0}
    {
        assert(inputChannels.size() == gridParam.size());
        std::vector<ov::InferRequest> detectorInferRequests;
        std::vector<ov::InferRequest> reidInferRequests;
        detectorInferRequests.reserve(nireq);
        reidInferRequests.reserve(nreidireq);
        std::generate_n(std::back_inserter(detectorInferRequests), nireq, [&]{
            return inferTasksContext.detector.createInferRequest();});
        std::generate_n(std::back_inserter(reidInferRequests), nreidireq, [&]{
            return detectionsProcessorsContext.reid.createInferRequest();});
        detectorsInfers.assign(detectorInferRequests);
        reidInfers.assign(reidInferRequests);
    }

    struct {
        std::vector<std::shared_ptr<InputChannel>> inputChannels;
        std::vector<int64_t> lastCapturedFrameIds;
        std::vector<std::mutex> lastCapturedFrameIdsMutexes;
        std::weak_ptr<Worker> readersWorker;
    } readersContext;

    struct {
        PersonDetector detector;
        std::weak_ptr<Worker> inferTasksWorker;
    } inferTasksContext;

    struct {
        ReId reid;
        std::weak_ptr<Worker> reidTasksWorker;
    } detectionsProcessorsContext;

    struct DrawersContext {
        DrawersContext(int pause, const std::vector<cv::Size>& gridParam, cv::Size displayResolution, std::chrono::steady_clock::duration showPeriod,
                       const std::string& monitorsStr):
            pause{pause}, gridParam{gridParam}, displayResolution{displayResolution}, showPeriod{showPeriod},
            lastShownframeId{0}, prevShow{std::chrono::steady_clock::time_point()},
            presenter{monitorsStr,
                GridMat(gridParam, displayResolution).outimg.rows - 70,
                cv::Size{GridMat(gridParam, displayResolution).outimg.cols / 4, 60}} {}
        int pause;
        std::vector<cv::Size> gridParam;
        cv::Size displayResolution;
        std::chrono::steady_clock::duration showPeriod;  // desiered frequency of imshow
        std::weak_ptr<Worker> drawersWorker;
        int64_t lastShownframeId;
        std::chrono::steady_clock::time_point prevShow;  // time stamp of previous imshow
        std::map<int64_t, GridMat> gridMats;
        std::mutex drawerMutex;
        Presenter presenter;
    } drawersContext;

    struct {
        std::vector<uint64_t> lastframeIds;
        std::vector<std::mutex> lastFrameIdsMutexes;
    } videoFramesContext;

    struct TrackersContext {
        std::vector<PersonTrackers> personTracker;
        std::vector<int> minW;
        std::vector<int> maxW;
        std::vector<std::atomic<int32_t>> lastProcessedIds;

        //manual definition of constructor is needed only for creating vector<std::atomic<int32_t>>
        TrackersContext(std::vector<PersonTrackers>&& personTracker, std::vector<int>&& minW, std::vector<int>&& maxW)
            : personTracker(personTracker), minW(minW), maxW(maxW), lastProcessedIds(personTracker.size()) {
            for (size_t i = 0; i < lastProcessedIds.size(); ++i) {
                lastProcessedIds[i] = 0;
            }
        }
    } trackersContext;

    std::weak_ptr<Worker> resAggregatorsWorker;
    std::mutex classifiersAggregatorPrintMutex;
    uint64_t nireq;
    bool isVideo;
    std::atomic<std::vector<ov::InferRequest>::size_type> freeDetectionInfersCount;
    std::atomic<uint32_t> frameCounter;
    InferRequestsContainer detectorsInfers, reidInfers;
    PerformanceMetrics metrics;
};

class ReborningVideoFrame : public VideoFrame {
public:
    ReborningVideoFrame(Context& context, const unsigned sourceID, const int64_t frameId, const cv::Mat& frame = cv::Mat())
            : VideoFrame{sourceID, frameId, frame}, context(context) {} // can not write context{context} because of CentOS 7.4 compiler bug

    virtual ~ReborningVideoFrame();
    Context& context;
};

// accumulates and shows processed frames
class Drawer : public Task {
public:
    explicit Drawer(VideoFrame::Ptr sharedVideoFrame) :
        Task{sharedVideoFrame, 1.0} {}
    bool isReady() override;
    void process() override;

private:
};

// draws results on the frame
class ResAggregator : public Task {
public:
    ResAggregator(const VideoFrame::Ptr& sharedVideoFrame, std::list<cv::Rect>&& boxes,
        std::list<TrackableObject>&& trackables)
        : Task{ sharedVideoFrame, 4.0 },
        boxes{ std::move(boxes) },
        trackables{ std::move(trackables) } {}

    bool isReady() override {
        Context& context = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context;
        auto& lastProcessedFrameId = context.trackersContext.lastProcessedIds[sharedVideoFrame->sourceID];
        return lastProcessedFrameId == sharedVideoFrame->frameId;
    }

    void process() override;

private:
    std::list<cv::Rect> boxes;
    std::list<TrackableObject> trackables;
};

// waits for all reid results accumulating results
class ClassifiersAggregator {
public:
    explicit ClassifiersAggregator(const VideoFrame::Ptr& sharedVideoFrame) : sharedVideoFrame{sharedVideoFrame} {}

    ~ClassifiersAggregator() {
        std::mutex& printMutex = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context.classifiersAggregatorPrintMutex;
        printMutex.lock();
        if (FLAGS_r && !rawDetections.empty()) {
            slog::debug << "---------------------Frame #" << sharedVideoFrame->frameId << "---------------------" << slog ::endl;
            slog::debug << rawDetections;
        }
        printMutex.unlock();
        tryPush(static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context.resAggregatorsWorker,
                std::make_shared<ResAggregator>(sharedVideoFrame, std::move(boxes), std::move(trackables)));
    }

    void push(cv::Rect&& bbox) {
        boxes.lockedPushBack(std::move(bbox));
    }

    void push(TrackableObject&& trackable) {
        trackables.lockedPushBack(std::move(trackable));
    }

    const VideoFrame::Ptr sharedVideoFrame;
    std::vector<std::string> rawDetections;

private:
    ConcurrentContainer<std::list<cv::Rect>> boxes;
    ConcurrentContainer<std::list<TrackableObject>> trackables;
};

// extracts detections from blob InferRequests and runs Re-Id
class DetectionsProcessor : public Task {
public:
    DetectionsProcessor(VideoFrame::Ptr sharedVideoFrame, ov::InferRequest* inferRequest)
            : Task{sharedVideoFrame, 1.0},
              inferRequest{inferRequest},
              requireGettingNumberOfDetections{true} {}

    DetectionsProcessor(VideoFrame::Ptr sharedVideoFrame,
                        std::shared_ptr<ClassifiersAggregator>&& classifiersAggregator,
                        std::list<cv::Rect>&& personRects)
            : Task{sharedVideoFrame, 1.0},
              classifiersAggregator{std::move(classifiersAggregator)},
              inferRequest{nullptr},
              personRects{std::move(personRects)},
              requireGettingNumberOfDetections{false} {}

    bool isReady() override;
    void process() override;

private:
    std::shared_ptr<ClassifiersAggregator> classifiersAggregator; // when no one stores this object we will draw
    ov::InferRequest* inferRequest;
    std::list<cv::Rect> personRects;
    std::vector<TrackableObject> personTrackers;
    std::vector<std::reference_wrapper<ov::InferRequest>> reservedReIdRequests;
    bool requireGettingNumberOfDetections;
};

// runs detection
class InferTask : public Task {
public:
    explicit InferTask(VideoFrame::Ptr sharedVideoFrame) : Task{sharedVideoFrame, 5.0} {}

    bool isReady() override;
    void process() override;
};

class Reader : public Task {
public:
    explicit Reader(VideoFrame::Ptr sharedVideoFrame) : Task{sharedVideoFrame, 2.0} {}

    bool isReady() override;
    void process() override;
};

ReborningVideoFrame::~ReborningVideoFrame() {
    try {
        const std::shared_ptr<Worker>& worker = std::shared_ptr<Worker>(context.readersContext.readersWorker);
        context.videoFramesContext.lastFrameIdsMutexes[sourceID].lock();
        const auto frameId = ++context.videoFramesContext.lastframeIds[sourceID];
        context.videoFramesContext.lastFrameIdsMutexes[sourceID].unlock();
        std::shared_ptr<ReborningVideoFrame> reborn = std::make_shared<ReborningVideoFrame>(context, sourceID, frameId, frame);
        worker->push(std::make_shared<Reader>(reborn));
    } catch (const std::bad_weak_ptr&) {}
}

bool Drawer::isReady() {
    Context& context = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context;
    std::chrono::steady_clock::time_point prevShow = context.drawersContext.prevShow;
    std::chrono::steady_clock::duration showPeriod = context.drawersContext.showPeriod;
    if (1u == context.drawersContext.gridParam.size()) {
        if (std::chrono::steady_clock::now() - prevShow > showPeriod) {
            return true;
        } else {
            return false;
        }
    } else {
        std::map<int64_t, GridMat>& gridMats = context.drawersContext.gridMats;
        auto gridMatIt = gridMats.find(sharedVideoFrame->frameId);
        if (gridMats.end() == gridMatIt) {
            if (2 > gridMats.size()) {  // buffer size
                return true;
            } else {
                return false;
            }
        } else {
            if (1u == gridMatIt->second.getUnupdatedSourceIDs().size()) {
                if (context.drawersContext.lastShownframeId == sharedVideoFrame->frameId
                    && std::chrono::steady_clock::now() - prevShow > showPeriod) {
                    return true;
                } else {
                    return false;
                }
            } else {
                return true;
            }
        }
    }
}

void Drawer::process() {
    const int64_t frameId = sharedVideoFrame->frameId;
    Context& context = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context;
    std::map<int64_t, GridMat>& gridMats = context.drawersContext.gridMats;
    context.drawersContext.drawerMutex.lock();
    auto gridMatIt = gridMats.find(frameId);
    if (gridMats.end() == gridMatIt) {
        gridMatIt = gridMats.emplace(frameId, GridMat(context.drawersContext.gridParam,
                                                      context.drawersContext.displayResolution)).first;
    }

    gridMatIt->second.update(sharedVideoFrame->frame, sharedVideoFrame->sourceID);
    auto firstGridIt = gridMats.begin();
    int64_t& lastShownframeId = context.drawersContext.lastShownframeId;
    if (firstGridIt->first == lastShownframeId && firstGridIt->second.isFilled()) {
        lastShownframeId++;
        cv::Mat mat = firstGridIt->second.getMat();
        constexpr float OPACITY = 0.6f;
        fillROIColor(mat, cv::Rect(5, 5, 390, 125), cv::Scalar(255, 0, 0), OPACITY);
        cv::putText(mat, "Detection InferRequests usage:", cv::Point2f(15, 95), cv::FONT_HERSHEY_TRIPLEX, 0.7, cv::Scalar{255, 255, 255});
        cv::Rect usage(15, 105, 370, 20);
        cv::rectangle(mat, usage, {0, 255, 0}, 2);
        uint64_t nireq = context.nireq;
        uint32_t frameCounter = context.frameCounter;
        usage.width = static_cast<int>(usage.width * static_cast<float>(frameCounter * nireq - context.freeDetectionInfersCount) / (frameCounter * nireq));
        cv::rectangle(mat, usage, {0, 255, 0}, cv::FILLED);

        context.drawersContext.presenter.drawGraphs(mat);
        context.metrics.update(sharedVideoFrame->timestamp, mat, { 15, 35 }, cv::FONT_HERSHEY_TRIPLEX, 0.7, cv::Scalar{ 255, 255, 255 }, 0);
        if (!FLAGS_no_show) {
            cv::imshow("Detection results", firstGridIt->second.getMat());
            context.drawersContext.prevShow = std::chrono::steady_clock::now();
            const int key = cv::waitKey(context.drawersContext.pause);
            if (key == 27 || 'q' == key || 'Q' == key || !context.isVideo) {
                try {
                    std::shared_ptr<Worker>(context.drawersContext.drawersWorker)->stop();
                }
                catch (const std::bad_weak_ptr&) {}
            }
            else {
                context.drawersContext.presenter.handleKey(key);
            }
        }
        else {
            if (!context.isVideo) {
                try {
                    std::shared_ptr<Worker>(context.drawersContext.drawersWorker)->stop();
                }
                catch (const std::bad_weak_ptr&) {}
            }
        }
        firstGridIt->second.clear();
        gridMats.emplace((--gridMats.end())->first + 1, firstGridIt->second);
        gridMats.erase(firstGridIt);
    }
    context.drawersContext.drawerMutex.unlock();
}

void ResAggregator::process() {
    Context& context = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context;
    context.freeDetectionInfersCount += context.detectorsInfers.inferRequests.lockedSize();
    context.frameCounter++;
    unsigned sourceID = sharedVideoFrame->sourceID;
    auto& personTracker = context.trackersContext.personTracker[sourceID];
    for (const cv::Rect& bbox : boxes) {
        cv::rectangle(sharedVideoFrame->frame, bbox, { 0, 255, 0 }, 2);
        if (bbox.width < context.trackersContext.minW[sourceID]) {
            context.trackersContext.minW[sourceID] = bbox.width;
        }
        if (bbox.width > context.trackersContext.maxW[sourceID]) {
            context.trackersContext.maxW[sourceID] = bbox.width;
        }
    }

    personTracker.similarity(trackables);

    int w = sharedVideoFrame->frame.size().width;
    int h = sharedVideoFrame->frame.size().height;
    for (auto it1 = personTracker.trackables.begin(); it1 != personTracker.trackables.end(); ++it1) {
        cv::Rect2d  l1 = it1->second.bbox;
        auto it2 = it1;
        ++it2;
        for (; it2 != personTracker.trackables.end(); ++it2) {
            cv::Rect2d  l2 = it2->second.bbox;
            cv::Point2d a, b, c, d;
            if (l1.y + l1.height < l2.y + l2.height) {
                a = { l1.x, l1.y + l1.height };
                b = { l1.x + l1.width, l1.y + l1.height };
                c = { l2.x, l2.y + l2.height };
                d = { l2.x + l2.width, l2.y + l2.height };
            }
            else {
                c = { l1.x, l1.y + l1.height };
                d = { l1.x + l1.width, l1.y + l1.height };
                a = { l2.x, l2.y + l2.height };
                b = { l2.x + l2.width, l2.y + l2.height };
            }

            std::tuple<int, int> frame_shape(h, w);
            auto result = socialDistance(frame_shape, a, b, c, d, 4 /* ~ 5 feets */,
                context.trackersContext.minW[sourceID],
                context.trackersContext.maxW[sourceID]);

            if (std::get<1>(result)) {
                cv::rectangle(sharedVideoFrame->frame, l1, { 0, 255, 255 }, 2);
                cv::rectangle(sharedVideoFrame->frame, l2, { 0, 255, 255 }, 2);
                cv::Point2d rect1center = { l1.x + l1.width / 2, l1.y + l1.height / 2 };
                cv::Point2d rect2center = { l2.x + l2.width / 2, l2.y + l2.height / 2 };
                cv::line(sharedVideoFrame->frame, rect1center, rect2center, { 0, 0, 255 }, 3);
            }
        }
    }

    tryPush(context.drawersContext.drawersWorker, std::make_shared<Drawer>(sharedVideoFrame));
    ++context.trackersContext.lastProcessedIds[sourceID];
}

bool DetectionsProcessor::isReady() {
    Context& context = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context;

    if (requireGettingNumberOfDetections) {
        classifiersAggregator = std::make_shared<ClassifiersAggregator>(sharedVideoFrame);
        std::list<PersonDetector::Result> results;
        results = context.inferTasksContext.detector.getResults(*inferRequest, sharedVideoFrame->frame.size(), classifiersAggregator->rawDetections);
        for (PersonDetector::Result result : results) {
            personRects.emplace_back(result.location & cv::Rect{ cv::Point(0, 0), sharedVideoFrame->frame.size() });
        }

        context.detectorsInfers.inferRequests.lockedPushBack(*inferRequest);
        requireGettingNumberOfDetections = false;
    }

    if (personRects.empty() || FLAGS_m_reid.empty()) {
        return true;
    } else {
        // isReady() is called under mutexes so it is assured that available InferRequests will not be taken, but new InferRequests can come in
        // acquire as many InferRequests as it is possible or needed
        InferRequestsContainer& reidInfers = context.reidInfers;
        reidInfers.inferRequests.mutex.lock();
        const std::size_t numberOfReIdInferRequestsAcquired = std::min(personRects.size(), reidInfers.inferRequests.container.size());
        reservedReIdRequests.assign(reidInfers.inferRequests.container.end() - numberOfReIdInferRequestsAcquired, reidInfers.inferRequests.container.end());
        reidInfers.inferRequests.container.erase(reidInfers.inferRequests.container.end() - numberOfReIdInferRequestsAcquired,
                                                 reidInfers.inferRequests.container.end());
        reidInfers.inferRequests.mutex.unlock();
        return numberOfReIdInferRequestsAcquired;
    }
}

void DetectionsProcessor::process() {
    if (!personRects.empty()) {
        Context& context = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context;

        if (!FLAGS_m_reid.empty()) {
            auto personRectsIt = personRects.begin();

            for (auto reidRequestsIt = reservedReIdRequests.begin(); reidRequestsIt != reservedReIdRequests.end(); personRectsIt++, reidRequestsIt++) {
                const cv::Rect personRect = *personRectsIt;
                ov::InferRequest& reidRequest = *reidRequestsIt;
                context.detectionsProcessorsContext.reid.setImage(reidRequest, sharedVideoFrame->frame, personRect);

                reidRequest.set_callback(
                    std::bind([](std::shared_ptr<ClassifiersAggregator> classifiersAggregator,
                        ov::InferRequest& reidRequest, cv::Rect rect, Context& context) {
                                    reidRequest.set_callback([](std::exception_ptr) {}); // destroy the stored bind object
                                    std::vector<float> result = context.detectionsProcessorsContext.reid.getResults(reidRequest);

                                    classifiersAggregator->push(cv::Rect(rect));
                                    classifiersAggregator->push(TrackableObject{rect,
                                            std::move(result), {rect.x + rect.width / 2, rect.y + rect.height } });
                                    context.reidInfers.inferRequests.lockedPushBack(reidRequest);
                            },
                            classifiersAggregator, std::ref(reidRequest), personRect, std::ref(context)));

                reidRequest.start_async();
            }
            personRects.erase(personRects.begin(), personRectsIt);
        } else {
            for (const cv::Rect& personRect : personRects) {
                classifiersAggregator->push(cv::Rect(personRect));
            }
            personRects.clear();
        }

        // Run DetectionsProcessor for remaining persons
        if (!personRects.empty()) {
            tryPush(context.detectionsProcessorsContext.reidTasksWorker,
                    std::make_shared<DetectionsProcessor>(sharedVideoFrame, std::move(classifiersAggregator), std::move(personRects)));
        }
    }
}

bool InferTask::isReady() {
    InferRequestsContainer& detectorsInfers = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context.detectorsInfers;
    if (detectorsInfers.inferRequests.container.empty()) {
        return false;
    } else {
        detectorsInfers.inferRequests.mutex.lock();
        if (detectorsInfers.inferRequests.container.empty()) {
            detectorsInfers.inferRequests.mutex.unlock();
            return false;
        } else {
            return true;  // process() will unlock the mutex
        }
    }
}

void InferTask::process() {
    Context& context = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context;
    InferRequestsContainer& detectorsInfers = context.detectorsInfers;
    std::reference_wrapper<ov::InferRequest> inferRequest = detectorsInfers.inferRequests.container.back();
    detectorsInfers.inferRequests.container.pop_back();
    detectorsInfers.inferRequests.mutex.unlock();

    context.inferTasksContext.detector.setImage(inferRequest, sharedVideoFrame->frame);

    inferRequest.get().set_callback(
        std::bind(
            [](VideoFrame::Ptr sharedVideoFrame,
                ov::InferRequest& inferRequest,
                Context& context) {
                    inferRequest.set_callback([](std::exception_ptr) {});  // destroy the stored bind object
                    tryPush(context.detectionsProcessorsContext.reidTasksWorker,
                            std::make_shared<DetectionsProcessor>(sharedVideoFrame, &inferRequest));
                }, sharedVideoFrame,
                   inferRequest,
                   std::ref(context)));
    inferRequest.get().start_async();
    // do not push as callback does it
}

bool Reader::isReady() {
    Context& context = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context;
    context.readersContext.lastCapturedFrameIdsMutexes[sharedVideoFrame->sourceID].lock();
    if (context.readersContext.lastCapturedFrameIds[sharedVideoFrame->sourceID] + 1 == sharedVideoFrame->frameId) {
        return true;
    } else {
        context.readersContext.lastCapturedFrameIdsMutexes[sharedVideoFrame->sourceID].unlock();
        return false;
    }
}

void Reader::process() {
    unsigned sourceID = sharedVideoFrame->sourceID;
    sharedVideoFrame->timestamp = std::chrono::steady_clock::now();
    Context& context = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context;
    const std::vector<std::shared_ptr<InputChannel>>& inputChannels = context.readersContext.inputChannels;
    if (inputChannels[sourceID]->read(sharedVideoFrame->frame)) {
        context.readersContext.lastCapturedFrameIds[sourceID]++;
        context.readersContext.lastCapturedFrameIdsMutexes[sourceID].unlock();
        tryPush(context.inferTasksContext.inferTasksWorker, std::make_shared<InferTask>(sharedVideoFrame));
    } else {
        context.readersContext.lastCapturedFrameIds[sourceID]++;
        context.readersContext.lastCapturedFrameIdsMutexes[sourceID].unlock();
        try {
            std::shared_ptr<Worker>(context.drawersContext.drawersWorker)->stop();
        } catch (const std::bad_weak_ptr&) {}
    }
}

int main(int argc, char* argv[]) {
    try {
        // ------------------------------ Parsing and validation of input args ---------------------------------
        try {
            if (!ParseAndCheckCommandLine(argc, argv)) {
                return 0;
            }
        } catch (std::logic_error& error) {
            slog::err << error.what() << slog::endl;
            return 1;
        }

        std::vector<std::string> files;
        parseInputFilesArguments(files);
        if (files.empty() && 0 == FLAGS_nc) {
            throw std::logic_error("No inputs were found");
        }

        std::vector<std::shared_ptr<VideoCaptureSource>> videoCapturSources;
        std::vector<std::shared_ptr<ImageSource>> imageSources;
        if (FLAGS_nc) {
            for (size_t i = 0; i < FLAGS_nc; ++i) {
                cv::VideoCapture videoCapture(i);
                if (!videoCapture.isOpened()) {
                    slog::info << "Cannot open web cam [" << i << "]" << slog::endl;
                    return 1;
                }
                videoCapture.set(cv::CAP_PROP_FPS , 30);
                videoCapture.set(cv::CAP_PROP_BUFFERSIZE , 1);
                videoCapture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
                videoCapture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
                videoCapturSources.push_back(std::make_shared<VideoCaptureSource>(videoCapture, FLAGS_loop_video));
            }
        }

        for (const std::string& file : files) {
            cv::Mat frame = cv::imread(file, cv::IMREAD_COLOR);
            if (frame.empty()) {
                cv::VideoCapture videoCapture(file);
                if (!videoCapture.isOpened()) {
                    slog::info << "Cannot open " << file << slog::endl;
                    return 1;
                }
                videoCapturSources.push_back(std::make_shared<VideoCaptureSource>(videoCapture, FLAGS_loop_video));
            } else {
                imageSources.push_back(std::make_shared<ImageSource>(frame, true));
            }
        }
        uint32_t channelsNum = 0 == FLAGS_ni ? videoCapturSources.size() + imageSources.size() : FLAGS_ni;
        std::vector<std::shared_ptr<IInputSource>> inputSources;
        inputSources.reserve(videoCapturSources.size() + imageSources.size());
        for (const std::shared_ptr<VideoCaptureSource>& videoSource : videoCapturSources) {
            inputSources.push_back(videoSource);
        }
        for (const std::shared_ptr<ImageSource>& imageSource : imageSources) {
            inputSources.push_back(imageSource);
        }

        std::vector<std::shared_ptr<InputChannel>> inputChannels;
        inputChannels.reserve(channelsNum);
        for (decltype(inputSources.size()) channelI = 0, counter = 0; counter < channelsNum; channelI++, counter++) {
            if (inputSources.size() == channelI) {
                channelI = 0;
            }
            inputChannels.push_back(InputChannel::create(inputSources[channelI]));
        }

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load OpenVINO runtime -------------------------------------
        slog::info << ov::get_openvino_version() << slog::endl;
        ov::Core core;

        std::set<std::string> devices;
        for (const std::string& netDevices : {FLAGS_d_det, FLAGS_d_reid}) {
            if (netDevices.empty()) {
                continue;
            }
            for (const std::string& device : parseDevices(netDevices)) {
                devices.insert(device);
            }
        }
        std::map<std::string, int32_t> deviceNStreams = parseValuePerDevice(devices, FLAGS_nstreams);

        for (const std::string& device : devices) {
            if ("CPU" == device) {
                if (FLAGS_nthreads != 0) {
                    core.set_property("CPU", ov::inference_num_threads(FLAGS_nthreads));
                }
                core.set_property("CPU", ov::affinity(ov::Affinity::NONE));
                core.set_property("CPU", ov::streams::num((deviceNStreams.count("CPU") > 0 ? ov::streams::Num(deviceNStreams["CPU"]) : ov::streams::AUTO)));
                deviceNStreams["CPU"] = core.get_property("CPU", ov::streams::num);
            }

            if ("GPU" == device) {
                core.set_property("GPU", ov::streams::num(deviceNStreams.count("GPU") > 0 ? ov::streams::Num(deviceNStreams["GPU"]) : ov::streams::AUTO));

                deviceNStreams["GPU"] = core.get_property("GPU", ov::streams::num);
                if (devices.end() != devices.find("CPU")) {
                    // multi-device execution with the CPU + GPU performs best with GPU trottling hint,
                    // which releases another CPU thread (that is otherwise used by the GPU driver for active polling)
                    core.set_property("GPU", ov::intel_gpu::hint::queue_throttle(ov::intel_gpu::hint::ThrottleLevel(1)));
                }
            }
        }

        // -----------------------------------------------------------------------------------------------------
        unsigned nireq = FLAGS_nireq == 0 ? inputChannels.size() : FLAGS_nireq;
        PersonDetector detector(core, FLAGS_d_det, FLAGS_m_det,
            {static_cast<float>(FLAGS_t), static_cast<float>(FLAGS_t)}, FLAGS_auto_resize);
        slog::info << "\tNumber of network inference requests: " << nireq << slog::endl;
        ReId reid;
        std::size_t nreidireq{0};
        if (!FLAGS_m_reid.empty()) {
            reid = ReId(core, FLAGS_d_reid, FLAGS_m_reid, FLAGS_auto_resize);
            nreidireq = nireq * 3;
            slog::info << "\tNumber of network inference requests: " << nreidireq << slog::endl;
        }

        bool isVideo = imageSources.empty() ? true : false;
        int pause = imageSources.empty() ? 1 : 0;
        std::chrono::steady_clock::duration showPeriod = 0 == FLAGS_fps ? std::chrono::steady_clock::duration::zero()
            : std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::seconds{1}) / FLAGS_fps;
        std::vector<cv::Size> gridParam;
        gridParam.reserve(inputChannels.size());
        for (const auto& inputChannel : inputChannels) {
            gridParam.emplace_back(inputChannel->getSize());
        }
        size_t found = FLAGS_display_resolution.find("x");
        cv::Size displayResolution = cv::Size{std::stoi(FLAGS_display_resolution.substr(0, found)),
                                              std::stoi(FLAGS_display_resolution.substr(found + 1, FLAGS_display_resolution.length()))};

        Context context{inputChannels,
                        detector,
                        reid,
                        pause, gridParam, displayResolution, showPeriod, FLAGS_u,
                        FLAGS_n_iqs - 1,
                        nireq,
                        isVideo,
                        nreidireq};
        // Create a worker after a context because the context has only weak_ptr<Worker>, but the worker is going to
        // indirectly store ReborningVideoFrames which have a reference to the context. So there won't be a situation
        // when the context is destroyed and the worker still lives with its ReborningVideoFrames referring to the
        // destroyed context.
        std::shared_ptr<Worker> worker = std::make_shared<Worker>(FLAGS_n_wt - 1);
        context.readersContext.readersWorker = context.inferTasksContext.inferTasksWorker
            = context.detectionsProcessorsContext.reidTasksWorker = context.drawersContext.drawersWorker
            = context.resAggregatorsWorker = worker;

        for (uint64_t i = 0; i < FLAGS_n_iqs; i++) {
            for (unsigned sourceID = 0; sourceID < inputChannels.size(); sourceID++) {
                VideoFrame::Ptr sharedVideoFrame = std::make_shared<ReborningVideoFrame>(context, sourceID, i);
                worker->push(std::make_shared<Reader>(sharedVideoFrame));
            }
        }

        // Running
        worker->runThreads();
        worker->threadFunc();
        worker->join();

        uint32_t frameCounter = context.frameCounter;
        double detectionsInfersUsage = 0;
        if (0 != frameCounter) {
            detectionsInfersUsage = static_cast<float>(frameCounter * context.nireq - context.freeDetectionInfersCount)
                / (frameCounter * context.nireq) * 100;
        }

        slog::info << "Metrics report:" << slog::endl;
        context.metrics.logTotal();
        slog::info << "\tDetection InferRequests usage: " << detectionsInfersUsage << "%" << slog::endl;
        slog::info << context.drawersContext.presenter.reportMeans() << slog::endl;
    } catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    } catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    return 0;
}
