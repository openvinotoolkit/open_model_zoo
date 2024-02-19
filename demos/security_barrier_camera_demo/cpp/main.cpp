// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <list>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <set>

#include "openvino/openvino.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"

#include "monitors/presenter.h"
#include "utils/args_helper.hpp"
#include "utils/grid_mat.hpp"
#include "utils/input_wrappers.hpp"
#include "utils/ocv_common.hpp"
#include "utils/slog.hpp"
#include "utils/threads_common.hpp"

#include "net_wrappers.hpp"
#include "security_barrier_camera_demo.hpp"

typedef std::chrono::duration<float, std::chrono::seconds::period> Sec;

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // Parsing and validation of input args
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    if (FLAGS_m.empty()) {
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

struct BboxAndDescr {
    enum class ObjectType {
        NONE,
        VEHICLE,
        PLATE,
    } objectType;
    cv::Rect rect;
    std::string descr;
};

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

    ConcurrentContainer<std::vector<std::reference_wrapper<ov::InferRequest>>> inferRequests;

private:
    std::vector<ov::InferRequest> actualInferRequests;
};

// stores all global data for tasks
struct Context {
    Context(const std::vector<std::shared_ptr<InputChannel>>& inputChannels,
            const Detector& detector,
            const VehicleAttributesClassifier& vehicleAttributesClassifier, const Lpr& lpr,
            int pause, const std::vector<cv::Size>& gridParam, cv::Size displayResolution, std::chrono::steady_clock::duration showPeriod,
                const std::string& monitorsStr,
            uint64_t lastFrameId,
            uint64_t nireq,
            bool isVideo,
            std::size_t nclassifiersireq, std::size_t nrecognizersireq) :
        readersContext{inputChannels, std::vector<int64_t>(inputChannels.size(), -1), std::vector<std::mutex>(inputChannels.size())},
        inferTasksContext{detector},
        detectionsProcessorsContext{vehicleAttributesClassifier, lpr},
        drawersContext{pause, gridParam, displayResolution, showPeriod, monitorsStr},
        videoFramesContext{std::vector<uint64_t>(inputChannels.size(), lastFrameId), std::vector<std::mutex>(inputChannels.size())},
        nireq{nireq},
        isVideo{isVideo},
        freeDetectionInfersCount{0},
        frameCounter{0}
    {
        assert(inputChannels.size() == gridParam.size());
        std::vector<ov::InferRequest> detectorInferRequests;
        std::vector<ov::InferRequest> attributesInferRequests;
        std::vector<ov::InferRequest> lprInferRequests;
        detectorInferRequests.reserve(nireq);
        attributesInferRequests.reserve(nclassifiersireq);
        lprInferRequests.reserve(nrecognizersireq);
        std::generate_n(std::back_inserter(detectorInferRequests), nireq, [&]{
            return inferTasksContext.detector.createInferRequest();});
        std::generate_n(std::back_inserter(attributesInferRequests), nclassifiersireq, [&]{
            return detectionsProcessorsContext.vehicleAttributesClassifier.createInferRequest();});
        std::generate_n(std::back_inserter(lprInferRequests), nrecognizersireq, [&]{
            return detectionsProcessorsContext.lpr.createInferRequest();});
        detectorsInfers.assign(detectorInferRequests);
        attributesInfers.assign(attributesInferRequests);
        platesInfers.assign(lprInferRequests);
    }

    struct {
        std::vector<std::shared_ptr<InputChannel>> inputChannels;
        std::vector<int64_t> lastCapturedFrameIds;
        std::vector<std::mutex> lastCapturedFrameIdsMutexes;
        std::weak_ptr<Worker> readersWorker;
    } readersContext;

    struct {
        Detector detector;
        std::weak_ptr<Worker> inferTasksWorker;
    } inferTasksContext;

    struct {
        VehicleAttributesClassifier vehicleAttributesClassifier;
        Lpr lpr;
        std::weak_ptr<Worker> detectionsProcessorsWorker;
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
        std::chrono::steady_clock::duration showPeriod; // desired frequency of imshow
        std::weak_ptr<Worker> drawersWorker;
        int64_t lastShownframeId;
        std::chrono::steady_clock::time_point prevShow; // time stamp of previous imshow
        std::map<int64_t, GridMat> gridMats;
        std::mutex drawerMutex;
        Presenter presenter;
    } drawersContext;

    struct {
        std::vector<uint64_t> lastframeIds;
        std::vector<std::mutex> lastFrameIdsMutexes;
    } videoFramesContext;

    std::weak_ptr<Worker> resAggregatorsWorker;
    std::mutex classifiersAggregatorPrintMutex;
    uint64_t nireq;
    bool isVideo;
    std::atomic<std::vector<ov::InferRequest>::size_type> freeDetectionInfersCount;
    std::atomic<uint32_t> frameCounter;
    InferRequestsContainer detectorsInfers, attributesInfers, platesInfers;
    PerformanceMetrics metrics;
};

class ReborningVideoFrame: public VideoFrame {
public:
    ReborningVideoFrame(Context& context, const unsigned sourceID, const int64_t frameId, const cv::Mat& frame = cv::Mat()) :
        VideoFrame{sourceID, frameId, frame}, context(context) {}  // can not write context{context} because of CentOS 7.4 compiler bug
    virtual ~ReborningVideoFrame();
    Context& context;
};

// accumulates and shows processed frames
class Drawer : public Task {
public:
    explicit Drawer(VideoFrame::Ptr sharedVideoFrame) :
        Task{ sharedVideoFrame, 1.0 } {}
    bool isReady() override;
    void process() override;
};

// draws results on the frame
class ResAggregator : public Task {
public:
    ResAggregator(const VideoFrame::Ptr& sharedVideoFrame, std::list<BboxAndDescr>&& boxesAndDescrs):
        Task{sharedVideoFrame, 4.0}, boxesAndDescrs{std::move(boxesAndDescrs)} {}

    bool isReady() override {
        return true;
    }
    void process() override;
private:
    std::list<BboxAndDescr> boxesAndDescrs;
};

// waits for all classifiers and recognisers accumulating results
class ClassifiersAggregator {
public:
    std::vector<std::string> rawDetections;
    ConcurrentContainer<std::list<std::string>> rawAttributes;
    ConcurrentContainer<std::list<std::string>> rawDecodedPlates;

    explicit ClassifiersAggregator(const VideoFrame::Ptr& sharedVideoFrame):
        sharedVideoFrame{sharedVideoFrame} {}
    ~ClassifiersAggregator() {
        std::mutex& printMutex = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context.classifiersAggregatorPrintMutex;
        printMutex.lock();
        if (FLAGS_r && !rawDetections.empty()) {
            slog::debug << "Frame #: " << sharedVideoFrame->frameId << slog::endl;
            slog::debug << rawDetections;
            // destructor assures that none uses the container
            for (const std::string& rawAttribute : rawAttributes.container) {
                slog::debug << rawAttribute << slog::endl;
            }
            for (const std::string& rawDecodedPlate : rawDecodedPlates.container) {
                slog::debug << rawDecodedPlate << slog::endl;
            }
        }
        printMutex.unlock();
        tryPush(static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context.resAggregatorsWorker,
                std::make_shared<ResAggregator>(sharedVideoFrame, std::move(boxesAndDescrs)));
    }

    void push(BboxAndDescr&& bboxAndDescr) {
        boxesAndDescrs.lockedPushBack(std::move(bboxAndDescr));
    }

    const VideoFrame::Ptr sharedVideoFrame;

private:
    ConcurrentContainer<std::list<BboxAndDescr>> boxesAndDescrs;
};

// extracts detections from blob InferRequests and runs classifiers and recognisers
class DetectionsProcessor : public Task {
public:
    DetectionsProcessor(VideoFrame::Ptr sharedVideoFrame, ov::InferRequest* inferRequest) :
        Task{sharedVideoFrame, 1.0}, inferRequest{inferRequest}, requireGettingNumberOfDetections{true} {}

    DetectionsProcessor(VideoFrame::Ptr sharedVideoFrame, std::shared_ptr<ClassifiersAggregator>&& classifiersAggregator, std::list<cv::Rect>&& vehicleRects,
        std::list<cv::Rect>&& plateRects) :
            Task{sharedVideoFrame, 1.0}, classifiersAggregator{std::move(classifiersAggregator)}, inferRequest{nullptr},
            vehicleRects{std::move(vehicleRects)}, plateRects{std::move(plateRects)}, requireGettingNumberOfDetections{false} {}

    bool isReady() override;
    void process() override;

private:
    std::shared_ptr<ClassifiersAggregator> classifiersAggregator; // when no one stores this object we will draw
    ov::InferRequest* inferRequest;
    std::list<cv::Rect> vehicleRects;
    std::list<cv::Rect> plateRects;
    std::vector<std::reference_wrapper<ov::InferRequest>> reservedAttributesRequests;
    std::vector<std::reference_wrapper<ov::InferRequest>> reservedLprRequests;
    bool requireGettingNumberOfDetections;
};

// runs detection
class InferTask: public Task {
public:
    explicit InferTask(VideoFrame::Ptr sharedVideoFrame) :
        Task{sharedVideoFrame, 5.0} {}
    bool isReady() override;
    void process() override;
};

class Reader: public Task {
public:
    explicit Reader(VideoFrame::Ptr sharedVideoFrame) :
        Task{sharedVideoFrame, 2.0} {}
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
            else if (key == 32) {
                context.drawersContext.pause = (context.drawersContext.pause + 1) & 1;
            } else {
                context.drawersContext.presenter.handleKey(key);
            }
        } else {
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
    for (const BboxAndDescr& bboxAndDescr : boxesAndDescrs) {
        switch (bboxAndDescr.objectType) {
            case BboxAndDescr::ObjectType::NONE:
                cv::rectangle(sharedVideoFrame->frame, bboxAndDescr.rect, {255, 255, 0},  4);
                break;

            case BboxAndDescr::ObjectType::VEHICLE:
                cv::rectangle(sharedVideoFrame->frame, bboxAndDescr.rect, {0, 255, 0},  4);
                putHighlightedText(sharedVideoFrame->frame, bboxAndDescr.descr,
                    cv::Point{bboxAndDescr.rect.x, bboxAndDescr.rect.y + 35},
                    cv::FONT_HERSHEY_COMPLEX, 1.3, cv::Scalar(0, 255, 0), 2);
                break;

            case BboxAndDescr::ObjectType::PLATE:
                cv::rectangle(sharedVideoFrame->frame, bboxAndDescr.rect, {0, 0, 255},  4);
                putHighlightedText(sharedVideoFrame->frame, bboxAndDescr.descr,
                    cv::Point{bboxAndDescr.rect.x, bboxAndDescr.rect.y - 10},
                    cv::FONT_HERSHEY_COMPLEX, 1.3, cv::Scalar(0, 0, 255), 2);
                break;

            default:
                throw std::runtime_error("Unexpected detection result"); // must never happen
                break;
        }
    }
    tryPush(context.drawersContext.drawersWorker, std::make_shared<Drawer>(sharedVideoFrame));
}

bool DetectionsProcessor::isReady() {
    Context& context = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context;
    if (requireGettingNumberOfDetections) {
        classifiersAggregator = std::make_shared<ClassifiersAggregator>(sharedVideoFrame);
        std::list<Detector::Result> results;
        results = context.inferTasksContext.detector.getResults(*inferRequest, sharedVideoFrame->frame.size(), classifiersAggregator->rawDetections);
        for (Detector::Result result : results) {
            switch (result.label) {
                case 1:
                {
                    vehicleRects.emplace_back(result.location & cv::Rect{cv::Point(0, 0), sharedVideoFrame->frame.size()});
                    break;
                }
                case 2:
                {
                    // expanding a bounding box a bit, better for the license plate recognition
                    result.location.x -= 5;
                    result.location.y -= 5;
                    result.location.width += 10;
                    result.location.height += 10;
                    plateRects.emplace_back(result.location & cv::Rect{cv::Point(0, 0), sharedVideoFrame->frame.size()});
                    break;
                }
                default:
                    throw std::runtime_error("Unexpected detection results"); // must never happen
                    break;
            }
        }
        context.detectorsInfers.inferRequests.lockedPushBack(*inferRequest);
        requireGettingNumberOfDetections = false;
    }

    if ((vehicleRects.empty() || FLAGS_m_va.empty()) && (plateRects.empty() || FLAGS_m_lpr.empty())) {
        return true;
    } else {
        // isReady() is called under mutexes so it is assured that available InferRequests will not be taken, but new InferRequests can come in
        // acquire as many InferRequests as it is possible or needed
        InferRequestsContainer& attributesInfers = context.attributesInfers;
        attributesInfers.inferRequests.mutex.lock();
        const std::size_t numberOfAttributesInferRequestsAcquired = std::min(vehicleRects.size(), attributesInfers.inferRequests.container.size());
        reservedAttributesRequests.assign(attributesInfers.inferRequests.container.end() - numberOfAttributesInferRequestsAcquired,
                                          attributesInfers.inferRequests.container.end());
        attributesInfers.inferRequests.container.erase(attributesInfers.inferRequests.container.end() - numberOfAttributesInferRequestsAcquired,
                                                       attributesInfers.inferRequests.container.end());
        attributesInfers.inferRequests.mutex.unlock();

        InferRequestsContainer& platesInfers = context.platesInfers;
        platesInfers.inferRequests.mutex.lock();
        const std::size_t numberOfLprInferRequestsAcquired = std::min(plateRects.size(), platesInfers.inferRequests.container.size());
        reservedLprRequests.assign(platesInfers.inferRequests.container.end() - numberOfLprInferRequestsAcquired, platesInfers.inferRequests.container.end());
        platesInfers.inferRequests.container.erase(platesInfers.inferRequests.container.end() - numberOfLprInferRequestsAcquired,
                                                   platesInfers.inferRequests.container.end());
        platesInfers.inferRequests.mutex.unlock();
        return numberOfAttributesInferRequestsAcquired || numberOfLprInferRequestsAcquired;
    }
}

void DetectionsProcessor::process() {
    Context& context = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context;
    if (!FLAGS_m_va.empty()) {
        auto vehicleRectsIt = vehicleRects.begin();
        for (auto attributesRequestIt = reservedAttributesRequests.begin(); attributesRequestIt != reservedAttributesRequests.end();
                vehicleRectsIt++, attributesRequestIt++) {
            const cv::Rect vehicleRect = *vehicleRectsIt;
            ov::InferRequest& attributesRequest = *attributesRequestIt;
            context.detectionsProcessorsContext.vehicleAttributesClassifier.setImage(attributesRequest, sharedVideoFrame->frame, vehicleRect);

            attributesRequest.set_callback(
                std::bind(
                    [](std::shared_ptr<ClassifiersAggregator> classifiersAggregator,
                        ov::InferRequest& attributesRequest,
                        cv::Rect rect,
                        Context& context) {
                            attributesRequest.set_callback([](std::exception_ptr) {}); // destroy the stored bind object

                            const std::pair<std::string, std::string>& attributes =
                                context.detectionsProcessorsContext.vehicleAttributesClassifier.getResults(attributesRequest);

                            if (FLAGS_r && ((classifiersAggregator->sharedVideoFrame->frameId == 0 && !context.isVideo) || context.isVideo)) {
                                classifiersAggregator->rawAttributes.lockedPushBack(
                                    "Vehicle Attributes results:" + attributes.first + ';' + attributes.second);
                            }
                            classifiersAggregator->push(
                                BboxAndDescr{BboxAndDescr::ObjectType::VEHICLE, rect, attributes.first + ' ' + attributes.second});
                            context.attributesInfers.inferRequests.lockedPushBack(attributesRequest);
                        }, classifiersAggregator,
                           std::ref(attributesRequest),
                           vehicleRect,
                           std::ref(context)));
            attributesRequest.start_async();
        }
        vehicleRects.erase(vehicleRects.begin(), vehicleRectsIt);
    } else {
        for (const cv::Rect vehicleRect : vehicleRects) {
            classifiersAggregator->push(BboxAndDescr{BboxAndDescr::ObjectType::NONE, vehicleRect, ""});
        }
        vehicleRects.clear();
    }

    if (!FLAGS_m_lpr.empty()) {
        auto plateRectsIt = plateRects.begin();
        for (auto lprRequestsIt = reservedLprRequests.begin(); lprRequestsIt != reservedLprRequests.end(); plateRectsIt++, lprRequestsIt++) {
            const cv::Rect plateRect = *plateRectsIt;
            ov::InferRequest& lprRequest = *lprRequestsIt;
            context.detectionsProcessorsContext.lpr.setImage(lprRequest, sharedVideoFrame->frame, plateRect);

            lprRequest.set_callback(
                std::bind(
                    [](std::shared_ptr<ClassifiersAggregator> classifiersAggregator,
                        ov::InferRequest& lprRequest,
                        cv::Rect rect,
                        Context& context) {
                            lprRequest.set_callback([](std::exception_ptr) {}); // destroy the stored bind object

                            std::string result = context.detectionsProcessorsContext.lpr.getResults(lprRequest);

                            if (FLAGS_r && ((classifiersAggregator->sharedVideoFrame->frameId == 0 && !context.isVideo) || context.isVideo)) {
                                classifiersAggregator->rawDecodedPlates.lockedPushBack("License Plate Recognition results:" + result);
                            }
                            classifiersAggregator->push(BboxAndDescr{BboxAndDescr::ObjectType::PLATE, rect, std::move(result)});
                            context.platesInfers.inferRequests.lockedPushBack(lprRequest);
                        }, classifiersAggregator,
                           std::ref(lprRequest),
                           plateRect,
                           std::ref(context)));

            lprRequest.start_async();
        }
        plateRects.erase(plateRects.begin(), plateRectsIt);
    } else {
        for (const cv::Rect& plateRect : plateRects) {
            classifiersAggregator->push(BboxAndDescr{BboxAndDescr::ObjectType::NONE, plateRect, ""});
        }
        plateRects.clear();
    }
    if (!vehicleRects.empty() || !plateRects.empty()) {
        tryPush(context.detectionsProcessorsContext.detectionsProcessorsWorker,
            std::make_shared<DetectionsProcessor>(sharedVideoFrame, std::move(classifiersAggregator), std::move(vehicleRects), std::move(plateRects)));
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
                    inferRequest.set_callback([](std::exception_ptr) {}); // destroy the stored bind object
                    tryPush(context.detectionsProcessorsContext.detectionsProcessorsWorker,
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
        // Parsing and validation of input args
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
        if (files.empty() && 0 == FLAGS_nc)
            throw std::logic_error("No inputs were found");

        std::vector<std::shared_ptr<VideoCaptureSource>> videoCapturSourcess;
        std::vector<std::shared_ptr<ImageSource>> imageSourcess;

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
                videoCapturSourcess.push_back(std::make_shared<VideoCaptureSource>(videoCapture, FLAGS_loop_video));
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
                videoCapturSourcess.push_back(std::make_shared<VideoCaptureSource>(videoCapture, FLAGS_loop_video));
            } else {
                imageSourcess.push_back(std::make_shared<ImageSource>(frame, true));
            }
        }
        uint32_t channelsNum = 0 == FLAGS_ni ? videoCapturSourcess.size() + imageSourcess.size() : FLAGS_ni;
        std::vector<std::shared_ptr<IInputSource>> inputSources;
        inputSources.reserve(videoCapturSourcess.size() + imageSourcess.size());
        for (const std::shared_ptr<VideoCaptureSource>& videoSource : videoCapturSourcess) {
            inputSources.push_back(videoSource);
        }
        for (const std::shared_ptr<ImageSource>& imageSource : imageSourcess) {
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

        // Load OpenVINO Runtime
        slog::info << ov::get_openvino_version() << slog::endl;
        ov::Core core;

        std::set<std::string> devices;
        for (const std::string& netDevices : {FLAGS_d, FLAGS_d_va, FLAGS_d_lpr}) {
            if (netDevices.empty()) {
                continue;
            }
            for (const std::string& device : parseDevices(netDevices)) {
                devices.insert(device);
            }
        }
        std::map<std::string, int32_t> device_nstreams = parseValuePerDevice(devices, FLAGS_nstreams);

        for (const std::string& device : devices) {
            if ("CPU" == device) {
                if (FLAGS_nthreads != 0) {
                    core.set_property("CPU", ov::inference_num_threads(FLAGS_nthreads));
                }
                core.set_property("CPU", ov::affinity(ov::Affinity::NONE));
                core.set_property("CPU", ov::streams::num((device_nstreams.count("CPU") > 0 ? ov::streams::Num(device_nstreams["CPU"]) : ov::streams::AUTO)));

                device_nstreams["CPU"] = core.get_property("CPU", ov::streams::num);
            }

            if ("GPU" == device) {
                core.set_property("GPU", ov::streams::num(device_nstreams.count("GPU") > 0 ? ov::streams::Num(device_nstreams["GPU"]) : ov::streams::AUTO));

                device_nstreams["GPU"] = core.get_property("GPU", ov::streams::num);
                if (devices.end() != devices.find("CPU")) {
                    // multi-device execution with the CPU + GPU performs best with GPU trottling hint,
                    // which releases another CPU thread (that is otherwise used by the GPU driver for active polling)
                    core.set_property("GPU", ov::intel_gpu::hint::queue_throttle(ov::intel_gpu::hint::ThrottleLevel(1)));
                }
            }
        }

        unsigned nireq = FLAGS_nireq == 0 ? inputChannels.size() : FLAGS_nireq;

        Detector detector(core, FLAGS_d, FLAGS_m,
            {static_cast<float>(FLAGS_t), static_cast<float>(FLAGS_t)}, FLAGS_auto_resize);
        slog::info << "\tNumber of network inference requests: " << nireq << slog::endl;

        VehicleAttributesClassifier vehicleAttributesClassifier;
        std::size_t nclassifiersireq{0};

        Lpr lpr;
        std::size_t nrecognizersireq{0};

        if (!FLAGS_m_va.empty()) {
            vehicleAttributesClassifier = VehicleAttributesClassifier(core, FLAGS_d_va, FLAGS_m_va, FLAGS_auto_resize);
            nclassifiersireq = nireq * 3;
            slog::info << "\tNumber of network inference requests: " << nclassifiersireq << slog::endl;
        } else {
            slog::info << "Vehicle Attributes Recognition DISABLED." << slog::endl;
        }

        if (!FLAGS_m_lpr.empty()) {
            lpr = Lpr(core, FLAGS_d_lpr, FLAGS_m_lpr, FLAGS_auto_resize);
            nrecognizersireq = nireq * 3;
            slog::info << "\tNumber of network inference requests: " << nrecognizersireq << slog::endl;
        } else {
            slog::info << "License Plate Recognition DISABLED." << slog::endl;
        }

        bool isVideo = imageSourcess.empty() ? true : false;
        int pause = imageSourcess.empty() ? 1 : 0;
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
                        vehicleAttributesClassifier, lpr,
                        pause, gridParam, displayResolution, showPeriod, FLAGS_u,
                        FLAGS_n_iqs - 1,
                        nireq,
                        isVideo,
                        nclassifiersireq, nrecognizersireq};
        // Create a worker after a context because the context has only weak_ptr<Worker>, but the worker is going to
        // indirectly store ReborningVideoFrames which have a reference to the context. So there won't be a situation
        // when the context is destroyed and the worker still lives with its ReborningVideoFrames referring to the
        // destroyed context.
        std::shared_ptr<Worker> worker = std::make_shared<Worker>(FLAGS_n_wt - 1);
        context.readersContext.readersWorker = worker;
        context.inferTasksContext.inferTasksWorker = worker;
        context.detectionsProcessorsContext.detectionsProcessorsWorker = worker;
        context.drawersContext.drawersWorker = worker;
        context.resAggregatorsWorker = worker;

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
