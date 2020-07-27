#pragma once
#include <string>
#include <deque>
#include <ngraph/ngraph.hpp>
#include <samples/ocv_common.hpp>
#include <map>
#include <condition_variable>
#include "config_factory.h"

/// This is base class for asynchronous pipeline
/// Derived classes should add functions for data submission and output processing
class PipelineBase
{
public:
    struct RequestResult {
        int64_t frameId;
        InferenceEngine::MemoryBlob::Ptr output=nullptr;
        std::chrono::steady_clock::time_point startTime;

        bool IsEmpty() { return output == nullptr; }
    };

    struct PerformanceInfo
    {
        int64_t framesCount = 0;
        std::chrono::steady_clock::duration latencySum;
        std::chrono::steady_clock::time_point startTime;
        double FPS=0;
    };

public:
    PipelineBase();
    virtual ~PipelineBase();

    /// Loads model and performs required initialization
    /// @param model_name name of model to load
    virtual void init(const std::string& model_name, const CnnConfig& cnnConfig);

    /// Waits until output data becomes available
    ///
    void waitForData();

    PerformanceInfo getPerformanceInfo() { std::lock_guard<std::mutex> lock(mtx); return perfInfo; }

    void waitForCompletion();

protected:
    std::map<InferenceEngine::InferRequest::Ptr, std::atomic_bool> requestsPool;
    std::unordered_map<int64_t, RequestResult> completedRequestResults;

    InferenceEngine::ExecutableNetwork execNetwork;

    PerformanceInfo perfInfo;

    std::mutex mtx;
    std::condition_variable condVar;

    int64_t inputFrameId;
    int64_t outputFrameId;
    std::string outputName;

    std::exception_ptr callbackException = nullptr;

    virtual void PrepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {}

    InferenceEngine::InferRequest::Ptr getIdleRequest();
    virtual int64_t submitRequest(InferenceEngine::InferRequest::Ptr request);
    virtual RequestResult getResult();

    bool isRequestsPoolEmpty(){
        return std::find_if(requestsPool.begin(), requestsPool.end(), [](std::pair<const InferenceEngine::InferRequest::Ptr,std::atomic_bool>& x) {return !x.second; })==requestsPool.end();
    }

    void setRequestIdle(const InferenceEngine::InferRequest::Ptr& request) {
        this->requestsPool.at(request) = false;
    }
};

