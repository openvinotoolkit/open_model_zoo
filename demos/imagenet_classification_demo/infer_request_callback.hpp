#include <cstdio>
#include <string>
#include <functional>
#include <atomic>

#include <inference_engine.hpp>

#include <ie_iextension.h>

using namespace InferenceEngine;

std::exception_ptr irCallbackException;

class IRInfo {
public:
    class IRImage {
    public:
        cv::Mat mat;
        unsigned rightClass;
        long long startTime;

        IRImage(cv::Mat &mat, unsigned rightClass, long long startTime):
            mat(mat), rightClass(rightClass), startTime(startTime) {}
    };

    InferRequest &ir;
    std::vector<IRImage> images;

    IRInfo(InferRequest &ir, std::vector<IRImage> images): ir(ir), images(images) {}
};

class InferRequestCallback
{
public:
    InferRequestCallback(InferRequest& ir,
                         std::vector<IRInfo::IRImage> inputBlobImages,
                         std::queue<IRInfo>& completedInferRequests,
                         std::mutex& mutex,
                         std::condition_variable& condVar
                         ):
                         ir(ir),
                         inputBlobImages(inputBlobImages),
                         completedInferRequests(completedInferRequests),
                         mutex(mutex),
                         condVar(condVar) {}

    void operator()() const {
        try {
            {
                std::lock_guard<std::mutex> lock(mutex);

                completedInferRequests.push(IRInfo(ir, inputBlobImages));
            }
            condVar.notify_one();
        }
        catch(...) {
            irCallbackException = std::current_exception();
        }
    }

private:
    InferRequest& ir;
    std::vector<IRInfo::IRImage> inputBlobImages;
    std::queue<IRInfo>& completedInferRequests;
    std::mutex& mutex;
    std::condition_variable& condVar;
};
