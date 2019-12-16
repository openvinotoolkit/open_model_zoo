#include <cstdio>
#include <string>
#include <functional>
#include <atomic>

#include <inference_engine.hpp>

#include <ie_iextension.h>

std::exception_ptr irCallbackException;

class InferRequestCallback
{
public:
    InferRequestCallback(InferenceEngine::InferRequest& ir,
                         std::vector<cv::Mat> inputBlobImages,
                         std::queue<std::pair<InferenceEngine::InferRequest&,
                                              std::vector<cv::Mat>>>& completedInferRequests,
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

                completedInferRequests.push({ir, inputBlobImages});
            }
            condVar.notify_one();
        }
        catch(...) {
            irCallbackException = std::current_exception();
        }
    }

private:
    InferenceEngine::InferRequest& ir;
    std::vector<cv::Mat> inputBlobImages;
    std::queue<std::pair<InferenceEngine::InferRequest&, std::vector<cv::Mat>>>& completedInferRequests;
    std::mutex& mutex;
    std::condition_variable& condVar;
};
