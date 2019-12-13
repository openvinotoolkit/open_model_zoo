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
                                              std::vector<cv::Mat>>>& completedInferRequests
                         ):
                         ir(ir),
                         inputBlobImages(inputBlobImages),
                         completedInferRequests(completedInferRequests) {}

    void operator()() const {
        try {
            completedInferRequests.push({ir, inputBlobImages});
        }
        catch(...) {
            irCallbackException = std::current_exception();
        }
    }

private:
    InferenceEngine::InferRequest& ir;
    std::vector<cv::Mat> inputBlobImages;
    std::queue<std::pair<InferenceEngine::InferRequest&, std::vector<cv::Mat>>>& completedInferRequests;
};
