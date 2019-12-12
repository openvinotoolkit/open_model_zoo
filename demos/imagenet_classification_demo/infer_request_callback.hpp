#include <cstdio>
#include <string>
#include <functional>
#include <atomic>

#include <inference_engine.hpp>

#include <ie_iextension.h>

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
        completedInferRequests.push({ir, inputBlobImages});
    }

private:
    InferenceEngine::InferRequest& ir;
    std::vector<cv::Mat> inputBlobImages;
    std::queue<std::pair<InferenceEngine::InferRequest&, std::vector<cv::Mat>>>& completedInferRequests;
};
