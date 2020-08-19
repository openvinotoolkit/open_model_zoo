#pragma once
#include "model.hpp"
#include <ngraph/ngraph.hpp>


class Retinaface :public Model
{
public:
    Retinaface(const InferenceEngine::Core& ie, std::string networkModel);
    void prepareInputBlobs(bool autoResize);
    void prepareOutputBlobs();
    void setConstInput(InferenceEngine::InferRequest::Ptr& inferReq);
    void processOutput(std::map< std::string, InferenceEngine::Blob::Ptr>& outputs, cv::Mat frame, bool printOutput, double threshold);
   
};
