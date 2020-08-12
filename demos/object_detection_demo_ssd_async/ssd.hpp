#pragma once
#include "model.hpp"

class Ssd :public Model
{

public:
    Ssd(const InferenceEngine::Core &ie, std::string networkModel);
    void prepareInputBlobs(bool autoResize);
    void prepareOutputBlobs();
    void setConstInput(InferenceEngine::InferRequest::Ptr& inferReq);
    void processOutput(std::map< std::string, InferenceEngine::Blob::Ptr>& outputs, cv::Mat frame, bool printOutput, double threshold);
private:
    int maxProposalCount;
    int objectSize;
    std::string imageInfoInputName;
};
