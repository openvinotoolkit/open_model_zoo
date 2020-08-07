#pragma once
#include "models.hpp"

class Ssd :public Model
{
private:
    
     int maxProposalCount;
     int objectSize;

public:

    Ssd(const InferenceEngine::Core &ie, std::string FLAGS_m);

    void prepareInputBlobs();
    void prepareOutputBlobs();
    void setConstInput(InferenceEngine::InferRequest::Ptr& inferReq);


    void processOutput(std::map<const std::string, InferenceEngine::Blob::Ptr>& outputs, cv::Mat frame, const size_t width, const size_t height, std::vector<std::string>& labels, bool FLAGS_r, double threshold);
        
    const int getMaxProposalCount();

    const int getObjectSize();

 
  

   
};
