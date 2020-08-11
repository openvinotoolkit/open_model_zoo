#pragma once
#include "model.hpp"

class Ssd :public Model
{
private:
    
     int maxProposalCount;
     int objectSize;
     std::string imageInfoInputName;

public:

    Ssd(const InferenceEngine::Core &ie, std::string FLAGS_m);

    void prepareInputBlobs(bool FLAGS_auto_resize);
    void prepareOutputBlobs();
    void setConstInput(InferenceEngine::InferRequest::Ptr& inferReq, std::vector<InferenceEngine::InferRequest::Ptr>& userSpecifiedInferRequests);
    void processOutput(std::map< std::string, InferenceEngine::Blob::Ptr>& outputs, cv::Mat frame, bool FLAGS_r, double threshold);
        
   

 
  

   
};
