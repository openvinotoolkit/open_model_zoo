#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>

#include <ie_core.hpp>
//#include <ngraph/ngraph.hpp>

//#include <monitors/presenter.h>
#include <samples/ocv_common.hpp>

#include <cldnn/cldnn_config.hpp>


//#include "object_detection_demo_ssd_async.hpp"
class Model
{
protected:
	InferenceEngine::CNNNetwork cnnNetwork;
	std::map<std::string, InferenceEngine::SizeVector > inputs;
	std::map<std::string, InferenceEngine::DataPtr& > outputs;
    std::size_t inputHeight;
    std::size_t inputWidth;
    std::vector<std::string>outputsNames;
    std::string imageInputName;// input contains images for function frameToBlob
    std::vector<std::string> labels;
	

public:
    Model(const InferenceEngine::Core &ie, std::string FLAGS_m);
    void loadLables(std::string FLAGS_labels);
	virtual void prepareInputBlobs(bool FLAGS_auto_resize) = 0;
	virtual void prepareOutputBlobs() = 0;
    virtual void setConstInput(InferenceEngine::InferRequest::Ptr& inferReq, std::vector<InferenceEngine::InferRequest::Ptr>& userSpecifiedInferRequests) = 0;
    virtual void processOutput(std::map< std::string, InferenceEngine::Blob::Ptr>& outputs,
                               cv::Mat frame, bool FLAGS_r, double threshold)=0;

    InferenceEngine::CNNNetwork getCnnNetwork() const;
    std::size_t getInputHeight() const;
    std::size_t getInputWidth() const;
    const std::map<std::string, InferenceEngine::SizeVector >& getInputs() const;
    const std::map< std::string, InferenceEngine::DataPtr&>& getOutputs()const;
    const std::vector< std::string>& getOutputsNames()const;
    std::string getImageInputName() const;
};
