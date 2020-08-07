#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>

#include <ie_core.hpp>
#include <ngraph/ngraph.hpp>

#include <monitors/presenter.h>
#include <samples/ocv_common.hpp>
#include <samples/args_helper.hpp>
#include <cldnn/cldnn_config.hpp>


#include "object_detection_demo_ssd_async.hpp"
class Model
{
protected:
	InferenceEngine::CNNNetwork cnnNetwork;
	std::map<std::string, InferenceEngine::SizeVector > inputs;
	std::map<const std::string, InferenceEngine::DataPtr& > outputs;
    std::size_t inputHeight;
    std::size_t inputWidth;
    std::vector<const std::string>outputsNames;
    std::string imageInfoInputName;//нужно будет что-то сделать с этим полем
	//std vector<std::string> outputs;

public:
    Model(const InferenceEngine::Core &ie, std::string FLAGS_m);
    InferenceEngine::CNNNetwork getCnnNetwork();

	virtual void prepareInputBlobs() = 0;
	virtual void prepareOutputBlobs() = 0;
    virtual void setConstInput(InferenceEngine::InferRequest::Ptr& inferReq) = 0;
    virtual void processOutput(std::map<const std::string, InferenceEngine::Blob::Ptr>& outputs, cv::Mat frame, const size_t width, const size_t height, std::vector<std::string>& labels, bool FLAGS_r, double threshold);

    std::size_t getInputHeight();
    std::size_t getInputWidth();
    std::map<std::string, InferenceEngine::SizeVector >& getInputs();
    std::map<const std::string, InferenceEngine::DataPtr&>& getOutputs();
    std::vector<const std::string>& getOutputsNames();
    std::string getImageInfoInputName();
	
};
