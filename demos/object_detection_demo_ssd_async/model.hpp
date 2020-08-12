#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>

#include <ie_core.hpp>
#include <samples/ocv_common.hpp>
#include <cldnn/cldnn_config.hpp>



class Model
{
public:
    Model(const InferenceEngine::Core &ie, std::string networkModel);
    void loadLables(std::string labelsFile);
	virtual void prepareInputBlobs(bool autoResize) = 0;
	virtual void prepareOutputBlobs() = 0;
    virtual void setConstInput(InferenceEngine::InferRequest::Ptr& inferReq) = 0;
    virtual void processOutput(std::map< std::string, InferenceEngine::Blob::Ptr>& outputs,
                               cv::Mat frame, bool printOutput, double threshold)=0;

    InferenceEngine::CNNNetwork getCnnNetwork() const;
    std::size_t getInputHeight() const;
    std::size_t getInputWidth() const;
    const std::map<std::string, InferenceEngine::SizeVector >& getInputs() const;
    const std::map< std::string, InferenceEngine::DataPtr&>& getOutputs()const;
    const std::vector< std::string>& getOutputsNames()const;
    std::string getImageInputName() const;

protected:
    InferenceEngine::CNNNetwork cnnNetwork;
    std::map<std::string, InferenceEngine::SizeVector > inputs;
    std::map<std::string, InferenceEngine::DataPtr& > outputs;
    std::size_t inputHeight;
    std::size_t inputWidth;
    std::vector<std::string>outputsNames;
    std::string imageInputName;// input contains images for function frameToBlob
    std::vector<std::string> labels;
};
