#include "pipeline_base.h"
#include "opencv2/core.hpp"
#pragma once
class DetectionPipeline :
    public PipelineBase
{
public:
    struct ObjectDesc : public cv::Rect2f
    {
        unsigned int labelID;
        std::string label;
        float confidence;
    };

    struct DetectionResults
    {
        int64_t frameId=-1;
        std::vector<ObjectDesc> objects;

        bool IsEmpty() { return frameId < 0; }
    };

public:
    DetectionPipeline();
    virtual ~DetectionPipeline();

    /// Loads model and performs required initialization
    /// @param model_name name of model to load
    virtual void init(const std::string& model_name, const CnnConfig& config,
        float confidenceThreshold, bool useAutoResize);

    virtual void PrepareInputsOutputs(InferenceEngine::CNNNetwork & cnnNetwork);

    int64_t submitImage(cv::Mat img);
    DetectionResults getDetectionResults();

    void loadLabels(const std::string& labelFilename);
    std::vector<std::string> labels;

protected:

    std::string imageInputName;
    std::string imageInfoInputName;
    size_t netInputHeight=0;
    size_t netInputWidth=0;

    bool useAutoResize=false;
    size_t maxProposalCount=0;
    size_t objectSize=0;
    float confidenceThreshold=0;
};

