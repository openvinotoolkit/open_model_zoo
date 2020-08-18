#pragma once
#include "model.hpp"
#include <ngraph/ngraph.hpp>

struct DetectionObject {
    int xmin, ymin, xmax, ymax, class_id;
    float confidence;

    DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale) {
        this->xmin = static_cast<int>((x - w / 2) * w_scale);
        this->ymin = static_cast<int>((y - h / 2) * h_scale);
        this->xmax = static_cast<int>(this->xmin + w * w_scale);
        this->ymax = static_cast<int>(this->ymin + h * h_scale);
        this->class_id = class_id;
        this->confidence = confidence;
    }

    bool operator <(const DetectionObject& s2) const {
        return this->confidence < s2.confidence;
    }
    bool operator >(const DetectionObject& s2) const {
        return this->confidence > s2.confidence;
    }
};

class Params {
    template <typename T>
    void computeAnchors(const std::vector<T>& mask);

public:
    int num = 0, classes = 0, coords = 0;
    std::vector<float> anchors = { 10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0,
                                  156.0, 198.0, 373.0, 326.0 };

    Params() {}

    Params(const std::shared_ptr<ngraph::op::RegionYolo> regionYolo);

    friend class Yolov3;
};

class Yolov3 :public Model
{
public:
    Yolov3(const InferenceEngine::Core& ie, std::string networkModel);
    void prepareInputBlobs(bool autoResize);
    void prepareOutputBlobs();
    void setConstInput(InferenceEngine::InferRequest::Ptr& inferReq);
    void processOutput(std::map< std::string, InferenceEngine::Blob::Ptr>& outputs, cv::Mat frame, bool printOutput, double threshold);
    void parseYOLOV3Output( const std::string& output_name,
        const InferenceEngine::Blob::Ptr& blob, const unsigned long resized_im_h,
        const unsigned long resized_im_w, const unsigned long original_im_h,
        const unsigned long original_im_w,
        const double threshold, std::vector<DetectionObject>& objects);
private:
    std::map<std::string, Params*> params;
};
