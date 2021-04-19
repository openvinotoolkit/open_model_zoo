#pragma once
#include "opencv2/core.hpp"
#include "inference_engine.hpp"

class UniImage {
public:
    enum RESIZE_MODE {
        RESIZE_FILL,
        RESIZE_KEEP_ASPECT,
        RESIZE_KEEP_ASPECT_LETTERBOX
    };

    using Ptr = std::shared_ptr<UniImage>;

    UniImage();
    virtual ~UniImage();
    virtual const cv::Mat toMat() = 0;
    virtual InferenceEngine::Blob::Ptr toBlob() = 0;
    virtual UniImage::Ptr resize(int width, int height, RESIZE_MODE resizeMode = RESIZE_FILL, bool hqResize = false, cv::Rect* dataRect = nullptr) = 0;
    virtual cv::Size size() = 0;
};

class UniImageMat : public UniImage {
public:
    UniImageMat();
    UniImageMat(const cv::Mat& mat);
    ~UniImageMat() override;
    const cv::Mat toMat() override;
    InferenceEngine::Blob::Ptr toBlob() override;
    UniImage::Ptr resize(int width, int height, RESIZE_MODE resizeMode, bool hqResize, cv::Rect* dataRect) override;
    cv::Size size() override { return mat.size(); }
protected:
    cv::Mat mat;
};
