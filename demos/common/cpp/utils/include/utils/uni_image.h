#pragma once
#include "opencv2/core.hpp"
#include "inference_engine.hpp"
#include "utils/uni_image_defs.h"
#include "vaapi_context.h"

class UniImage {
public:
    using Ptr = std::shared_ptr<UniImage>;

    virtual ~UniImage(){};
    virtual const cv::Mat toMat(IMG_CONVERSION_TYPE convType = CONVERT_TO_BGR) = 0;
    virtual InferenceEngine::Blob::Ptr toBlob(bool isNHWCModelInput = false) = 0;
    virtual UniImage::Ptr resize(int width, int height, IMG_RESIZE_MODE resizeMode = RESIZE_FILL, bool hqResize = false) = 0;
    virtual cv::Size size() = 0;
    virtual cv::Rect getRoi(){ return roi;};
protected:
    cv::Rect roi;
};

class UniImageMat : public UniImage {
public:
    UniImageMat(const cv::Mat& mat) : mat(mat) {roi = cv::Rect(cv::Point(0,0), mat.size());}
    const cv::Mat toMat(IMG_CONVERSION_TYPE convType = CONVERT_TO_BGR) override;
    InferenceEngine::Blob::Ptr toBlob(bool isNHWCModelInput = false) override;
    UniImage::Ptr resize(int width, int height, IMG_RESIZE_MODE resizeMode = RESIZE_FILL, bool hqResize = false) override;
    cv::Size size() override { return mat.size(); }
protected:
    cv::Mat mat;
private:
    UniImageMat(){}
};

inline UniImage::Ptr mat2Img(const cv::Mat& mat) {
    return std::make_shared<UniImageMat>(mat);
};

#ifdef USE_VA
#include "vaapi_images.h"
class UniImageVA : public UniImage {
public:
    UniImageVA(const InferenceBackend::VaApiImage::Ptr& vaImg, InferenceBackend::VaApiContext::Ptr context = nullptr);
    const cv::Mat toMat(IMG_CONVERSION_TYPE convType = CONVERT_TO_BGR) override;
    InferenceEngine::Blob::Ptr toBlob(bool isNHWCModelInput = false) override;
    UniImage::Ptr resize(int width, int height, IMG_RESIZE_MODE resizeMode = RESIZE_FILL, bool hqResize = false) override;
    cv::Size size() override { return cv::Size(img->width,img->height); }
protected:
    InferenceBackend::VaApiImage::Ptr img;
    InferenceBackend::VaApiImage::Ptr getVaImageFromPool(const InferenceBackend::VaApiContext::Ptr& context, int width, int height);
private:
    UniImageVA(){}
};

inline UniImage::Ptr VA2Img(const InferenceBackend::VaApiImage::Ptr& img, InferenceBackend::VaApiContext::Ptr context = nullptr) {
    return std::make_shared<UniImageVA>(img,context);
};

#endif