#include "utils/uni_image.h"
#include <utils/ocv_common.hpp>
#include <opencv2/imgproc.hpp>
#include <utils/slog.hpp>
#include <mutex>

const cv::Mat UniImageMat::toMat(IMG_CONVERSION_TYPE convType) {
    if(convType==CONVERT_TO_RGB) {
        cv::Mat retVal;
        cv::cvtColor(mat, retVal, cv::COLOR_BGR2RGB);
        return retVal;
    }
    return mat;
}

InferenceEngine::Blob::Ptr UniImageMat::toBlob(bool isNHWCModelInput) {
    return wrapMat2Blob(mat, isNHWCModelInput);
}

UniImage::Ptr UniImageMat::resize(int width, int height, IMG_RESIZE_MODE resizeMode, bool hqResize) {
    if (width == mat.cols && height == mat.rows) {
        return std::make_shared<UniImageMat>(mat.clone());
    }

    auto dst = std::shared_ptr<UniImageMat>(new UniImageMat());
    int interpMode = hqResize ? cv::INTER_LINEAR : cv::INTER_CUBIC;

    switch (resizeMode) {
    case RESIZE_FILL:
    {
        cv::resize(mat, dst->mat, cv::Size(width, height), interpMode);
        dst->roi = cv::Rect(0, 0, width, height);
        break;
    }
    case RESIZE_KEEP_ASPECT:
    {
        double scale = std::max(static_cast<double>(width) / mat.cols, static_cast<double>(height) / mat.rows);
        int newW = static_cast<int>(mat.cols * scale);
        int newH = static_cast<int>(mat.rows * scale);
        cv::Mat resizedImage;
        cv::resize(mat, resizedImage, cv::Size(0, 0), scale, scale, interpMode);
        cv::copyMakeBorder(resizedImage, dst->mat, 0, height - newH,
            0, width - newW, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        dst->roi = cv::Rect(0, 0, newW, newH);
        break;
    }
    case RESIZE_KEEP_ASPECT_LETTERBOX:
    {
        double scale = std::min(static_cast<double>(width) / mat.cols, static_cast<double>(height) / mat.rows);
        int newW = static_cast<int>(mat.cols * scale);
        int newH = static_cast<int>(mat.rows * scale);
        cv::Mat resizedImage;
        int dx = (width - newW) / 2;
        int dy = (height - newH) / 2;
        cv::resize(mat, resizedImage, cv::Size(0, 0), scale, scale, interpMode);
        cv::copyMakeBorder(resizedImage, dst->mat, dy, dy,
            dx, dx, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        dst->roi = cv::Rect(dx, dy, newW, newH);
        break;
    }
    }
    return dst;
}

#ifdef USE_VA
UniImageVA::UniImageVA(const InferenceBackend::VaApiImage::Ptr& vaImg, InferenceBackend::VaApiContext::Ptr context) {
    img = (context && vaImg->context->display() != context->display()) ? vaImg->cloneToAnotherContext(context) : vaImg;
    roi = cv::Rect(0,0,img->width,img->height);
}

const cv::Mat UniImageVA::toMat(IMG_CONVERSION_TYPE convType) {
    return img->copyToMat((IMG_CONVERSION_TYPE)convType);
}

InferenceEngine::Blob::Ptr UniImageVA::toBlob(bool isNHWCModelInput) {
    if(isNHWCModelInput) {
        slog::warn << "VA Image use NV12 conversion, so NHWC layout parameter will be ignored";
    }
    if(!img->context->sharedContext()) {
        throw std::runtime_error("toBlob: shared context is not initialized, cannot share VA surface with blob");
    }

    return InferenceEngine::gpu::make_shared_blob_nv12(img->height, img->width, img->context->sharedContext(), img->va_surface_id);
}

UniImage::Ptr UniImageVA::resize(int width, int height, IMG_RESIZE_MODE resizeMode, bool hqResize) {
    // TODO: this is dumy code, image should be taken from pool
    return std::make_shared<UniImageVA>(img->resizeUsingPooledSurface((uint16_t)width, (uint16_t)height, resizeMode, hqResize));
}
#endif
