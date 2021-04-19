#include "utils/uni_image.h"
#include <utils/ocv_common.hpp>
#include <opencv2/imgproc.hpp>

UniImage::UniImage() {
}

UniImage::~UniImage() {
}

UniImageMat::UniImageMat() {
}

UniImageMat::UniImageMat(const cv::Mat& mat) :
    mat(mat) {
}

UniImageMat::~UniImageMat() {
}

const cv::Mat UniImageMat::toMat() {
    return mat;
}

InferenceEngine::Blob::Ptr UniImageMat::toBlob() {
    return wrapMat2Blob(mat);
}

UniImage::Ptr UniImageMat::resize(int width, int height, RESIZE_MODE resizeMode, bool hqResize, cv::Rect* dataRect) {
    if (width == mat.cols && height == mat.rows) {
        return std::make_shared<UniImageMat>(mat.clone());
    }

    auto dst = std::make_shared<UniImageMat>();
    int interpMode = hqResize ? cv::INTER_LINEAR : cv::INTER_CUBIC;

    switch (resizeMode) {
    case RESIZE_FILL:
    {
        cv::resize(mat, dst->mat, cv::Size(width, height), interpMode);
        if(dataRect) {
            *dataRect = cv::Rect(0, 0, width, height);
        }
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
        if (dataRect) {
            *dataRect = cv::Rect(0, 0, newW, newH);
        }
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
        if (dataRect) {
            *dataRect = cv::Rect(dx, dy, newW, newH);
        }
        break;
    }
    }
    return dst;
}
