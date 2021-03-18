// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with common samples functionality using OpenCV
 * @file ocv_common.hpp
 */

#pragma once

#include <opencv2/opencv.hpp>

#include "utils/common.hpp"

class Resizer {
protected:
    size_t dstW;
    size_t dstH;
    size_t origW;
    size_t origH;
    float scaleX;
    float scaleY;
public:
    Resizer(size_t w, size_t h) : dstW(w), dstH(h), origW(0), origH(0),
                scaleX(0.0f), scaleY(0.0f){}
    virtual cv::Mat resize(const cv::Mat& img) = 0;
    virtual void scaleCoord2Origin(float& x, float& y) = 0;
};

class StretchResizer : public Resizer {
public:
    StretchResizer(size_t w, size_t h) : Resizer(w, h) {};
    cv::Mat resize(const cv::Mat& img) override {
        origW = img.size().width;
        origH = img.size().height;
        scaleX = static_cast<float>(origW) / dstW;
        scaleY = static_cast<float>(origH) / dstH;
        
        cv::Mat resizedImage(img);
        if (dstW != origW ||dstH != origH) {
            cv::resize(img, resizedImage, cv::Size(dstW, dstH));
        }
        return resizedImage;
    }

    void scaleCoord2Origin(float& x, float& y) override {
        x *= scaleX;
        y *= scaleY;
    }
};

class LetterboxResizer : public Resizer {
protected:
    size_t dx;
    size_t dy;
public:
    LetterboxResizer(size_t w, size_t h) : Resizer(w, h), dx(0), dy(0) {};
    cv::Mat resize(const cv::Mat & img) override {
        origW = img.size().width;
        origH = img.size().height;
        scaleX = scaleY = std::fmaxf(static_cast<float>(origW) / dstW, static_cast<float>(origH) / dstH);
        size_t newW = static_cast<size_t>(origW / scaleX);
        size_t newH = static_cast<size_t>(origH / scaleY);
        cv::Mat resizedImage(img);
        dx = (dstW - newW) / 2;
        dy = (dstH - newH) / 2;
        if (static_cast<int>(dstW) != img.size().width ||
            static_cast<int>(dstH) != img.size().height) {
            cv::resize(img, resizedImage, cv::Size(0, 0), 1 / scaleX, 1 / scaleY);
        }
        cv::Mat imgBuf(resizedImage.rows + dx * 2, resizedImage.cols + dy * 2, resizedImage.depth());
        cv::copyMakeBorder(resizedImage, imgBuf, dy, dy,
            dx, dx, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        cv::imshow("res", imgBuf);
        cv::waitKey(0);
        return imgBuf;
    }

    void scaleCoord2Origin(float& x, float& y) override {
        x -= dx;
        y -= dy;
        x *= scaleX;
        y *= scaleY;
    }
};

/**
* @brief Sets image data stored in cv::Mat object to a given Blob object.
* @param origImage - given cv::Mat object with an image data.
* @param blob - Blob object which to be filled by an image data.
* @param batchIndex - batch index of an image inside of the blob.
*/
template <typename T>
void matU8ToBlob(const cv::Mat& origImage, InferenceEngine::Blob::Ptr& blob, int batchIndex = 0) {
    InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
    const size_t width = blobSize[3];
    const size_t height = blobSize[2];
    const size_t channels = blobSize[1];
    if (static_cast<size_t>(origImage.channels()) != channels) {
        THROW_IE_EXCEPTION << "The number of channels for net input and image must match";
    }
    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->wmap();
    T* blob_data = blobMapped.as<T*>();

    //auto resizedImage = r.resize(origImage);
    // cv::Mat resizedImage(origImage);
    // if (static_cast<int>(width) != origImage.size().width ||
    //         static_cast<int>(height) != origImage.size().height) {
    //     cv::resize(origImage, resizedImage, cv::Size(width, height));
    // }

    int batchOffset = batchIndex * width * height * channels;

    if (channels == 1) {
        for (size_t  h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                blob_data[batchOffset + h * width + w] = origImage.at<uchar>(h, w);
            }
        }
    } else if (channels == 3) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t  h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    blob_data[batchOffset + c * width * height + h * width + w] =
                            origImage.at<cv::Vec3b>(h, w)[c];
                }
            }
        }
    } else {
        THROW_IE_EXCEPTION << "Unsupported number of channels";
    }
}

/**
 * @brief Wraps data stored inside of a passed cv::Mat object by new Blob pointer.
 * @note: No memory allocation is happened. The blob just points to already existing
 *        cv::Mat data.
 * @param mat - given cv::Mat object with an image data.
 * @return resulting Blob pointer.
 */
static UNUSED InferenceEngine::Blob::Ptr wrapMat2Blob(const cv::Mat &mat) {
    size_t channels = mat.channels();
    size_t height = mat.size().height;
    size_t width = mat.size().width;

    size_t strideH = mat.step.buf[0];
    size_t strideW = mat.step.buf[1];

    bool is_dense =
            strideW == channels &&
            strideH == channels * width;

    if (!is_dense) THROW_IE_EXCEPTION
                << "Doesn't support conversion from not dense cv::Mat";

    InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
                                      {1, channels, height, width},
                                      InferenceEngine::Layout::NHWC);

    return InferenceEngine::make_shared_blob<uint8_t>(tDesc, mat.data);
}

/**
 * @brief Puts text message on the frame, highlights the text with a white border to make it distinguishable from
 *        the background.
 * @param frame - frame to put the text on.
 * @param message - text of the message.
 * @param position - bottom-left corner of the text string in the image.
 * @param fontFace - font type.
 * @param fontScale - font scale factor that is multiplied by the font-specific base size.
 * @param color - text color.
 * @param thickness - thickness of the lines used to draw a text.
 */
inline void putHighlightedText(cv::Mat& frame,
                               const std::string& message,
                               cv::Point position,
                               int fontFace,
                               double fontScale,
                               cv::Scalar color,
                               int thickness) {
    cv::putText(frame, message, position, fontFace, fontScale, cv::Scalar(255, 255, 255), thickness + 1);
    cv::putText(frame, message, position, fontFace, fontScale, color, thickness);
}
