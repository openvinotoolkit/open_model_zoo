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

/**
* @brief Sets image data stored in cv::Mat object to a given Blob object.
* @param orig_image - given cv::Mat object with an image data.
* @param blob - Blob object which to be filled by an image data.
* @param batchIndex - batch index of an image inside of the blob.
*/
template <typename T>
void matU8ToBlob(const cv::Mat& orig_image, const InferenceEngine::Blob::Ptr& blob, int batchIndex = 0) {
    InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
    const size_t width = blobSize[3];
    const size_t height = blobSize[2];
    const size_t channels = blobSize[1];
    if (static_cast<size_t>(orig_image.channels()) != channels) {
        throw std::runtime_error("The number of channels for net input and image must match");
    }
    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->wmap();
    T* blob_data = blobMapped.as<T*>();

    cv::Mat resized_image(orig_image);
    if (static_cast<int>(width) != orig_image.size().width ||
            static_cast<int>(height) != orig_image.size().height) {
        cv::resize(orig_image, resized_image, cv::Size(width, height));
    }

    int batchOffset = batchIndex * width * height * channels;

    if (channels == 1) {
        for (size_t  h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                blob_data[batchOffset + h * width + w] = resized_image.at<uchar>(h, w);
            }
        }
    } else if (channels == 3) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t  h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    blob_data[batchOffset + c * width * height + h * width + w] =
                            resized_image.at<cv::Vec3b>(h, w)[c];
                }
            }
        }
    } else {
        throw std::runtime_error("Unsupported number of channels");
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

    if (!is_dense)
        throw std::runtime_error("Doesn't support conversion from not dense cv::Mat");

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


class OutputTransform {
    public:
        OutputTransform() : doResize(false), scaleFactor(1) {}

        OutputTransform(cv::Size inputSize, cv::Size outputResolution) :
            doResize(true), scaleFactor(1), inputSize(inputSize), outputResolution(outputResolution) {}

        cv::Size computeResolution() {
            float inputWidth = static_cast<float>(inputSize.width);
            float inputHeight = static_cast<float>(inputSize.height);
            scaleFactor = std::min(outputResolution.height / inputHeight, outputResolution.width / inputWidth);
            newResolution = cv::Size{static_cast<int>(inputWidth * scaleFactor), static_cast<int>(inputHeight * scaleFactor)};
            return newResolution;
        }

        void resize(cv::Mat& image) {
            if (!doResize) { return; }
            cv::Size currSize = image.size();
            if (currSize != inputSize) {
                inputSize = currSize;
                computeResolution();
            }
            if (scaleFactor == 1) { return; }
            cv::resize(image, image, newResolution);
        }

        template<typename T>
        void scaleCoord(T& coord) {
            if (!doResize || scaleFactor == 1) { return; }
            coord.x = std::floor(coord.x * scaleFactor);
            coord.y = std::floor(coord.y * scaleFactor);
        }

        template<typename T>
        void scaleRect(T& rect) {
            if (!doResize || scaleFactor == 1) { return; }
            scaleCoord(rect);
            rect.width = std::floor(rect.width * scaleFactor);
            rect.height = std::floor(rect.height * scaleFactor);
        }

        bool doResize;

    private:
        float scaleFactor;
        cv::Size inputSize;
        cv::Size outputResolution;
        cv::Size newResolution;
};
