// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with common samples functionality using OpenCV
 * @file ocv_common.hpp
 */

#pragma once

#include <opencv2/opencv.hpp>

#include "utils/common.hpp"
#include "utils/shared_blob_allocator.h"


/**
* @brief Get cv::Mat value in the correct format.
*/
template <typename T>
static const T getMatValue(const cv::Mat& mat, size_t h, size_t w, size_t c) {
    switch (mat.type()) {
        case CV_8UC1:  return (T)mat.at<uchar>(h, w);
        case CV_8UC3:  return (T)mat.at<cv::Vec3b>(h, w)[c];
        case CV_32FC1: return (T)mat.at<float>(h, w);
        case CV_32FC3: return (T)mat.at<cv::Vec3f>(h, w)[c];
    }
    throw std::runtime_error("cv::Mat type is not recognized");
};

/**
* @brief Sets image data stored in cv::Mat object to a given Blob object.
* @param mat - given cv::Mat object with an image data.
* @param blob - Blob object which to be filled by an image data.
* @param batchIndex - batch index of an image inside of the blob.
*/
static UNUSED void matToBlob(const cv::Mat& mat, const InferenceEngine::Blob::Ptr& blob, int batchIndex = 0) {
    InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
    const size_t width = blobSize[3];
    const size_t height = blobSize[2];
    const size_t channels = blobSize[1];
    if (static_cast<size_t>(mat.channels()) != channels) {
        throw std::runtime_error("The number of channels for net input and image must match");
    }
    if (channels != 1 && channels != 3) {
        throw std::runtime_error("Unsupported number of channels");
    }
    int batchOffset = batchIndex * width * height * channels;

    cv::Mat resizedMat(mat);
    if (static_cast<int>(width) != mat.size().width || static_cast<int>(height) != mat.size().height) {
        cv::resize(mat, resizedMat, cv::Size(width, height));
    }

    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->wmap();
    if (blob->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32) {
        float_t* blobData = blobMapped.as<float_t*>();
        for (size_t c = 0; c < channels; c++)
            for (size_t h = 0; h < height; h++)
                for (size_t w = 0; w < width; w++)
                    blobData[batchOffset + c * width * height + h * width + w] =
                        getMatValue<float_t>(resizedMat, h, w, c);
    }
    else {
        uint8_t* blobData = blobMapped.as<uint8_t*>();
        if ((resizedMat.type() & CV_MAT_DEPTH_MASK) == CV_32F) {
            throw std::runtime_error("Conversion of cv::Mat from float_t to uint8_t is forbidden");
        }
        for (size_t c = 0; c < channels; c++)
            for (size_t h = 0; h < height; h++)
                for (size_t w = 0; w < width; w++)
                    blobData[batchOffset + c * width * height + h * width + w] =
                        getMatValue<uint8_t>(resizedMat, h, w, c);
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
    auto matType = mat.type() & CV_MAT_DEPTH_MASK;
    if (matType != CV_8U && matType != CV_32F) {
        throw std::runtime_error("Unsupported mat type for wrapping");
    }
    bool isMatFloat = matType == CV_32F;

    size_t channels = mat.channels();
    size_t height = mat.size().height;
    size_t width = mat.size().width;

    size_t strideH = mat.step.buf[0];
    size_t strideW = mat.step.buf[1];

    bool isDense = !isMatFloat ? (strideW == channels && strideH == channels * width) :
        (strideW == channels * sizeof(float) && strideH == channels * width * sizeof(float));
    if (!isDense) {
        throw std::runtime_error("Doesn't support conversion from not dense cv::Mat");
    }
    InferenceEngine::Precision precision = isMatFloat ?
        InferenceEngine::Precision::FP32 : InferenceEngine::Precision::U8;
    InferenceEngine::TensorDesc tDesc(precision,
                                      {1, channels, height, width},
                                      InferenceEngine::Layout::NHWC);

    const auto& blob = isMatFloat ?
        InferenceEngine::make_shared_blob<float>(tDesc, std::make_shared<SharedBlobAllocator>(mat)) :
        InferenceEngine::make_shared_blob<uint8_t>(tDesc, std::make_shared<SharedBlobAllocator>(mat));

    blob->allocate();
    return blob;
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
inline void putHighlightedText(const cv::Mat& frame,
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


class InputTransform {
public:
    InputTransform() : reverseInputChannels(false), isTrivial(true) {}

    InputTransform(bool reverseInputChannels, const std::string &meanValues, const std::string &scaleValues) :
        reverseInputChannels(reverseInputChannels),
        isTrivial(!reverseInputChannels && meanValues.empty() && scaleValues.empty()),
        means(meanValues.empty() ? cv::Scalar(0.0, 0.0, 0.0) : string2Vec(meanValues)),
        stdScales(scaleValues.empty() ? cv::Scalar(1.0, 1.0, 1.0) : string2Vec(scaleValues)) {
    }

    cv::Scalar string2Vec(const std::string &string) {
        const auto& strValues = split(string, ' ');
        std::vector<float> values;
        try {
            for (auto& str : strValues)
                values.push_back(std::stof(str));
        } catch (const std::invalid_argument&) {
            throw std::runtime_error("Invalid parameter --mean_values or --scale_values is provided.");
        }
        if (values.size() != 3) {
            throw std::runtime_error("InputTransform expects 3 values per channel, but get \"" + string + "\".");
        }
        return cv::Scalar(values[0], values[1], values[2]);
    }

    void setPrecision(const InferenceEngine::InputInfo::Ptr& input) {
        const auto precision = isTrivial ? InferenceEngine::Precision::U8 : InferenceEngine::Precision::FP32;
        input->setPrecision(precision);
    }

    cv::Mat operator()(const cv::Mat& inputs) {
        if (isTrivial) { return inputs; }
        cv::Mat result;
        inputs.convertTo(result, CV_32F);
        if (reverseInputChannels) {
            cv::cvtColor(result, result, cv::COLOR_BGR2RGB);
        }
        return (result - means) / stdScales;
    }

private:
    bool reverseInputChannels;
    bool isTrivial;
    cv::Scalar means;
    cv::Scalar stdScales;
};
