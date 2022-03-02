// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with common samples functionality using OpenCV
 * @file ocv_common.hpp
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "utils/common.hpp"
#include "utils/shared_tensor_allocator.hpp"

/**
* @brief Get cv::Mat value in the correct format.
*/
template <typename T>
const T getMatValue(const cv::Mat& mat, size_t h, size_t w, size_t c) {
    switch (mat.type()) {
        case CV_8UC1:  return (T)mat.at<uchar>(h, w);
        case CV_8UC3:  return (T)mat.at<cv::Vec3b>(h, w)[c];
        case CV_32FC1: return (T)mat.at<float>(h, w);
        case CV_32FC3: return (T)mat.at<cv::Vec3f>(h, w)[c];
    }
    throw std::runtime_error("cv::Mat type is not recognized");
};

/**
* @brief Resize and copy image data from cv::Mat object to a given Tensor object.
* @param mat - given cv::Mat object with an image data.
* @param tensor - Tensor object which to be filled by an image data.
* @param batchIndex - batch index of an image inside of the blob.
*/
static UNUSED void matToTensor(const cv::Mat& mat, const ov::Tensor& tensor, int batchIndex = 0) {
    ov::Shape tensorShape = tensor.get_shape();
    static const ov::Layout layout("NCHW");
    const size_t width = tensorShape[ov::layout::width_idx(layout)];
    const size_t height = tensorShape[ov::layout::height_idx(layout)];
    const size_t channels = tensorShape[ov::layout::channels_idx(layout)];
    if (static_cast<size_t>(mat.channels()) != channels) {
        throw std::runtime_error("The number of channels for model input and image must match");
    }
    if (channels != 1 && channels != 3) {
        throw std::runtime_error("Unsupported number of channels");
    }
    int batchOffset = batchIndex * width * height * channels;

    cv::Mat resizedMat;
    if (static_cast<int>(width) != mat.size().width || static_cast<int>(height) != mat.size().height) {
        cv::resize(mat, resizedMat, cv::Size(width, height));
    } else {
        resizedMat = mat;
    }

    if (tensor.get_element_type() == ov::element::f32) {
        float_t* tensorData = tensor.data<float_t>();
        for (size_t c = 0; c < channels; c++)
            for (size_t h = 0; h < height; h++)
                for (size_t w = 0; w < width; w++)
                    tensorData[batchOffset + c * width * height + h * width + w] =
                        getMatValue<float_t>(resizedMat, h, w, c);
    } else {
        uint8_t* tensorData = tensor.data<uint8_t>();
        if (resizedMat.depth() == CV_32F) {
            throw std::runtime_error("Conversion of cv::Mat from float_t to uint8_t is forbidden");
        }
        for (size_t c = 0; c < channels; c++)
            for (size_t h = 0; h < height; h++)
                for (size_t w = 0; w < width; w++)
                    tensorData[batchOffset + c * width * height + h * width + w] =
                        getMatValue<uint8_t>(resizedMat, h, w, c);
    }
}

static UNUSED ov::Tensor wrapMat2Tensor(const cv::Mat& mat) {
    auto matType = mat.type() & CV_MAT_DEPTH_MASK;
    if (matType != CV_8U && matType != CV_32F) {
        throw std::runtime_error("Unsupported mat type for wrapping");
    }
    bool isMatFloat = matType == CV_32F;

    const size_t channels = mat.channels();
    const size_t height = mat.rows;
    const size_t width = mat.cols;

    const size_t strideH = mat.step.buf[0];
    const size_t strideW = mat.step.buf[1];

    const bool isDense = !isMatFloat ? (strideW == channels && strideH == channels * width) :
        (strideW == channels * sizeof(float) && strideH == channels * width * sizeof(float));
    if (!isDense) {
        throw std::runtime_error("Doesn't support conversion from not dense cv::Mat");
    }
    auto precision = isMatFloat ? ov::element::f32 : ov::element::u8;
    auto allocator = std::make_shared<SharedTensorAllocator>(mat);
    return ov::Tensor(precision, ov::Shape{ 1, height, width, channels }, ov::Allocator(allocator));
}

static inline void resize2tensor(const cv::Mat& mat, const ov::Tensor& tensor) {
    static const ov::Layout layout{"NHWC"};
    const ov::Shape& shape = tensor.get_shape();
    cv::Size size{int(shape[ov::layout::width_idx(layout)]), int(shape[ov::layout::height_idx(layout)])};
    assert(tensor.get_element_type() == ov::element::u8);
    assert(shape.size() == 4);
    assert(shape[ov::layout::batch_idx(layout)] == 1);
    assert(shape[ov::layout::channels_idx(layout)] == 3);
    cv::resize(mat, cv::Mat{size, CV_8UC3, tensor.data()}, size);
}

static inline ov::Layout getLayoutFromShape(const ov::Shape& shape) {
    if (shape.size() == 2) {
        return "NC";
    }
    else if (shape.size() == 3) {
        return (shape[0] >= 1 && shape[0] <= 4) ? "CHW" :
                                                  "HWC";
    }
    else if (shape.size() == 4) {
        return (shape[1] >= 1 && shape[1] <= 4) ? "NCHW" :
                                                  "NHWC";
    }
    else {
        throw std::runtime_error("Usupported " + std::to_string(shape.size()) + "D shape");
    }
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

    InputTransform(bool reverseInputChannels, const std::string& meanValues, const std::string& scaleValues) :
        reverseInputChannels(reverseInputChannels),
        isTrivial(!reverseInputChannels && meanValues.empty() && scaleValues.empty()),
        means(meanValues.empty() ? cv::Scalar(0.0, 0.0, 0.0) : string2Vec(meanValues)),
        stdScales(scaleValues.empty() ? cv::Scalar(1.0, 1.0, 1.0) : string2Vec(scaleValues)) {
    }

    cv::Scalar string2Vec(const std::string& string) {
        const auto& strValues = split(string, ' ');
        std::vector<float> values;
        try {
            for (auto& str : strValues)
                values.push_back(std::stof(str));
        }
        catch (const std::invalid_argument&) {
            throw std::runtime_error("Invalid parameter --mean_values or --scale_values is provided.");
        }
        if (values.size() != 3) {
            throw std::runtime_error("InputTransform expects 3 values per channel, but get \"" + string + "\".");
        }
        return cv::Scalar(values[0], values[1], values[2]);
    }

    void setPrecision(ov::preprocess::PrePostProcessor& ppp, const std::string& tensorName) {
        const auto precision = isTrivial ? ov::element::u8 : ov::element::f32;
        ppp.input(tensorName).tensor().
                set_element_type(precision);
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

class LazyVideoWriter {
    cv::VideoWriter writer;
    unsigned nwritten;
public:
    const std::string filenames;
    const double fps;
    const unsigned lim;

    LazyVideoWriter(const std::string& filenames, double fps, unsigned lim) :
        nwritten{1}, filenames{filenames}, fps{fps}, lim{lim} {}
    void write(cv::InputArray im) {
        if (writer.isOpened() && (nwritten < lim || 0 == lim)) {
            writer.write(im);
            ++nwritten;
            return;
        }
        if (!writer.isOpened() && !filenames.empty()) {
            if (!writer.open(filenames, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, im.size())) {
                throw std::runtime_error("Can't open video writer");
            }
            writer.write(im);
        }
    }
};
