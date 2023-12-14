/*
// Copyright (C) 2021-2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "models/image_model.h"

#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>

#include <utils/image_utils.h>
#include <utils/ocv_common.hpp>

#include "models/input_data.h"
#include "models/internal_model_data.h"

ImageModel::ImageModel(const std::string& modelFileName, bool useAutoResize, const std::string& layout)
    : ModelBase(modelFileName, layout),
      useAutoResize(useAutoResize) {}

std::shared_ptr<InternalModelData> ImageModel::preprocess(std::vector<std::shared_ptr<InputData>>::iterator inputDataBegin,
                                                          std::vector<std::shared_ptr<InputData>>::iterator inputDataEnd,
                                                          ov::InferRequest& request) {

    const ov::Tensor& frameTensor = request.get_tensor(inputsNames[0]);  // first input should be image
    const ov::Shape& tensorShape = frameTensor.get_shape();
    const ov::Layout layout("NHWC");
    const size_t batch = tensorShape[ov::layout::batch_idx(layout)];
    const size_t width = tensorShape[ov::layout::width_idx(layout)];
    const size_t height = tensorShape[ov::layout::height_idx(layout)];
    const size_t channels = tensorShape[ov::layout::channels_idx(layout)];
    char* memoryBlob = nullptr;
    size_t image_index = 0;
    bool isMatFloat = false;
    int origImg_cols = 0;
    int origImg_rows = 0;
    size_t image_count = std::distance(inputDataBegin, inputDataEnd);
    if (image_count != batch) {
        throw std::runtime_error("Image count in preprocess must repeat batch count");
    }
    for (auto inputDataIt = inputDataBegin; inputDataIt != inputDataEnd; ++inputDataIt ) {
        const auto& origImg = (*inputDataIt)->asRef<ImageInputData>().inputImage;
        origImg_cols = origImg.cols;
        origImg_rows = origImg.rows;
        auto img = inputTransform(origImg);

        auto matType = img.type() & CV_MAT_DEPTH_MASK;
        if (matType != CV_8U && matType != CV_32F) {
            throw std::runtime_error("Unsupported mat type for wrapping");
        }
        isMatFloat = matType == CV_32F;

        if (!useAutoResize) {
            // /* Resize and copy data from the image to the input tensor */

            if (static_cast<size_t>(img.channels()) != channels) {
                throw std::runtime_error(std::string("The number of channels for model input: ") +
                                         std::to_string(channels) + " and image: " +
                                         std::to_string(img.channels()) + " - must match");
            }
            if (channels != 1 && channels != 3) {
                throw std::runtime_error("Unsupported number of channels");
            }
            img = resizeImageExt(img, width, height, resizeMode, interpolationMode);
        }
        size_t sizeInBytes = img.total() * img.elemSize();
        if (!memoryBlob) {
            memoryBlob = new char[sizeInBytes * batch]; // intended memory leak
        }

        // fill continuous batch
        memcpy(memoryBlob + sizeInBytes * image_index, img.ptr(), sizeInBytes);
        image_index++;
    }

    auto precision = isMatFloat ? ov::element::f32 : ov::element::u8;
    auto batched_tensor =  ov::Tensor(precision, ov::Shape{ batch, height, width, channels }, memoryBlob);
    request.set_tensor(inputsNames[0], batched_tensor);
    return std::make_shared<InternalImageModelData>(origImg_cols, origImg_rows);
}

std::shared_ptr<InternalModelData> ImageModel::preprocess(const InputData& inputData, ov::InferRequest& request) {
    const auto& origImg = inputData.asRef<ImageInputData>().inputImage;
    auto img = inputTransform(origImg);

    if (!useAutoResize) {
        // /* Resize and copy data from the image to the input tensor */
        const ov::Tensor& frameTensor = request.get_tensor(inputsNames[0]);  // first input should be image
        const ov::Shape& tensorShape = frameTensor.get_shape();
        const ov::Layout layout("NHWC");
        const size_t width = tensorShape[ov::layout::width_idx(layout)];
        const size_t height = tensorShape[ov::layout::height_idx(layout)];
        const size_t channels = tensorShape[ov::layout::channels_idx(layout)];
        if (static_cast<size_t>(img.channels()) != channels) {
            throw std::runtime_error(std::string("The number of channels for model input: ") +
                                     std::to_string(channels) + " and image: " +
                                     std::to_string(img.channels()) + " - must match");
        }
        if (channels != 1 && channels != 3) {
            throw std::runtime_error("Unsupported number of channels");
        }
        img = resizeImageExt(img, width, height, resizeMode, interpolationMode);
    }
    request.set_tensor(inputsNames[0], wrapMat2Tensor(img));
    return std::make_shared<InternalImageModelData>(origImg.cols, origImg.rows);
}
