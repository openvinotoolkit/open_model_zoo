/*
// Copyright (C) 2021 Intel Corporation
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

#include <openvino/openvino.hpp>
#include "models/image_model.h"

ImageModel::ImageModel(const std::string& modelFileName, bool useAutoResize) :
    ModelBase(modelFileName),
    useAutoResize(useAutoResize) {
}

std::shared_ptr<InternalModelData> ImageModel::preprocess(const InputData& inputData, ov::runtime::InferRequest& request) {
    const auto& origImg = inputData.asRef<ImageInputData>().inputImage;
    const auto& img = inputTransform(origImg);

    if (useAutoResize) {
        /* Just set input tensor containing read image. Resize and layout conversion will be done automatically */
        request.set_input_tensor(wrapMat2Tensor(img));
    }
    else {
        /* Resize and copy data from the image to the input tensor */
        ov::runtime::Tensor frameTensor = request.get_input_tensor();
        matToTensor(img, frameTensor);
    }
    return std::make_shared<InternalImageModelData>(img.cols, img.rows);
}
