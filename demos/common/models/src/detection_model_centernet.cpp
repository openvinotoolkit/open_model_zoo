/*
// Copyright (C) 2018-2021 Intel Corporation
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

#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/imgproc.hpp>

#include <ngraph/ngraph.hpp>
#include <samples/slog.hpp>
#include <samples/common.hpp>
#include <samples/ocv_common.hpp>
#include "models/detection_model_centernet.h"

using namespace InferenceEngine;

ModelCenterNet::ModelCenterNet(const std::string& modelFileName,
    float confidenceThreshold, bool useAutoResize, float boxIOUThreshold, const std::vector<std::string>& labels)
    : DetectionModel(modelFileName, confidenceThreshold, useAutoResize, labels),
    boxIOUThreshold(boxIOUThreshold) {
}

void ModelCenterNet::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input blobs ------------------------------------------------------
    slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());

    if (inputInfo.size() != 1) {
        throw std::logic_error("This demo accepts networks that have only one input");
    }

    InputInfo::Ptr& input = inputInfo.begin()->second;
    const TensorDesc& inputDesc = input->getTensorDesc();
    input->setPrecision(Precision::U8);

    if (inputDesc.getDims()[1] != 3) {
        throw std::logic_error("Expected 3-channel input");
    }

    if (useAutoResize) {
        input->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
        input->getInputData()->setLayout(Layout::NHWC);
    }
    else {
        input->getInputData()->setLayout(Layout::NCHW);
    }

    // --------------------------- Reading image input parameters -------------------------------------------
    std::string imageInputName = inputInfo.begin()->first;
    inputsNames.push_back(imageInputName);
    netInputHeight = getTensorHeight(inputDesc);
    netInputWidth = getTensorWidth(inputDesc);

    // --------------------------- Prepare output blobs -----------------------------------------------------
    slog::info << "Checking that the outputs are as the demo expects" << slog::endl;

    InferenceEngine::OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());

    if (outputInfo.size() != 3) {
        throw std::logic_error("This demo expect networks that have 3 outputs blobs");
    }

    const TensorDesc& outputDesc = outputInfo.begin()->second->getTensorDesc();
    maxProposalsCount = outputDesc.getDims()[1];

    for (auto& output : outputInfo) {
        output.second->setPrecision(InferenceEngine::Precision::FP32);
        output.second->setLayout(InferenceEngine::Layout::NCHW);
        outputsNames.push_back(output.first);
    }

}

cv::Point2f getDir(const cv::Point2f& srcPoint, float rotRadius) {
    float sn = std::sinf(rotRadius);
    float cs = std::cosf(rotRadius);

    cv::Point2f srcResult({ 0, 0 });
    srcResult.x = srcPoint.x * cs - srcPoint.y * sn;
    srcResult.y = srcPoint.x * sn + srcPoint.y * cs;

    return srcResult;
}

cv::Point2f get3rdPoint(const cv::Point2f& a, const cv::Point2f& b) {
    cv::Point2f direct = a - b;
    return b + cv::Point2f({ -direct.y, direct.x });
}

cv::Mat getAffineTransform(float centerX, float centerY, float scale, float rot, int outputWidth, int outputHeight, bool inv = false) {
    float srcW = scale;
    float rotRad =  M_PI * rot / 180.0f;
    auto srcDir = getDir({ 0, -0.5f * srcW }, rotRad);
    cv::Point2f dstDir({ 0,  -0.5f * outputWidth });
    std::vector<cv::Point2f> src(3, { 0 , 0 });
    std::vector<cv::Point2f> dst(3, { 0 , 0 });

    src[0] = { centerX, centerY };
    src[1] = srcDir + src[0];
    src[2] = get3rdPoint(src[0], src[1]);

    dst[0] = { outputWidth * 0.5f, outputHeight * 0.5f };
    dst[1] = dst[0] + dstDir;
    dst[2] = get3rdPoint(dst[0], dst[1]);

    cv::Mat trans;
    if (inv) {
        trans = cv::getAffineTransform(src, dst);
    }
    else {
        trans = cv::getAffineTransform(src, dst);
    }

    return trans;
}

void hwcTochw(cv::InputArray src, cv::OutputArray dst) {
    const int srcH = src.rows();
    const int srcW = src.cols();
    const int srcC = src.channels();

    cv::Mat hwC = src.getMat().reshape(1, srcH * srcW);

    const std::array<int, 3> dims = { srcC, srcH, srcW };
    dst.create(3, &dims[0], CV_MAKETYPE(src.depth(), 1));
    cv::Mat dst1d = dst.getMat().reshape(1, { srcC, srcH, srcW });

    cv::transpose(hwC, dst1d);
}

std::shared_ptr<InternalModelData> ModelCenterNet::preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) {

    //height, width = image.shape[0:2]
    //    center = np.array([width / 2., height / 2.], dtype = np.float32)
    //    scale = max(height, width)
    //    trans_input = self.get_affine_transform(center, scale, 0, [self.w, self.h])
    //    resized_image = cv2.warpAffine(image, trans_input, (self.w, self.h), flags = cv2.INTER_LINEAR)
    //    resized_image = np.transpose(resized_image, (2, 0, 1))
    auto& img = inputData.asRef<ImageInputData>().inputImage;

    int imgWidth = img.cols;
    int imgHeight = img.rows;

    float centerX = imgWidth / 2.0f;
    float centerY = imgHeight / 2.0f;

    int scale = std::max(imgWidth, imgHeight);
    auto transInput = getAffineTransform(centerX, centerY, scale, 0, netInputWidth, netInputHeight);
    cv::Mat resizedImg;
    cv::warpAffine(img, resizedImg, transInput, cv::Size(netInputWidth, netInputHeight), cv::INTER_LINEAR);
    //change order
    cv::Mat chwImg;
    //hwcTochw(resizedImg, chwImg);

    if (useAutoResize) {
        /* Just set input blob containing read image. Resize and layout conversionx will be done automatically */
        request->SetBlob(inputsNames[0], wrapMat2Blob(resizedImg));
    }
    else {
        /* Resize and copy data from the image to the input blob */
        Blob::Ptr frameBlob = request->GetBlob(inputsNames[0]);
        matU8ToBlob<uint8_t>(resizedImg, frameBlob);
    }

    return std::shared_ptr<InternalModelData>(new InternalImageModelData(img.cols, img.rows));
}

std::vector<float> maxPool2d(const std::vector<float>& mat, int kernelSize, int stride = 1) {

    return {};
}

std::vector<int> nms(const std::vector<float>& heat, int kernel = 3) {
    //def max_pool2d(A, kernel_size, padding = 1, stride = 1) :
    //    A = np.pad(A, padding, mode = 'constant')
    //    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
    //    (A.shape[1] - kernel_size)//stride + 1)
    //        kernel_size = (kernel_size, kernel_size)
    //        A_w = as_strided(A, shape = output_shape + kernel_size,
    //            strides = (stride*A.strides[0],
    //                stride*A.strides[1]) + A.strides)
    //        A_w = A_w.reshape(-1, *kernel_size)

    //        return A_w.max(axis = (1, 2)).reshape(output_shape)

//        pad = (kernel - 1) // 2

          
//        hmax = np.array([max_pool2d(channel, kernel, pad) for channel in heat])
//        keep = (hmax == heat)
//        return heat * keep
    int pad = (kernel - 1) / 2;
    std::vector<int> indices;
    for (int ch = 0; ch < 80; ++ch) {
        for (int w = 0; w < 128; ++w) {
            for (int h = 0; h < 128; ++h) {
                float max = heat[128 * w + h];
                indices.push_back(128 * w + h);
                bool next = true;
                for (int i = -kernel / 2; i < kernel / 2 + 1 && next; ++i) {
                    for (int j = -kernel / 2; j < kernel / 2 + 1; ++j) {
                        if (w + i >= 0 && w + i < 128 && h + j >= 0 && h + j <= 128) {
                            if (heat[128 * (w + i) + h + j] > max) {
                                indices.pop_back();
                                next = false;
                                break;
                            }
                        }
                        else {
                            if (max < 0) {
                                indices.pop_back();
                                next = false;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    return indices;
}

std::vector<float> heatProcessing(InferenceEngine::MemoryBlob::Ptr heat) {
    LockedMemory<const void> bboxesOutputMapped = heat->rmap();
    auto desc = heat->getTensorDesc();
    auto sz = desc.getDims();
    const float *heatPtr = bboxesOutputMapped.as<float*>();

    std::vector<float> retVal;

    for (int i = 0; i < sz[0] * sz[1] * sz[2] * sz[3]; ++i) {
        retVal.push_back(expf(heatPtr[i]) / (1 + expf(heatPtr[i])));
    }

    nms(retVal);
    return retVal;
}

std::vector<float> getTopKScores() {

    return {};
}

void topK() {
}


std::unique_ptr<ResultBase> ModelCenterNet::postprocess(InferenceResult& infResult) {
        //heat = outputs[self._output_layer_names[0]][0]
        //reg = outputs[self._output_layer_names[1]][0]
        //wh = outputs[self._output_layer_names[2]][0]
        //heat = np.exp(heat) / (1 + np.exp(heat))
        //height, width = heat.shape[1:3]
        //num_predictions = 100

        //heat = self._nms(heat)
        //scores, inds, clses, ys, xs = self._topk(heat, K = num_predictions)
        //reg = self._tranpose_and_gather_feat(reg, inds)

        //reg = reg.reshape((num_predictions, 2))
        //xs = xs.reshape((num_predictions, 1)) + reg[:, 0 : 1]
        //ys = ys.reshape((num_predictions, 1)) + reg[:, 1 : 2]

        //wh = self._tranpose_and_gather_feat(wh, inds)
        //wh = wh.reshape((num_predictions, 2))
        //clses = clses.reshape((num_predictions, 1))
        //scores = scores.reshape((num_predictions, 1))
        //bboxes = np.concatenate((xs - wh[..., 0:1] / 2,
        //    ys - wh[..., 1:2] / 2,
        //    xs + wh[..., 0:1] / 2,
        //    ys + wh[..., 1:2] / 2), axis = 1)
        //detections = np.concatenate((bboxes, scores, clses), axis = 1)
        //mask = detections[..., 4] >= self._threshold
        //filtered_detections = detections[mask]
        //scale = max(meta['original_shape'])
        //center = np.array(meta['original_shape'][:2]) / 2.0
        //dets = self._transform(filtered_detections, np.flip(center, 0), scale, height, width)
        //dets = [Detection(x[0], x[1], x[2], x[3], score = x[4], id = x[5]) for x in dets]
        //return dets

    auto heat = infResult.outputsData[outputsNames[0]];
    heatProcessing(heat);

    auto reg = infResult.outputsData[outputsNames[1]];

    auto wh = infResult.outputsData[outputsNames[2]];


    return {};
}
