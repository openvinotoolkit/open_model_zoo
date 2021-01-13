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
        trans = cv::getAffineTransform(dst, src);
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

std::vector<std::pair<size_t, float>> nms(const float *scoresPtr, SizeVector sz, float threshold, int kernel = 3) {
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
    std::vector<std::pair<size_t, float>> scores;
    scores.reserve(100);
    //std::vector<float> scores;
    //std::vector<float> classes;
    //std::vector<float> x;
    //std::vector<float> y;
    int chSize = sz[2] * sz[3];
    int k = 100;
    for (int ch = 0; ch < sz[1]; ++ch) {
        int count = 0;
        for (int w = 0; w < sz[2]; ++w) {
            for (int h = 0; h < sz[3]; ++h) {
                float max = scoresPtr[chSize * ch + sz[2] * w + h];
                if (max < threshold) {
                    continue;
                }
                scores.push_back({ chSize * ch + sz[2] * w + h, max });
                bool next = true;
                for (int i = -kernel / 2; i < kernel / 2 + 1 && next; ++i) {
                    for (int j = -kernel / 2; j < kernel / 2 + 1; ++j) {
                        if (w + i >= 0 && w + i < sz[2] && h + j >= 0 && h + j < sz[3]) {
                            if (scoresPtr[chSize * ch + sz[2] * (w + i) + h + j] > max) {
                                scores.pop_back();
                                next = false;
                                break;
                            }
                        }
                        else {
                            if (max < 0) {
                                scores.pop_back();
                                next = false;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    //for (int i = 0; i < k; ++i) {
    //    auto ki = q.top();
    //    newScores[k - i - 1] = ki;
    //    q.pop();
    //}

    //for (auto i : indices) {
    //    scores.push_back(heat[i]);
    //    classes.push_back(i / chSize);
    //    x.push_back(i / sz[2]);
    //    y.push_back(i % sz[2]);
    //}
    return scores;
}


std::vector<std::pair<size_t, float>> filterScores(InferenceEngine::MemoryBlob::Ptr scoresInfRes, float threshold) {
    LockedMemory<const void> scoresOutputMapped = scoresInfRes->rmap();
    auto desc = scoresInfRes->getTensorDesc();
    auto sz = desc.getDims();
    const float *scoresPtr = scoresOutputMapped.as<float*>();

    return nms(scoresPtr, sz, threshold);
}

std::vector<std::pair<float, float>> filterReg(InferenceEngine::MemoryBlob::Ptr regInfRes, const std::vector<std::pair<size_t, float>>& scores, size_t chSize) {
    LockedMemory<const void> bboxesOutputMapped = regInfRes->rmap();
    //auto desc = regInfRes->getTensorDesc();
    //auto sz = desc.getDims();
    //auto chSize = sz[1] * sz[2];
    const float *regPtr = bboxesOutputMapped.as<float*>();
    std::vector<std::pair<float, float>> reg;

    //for (int i = 0; i < sz[1] * sz[2] * sz[3]; ++i) {
    //    reg.push_back({ regPtr[i], regPtr[i] } );
    //}

    for (auto s : scores) {
        reg.push_back({ regPtr[s.first % chSize], regPtr[chSize + s.first % chSize] });
    }
    return reg;
}

std::vector<std::pair<float, float>> filterWH(InferenceEngine::MemoryBlob::Ptr whInfRes, const std::vector<std::pair<size_t, float>>& scores, size_t chSize) {
    LockedMemory<const void> bboxesOutputMapped = whInfRes->rmap();
    //auto desc = whInfRes->getTensorDesc();
    //auto sz = desc.getDims();
    const float *whPtr = bboxesOutputMapped.as<float*>();
    std::vector<std::pair<float, float>> wh;

    //for (int i = 0; i < sz[1] * sz[2] * sz[3]; ++i) {
    //    wh.push_back({ whPtr[i], whPtr[i] } );
    //}

    for (auto s : scores) {
        wh.push_back({ whPtr[s.first % chSize], whPtr[chSize + s.first % chSize] });
    }

    return wh;
}

std::vector<ModelCenterNet::BBoxes> calcBBoxes(const std::vector<std::pair<size_t, float>>& scores, const std::vector<std::pair<float, float>>& reg,
    const std::vector<std::pair<float, float>>& wh, const SizeVector& sz) {
    std::vector<ModelCenterNet::BBoxes> bboxes(scores.size());

    for (int i = 0; i < bboxes.size(); ++i) {
        size_t chIdx = scores[i].first % (sz[2] * sz[3]);
        auto xCenter = chIdx % sz[3];
        auto yCenter = chIdx / sz[3];

        bboxes[i].left = xCenter + reg[i].first - wh[i].first / 2.0f;
        bboxes[i].top = yCenter + reg[i].second - wh[i].second / 2.0f;
        bboxes[i].right = xCenter + reg[i].first + wh[i].first / 2.0f;
        bboxes[i].bottom = yCenter + reg[i].second + wh[i].second / 2.0f;
    }

    return bboxes;
}

void transform(std::vector<ModelCenterNet::BBoxes>* bboxes, const SizeVector& sz, int scale, float centerX, float centerY) {
    cv::Mat1f trans = getAffineTransform(centerX, centerY, scale, 0, sz[2], sz[3], true);
    for (auto& b : *bboxes) {
        ModelCenterNet::BBoxes newbb;

        newbb.left = trans.at<float>(0, 0) *  b.left + trans.at<float>(0, 1) *  b.top + trans.at<float>(0, 2);
        newbb.top = trans.at<float>(1, 0) *  b.left + trans.at<float>(1, 1) *  b.top + trans.at<float>(1, 2);
        newbb.right = trans.at<float>(0, 0) *  b.right + trans.at<float>(0, 1) *  b.bottom + trans.at<float>(0, 2);
        newbb.bottom = trans.at<float>(1, 0) *  b.right + trans.at<float>(1, 1) *  b.bottom + trans.at<float>(1, 2);

        b = newbb;
    }
}

std::unique_ptr<ResultBase> ModelCenterNet::postprocess(InferenceResult& infResult) {
    auto heatInfRes = infResult.outputsData[outputsNames[0]];
    auto sz = heatInfRes->getTensorDesc().getDims();;
    auto chSize = sz[2] * sz[3];

    auto scores = filterScores(heatInfRes, confidenceThreshold);

    for (auto& s : scores) {
        s.second = expf(s.second) / (1 + expf(s.second));
    }

    auto regInfRes = infResult.outputsData[outputsNames[1]];

    auto reg = filterReg(regInfRes, scores, chSize);

    auto whInfRes = infResult.outputsData[outputsNames[2]];

    auto wh = filterWH(whInfRes, scores, chSize);

    auto bboxes = calcBBoxes(scores, reg, wh, sz);

    auto imgWidth = infResult.internalModelData->asRef<InternalImageModelData>().inputImgWidth;
    auto imgHeight = infResult.internalModelData->asRef<InternalImageModelData>().inputImgHeight;
    auto scale = std::max(imgWidth, imgHeight);
    float centerX = imgWidth / 2.0f;
    float centerY = imgHeight / 2.0f;

    transform(&bboxes, sz, scale, centerX, centerY);

    // --------------------------- Create detection result objects --------------------------------------------------------
    DetectionResult* result = new DetectionResult;
    *static_cast<ResultBase*>(result) = static_cast<ResultBase&>(infResult);

    result->objects.reserve(scores.size());
    for (int i = 0; i < scores.size(); ++i) {
        size_t chIdx = scores[i].first % chSize;
        DetectedObject desc;
        desc.confidence = static_cast<float>(scores[i].second);
        desc.labelID = scores[i].first / chSize;
        desc.label = getLabelName(desc.labelID);
        desc.x = bboxes[i].left;
        desc.y = bboxes[i].top;
        desc.width = bboxes[i].getWidth(); 
        desc.height = bboxes[i].getHeight(); 

        result->objects.push_back(desc);
    }

    return std::unique_ptr<ResultBase>(result);
}
