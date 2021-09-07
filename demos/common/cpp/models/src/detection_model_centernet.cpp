/*
// Copyright (C) 2020-2021 Intel Corporation
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

#include <ngraph/ngraph.hpp>
#include <opencv2/imgproc.hpp>
#include <utils/common.hpp>
#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>
#include "models/detection_model_centernet.h"


ModelCenterNet::ModelCenterNet(const std::string& modelFileName,
    float confidenceThreshold, const std::vector<std::string>& labels)
    : DetectionModel(modelFileName, confidenceThreshold, false , labels) {
}

void ModelCenterNet::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input blobs ------------------------------------------------------
    slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
    InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::logic_error("This demo accepts networks that have only one input");
    }

    InferenceEngine::InputInfo::Ptr& input = inputInfo.begin()->second;
    const InferenceEngine::TensorDesc& inputDesc = input->getTensorDesc();
    input->setPrecision(InferenceEngine::Precision::U8);

    if (inputDesc.getDims()[1] != 3) {
        throw std::logic_error("Expected 3-channel input");
    }
    input->getInputData()->setLayout(InferenceEngine::Layout::NHWC);

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

    for (auto& output : outputInfo) {
        output.second->setPrecision(InferenceEngine::Precision::FP32);
        output.second->setLayout(InferenceEngine::Layout::NCHW);
        outputsNames.push_back(output.first);
    }
}

cv::Point2f getDir(const cv::Point2f& srcPoint, float rotRadius) {
    float sn = sinf(rotRadius);
    float cs = cosf(rotRadius);

    cv::Point2f srcResult(0.0f, 0.0f);
    srcResult.x = srcPoint.x * cs - srcPoint.y * sn;
    srcResult.y = srcPoint.x * sn + srcPoint.y * cs;

    return srcResult;
}

cv::Point2f get3rdPoint(const cv::Point2f& a, const cv::Point2f& b) {
    cv::Point2f direct = a - b;
    return b + cv::Point2f(-direct.y, direct.x);
}

cv::Mat getAffineTransform(float centerX, float centerY, int srcW, float rot, size_t outputWidth, size_t outputHeight, bool inv = false) {
    float rotRad =  static_cast<float>(CV_PI) * rot / 180.0f;
    auto srcDir = getDir({ 0.0f, -0.5f * srcW }, rotRad);
    cv::Point2f dstDir(0.0f,  -0.5f * outputWidth);
    std::vector<cv::Point2f> src(3, { 0.0f, 0.0f });
    std::vector<cv::Point2f> dst(3, { 0.0f, 0.0f });

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

std::shared_ptr<InternalModelData> ModelCenterNet::preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) {
    auto& img = inputData.asRef<ImageInputData>().inputImage;

    int imgWidth = img.cols;
    int imgHeight = img.rows;
    float centerX = imgWidth / 2.0f;
    float centerY = imgHeight / 2.0f;
    int scale = std::max(imgWidth, imgHeight);

    auto transInput = getAffineTransform(centerX, centerY, scale, 0, netInputWidth, netInputHeight);
    cv::Mat resizedImg;
    cv::warpAffine(img, resizedImg, transInput, cv::Size(netInputWidth, netInputHeight), cv::INTER_LINEAR);
    request->SetBlob(inputsNames[0], wrapMat2Blob(resizedImg));
    /* IE::Blob::Ptr from wrapMat2Blob() doesn't own data. Save the image to avoid deallocation before inference */
    return std::make_shared<InternalImageMatModelData>(resizedImg, img.cols, img.rows);
}

std::vector<std::pair<size_t, float>> nms(float* scoresPtr, InferenceEngine::SizeVector sz, float threshold, int kernel = 3) {
    std::vector<std::pair<size_t, float>> scores;
    scores.reserve(ModelCenterNet::INIT_VECTOR_SIZE);
    auto chSize = sz[2] * sz[3];

    for (size_t i = 0; i < sz[1] * sz[2] * sz[3]; ++i) {
        scoresPtr[i] = expf(scoresPtr[i]) / (1 + expf(scoresPtr[i]));
    }

    for (size_t ch = 0; ch < sz[1]; ++ch) {
        for (size_t w = 0; w < sz[2]; ++w) {
            for (size_t h = 0; h < sz[3]; ++h) {
                float max = scoresPtr[chSize * ch + sz[2] * w + h];

                // ---------------------  filter on threshold--------------------------------------
                if (max < threshold) {
                    continue;
                }

                // ---------------------  store index and score------------------------------------
                scores.push_back({ chSize * ch + sz[2] * w + h, max });

                bool next = true;
                // ---------------------- maxpool2d -----------------------------------------------
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

    return scores;
}


static std::vector<std::pair<size_t, float>> filterScores(const InferenceEngine::MemoryBlob::Ptr& scoresInfRes, float threshold) {
    InferenceEngine::LockedMemory<const void> scoresOutputMapped = scoresInfRes->rmap();
    auto desc = scoresInfRes->getTensorDesc();
    auto sz = desc.getDims();
    float *scoresPtr = scoresOutputMapped.as<float*>();

    return nms(scoresPtr, sz, threshold);
}

std::vector<std::pair<float, float>> filterReg(const InferenceEngine::MemoryBlob::Ptr& regInfRes, const std::vector<std::pair<size_t, float>>& scores, size_t chSize) {
    InferenceEngine::LockedMemory<const void> bboxesOutputMapped = regInfRes->rmap();
    const float *regPtr = bboxesOutputMapped.as<float*>();
    std::vector<std::pair<float, float>> reg;

    for (auto s : scores) {
        reg.push_back({ regPtr[s.first % chSize], regPtr[chSize + s.first % chSize] });
    }

    return reg;
}

std::vector<std::pair<float, float>> filterWH(const InferenceEngine::MemoryBlob::Ptr& whInfRes, const std::vector<std::pair<size_t, float>>& scores, size_t chSize) {
    InferenceEngine::LockedMemory<const void> bboxesOutputMapped = whInfRes->rmap();
    const float *whPtr = bboxesOutputMapped.as<float*>();
    std::vector<std::pair<float, float>> wh;

    for (auto s : scores) {
        wh.push_back({ whPtr[s.first % chSize], whPtr[chSize + s.first % chSize] });
    }

    return wh;
}

std::vector<ModelCenterNet::BBox> calcBBoxes(const std::vector<std::pair<size_t, float>>& scores, const std::vector<std::pair<float, float>>& reg,
    const std::vector<std::pair<float, float>>& wh, const InferenceEngine::SizeVector& sz) {
    std::vector<ModelCenterNet::BBox> bboxes(scores.size());

    for (size_t i = 0; i < bboxes.size(); ++i) {
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

void transform(std::vector<ModelCenterNet::BBox>& bboxes, const InferenceEngine::SizeVector& sz, int scale, float centerX, float centerY) {
    cv::Mat1f trans = getAffineTransform(centerX, centerY, scale, 0, sz[2], sz[3], true);

    for (auto& b : bboxes) {
        ModelCenterNet::BBox newbb;

        newbb.left = trans.at<float>(0, 0) *  b.left + trans.at<float>(0, 1) *  b.top + trans.at<float>(0, 2);
        newbb.top = trans.at<float>(1, 0) *  b.left + trans.at<float>(1, 1) *  b.top + trans.at<float>(1, 2);
        newbb.right = trans.at<float>(0, 0) *  b.right + trans.at<float>(0, 1) *  b.bottom + trans.at<float>(0, 2);
        newbb.bottom = trans.at<float>(1, 0) *  b.right + trans.at<float>(1, 1) *  b.bottom + trans.at<float>(1, 2);

        b = newbb;
    }
}

std::unique_ptr<ResultBase> ModelCenterNet::postprocess(InferenceResult& infResult) {
    // --------------------------- Filter data and get valid indices ---------------------------------
    auto heatInfRes = infResult.outputsData[outputsNames[0]];
    auto sz = heatInfRes->getTensorDesc().getDims();;
    auto chSize = sz[2] * sz[3];
    auto scores = filterScores(heatInfRes, confidenceThreshold);

    auto regInfRes = infResult.outputsData[outputsNames[1]];
    auto reg = filterReg(regInfRes, scores, chSize);

    auto whInfRes = infResult.outputsData[outputsNames[2]];
    auto wh = filterWH(whInfRes, scores, chSize);

    // --------------------------- Calculate bounding boxes & apply inverse affine transform ----------
    auto bboxes = calcBBoxes(scores, reg, wh, sz);

    auto imgWidth = infResult.internalModelData->asRef<InternalImageModelData>().inputImgWidth;
    auto imgHeight = infResult.internalModelData->asRef<InternalImageModelData>().inputImgHeight;
    auto scale = std::max(imgWidth, imgHeight);
    float centerX = imgWidth / 2.0f;
    float centerY = imgHeight / 2.0f;

    transform(bboxes, sz, scale, centerX, centerY);

    // --------------------------- Create detection result objects ------------------------------------
    DetectionResult* result = new DetectionResult(infResult.frameId, infResult.metaData);

    result->objects.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        DetectedObject desc;
        desc.confidence = scores[i].second;
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
