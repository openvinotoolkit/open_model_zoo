// Copyright (C) 2023 KNS Group LLC (YADRO)
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "models.hpp"
#include "reid_gallery.hpp"

#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>
#include <vector>

#include "utils/ocv_common.hpp"
#include "utils/image_utils.h"

// Classes for using in main
struct Result {
    cv::Rect face;
    std::vector<cv::Point> landmarks;
    size_t id;
    float distance;
    std::string label;
    bool real;
    Result(cv::Rect face, std::vector<cv::Point> landmarks, size_t id,
           float distance, const std::string& label, bool real = true) :
        face(face), landmarks(landmarks), id(id), distance(distance), label(label), real(real) {}
};

class FaceRecognizer {
public:
    virtual ~FaceRecognizer() = default;

    virtual std::vector<Result> recognize(const cv::Mat& frame, const std::vector<FaceBox>& faces) = 0;
};

class FaceRecognizerDefault : public FaceRecognizer {
public:
    static const int MAX_NUM_REQUESTS;
    static const int LANDMARKS_NUM;
    FaceRecognizerDefault(
        const BaseConfig& landmarksDetectorConfig,
        const BaseConfig& reidConfig,
        const DetectorConfig& faceRegistrationDetConfig,
        const std::string& faceGalleryPath,
        const double reidThreshold,
        const bool cropGallery,
        const bool allowGrow,
        const bool greedyReidMatching) :
        allowGrow(allowGrow),
        landmarksDetector(landmarksDetectorConfig),
        faceReid(reidConfig),
        faceGallery(faceGalleryPath, reidThreshold, cropGallery,
                    faceRegistrationDetConfig, landmarksDetector, faceReid,
                    greedyReidMatching)
     {};

    std::vector<Result> recognize(const cv::Mat& frame, const std::vector<FaceBox>& faces) {
        cv::Mat origImg = frame.clone();

        std::vector<cv::Mat> landmarks;
        std::vector<cv::Mat> embeddings;
        std::vector<cv::Mat> faceRois;

        auto faceRoi = [&](const FaceBox& face) {
            return frame(face.face);
        };

        int numFaces = faces.size();
        if (numFaces < MAX_NUM_REQUESTS) {
            std::transform(faces.begin(), faces.end(), std::back_inserter(faceRois), faceRoi);
            landmarks = landmarksDetector.infer(faceRois);
            alignFaces(faceRois, landmarks);
            embeddings = faceReid.infer(faceRois);
        } else {
            auto embedding = [&](cv::Mat& emb) { return emb; };
            for (int n = numFaces; n > 0; n -= MAX_NUM_REQUESTS) {
                landmarks.clear();
                faceRois.clear();
                size_t start_idx = size_t(numFaces) - n;
                size_t end_idx = start_idx + std::min(numFaces, MAX_NUM_REQUESTS);
                std::transform(faces.begin() + start_idx, faces.begin() + end_idx, std::back_inserter(faceRois), faceRoi);

                landmarks = landmarksDetector.infer(faceRois);
                alignFaces(faceRois, landmarks);

                std::vector<cv::Mat> tmpEmbeddings = faceReid.infer(faceRois);

                std::transform(tmpEmbeddings.begin(), tmpEmbeddings.end(), std::back_inserter(embeddings), embedding);
            }
        }
        std::vector<std::pair<int, float>> matches = faceGallery.getIDsByEmbeddings(embeddings);
        std::vector<Result> results;
        for (size_t faceIndex = 0; faceIndex < faces.size(); ++faceIndex) {
            if (matches[faceIndex].first == EmbeddingsGallery::unknownId && allowGrow) {
                std::string personName = faceGallery.tryToSave(origImg(faces[faceIndex].face));
                if (personName != "")
                    faceGallery.addFace(origImg(faces[faceIndex].face), embeddings[faceIndex], personName);
            }
            std::vector<cv::Point> lms;
            for (int i = 0; i < LANDMARKS_NUM; ++i) {
                std::cout << landmarks[0] << std::endl;
                int x = static_cast<int>(faces[faceIndex].face.x + landmarks[faceIndex].at<float>(2* i, 0) * faces[faceIndex].face.width);
                int y = static_cast<int>(faces[faceIndex].face.y + landmarks[faceIndex].at<float>(2* i + 1, 0) * faces[faceIndex].face.height);
                lms.emplace_back(x, y);
            }
            results.emplace_back(faces[faceIndex].face, lms, matches[faceIndex].first,
                                 matches[faceIndex].second, faceGallery.getLabelByID(matches[faceIndex].first));
        }
        return results;
    }
protected:
    bool allowGrow;
    AsyncModel landmarksDetector;
    AsyncModel faceReid;
    EmbeddingsGallery faceGallery;
};

class AntiSpoofer {
public:
    static const int MAX_NUM_REQUESTS;
    AntiSpoofer(const BaseConfig& antiSpoofConfig, const float spoofThreshold=40.0) :
        antiSpoof(antiSpoofConfig), spoofThreshold(spoofThreshold)
     {}

    void process(const cv::Mat& frame, const std::vector<FaceBox>& faces, std::vector<Result>& results) {
        if (!antiSpoof.enabled()) {
            return;
        }
        cv::Mat origImg = frame.clone();

        std::vector<cv::Mat> faceRois;
        std::vector<cv::Mat> spoofs;

        auto faceRoi = [&](const FaceBox& face) {
            return frame(face.face);
        };
        int numFaces = faces.size();
        if (numFaces < MAX_NUM_REQUESTS) {
            std::transform(faces.begin(), faces.end(), std::back_inserter(faceRois), faceRoi);
            spoofs = antiSpoof.infer(faceRois);
        } else {
            auto func = [&](cv::Mat& spoof) { return spoof; };
            for (int n = numFaces; n > 0; n -= MAX_NUM_REQUESTS) {
                faceRois.clear();
                size_t startIdx = size_t(numFaces) - n;
                size_t endIdx = startIdx + std::min(numFaces, MAX_NUM_REQUESTS);
                std::transform(faces.begin() + startIdx, faces.begin() + endIdx, std::back_inserter(faceRois), faceRoi);
                std::vector<cv::Mat> tmpSpoofs = antiSpoof.infer(faceRois);
                std::transform(tmpSpoofs.begin(), tmpSpoofs.end(), std::back_inserter(spoofs), func);
            }
        }
        for (size_t faceIndex = 0; faceIndex < faces.size(); ++faceIndex) {
            results[faceIndex].real = isReal(spoofs[faceIndex]);
        }
    }
private:
    AsyncModel antiSpoof;
    float spoofThreshold;

    bool isReal(cv::Mat& spoof) {
        float probability = spoof.at<float>(0) * 100;
        return probability > spoofThreshold;
    }
};

const int FaceRecognizerDefault::LANDMARKS_NUM = 5;
const int FaceRecognizerDefault::MAX_NUM_REQUESTS = 16;
const int AntiSpoofer::MAX_NUM_REQUESTS = 16;
