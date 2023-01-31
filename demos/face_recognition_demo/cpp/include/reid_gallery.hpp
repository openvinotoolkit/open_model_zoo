// Copyright (C) 2023 KNS Group LLC (YADRO)
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "models.hpp"

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

enum class RegistrationStatus {
  SUCCESS,
  FAILURE_LOW_QUALITY,
  FAILURE_NOT_DETECTED,
};

struct GalleryObject {
    std::vector<cv::Mat> embeddings;
    std::string label;
    int id;

    GalleryObject(const std::vector<cv::Mat>& embeddings,
                  const std::string& label, int id) :
        embeddings(embeddings), label(label), id(id) {}
};

class EmbeddingsGallery {
public:
    static constexpr char unknownLabel[] = "Unknown";
    static constexpr int unknownId = -1;
    static constexpr float unknownDistance = 1.0;
    EmbeddingsGallery(const std::string& ids_list, double threshold,
                      bool crop_gallery, const DetectorConfig& detector_config,
                      AsyncModel& landmarks_det,
                      AsyncModel& image_reid,
                      bool use_greedy_matcher=false);
    size_t size() const;
    std::vector<std::pair<int, float>> getIDsByEmbeddings(const std::vector<cv::Mat>& embeddings) const;
    std::string getLabelByID(int id) const;
    bool labelExists(const std::string& label) const;
    std::string tryToSave(cv::Mat new_face);
    void addFace(cv::Mat new_face, cv::Mat embedding, std::string label);

private:
    RegistrationStatus registerIdentity(const std::string& identity_label,
                                        const cv::Mat& image,
                                        const bool crop_gallery,
                                        FaceDetector& detector,
                                        AsyncModel& landmarks_det,
                                        AsyncModel& image_reid,
                                        cv::Mat& embedding);
    std::vector<int> idxToId;
    double reidThreshold;
    std::vector<GalleryObject> identities;
    bool useGreedyMatcher;
    std::string faceGalleryPath;
};
