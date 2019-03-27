// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

#include "cnn.hpp"
#include "detector.hpp"

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
                  const std::string& label, int id)
        : embeddings(embeddings), label(label), id(id) {}
};

class EmbeddingsGallery {
public:
    static const char unknown_label[];
    static const int unknown_id;
    EmbeddingsGallery(const std::string& ids_list, double threshold, int min_size_fr,
                      bool crop_gallery, detection::FaceDetection& detector,
                      const VectorCNN& landmarks_det,
                      const VectorCNN& image_reid);
    size_t size() const;
    std::vector<int> GetIDsByEmbeddings(const std::vector<cv::Mat>& embeddings) const;
    std::string GetLabelByID(int id) const;
    std::vector<std::string> GetIDToLabelMap() const;
    bool LabelExists(const std::string& label) const;

private:
    RegistrationStatus RegisterIdentity(const std::string& identity_label,
                                        const cv::Mat& image,
                                        int min_size_fr,
                                        bool crop_gallery,
                                        detection::FaceDetection& detector,
                                        const VectorCNN& landmarks_det,
                                        const VectorCNN& image_reid,
                                        cv::Mat & embedding);
    std::vector<int> idx_to_id;
    double reid_threshold;
    std::vector<GalleryObject> identities;
};

void AlignFaces(std::vector<cv::Mat>* face_images,
                std::vector<cv::Mat>* landmarks_vec);
