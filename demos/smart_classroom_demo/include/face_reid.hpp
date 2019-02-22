// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

#include "cnn.hpp"

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
    static const std::string unknown_label;
    static const int unknown_id;
    EmbeddingsGallery(const std::string& ids_list, double threshold,
                      const VectorCNN& landmarks_det,
                      const VectorCNN& image_reid);
    size_t size() const;
    std::vector<int> GetIDsByEmbeddings(const std::vector<cv::Mat>& embeddings) const;
    std::string GetLabelByID(int id) const;
    std::vector<std::string> GetIDToLabelMap() const;

private:
    std::vector<int> idx_to_id;
    double reid_threshold;
    std::vector<GalleryObject> identities;
};

void AlignFaces(std::vector<cv::Mat>* face_images,
                std::vector<cv::Mat>* landmarks_vec);
