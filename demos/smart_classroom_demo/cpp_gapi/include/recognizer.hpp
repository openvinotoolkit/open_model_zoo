// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "face_reid.hpp"

struct FaceRecognizerConfig {
    double reid_threshold;
    std::vector<GalleryObject> identities;
    std::vector<int> idx_to_id;
    bool greedy_reid_matching;
};

class FaceRecognizer {
public:
    FaceRecognizer(FaceRecognizerConfig config)
        : face_gallery(config.reid_threshold, config.identities, config.idx_to_id, config.greedy_reid_matching) {}

    bool LabelExists(const std::string& label) const {
        return face_gallery.LabelExists(label);
    }

    std::string GetLabelByID(int id) const {
        return face_gallery.GetLabelByID(id);
    }

    std::vector<std::string> GetIDToLabelMap() const {
        return face_gallery.GetIDToLabelMap();
    }

    std::vector<int> Recognize(const std::vector<cv::Rect>& face_rois, std::vector<cv::Mat>& embeddings) {
        if (embeddings.empty()) {
            return std::vector<int>(face_rois.size(), EmbeddingsGallery::unknown_id);
        }
        for (auto& emb : embeddings) {
            emb = emb.reshape(1, {static_cast<int>(emb.total()), 1});
        }
        return face_gallery.GetIDsByEmbeddings(embeddings);
    }

private:
    EmbeddingsGallery face_gallery;
};
