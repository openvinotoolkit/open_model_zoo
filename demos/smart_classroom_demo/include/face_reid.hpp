/*
// Copyright (c) 2018 Intel Corporation
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

    std::vector<int> GetIDsByEmbeddings(const std::vector<cv::Mat>& embeddings) const;
    std::string GetLabelByID(int id) const;

private:
    std::vector<int> idx_to_id;
    double reid_threshold;
    std::vector<GalleryObject> identities;
};

void AlignFaces(std::vector<cv::Mat>* face_images,
                std::vector<cv::Mat>* landmarks_vec);
