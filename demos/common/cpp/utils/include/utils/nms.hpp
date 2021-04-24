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

#pragma once

#include "opencv2/core.hpp"
#include <vector>


template <typename Anchor>
std::vector<int> nms(const std::vector<Anchor>& boxes, const std::vector<float>& scores,
                     const float thresh, bool includeBoundaries=false) {
    std::vector<float> areas(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        areas[i] = (boxes[i].right - boxes[i].left + includeBoundaries) * (boxes[i].bottom - boxes[i].top + includeBoundaries);
    }
    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&scores](int o1, int o2) { return scores[o1] > scores[o2]; });

    size_t ordersNum = 0;
    for (; ordersNum < order.size() && scores[order[ordersNum]] >= 0; ordersNum++);

    std::vector<int> keep;
    bool shouldContinue = true;
    for (size_t i = 0; shouldContinue && i < ordersNum; ++i) {
        auto idx1 = order[i];
        if (idx1 >= 0) {
            keep.push_back(idx1);
            shouldContinue = false;
            for (size_t j = i + 1; j < ordersNum; ++j) {
                auto idx2 = order[j];
                if (idx2 >= 0) {
                    shouldContinue = true;
                    auto overlappingWidth = std::fminf(boxes[idx1].right, boxes[idx2].right) - std::fmaxf(boxes[idx1].left, boxes[idx2].left);
                    auto overlappingHeight = std::fminf(boxes[idx1].bottom, boxes[idx2].bottom) - std::fmaxf(boxes[idx1].top, boxes[idx2].top);
                    auto intersection = overlappingWidth > 0 && overlappingHeight > 0 ? overlappingWidth * overlappingHeight : 0;
                    auto overlap = intersection / (areas[idx1] + areas[idx2] - intersection);

                    if (overlap >= thresh) {
                        order[j] = -1;
                    }
                }
            }
        }
    }
    return keep;
}
