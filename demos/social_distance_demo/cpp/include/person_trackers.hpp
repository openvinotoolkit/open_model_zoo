// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <unordered_map>
#include <tuple>
#include <deque>

#include <opencv2/core.hpp>

struct TrackableObject {
    TrackableObject(cv::Rect2i bb, const std::vector<float>& r, cv::Point centroid)
            : bbox{bb}, reid{r}, updated{false}, disappeared(0) {
        centroids.push_back(centroid);
    }

    cv::Rect bbox;
    std::vector<float> reid;
    std::vector<cv::Point> centroids;
    bool updated;
    int disappeared;
};

class PersonTrackers {
public:
    PersonTrackers() : trackIdGenerator{0}, similarityThreshold{0.7f}, maxDisappeared{10} {}

    void similarity(std::list<TrackableObject>& tos) {
        for (const auto& to : tos) {
            std::deque<std::pair<int, float>> sim;
            for (auto& tracker : trackables) {
                if (!tracker.second.updated) {
                    float cosine = cosineSimilarity(to.reid, tracker.second.reid);
                    if (cosine > similarityThreshold) {
                        sim.push_back(std::make_pair(tracker.first, cosine));
                    }
                }
            }

            if (sim.empty()) {
                trackables.insert({ trackIdGenerator, to });
                trackables.at(trackIdGenerator).updated = true;
                trackables.at(trackIdGenerator).disappeared = 0;
                trackIdGenerator += 1;
            } else {
                int maxSimilarity = std::max_element(sim.begin(), sim.end(), [](std::pair<int, float> a, std::pair<int, float> b) {
                    return std::get<1>(a) > std::get<1>(b);
                    })->first;
                trackables.at(maxSimilarity) = to;
                trackables.at(maxSimilarity).updated = true;
                trackables.at(maxSimilarity).disappeared = 0;
            }
        }

        for (auto it = trackables.begin(); it != trackables.end(); ) {
            if (!it->second.updated) {
                it->second.disappeared += 1;
                if (it->second.disappeared > maxDisappeared) {
                    it = trackables.erase(it);
                    continue;
                }
            }
            it->second.updated = false;
            ++it;
        }
    }

    float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            throw "Vector sizes don't match!";
        }

        float dot = 0.0, denomA = 0.0, denomB = 0.0;
        for (std::vector<float>::size_type i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            denomA += a[i] * a[i];
            denomB += b[i] * b[i];
        }
        return static_cast<float>(dot / (sqrt(denomA) * sqrt(denomB) + 1e-6));
    }

public:
    std::unordered_map<int, TrackableObject> trackables;

private:
    int trackIdGenerator;
    float similarityThreshold;
    int maxDisappeared;
};
