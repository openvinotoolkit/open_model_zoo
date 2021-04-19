// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <unordered_map>
#include <tuple>
#include <deque>

#include <opencv2/core.hpp>

class TrackableObject {
public:
    TrackableObject(cv::Rect2i bb, const std::vector<float> &r, cv::Point centroid)
            : bbox{bb}, reid{r}, updated{false}, disappeared(0) {
        centroids.push_back(centroid);
    }

    //std::tuple<int, int, int, int> bbox;
    cv::Rect bbox;
    std::vector<float> reid;
    std::vector<cv::Point> centroids;
    bool updated;
    int disappeared;
};

class PersonTrackers {
public:
    PersonTrackers() : trackIdGenerator{0}, similarityThreshold{0.7f}, maxDisappeared{10} {}

    void similarity(std::list<TrackableObject> &tos) {
        std::deque<std::pair<int, float>> sim;
        if (trackers.size() > 0) {
            auto ntos = tos.size();
            for (const auto& to : tos) {
                for (auto &tracker : trackers) {
                    float cosine = 0;
                    cosine = cosineSimilarity(to.reid, tracker.second.reid);
     
                    if (cosine > similarityThreshold)
                        sim.push_back(std::make_pair(tracker.first, cosine));
                }
                if (!sim.empty()) {
                    int maxSimilarity = getMaxSimilarity(sim);
                    if (maxSimilarity < 0)
                        continue;
                    trackers.at(maxSimilarity) = to;
                    trackers.at(maxSimilarity).updated = true;
                    trackers.at(maxSimilarity).disappeared = 0;
                } else {
                    trackers.insert({trackIdGenerator, to});
                    trackers.at(trackIdGenerator).updated = true;
                    trackers.at(trackIdGenerator).disappeared = 0;
                    trackIdGenerator += 1;
                }
            }

            for (auto it = trackers.begin(); it != trackers.end(); ) {
                if (!it->second.updated) {
                    it->second.disappeared += 1;
                    if (it->second.disappeared > maxDisappeared) {
                        it = trackers.erase(it);
                        continue;
                    }
                }
                if (ntos <= trackers.size()) {
                    it->second.updated = false;
                }
                ++it;
            }
        } else {
            registerTrackers(tos);
        }
    }

    float cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b) {
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

    void registerTrackers(std::list<TrackableObject> &tos) {
        for (auto &to : tos) {
            to.disappeared = 0;
            trackers.insert({trackIdGenerator, to});
            trackIdGenerator += 1;
        }
    }

    int getMaxSimilarity(std::deque<std::pair<int, float>> &similList) {
        std::sort(similList.begin(), similList.end(), [](std::pair<int, float> a, std::pair<int, float> b) {
            return std::get<1>(a) > std::get<1>(b);
        });

        for (auto &sim : similList) {
            if (!trackers.at(std::get<0>(sim)).updated)
                return std::get<0>(sim);
        }

        return -1;
    }

    void clear() {
        trackers.clear();
        trackIdGenerator = 0;
    }

public:
    std::unordered_map<int, TrackableObject> trackers;

private:
    int trackIdGenerator;
    float similarityThreshold;
    int maxDisappeared;
};
