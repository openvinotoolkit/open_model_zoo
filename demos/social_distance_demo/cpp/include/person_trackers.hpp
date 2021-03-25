// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <unordered_map>
#include <tuple>
#include <deque>

class TrackableObject {
public:
    TrackableObject(std::tuple<int, int, int, int> bb, const std::vector<float> &r, std::tuple<int, int> centroid)
            : bbox{bb}, reid{r}, updated{false} {
        centroids.push_back(centroid);
    }

    std::tuple<int, int, int, int> bbox;
    std::vector<float> reid;
    std::vector<std::tuple<int, int>> centroids;
    bool updated;
};

class PersonTrackers {
public:
    PersonTrackers() : trackId_generator{0}, similarity_threshold{0.7}, max_disappeared{10} {}

    void similarity(std::list<TrackableObject> &tos) {
        std::deque<std::pair<int, float>> sim;
        if (trackers.size() > 0) {
            auto ntos = tos.size();
            if (ntos == 0) {
                for (auto it = trackers.begin(); it != trackers.end(); ) {
                    dissapeared.at(it->first) += 1;
                    if (dissapeared.at(it->first) > max_disappeared) {
                        dissapeared.erase(it->first);
                        it = trackers.erase(it);
                        continue;
                    }
                    ++it;
                }
            } else {
                for (const auto& to : tos) {
                    for (auto &tracker : trackers) {
                        float cosine = 0;
                        try {
                            cosine = cosine_similarity(to.reid, tracker.second.reid);
                        } catch (std::exception& e) {
                            slog::err << e.what();
                            continue;
                        }
                        if (cosine > similarity_threshold)
                            sim.push_back(std::make_pair(tracker.first, cosine));
                    }
                    if (!sim.empty()) {
                        int max_similarity = get_max_similarity(sim);
                        if (max_similarity < 0)
                            continue;
                        trackers.at(max_similarity).reid = to.reid;
                        trackers.at(max_similarity).bbox = to.bbox;
                        trackers.at(max_similarity).centroids.push_back(to.centroids[0]);
                        trackers.at(max_similarity).updated = true;
                        dissapeared.at(max_similarity) = 0;
                    } else {
                        trackers.insert({trackId_generator, to});
                        trackers.at(trackId_generator).updated = true;
                        dissapeared.insert({trackId_generator, 0});
                        trackId_generator += 1;
                    }
                }

                if (ntos <= trackers.size()) {
                    for (auto it = trackers.begin(); it != trackers.end(); ) {
                        if (!it->second.updated) {
                            dissapeared.at(it->first) += 1;
                            if (dissapeared.at(it->first) > max_disappeared) {
                                dissapeared.erase(it->first);
                                it = trackers.erase(it);
                                continue;
                            }
                        }
                        it->second.updated = false;
                        ++it;
                    }
                }
            }
        } else {
            register_trackers(tos);
        }
    }

    float cosine_similarity(const std::vector<float> &a, const std::vector<float> &b) {
        if (a.size() != b.size()) {
            throw "Vector sizes don't match!";
        }

        float dot = 0.0, denom_a = 0.0, denom_b = 0.0;
        for (std::vector<float>::size_type i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            denom_a += a[i] * a[i];
            denom_b += b[i] * b[i];
        }
        return dot / (sqrt(denom_a) * sqrt(denom_b) + 1e-6);
    }

    void register_trackers(const std::list<TrackableObject> &tos) {
        for (auto &to : tos) {
            trackers.insert({trackId_generator, to});
            dissapeared.insert({trackId_generator, 0});
            trackId_generator += 1;
        }
    }

    int get_max_similarity(std::deque<std::pair<int, float>> &simil_list) {
        std::sort(simil_list.begin(), simil_list.end(), [](std::pair<int, float> a, std::pair<int, float> b) {
            return std::get<1>(a) > std::get<1>(b);
        });

        for (auto &sim : simil_list) {
            if (!trackers.at(std::get<0>(sim)).updated)
                return std::get<0>(sim);
        }

        return -1;
    }

    void clear() {
        trackers.clear();
        trackId_generator = 0;
    }

public:
    std::unordered_map<int, TrackableObject> trackers;

private:
    std::unordered_map<int, int> dissapeared;
    int trackId_generator;
    float similarity_threshold;
    int max_disappeared;
};
