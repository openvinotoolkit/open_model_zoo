// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <opencv2/imgproc/imgproc.hpp>

#include "extract_poses.hpp"
#include "peak.hpp"

namespace human_pose_estimation {
static void resizeFeatureMaps(std::vector<cv::Mat>& featureMaps, int upsampleRatio) {
    for (auto& featureMap : featureMaps) {
        cv::resize(featureMap, featureMap, cv::Size(),
                   upsampleRatio, upsampleRatio, cv::INTER_CUBIC);
    }
}

class FindPeaksBody: public cv::ParallelLoopBody {
public:
    FindPeaksBody(const std::vector<cv::Mat>& heatMaps, float minPeaksDistance,
                  std::vector<std::vector<Peak> >& peaksFromHeatMap)
        : heatMaps(heatMaps),
          minPeaksDistance(minPeaksDistance),
          peaksFromHeatMap(peaksFromHeatMap) {}

    virtual void operator()(const cv::Range& range) const {
        for (int i = range.start; i < range.end; i++) {
            findPeaks(heatMaps, minPeaksDistance, peaksFromHeatMap, i);
        }
    }

private:
    const std::vector<cv::Mat>& heatMaps;
    float minPeaksDistance;
    std::vector<std::vector<Peak> >& peaksFromHeatMap;
};

std::vector<HumanPose> extractPoses(
        std::vector<cv::Mat>& heatMaps,
        std::vector<cv::Mat>& pafs,
        int upsampleRatio) {
    resizeFeatureMaps(heatMaps, upsampleRatio);
    resizeFeatureMaps(pafs, upsampleRatio);
    std::vector<std::vector<Peak> > peaksFromHeatMap(heatMaps.size());
    float minPeaksDistance = 3.0f;
    FindPeaksBody findPeaksBody(heatMaps, minPeaksDistance, peaksFromHeatMap);
    cv::parallel_for_(cv::Range(0, static_cast<int>(heatMaps.size())),
                      findPeaksBody);
    int peaksBefore = 0;
    for (size_t heatmapId = 1; heatmapId < heatMaps.size(); heatmapId++) {
        peaksBefore += static_cast<int>(peaksFromHeatMap[heatmapId - 1].size());
        for (auto& peak : peaksFromHeatMap[heatmapId]) {
            peak.id += peaksBefore;
        }
    }
    int keypointsNumber = 18;
    float midPointsScoreThreshold = 0.05f;
    float foundMidPointsRatioThreshold = 0.8f;
    int minJointsNumber = 3;
    float minSubsetScore = 0.2f;
    std::vector<HumanPose> poses = groupPeaksToPoses(
                peaksFromHeatMap, pafs, keypointsNumber, midPointsScoreThreshold,
                foundMidPointsRatioThreshold, minJointsNumber, minSubsetScore);
    return poses;
}
} // namespace human_pose_estimation
