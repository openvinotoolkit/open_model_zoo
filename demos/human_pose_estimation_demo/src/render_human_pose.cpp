// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>

#include "human_pose_estimator.hpp"
#include "render_human_pose.hpp"


namespace human_pose_estimation {
void renderHumanPose(const std::vector<HumanPose>& poses, cv::Mat& image, const std::string& modeltype) {
    CV_Assert(image.type() == CV_8UC3);

    const std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0), cv::Scalar(255, 85, 0), cv::Scalar(255, 170, 0),
        cv::Scalar(255, 255, 0), cv::Scalar(170, 255, 0), cv::Scalar(85, 255, 0),
        cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 85), cv::Scalar(0, 255, 170),
        cv::Scalar(0, 255, 255), cv::Scalar(0, 170, 255), cv::Scalar(0, 85, 255),
        cv::Scalar(0, 0, 255), cv::Scalar(85, 0, 255), cv::Scalar(170, 0, 255),
        cv::Scalar(255, 0, 255), cv::Scalar(255, 0, 170), cv::Scalar(255, 0, 85),
		cv::Scalar(255, 128, 0), cv::Scalar(255, 85, 128), cv::Scalar(255, 170, 128),
		cv::Scalar(255, 255, 128), cv::Scalar(170, 255, 128), cv::Scalar(85, 255, 128),
		cv::Scalar(128, 255, 0), cv::Scalar(128, 255, 85), cv::Scalar(128, 255, 170),
		cv::Scalar(128, 255, 255), cv::Scalar(128, 170, 255), cv::Scalar(128, 85, 255),
		cv::Scalar(128, 0, 255), cv::Scalar(85, 128, 255), cv::Scalar(170, 128, 255),
		cv::Scalar(255, 128, 255), cv::Scalar(255, 128, 170), cv::Scalar(255, 128, 85)
		,cv::Scalar(255, 0, 0), cv::Scalar(255, 85, 0), cv::Scalar(255, 170, 0),
		cv::Scalar(255, 255, 0), cv::Scalar(170, 255, 0), cv::Scalar(85, 255, 0),
		cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 85), cv::Scalar(0, 255, 170),
		cv::Scalar(0, 255, 255), cv::Scalar(0, 170, 255), cv::Scalar(0, 85, 255),
		cv::Scalar(0, 0, 255), cv::Scalar(85, 0, 255), cv::Scalar(170, 0, 255),
		cv::Scalar(255, 0, 255), cv::Scalar(255, 0, 170), cv::Scalar(255, 0, 85),
		cv::Scalar(255, 128, 0), cv::Scalar(255, 85, 128), cv::Scalar(255, 170, 128),
		cv::Scalar(255, 255, 128), cv::Scalar(170, 255, 128), cv::Scalar(85, 255, 128),
		cv::Scalar(128, 255, 0), cv::Scalar(128, 255, 85), cv::Scalar(128, 255, 170),
		cv::Scalar(128, 255, 255), cv::Scalar(128, 170, 255), cv::Scalar(128, 85, 255),
		cv::Scalar(128, 0, 255), cv::Scalar(85, 128, 255)
    };

	const std::vector<std::pair<int, int> > limbKeypointsIds;

	if (modeltype == "COCO")
	{
		std::vector<std::pair<int, int> > *ptr = (std::vector<std::pair<int, int> >*)&limbKeypointsIds;
		*ptr = {
			{1, 2},  {1, 5},   {2, 3},
			{3, 4},  {5, 6},   {6, 7},
			{1, 8},  {8, 9},   {9, 10},
			{1, 11}, {11, 12}, {12, 13},
			{1, 0},  {0, 14},  {14, 16},
			{0, 15}, {15, 17}
		};
	}

	else if (modeltype == "MPI")
	{
		std::vector<std::pair<int, int> > *ptr = (std::vector<std::pair<int, int> >*)&limbKeypointsIds;
		*ptr = {
			{0, 1}, { 1,2 }, { 2,3 },
			{ 3,4 }, { 1,5 }, { 5,6 },
			{ 6,7 }, { 1,14 }, { 14,8 }, { 8,9 },
			{ 9,10 }, { 14,11 }, { 11,12 }, { 12,13 }
		};
	}

	else if (modeltype == "BODY_25")
	{
		std::vector<std::pair<int, int> > *ptr = (std::vector<std::pair<int, int> >*)&limbKeypointsIds;
		*ptr = {
			{1, 8}, { 1,2 }, { 1,5 }, { 2,3 }, { 3,4 },
			{ 5,6 }, { 6,7 }, { 8,9 }, { 9,10 }, { 10,11 },
			{ 8,12 }, { 12,13 }, { 13,14 }, { 1,0 }, { 0,15 },
			{ 15,17 }, { 0,16 }, { 16,18 }, { 14,19 }, { 19,20 },
			{ 14,21 }, { 11,22 }, { 22,23 }, { 11,24 }
		};
	}

	else if (modeltype == "FACE")
	{
		std::vector<std::pair<int, int> > *ptr = (std::vector<std::pair<int, int> >*)&limbKeypointsIds;
		*ptr = {
			{0, 1}, { 1,2 }, { 1,3 }, { 3,4 }, { 4,5 },
			{ 5,6 }, { 6,7 }, { 7,8 }, { 8,9 }, { 9,10 },
			{ 10,11 }, { 11,12 }, { 12,13 }, { 13,14 }, { 14,15 },
			{ 15,16 }, { 17,18 }, { 18,19 }, { 19,20 }, { 20,21 },
			{ 22,23 }, { 23,24 }, { 24,25 }, { 25,26 }, { 27,28 },
			{ 28,29 }, { 29,30 }, { 31,32 }, { 32,33 }, { 33,34 },
			{ 34,35 }, { 36,37 }, { 37,38 }, { 38,39 }, { 39,40 },
			{ 40,41 }, { 41,36 }, { 42,43 }, { 43,44 }, { 44,45 },
			{ 45,46 }, { 46,47 }, { 47,42 }, { 48,49 }, { 49,50 },
			{ 50,51 }, { 51,52 }, { 52,53 }, { 53,54 }, { 54,55 },
			{ 55,56 }, { 56,57 }, { 57,58 }, { 58,59 }, { 59,48 },
			{ 60,61 }, { 61,62 }, { 62,63 }, { 63,64 }, { 64,65 },
			{ 65,66 }, { 66,67 }, { 67,60 }, { 37,68 }, { 44,69 }
		};
	}

	else if (modeltype == "HAND")
	{
		std::vector<std::pair<int, int> > *ptr = (std::vector<std::pair<int, int> >*)&limbKeypointsIds;
		*ptr = {
			{0, 1}, { 1,2 }, { 2,3 }, { 3,4 }, { 0,5 },
			{ 5,6 }, { 6,7 }, { 7,8 }, { 0,9 }, { 9,10 },
			{ 10,11 }, { 11,12 }, { 0,13 }, { 13,14 }, { 14,15 },
			{ 15,16 }, { 0,17 }, { 17,18 }, { 18,19 }, { 19,20 }
		};
	}


    const int stickWidth = 4;
    const cv::Point2f absentKeypoint(-1.0f, -1.0f);
    for (const auto& pose : poses) {
        CV_Assert( (pose.keypoints.size() == HumanPoseEstimator::ModelType::COCO) || 
				   (pose.keypoints.size() == HumanPoseEstimator::ModelType::MPI) ||
			       (pose.keypoints.size() == HumanPoseEstimator::ModelType::BODY_25) || 
			       (pose.keypoints.size() == HumanPoseEstimator::ModelType::FACE) || 
			       (pose.keypoints.size() == HumanPoseEstimator::ModelType::HAND) );
	
        for (size_t keypointIdx = 0; keypointIdx < pose.keypoints.size(); keypointIdx++) {
            if (pose.keypoints[keypointIdx] != absentKeypoint) {
                cv::circle(image, pose.keypoints[keypointIdx], 4, colors[keypointIdx], -1);
            }
        }
    }
    cv::Mat pane = image.clone();
    for (const auto& pose : poses) {
        for (const auto& limbKeypointsId : limbKeypointsIds) {
            std::pair<cv::Point2f, cv::Point2f> limbKeypoints(pose.keypoints[limbKeypointsId.first],
                    pose.keypoints[limbKeypointsId.second]);
            if (limbKeypoints.first == absentKeypoint
                    || limbKeypoints.second == absentKeypoint) {
                continue;
            }

            float meanX = (limbKeypoints.first.x + limbKeypoints.second.x) / 2;
            float meanY = (limbKeypoints.first.y + limbKeypoints.second.y) / 2;
            cv::Point difference = limbKeypoints.first - limbKeypoints.second;
            double length = std::sqrt(difference.x * difference.x + difference.y * difference.y);
            int angle = static_cast<int>(std::atan2(difference.y, difference.x) * 180 / CV_PI);
            std::vector<cv::Point> polygon;
            cv::ellipse2Poly(cv::Point2d(meanX, meanY), cv::Size2d(length / 2, stickWidth),
                             angle, 0, 360, 1, polygon);
            cv::fillConvexPoly(pane, polygon, colors[limbKeypointsId.second]);
        }
    }
    cv::addWeighted(image, 0.4, pane, 0.6, 0, image);
}
}  // namespace human_pose_estimation
