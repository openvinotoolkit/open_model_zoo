#include "opencv2/open_model_zoo.hpp"
#include "opencv2/open_model_zoo/human_pose_estimation.hpp"
#include "opencv2/open_model_zoo/dnn.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

// TODO: make it work without Inference Engine as well
#ifdef HAVE_INF_ENGINE
#include "human_pose.hpp"
#include "human_pose_estimator.hpp"
#include "render_human_pose.hpp"

using namespace human_pose_estimation;
#endif

namespace cv { namespace open_model_zoo {

struct HumanPoseEstimationImpl::Impl
{
#ifdef HAVE_INF_ENGINE
    Impl(const std::string& modelPath, const std::string& device)
        : estimator(modelPath, device == "GPU16" ? "GPU" : device, false) {}

    HumanPoseEstimator estimator;
#endif
};

HumanPoseEstimation::HumanPoseEstimationImpl(const std::string& device)
{
#ifndef HAVE_INF_ENGINE
    CV_Error(Error::StsNotImplemented, "Human pose estimation without Inference Engine");
#endif

    Topology t;
    if (device == "GPU16" || device == "MYRIAD")
        t = topologies::human_pose_estimation_fp16();
    else
        t = topologies::human_pose_estimation();

    impl.reset(new Impl(t.getConfigPath(), device));
}

HumanPoseEstimation::HumanPoseEstimationImpl(const Topology& t, const std::string& device)
{
#ifndef HAVE_INF_ENGINE
    CV_Error(Error::StsNotImplemented, "Human pose estimation without Inference Engine");
#endif
    impl.reset(new Impl(t.getConfigPath(), device));
}

void HumanPoseEstimation::process(InputArray frame, CV_OUT std::vector<HumanPose>& humanPoses)
{
#ifdef HAVE_INF_ENGINE
    auto poses = impl->estimator.estimate(frame.getMat());
    humanPoses.resize(poses.size());
    for (size_t i = 0; i < poses.size(); ++i)
    {
        humanPoses[i].keypoints = poses[i].keypoints;
        humanPoses[i].type = "COCO";
    }
#else
    CV_UNUSED(frame); CV_UNUSED(humanPoses);
    CV_Error(Error::StsNotImplemented, "Inference Engine is required");
#endif
}

void HumanPoseEstimation::render(InputOutputArray frame, const std::vector<HumanPose>& humanPoses)
{
#ifdef HAVE_INF_ENGINE
    Mat img = frame.getMat();

    std::vector<human_pose_estimation::HumanPose> poses(humanPoses.size());
    for (size_t i = 0; i < poses.size(); ++i)
    {
        poses[i].keypoints = humanPoses[i].keypoints;
    }
    renderHumanPose(poses, img);
#else
    CV_UNUSED(frame); CV_UNUSED(humanPoses);
    CV_Error(Error::StsNotImplemented, "Inference Engine is required");
#endif
}

}}  // namespace cv::open_model_zoo
