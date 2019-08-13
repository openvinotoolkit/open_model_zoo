#ifndef __OPENCV_OPEN_MODEL_ZOO_HUMAN_POSE_ESTIMATION_HPP__
#define __OPENCV_OPEN_MODEL_ZOO_HUMAN_POSE_ESTIMATION_HPP__

#include "opencv2/open_model_zoo.hpp"
#include "opencv2/open_model_zoo/topologies.hpp"

using namespace cv::open_model_zoo::topologies;

namespace cv { namespace open_model_zoo {

// This is a trick to enable open_model_zoo::HumanPoseEstimation both in Python and in C++
#if 0
/**
 * @brief Constructor
 * @param[in] device Computational device
 */
CV_WRAP_AS(HumanPoseEstimation)
Ptr<HumanPoseEstimation> createHumanPoseEstimation(const String& device = "CPU");

/**
 * @brief Constructor
 * @param[in] t open_model_zoo::Topology instance
 * @param[in] device Computational device
 */
CV_WRAP_AS(HumanPoseEstimation)
Ptr<HumanPoseEstimation> createHumanPoseEstimation(const Topology& t, const String& device = "CPU");
#endif

struct CV_EXPORTS_W_SIMPLE HumanPose
{
    CV_PROP std::vector<cv::Point2f> keypoints;
    CV_PROP String type;  // COCO or MPI
};

/**
 * @brief Human body pose estimation.
 */
class CV_EXPORTS_W HumanPoseEstimationImpl
{
public:
    /**
     * @brief Constructor
     * @param[in] device Computational device
     */
    CV_WRAP HumanPoseEstimationImpl(const String& device = "CPU");

    /**
     * @brief Constructor
     * @param[in] t open_model_zoo::Topology instance
     * @param[in] device Computational device
     */
    CV_WRAP HumanPoseEstimationImpl(const Topology& t, const String& device = "CPU");

    CV_WRAP void process(InputArray frame, CV_OUT std::vector<HumanPose>& poses);

    CV_WRAP static void render(InputOutputArray frame, const std::vector<HumanPose>& poses);

private:
    struct Impl;
    Ptr<Impl> impl;
};

typedef HumanPoseEstimationImpl HumanPoseEstimation;

}}  // namespace cv::open_model_zoo

#endif  // __OPENCV_OPEN_MODEL_ZOO_HUMAN_POSE_ESTIMATION_HPP__
