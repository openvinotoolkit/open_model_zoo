#ifndef __OPENCV_OPEN_MODEL_ZOO_TEXT_RECOGNITION_HPP__
#define __OPENCV_OPEN_MODEL_ZOO_TEXT_RECOGNITION_HPP__

#include "opencv2/open_model_zoo.hpp"
#include "opencv2/open_model_zoo/topologies.hpp"

using namespace cv::open_model_zoo::topologies;

namespace cv { namespace open_model_zoo {

class CV_EXPORTS_W TextRecognitionPipelineImpl
{
public:
    CV_WRAP TextRecognitionPipelineImpl(const Topology& detection = text_detection(),
                                        const Topology& recognition = text_recognition());

    CV_WRAP void process(InputArray frame, CV_OUT std::vector<RotatedRect>& rects,
                         CV_OUT std::vector<String>& texts);

private:
    struct Impl;
    Ptr<Impl> impl;
};

typedef TextRecognitionPipelineImpl TextRecognitionPipeline;

#if 0
// This is a trick to enable open_model_zoo::TextRecognitionPipeline both in Python and in C++
CV_WRAP_AS(TextRecognitionPipeline)
Ptr<TextRecognitionPipeline> createTextRecognitionPipeline(const Topology& detection = text_detection(),
                                                           const Topology& recognition = text_recognition());
#endif

}}  // namespace cv::open_model_zoo

#endif  // __OPENCV_OPEN_MODEL_ZOO_TEXT_RECOGNITION_HPP__
