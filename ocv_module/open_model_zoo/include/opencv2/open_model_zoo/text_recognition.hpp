#ifndef __OPENCV_OPEN_MODEL_ZOO_TEXT_RECOGNITION_HPP__
#define __OPENCV_OPEN_MODEL_ZOO_TEXT_RECOGNITION_HPP__

#include "opencv2/open_model_zoo.hpp"
#include "opencv2/open_model_zoo/topologies.hpp"

using namespace cv::open_model_zoo::topologies;

namespace cv { namespace open_model_zoo {


/**
 * @brief Text detection and recognition pipeline.
 */

#if 0
// This is a trick to enable open_model_zoo::TextRecognitionPipeline both in Python and in C++
CV_WRAP_AS(TextRecognitionPipeline)
Ptr<TextRecognitionPipeline> createTextRecognitionPipeline(const String& detectionDevice = "CPU",
                                                           const String& recognitionDevice = "CPU");

CV_WRAP_AS(TextRecognitionPipeline)
Ptr<TextRecognitionPipeline> createTextRecognitionPipeline(const Topology& detection,
                                                           const Topology& recognition,
                                                           const String& detectionDevice = "CPU",
                                                           const String& recognitionDevice = "CPU");
#endif

class CV_EXPORTS_W TextRecognitionPipelineImpl
{
public:
    /**
     * @brief Constructor
     * @param[in] device Computational device
     */
    CV_WRAP TextRecognitionPipelineImpl(const String& detectionDevice = "CPU",
                                        const String& recognitionDevice = "CPU");

    CV_WRAP TextRecognitionPipelineImpl(const Topology& detection,
                                        const Topology& recognition,
                                        const String& detectionDevice = "CPU",
                                        const String& recognitionDevice = "CPU");

    CV_WRAP void process(InputArray frame, CV_OUT std::vector<RotatedRect>& rects,
                         CV_OUT std::vector<String>& texts,
                         CV_OUT std::vector<float>& confidences);

    /**
     * @brief Set maximum number of text rectangles to recognize.
     * By default algorithm recognizes all of detected bounding boxes.
     */
    CV_WRAP void setMaxRectNum(int num);

    /**
     * @brief Define parameter for text recognition threshold
     * Value in range [0, 1]. Default value is 0.2
     */
    CV_WRAP void setRecognitionThresh(float thr);

    /**
     * @brief Define parameter for pixel classification threshold
     * Value in range [0, 1]. Default value is 0.8
     */
    CV_WRAP void setPixelClassificationThresh(float thr);

    /**
     * @brief Define parameter for pixel linking threshold
     * Value in range [0, 1]. Default value is 0.8
     */
    CV_WRAP void setPixelLinkThresh(float thr);

private:
    struct Impl;
    Ptr<Impl> impl;
};

typedef TextRecognitionPipelineImpl TextRecognitionPipeline;

}}  // namespace cv::open_model_zoo

#endif  // __OPENCV_OPEN_MODEL_ZOO_TEXT_RECOGNITION_HPP__
