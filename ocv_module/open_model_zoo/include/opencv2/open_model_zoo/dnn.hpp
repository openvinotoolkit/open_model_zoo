#ifndef __OPENCV_OPEN_MODEL_ZOO_DNN_HPP__
#define __OPENCV_OPEN_MODEL_ZOO_DNN_HPP__

#include "opencv2/open_model_zoo.hpp"
#include "opencv2/dnn.hpp"

namespace cv { namespace open_model_zoo {

CV_EXPORTS_W Ptr<dnn::Model> DnnModel(const Topology& topology);

CV_EXPORTS_W Ptr<dnn::ClassificationModel> DnnClassificationModel(const Topology& topology);

CV_EXPORTS_W Ptr<dnn::DetectionModel> DnnDetectionModel(const Topology& topology);

}}  // namespace cv::open_model_zoo

#endif  // __OPENCV_OPEN_MODEL_ZOO_DNN_HPP__
