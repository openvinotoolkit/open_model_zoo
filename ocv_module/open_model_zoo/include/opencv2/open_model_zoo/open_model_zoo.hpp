#ifndef __OPENCV_OPEN_MODEL_ZOO_OPEN_MODEL_ZOO_HPP__
#define __OPENCV_OPEN_MODEL_ZOO_OPEN_MODEL_ZOO_HPP__

#include "opencv2/core.hpp"

#include <iostream>

namespace cv { namespace open_model_zoo {

    class CV_EXPORTS_W Topology
    {
    public:
        Topology(const std::string& modelURL, const std::string& description);

        virtual ~Topology();

        CV_WRAP void download();

        CV_WRAP std::string getDescription() const;

        CV_WRAP std::string getModelURL() const;

    protected:
        struct Impl;
        Ptr<Impl> impl;
    };

    CV_EXPORTS_W Ptr<Topology> face_detection_retail();

}}  // namespace cv::open_model_zoo

#endif  // __OPENCV_OPEN_MODEL_ZOO_OPEN_MODEL_ZOO_HPP__
