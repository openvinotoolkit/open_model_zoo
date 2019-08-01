#ifndef __OPENCV_OPEN_MODEL_ZOO_OPEN_MODEL_ZOO_HPP__
#define __OPENCV_OPEN_MODEL_ZOO_OPEN_MODEL_ZOO_HPP__

#include "opencv2/core.hpp"

#include <iostream>
#include <map>

namespace cv { namespace open_model_zoo {

    class CV_EXPORTS_W Topology
    {
    public:
        Topology(const std::map<std::string, std::string>& config);

        virtual ~Topology();

        CV_WRAP void download();

        CV_WRAP std::string getDescription() const;

        CV_WRAP std::string getLicense() const;

        CV_WRAP void getModelInfo(CV_OUT String& url, CV_OUT String& sha256,
                                  CV_OUT String& path) const;

    protected:
        struct Impl;
        Ptr<Impl> impl;
    };

    CV_EXPORTS_W Ptr<Topology> densenet_161();

    // CV_EXPORTS_W Ptr<Topology> face_detection_retail();

}}  // namespace cv::open_model_zoo

#endif  // __OPENCV_OPEN_MODEL_ZOO_OPEN_MODEL_ZOO_HPP__
