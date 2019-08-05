#ifndef __OPENCV_OPEN_MODEL_ZOO_OPEN_MODEL_ZOO_HPP__
#define __OPENCV_OPEN_MODEL_ZOO_OPEN_MODEL_ZOO_HPP__

#include "opencv2/core.hpp"

#include <map>

namespace cv { namespace open_model_zoo {

    class CV_EXPORTS_W_SIMPLE Topology
    {
    public:
        Topology();

        Topology(const std::map<std::string, std::string>& config);

        CV_WRAP void download();

        CV_WRAP std::string getDescription() const;

        CV_WRAP std::string getLicense() const;

        CV_WRAP void getArchiveInfo(CV_OUT String& url, CV_OUT String& sha256,
                                    CV_OUT String& path) const;

        CV_WRAP void getModelInfo(CV_OUT String& url, CV_OUT String& sha256,
                                  CV_OUT String& path) const;

        CV_WRAP void getConfigInfo(CV_OUT String& url, CV_OUT String& sha256,
                                   CV_OUT String& path) const;

        CV_WRAP String getModelPath() const;

        CV_WRAP String getConfigPath() const;

        void getMeans(std::map<std::string, Scalar>& means) const;

        void getScales(std::map<std::string, double>& scales) const;

        std::map<String, String> getModelOptimizerArgs() const;

        CV_WRAP void convertToIR(CV_OUT String& xmlPath, CV_OUT String& binPath) const;

    protected:
        struct Impl;
        Ptr<Impl> impl;
    };

}}  // namespace cv::open_model_zoo

#endif  // __OPENCV_OPEN_MODEL_ZOO_OPEN_MODEL_ZOO_HPP__
