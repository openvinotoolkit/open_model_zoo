#include "opencv2/core.hpp"

#ifdef HAVE_OPENCV_DNN
#include "opencv2/open_model_zoo/dnn.hpp"

#include <iostream>

namespace cv { namespace open_model_zoo {

static void setParams(Ptr<dnn::Model> model, const Topology& topology)
{
    // Set input means
    std::map<std::string, Scalar> means;
    topology.getMeans(means);
    if (!means.empty())
    {
        if (means.size() > 1)
            CV_Error(Error::StsNotImplemented, "More than one means subtraction");
        model->setInputMean(means.begin()->second);
    }

    // Set input scale
    std::map<std::string, double> scales;
    topology.getScales(scales);
    if (!scales.empty())
    {
        if (scales.size() > 1)
            CV_Error(Error::StsNotImplemented, "More than one scale");
        model->setInputScale(1.0 / scales.begin()->second);
    }

    std::map<std::string, std::string> moArgs = topology.getModelOptimizerArgs();
    if (moArgs.find("--reverse_input_channels") != moArgs.end())
    {
        model->setInputSwapRB(true);
    }
    model->setInputSize(300, 300);
}

Ptr<dnn::Model> DnnModel(const Topology& topology)
{
    Ptr<dnn::Model> model(new dnn::Model(topology.getModelPath(), topology.getConfigPath()));
    setParams(model, topology);
    return model;
}

Ptr<dnn::DetectionModel> DnnDetectionModel(const Topology& topology)
{
    Ptr<dnn::DetectionModel> model(new dnn::DetectionModel(topology.getModelPath(),
                                                           topology.getConfigPath()));
    setParams(model, topology);
    return model;
}

}}  // namespace cv::omz

#endif  // HAVE_OPENCV_DNN
