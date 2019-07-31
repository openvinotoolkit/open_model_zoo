#include "opencv2/open_model_zoo.hpp"

#include <iostream>

namespace cv { namespace open_model_zoo {

struct Topology::Impl
{
    std::string modelURL, description;
};

Topology::Topology(const std::string& modelURL, const std::string& description)
    : impl(new Impl())
{
    impl->modelURL = modelURL;
    impl->description = description;
}

Topology::~Topology()
{

}

std::string Topology::getDescription() const
{
    return impl->description;
}

std::string Topology::getModelURL() const
{
    return impl->modelURL;
}

}}  // namespace cv::omz
