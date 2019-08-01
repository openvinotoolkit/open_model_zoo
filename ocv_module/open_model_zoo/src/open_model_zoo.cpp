#include "opencv2/open_model_zoo.hpp"

#include <iostream>

namespace cv { namespace open_model_zoo {

struct Topology::Impl
{
    std::string modelURL, modelSHA, modelPath, description, license;
};

Topology::Topology(const std::map<std::string, std::string>& config)
    : impl(new Impl())
{
    impl->modelURL = config.at("model_url");
    impl->modelSHA = config.at("model_sha256");
    impl->modelPath = config.at("model_name");  // TODO: detect cache
    impl->description = config.at("description");
    impl->license = config.at("license");
}

Topology::~Topology() {}

std::string Topology::getDescription() const { return impl->description; }
std::string Topology::getLicense() const { return impl->license; }

void Topology::getModelInfo(std::string& url, std::string& sha, std::string& path) const
{
    url = impl->modelURL;
    sha = impl->modelSHA;
    path = impl->modelPath;
}

}}  // namespace cv::omz
