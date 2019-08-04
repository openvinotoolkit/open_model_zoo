#include "opencv2/open_model_zoo.hpp"

#include <iostream>
#include <sstream>

namespace cv { namespace open_model_zoo {

struct Topology::Impl
{
    std::string modelURL, modelSHA, modelPath,
                configURL, configSHA, configPath,
                description, license;
    std::map<std::string, std::string> modelOptimizerArgs;
};

Topology::Topology() {}

Topology::Topology(const std::map<std::string, std::string>& config)
    : impl(new Impl())
{
    impl->modelURL = config.at("model_url");
    impl->modelSHA = config.at("model_sha256");
    impl->modelPath = config.at("model_name");  // TODO: detect cache
    impl->configURL = config.at("config_url");
    impl->configSHA = config.at("config_sha256");
    impl->configPath = config.at("config_name");  // TODO: detect cache
    impl->description = config.at("description");
    impl->license = config.at("license");

    auto it = config.find("model_optimizer_args");
    if (it != config.end())
    {
        std::stringstream s(it->second);
        std::string arg;
        while (s >> arg)
        {
            int delim = arg.find('=');
            impl->modelOptimizerArgs[arg.substr(0, delim)] = arg.substr(delim + 1);
        }
    }
}

std::string Topology::getDescription() const { return impl->description; }
std::string Topology::getLicense() const { return impl->license; }

void Topology::getModelInfo(std::string& url, std::string& sha, std::string& path) const
{
    url = impl->modelURL;
    sha = impl->modelSHA;
    path = impl->modelPath;
}

std::string Topology::getModelPath() const
{
    return impl->modelPath;
}

void Topology::getConfigInfo(String& url, String& sha, String& path) const
{
    url = impl->configURL;
    sha = impl->configSHA;
    path = impl->configPath;
}

std::string Topology::getConfigPath() const
{
    return impl->configPath;
}

void Topology::getMeans(std::map<std::string, Scalar>& means) const
{
    // mean_values has format "name1[v1,v2,v3] name2[v1,v2,v3]"
    means.clear();
    auto it = impl->modelOptimizerArgs.find("--mean_values");
    if (it == impl->modelOptimizerArgs.end())
        return;

    std::string arg, name, value;
    std::stringstream args(it->second);
    while (args >> arg)
    {
        int start = arg.find('[');
        int end = arg.find(']');
        name = arg.substr(0, start);
        std::stringstream ss(arg.substr(start + 1, end - 1));

        Scalar mean;
        int i = 0;
        while (std::getline(ss, value, ','))
        {
            std::stringstream(value) >> mean[i];
            i += 1;
        }
        arg = arg.substr(end + 1);

        means[name] = mean;
    }
}

void Topology::getScales(std::map<std::string, double>& scales) const
{
    // mean_values has format "name1[value1] name2[value2]"
    scales.clear();
    auto it = impl->modelOptimizerArgs.find("--scale_values");
    if (it == impl->modelOptimizerArgs.end())
        return;

    std::string arg, name;
    std::stringstream args(it->second);
    double value;
    while (args >> arg)
    {
        int start = arg.find('[');
        int end = arg.find(']');
        name = arg.substr(0, start);
        std::stringstream(arg.substr(start + 1, end - 1)) >> value;
        scales[name] = value;
    }
}

std::map<String, String> Topology::getModelOptimizerArgs() const
{
    return impl->modelOptimizerArgs;
}

}}  // namespace cv::omz
