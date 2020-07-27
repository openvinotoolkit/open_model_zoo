#pragma once
#include <map>
#include <string>
#include "gflags/gflags.h"

struct CnnConfig{
    std::string devices;
    std::string cpuExtensionsPath;
    std::string clKernelsConfigPath;
    unsigned int maxAsyncRequests=2;
    std::map < std::string, std::string> execNetworkConfig;
};

class ConfigFactory{
public:
    static CnnConfig GetUserConfig();
    static CnnConfig GetMinLatencyConfig();
protected:
    static CnnConfig GetCommonConfig();
};

