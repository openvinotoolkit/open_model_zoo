// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "execution_providers.hpp"


execution_providers_t createProvidersFromString(const std::string &str){
#ifndef GAPI_IE_EXECUTION_PROVIDERS_AVAILABLE
    if (!str.empty()) {
        std::cerr << "G-API execution providers are not available in OPENCV: " << CV_VERSION
                  << "\nPlease make sure you are using 4.9.0 at least" << std::endl;
        exit(-1);
    }
    return {};
#else // GAPI_IE_EXECUTION_PROVIDERS_AVAILABLE
    execution_providers_t providers;
    std::stringstream params_list(str);
    try {
        std::string line;
        std::string::size_type endline_pos = std::string::npos;
        while (std::getline(params_list, line, ';')) {
            std::string::size_type name_endline_pos = line.find(':');
            if (name_endline_pos == std::string::npos) {
                throw std::runtime_error("Cannot parse execution provider from string: " + line +
                                        ". Name and value should be separated by \":\"" );
            }

            std::string name = line.substr(0, name_endline_pos);
            std::string value = line.substr(name_endline_pos + 2);
            providers.emplace(std::move(name), std::move(value));
        }
    } catch (const std::exception& ex) {
        std::cerr << "Invalid -ep format: " << ex.what() << std::endl;
        exit(-1);
    }
    return providers;
#endif // GAPI_IE_EXECUTION_PROVIDERS_AVAILABLE
}
