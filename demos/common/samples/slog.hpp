/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

/**
 * @brief a header file with logging facility for common samples
 * @file log.hpp
 */

#pragma once

#include <string>

namespace slog {

/**
 * @class LogStreamEndLine
 * @brief The LogStreamEndLine class implements an end line marker for a log stream
 */
class LogStreamEndLine { };

static constexpr LogStreamEndLine endl;


/**
 * @class LogStream
 * @brief The LogStream class implements a stream for sample logging
 */
class LogStream {
    std::string _prefix;
    std::ostream* _log_stream;
    bool _new_line;

public:
    /**
     * @brief A constructor. Creates an LogStream object
     * @param prefix The prefix to print
     */
    LogStream(const std::string &prefix, std::ostream& log_stream)
            : _prefix(prefix), _new_line(true) {
        _log_stream = &log_stream;
    }

    /**
     * @brief A stream output operator to be used within the logger
     * @param arg Object for serialization in the logger message
     */
    template<class T>
    LogStream &operator<<(const T &arg) {
        if (_new_line) {
            (*_log_stream) << "[ " << _prefix << " ] ";
            _new_line = false;
        }

        (*_log_stream) << arg;
        return *this;
    }

    // Specializing for LogStreamEndLine to support slog::endl
    LogStream& operator<< (const LogStreamEndLine &arg) {
        _new_line = true;

        (*_log_stream) << std::endl;
        return *this;
    }
};


static LogStream info("INFO", std::cout);
static LogStream warn("WARNING", std::cout);
static LogStream err("ERROR", std::cerr);

}  // namespace slog
