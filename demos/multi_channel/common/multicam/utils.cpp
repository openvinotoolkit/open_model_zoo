// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <unistd.h>

namespace mcam {

string_ref::string_ref(const char* str):
    ptr(str),
    len(nullptr != str ? std::strlen(str) : 0) {
}

string_ref::string_ref(const std::string& str):
    ptr(!str.empty() ? str.c_str() : nullptr),
    len(str.size()) {
}

bool operator==(const string_ref& str, std::nullptr_t) {
    return str.empty();
}

bool operator==(std::nullptr_t, const string_ref& str) {
    return str.empty();
}

bool operator!=(const string_ref& str, std::nullptr_t) {
    return !str.empty();
}

bool operator!=(std::nullptr_t, const string_ref& str) {
    return !str.empty();
}

[[noreturn]] void throw_errno_error(string_ref desc, int err) {
    std::stringstream ss;
    if (nullptr != desc) {
        ss << desc.data();
    }
    ss << " " << err << ", " << strerror(err);
    throw std::logic_error(ss.str());
}

void throw_error(string_ref desc) {
    throw std::logic_error(desc == nullptr ? "Generic error" : desc.data());
}

file_descriptor::file_descriptor(int fd_):
    desc(fd_) {
}

file_descriptor::~file_descriptor() {
    if (-1 != desc) {
        close(desc);
    }
}

file_descriptor& file_descriptor::operator=(file_descriptor&& other) {
    if (this != &other) {
        swap(other);
    }
    return *this;
}

void file_descriptor::swap(file_descriptor& other) {
    std::swap(desc, other.desc);
}

}  // namespace mcam
