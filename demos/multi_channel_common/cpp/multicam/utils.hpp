// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <utility>

namespace mcam {

struct string_ref final {
public:
    // Non-explicit constructor is intentional here
    string_ref(const char* str); // NOLINT
    string_ref(const std::string& str); // NOLINT
    string_ref(const string_ref&) = default;

    const char* data() const { return ptr; }
    std::size_t size() const { return len; }
    bool empty() const { return 0 == len; }

private:
    const char* ptr = nullptr;
    std::size_t len = 0;
};

bool operator==(const string_ref& str, std::nullptr_t);
bool operator==(std::nullptr_t, const string_ref& str);
bool operator!=(const string_ref& str, std::nullptr_t);
bool operator!=(std::nullptr_t, const string_ref& str);

[[noreturn]] void throw_errno_error(string_ref desc, int err);
[[noreturn]] void throw_error(string_ref desc);

struct file_descriptor {
    explicit file_descriptor(int fd_ = -1);
    file_descriptor(const file_descriptor&) = delete;
    file_descriptor(file_descriptor&&) = default;
    ~file_descriptor();

    file_descriptor& operator=(file_descriptor&& other);

    int get() const { return desc; }
    bool valid() const { return -1 != desc; }

    void swap(file_descriptor& other);

private:
    int desc = -1;
};

inline unsigned make_4cc(char a, char b, char c, char d) {
    return (static_cast<unsigned>(a)) |
           (static_cast<unsigned>(b) << 8) |
           (static_cast<unsigned>(c) << 16) |
           (static_cast<unsigned>(d) << 24);
}

}   // namespace mcam
