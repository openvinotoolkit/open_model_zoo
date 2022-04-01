// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "utils/images_capture.h"

#include <string.h>

#ifdef _WIN32
#    include "w_dirent.hpp"
#else
#    include <dirent.h>  // for closedir, dirent, opendir, readdir, DIR
#endif

#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

class InvalidInput : public std::runtime_error {
public:
    explicit InvalidInput(const std::string& message) noexcept : std::runtime_error(message) {}
};

class OpenError : public std::runtime_error {
public:
    explicit OpenError(const std::string& message) noexcept : std::runtime_error(message) {}
};

class ImreadWrapper : public ImagesCapture {
    cv::Mat img;
    bool canRead;

public:
    ImreadWrapper(const std::string& input, bool loop) : ImagesCapture{loop}, canRead{true} {
        auto startTime = std::chrono::steady_clock::now();

        std::ifstream file(input.c_str());
        if (!file.good())
            throw InvalidInput("Can't find the image by " + input);

        img = cv::imread(input);
        if (!img.data)
            throw OpenError("Can't open the image from " + input);
        else
            readerMetrics.update(startTime);
    }

    double fps() const override {
        return 1.0;
    }

    std::string getType() const override {
        return "IMAGE";
    }

    cv::Mat read() override {
        if (loop)
            return img.clone();
        if (canRead) {
            canRead = false;
            return img.clone();
        }
        return cv::Mat{};
    }
};

class DirReader : public ImagesCapture {
    std::vector<std::string> names;
    size_t fileId;
    size_t nextImgId;
    const size_t initialImageId;
    const size_t readLengthLimit;
    const std::string input;

public:
    DirReader(const std::string& input, bool loop, size_t initialImageId, size_t readLengthLimit)
        : ImagesCapture{loop},
          fileId{0},
          nextImgId{0},
          initialImageId{initialImageId},
          readLengthLimit{readLengthLimit},
          input{input} {
        DIR* dir = opendir(input.c_str());
        if (!dir)
            throw InvalidInput("Can't find the dir by " + input);
        while (struct dirent* ent = readdir(dir))
            if (strcmp(ent->d_name, ".") && strcmp(ent->d_name, ".."))
                names.emplace_back(ent->d_name);
        closedir(dir);
        if (names.empty())
            throw OpenError("The dir " + input + " is empty");
        sort(names.begin(), names.end());
        size_t readImgs = 0;
        while (fileId < names.size()) {
            cv::Mat img = cv::imread(input + '/' + names[fileId]);
            if (img.data) {
                ++readImgs;
                if (readImgs - 1 >= initialImageId)
                    return;
            }
            ++fileId;
        }
        throw OpenError("Can't read the first image from " + input);
    }

    double fps() const override {
        return 1.0;
    }

    std::string getType() const override {
        return "DIR";
    }

    cv::Mat read() override {
        auto startTime = std::chrono::steady_clock::now();

        while (fileId < names.size() && nextImgId < readLengthLimit) {
            cv::Mat img = cv::imread(input + '/' + names[fileId]);
            ++fileId;
            if (img.data) {
                ++nextImgId;
                readerMetrics.update(startTime);
                return img;
            }
        }

        if (loop) {
            fileId = 0;
            size_t readImgs = 0;
            while (fileId < names.size()) {
                cv::Mat img = cv::imread(input + '/' + names[fileId]);
                ++fileId;
                if (img.data) {
                    ++readImgs;
                    if (readImgs - 1 >= initialImageId) {
                        nextImgId = 1;
                        readerMetrics.update(startTime);
                        return img;
                    }
                }
            }
        }
        return cv::Mat{};
    }
};

class VideoCapWrapper : public ImagesCapture {
    cv::VideoCapture cap;
    bool first_read;
    const read_type type;
    size_t nextImgId;
    const double initialImageId;
    size_t readLengthLimit;

public:
    VideoCapWrapper(const std::string& input, bool loop, read_type type, size_t initialImageId, size_t readLengthLimit)
        : ImagesCapture{loop},
          first_read{true},
          type{type},
          nextImgId{0},
          initialImageId{static_cast<double>(initialImageId)} {
        if (0 == readLengthLimit) {
            throw std::runtime_error("readLengthLimit must be positive");
        }
        if (cap.open(input)) {
            this->readLengthLimit = readLengthLimit;
            if (!cap.set(cv::CAP_PROP_POS_FRAMES, this->initialImageId))
                throw OpenError("Can't set the frame to begin with");
            return;
        }
        throw InvalidInput("Can't open the video from " + input);
    }

    double fps() const override {
        return cap.get(cv::CAP_PROP_FPS);
    }

    std::string getType() const override {
        return "VIDEO";
    }

    cv::Mat read() override {
        auto startTime = std::chrono::steady_clock::now();

        if (nextImgId >= readLengthLimit) {
            if (loop && cap.set(cv::CAP_PROP_POS_FRAMES, initialImageId)) {
                nextImgId = 1;
                cv::Mat img;
                cap.read(img);
                if (type == read_type::safe) {
                    img = img.clone();
                }
                readerMetrics.update(startTime);
                return img;
            }
            return cv::Mat{};
        }
        cv::Mat img;
        bool success = cap.read(img);
        if (!success && first_read) {
            throw std::runtime_error("The first image can't be read");
        }
        first_read = false;
        if (!success && loop && cap.set(cv::CAP_PROP_POS_FRAMES, initialImageId)) {
            nextImgId = 1;
            cap.read(img);
        } else {
            ++nextImgId;
        }
        if (type == read_type::safe) {
            img = img.clone();
        }
        readerMetrics.update(startTime);
        return img;
    }
};

class CameraCapWrapper : public ImagesCapture {
    cv::VideoCapture cap;
    const read_type type;
    size_t nextImgId;
    size_t readLengthLimit;

public:
    CameraCapWrapper(const std::string& input,
                     bool loop,
                     read_type type,
                     size_t readLengthLimit,
                     cv::Size cameraResolution)
        : ImagesCapture{loop},
          type{type},
          nextImgId{0} {
        if (0 == readLengthLimit) {
            throw std::runtime_error("readLengthLimit must be positive");
        }
        try {
            if (cap.open(std::stoi(input))) {
                this->readLengthLimit = loop ? std::numeric_limits<size_t>::max() : readLengthLimit;
                cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
                cap.set(cv::CAP_PROP_FRAME_WIDTH, cameraResolution.width);
                cap.set(cv::CAP_PROP_FRAME_HEIGHT, cameraResolution.height);
                cap.set(cv::CAP_PROP_AUTOFOCUS, true);
                cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
                return;
            }
            throw OpenError("Can't open the camera from " + input);
        } catch (const std::invalid_argument&) {
            throw InvalidInput("Can't find the camera " + input);
        } catch (const std::out_of_range&) { throw InvalidInput("Can't find the camera " + input); }
    }

    double fps() const override {
        return cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 30;
    }

    std::string getType() const override {
        return "CAMERA";
    }

    cv::Mat read() override {
        auto startTime = std::chrono::steady_clock::now();

        if (nextImgId >= readLengthLimit) {
            return cv::Mat{};
        }
        cv::Mat img;
        if (!cap.read(img)) {
            throw std::runtime_error("The image can't be captured from the camera");
        }
        if (type == read_type::safe) {
            img = img.clone();
        }
        ++nextImgId;

        readerMetrics.update(startTime);
        return img;
    }
};

std::unique_ptr<ImagesCapture> openImagesCapture(const std::string& input,
                                                 bool loop,
                                                 read_type type,
                                                 size_t initialImageId,
                                                 size_t readLengthLimit,
                                                 cv::Size cameraResolution) {
    if (readLengthLimit == 0)
        throw std::runtime_error{"Read length limit must be positive"};
    std::vector<std::string> invalidInputs, openErrors;
    try {
        return std::unique_ptr<ImagesCapture>(new ImreadWrapper{input, loop});
    } catch (const InvalidInput& e) { invalidInputs.push_back(e.what()); } catch (const OpenError& e) {
        openErrors.push_back(e.what());
    }

    try {
        return std::unique_ptr<ImagesCapture>(new DirReader{input, loop, initialImageId, readLengthLimit});
    } catch (const InvalidInput& e) { invalidInputs.push_back(e.what()); } catch (const OpenError& e) {
        openErrors.push_back(e.what());
    }

    try {
        return std::unique_ptr<ImagesCapture>(new VideoCapWrapper{input, loop, type, initialImageId, readLengthLimit});
    } catch (const InvalidInput& e) { invalidInputs.push_back(e.what()); } catch (const OpenError& e) {
        openErrors.push_back(e.what());
    }

    try {
        return std::unique_ptr<ImagesCapture>(
            new CameraCapWrapper{input, loop, type, readLengthLimit, cameraResolution});
    } catch (const InvalidInput& e) { invalidInputs.push_back(e.what()); } catch (const OpenError& e) {
        openErrors.push_back(e.what());
    }

    std::vector<std::string> errorMessages = openErrors.empty() ? invalidInputs : openErrors;
    std::string errorsInfo;
    for (const auto& message : errorMessages) {
        errorsInfo.append(message + "\n");
    }
    throw std::runtime_error(errorsInfo);
}
