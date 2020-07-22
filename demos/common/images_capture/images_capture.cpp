// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdexcept>
#include <string>
#include <memory>

#ifdef _WIN32
#include <os/windows/w_dirent.h>
#else
#include <dirent.h>
#endif

#include <opencv2/imgcodecs.hpp>

#include "images_capture.h"

class InvalidInput {};

class ImreadWrapper : public ImagesCapture {
public:
    cv::Mat img;
    bool canRead;

    ImreadWrapper(const std::string &input, bool loop) : ImagesCapture{loop}, canRead{true} {
        img = cv::imread(input);
        if(!img.data) throw InvalidInput{};
    }

    double fps() const override {return 0.0;}

    size_t lastImgId() const override {return 0;}

    cv::Mat read() override {
        if (loop) return img.clone();
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
    size_t imgId;
    const size_t posFrames;
    const size_t readLengthLimit;

public:
    const std::string input;

    DirReader(const std::string &input, bool loop, size_t posFrames, size_t readLengthLimit) : ImagesCapture{loop},
            fileId{0}, imgId{0}, posFrames{posFrames}, readLengthLimit{readLengthLimit}, input{input} {
        DIR *dir = opendir(input.c_str());
        if (!dir) throw InvalidInput{};
        for (struct dirent *ent = readdir(dir); ent != nullptr; ent = readdir(dir))
            if (strcmp(ent->d_name, ".") && strcmp(ent->d_name, ".."))
                names.emplace_back(ent->d_name);
        closedir(dir);
        if (names.empty()) throw InvalidInput{};
        sort(names.begin(), names.end());
        size_t readImgs = 0;
        bool readAtLeastOnce = false;
        while (fileId < names.size()) {
            cv::Mat img = cv::imread(input + '/' + names[fileId]);
            if (img.data) {
                ++readImgs;
                readAtLeastOnce = true;
                if (readImgs - 1 >= posFrames) break;
            }
            ++fileId;
        }
        if (!readAtLeastOnce && readImgs - 1 < posFrames)
            throw std::runtime_error{"Can't read the first image from " + input + " dir"};
    }

    double fps() const override {return 0.0;}

    size_t lastImgId() const override {return imgId - 1;}

    cv::Mat read() override {
        while (fileId < names.size() && imgId < readLengthLimit) {
            cv::Mat img = cv::imread(input + '/' + names[fileId]);
            ++fileId;
            if (img.data) {
                ++imgId;
                return img;
            }
        }
        if (loop) {
            fileId = 0;
            imgId = 0;
            while (fileId < names.size() && imgId < posFrames) {
                cv::Mat img = cv::imread(input + '/' + names[fileId]);
                ++fileId;
                if (img.data) {
                    ++imgId;
                    return img;
                }
            }
        }
        return cv::Mat{};
    }
};

class VideoCapWrapper : public ImagesCapture {
    cv::VideoCapture cap;
    size_t imgId;
    const double posFrames;
    size_t readLengthLimit_;

public:
    VideoCapWrapper(const std::string &input, bool loop, double posFrames, size_t readLengthLimit,
                double buffersize, double frameWidth, double frameHeight, double autofocus, double fourcc)
            : ImagesCapture{loop}, imgId{0}, posFrames{posFrames}, readLengthLimit_{readLengthLimit} {
        try {
            cap.open(std::stoi(input));
            readLengthLimit_ = loop ? std::numeric_limits<size_t>::max() : readLengthLimit;
            cap.set(cv::CAP_PROP_BUFFERSIZE, buffersize);
            cap.set(cv::CAP_PROP_FRAME_WIDTH, frameWidth);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, frameHeight);
            cap.set(cv::CAP_PROP_AUTOFOCUS, autofocus);
            cap.set(cv::CAP_PROP_FOURCC, fourcc);
        } catch (const std::invalid_argument&) {
            cap.open(input);
            readLengthLimit_ = readLengthLimit;
            if (!cap.set(cv::CAP_PROP_POS_FRAMES, posFrames))
                throw std::runtime_error{"Can't set the frame to begin with"};
        } catch (const std::out_of_range&) {
            cap.open(input);
            readLengthLimit_ = readLengthLimit;
            if (!cap.set(cv::CAP_PROP_POS_FRAMES, posFrames))
                throw std::runtime_error{"Can't set the frame to begin with"};
        }
        if (!cap.isOpened()) throw InvalidInput{};
    }

    double fps() const override {return cap.get(cv::CAP_PROP_FPS);}

    size_t lastImgId() const override {return imgId - 1;}

    cv::Mat read() override {
        if (imgId >= readLengthLimit_) {
            if (loop && cap.set(cv::CAP_PROP_POS_FRAMES, posFrames)) {
                imgId = 1;
                cv::Mat img;
                cap.read(img);
                return img;
            }
            return cv::Mat{};
        }
        cv::Mat img;
        if (!cap.read(img) && loop && cap.set(cv::CAP_PROP_POS_FRAMES, posFrames)) {
            imgId = 1;
            cap.read(img);
        } else {
            ++imgId;
        }
        return img;
    }
};

std::unique_ptr<ImagesCapture> openImagesCapture(const std::string &input, bool loop, size_t posFrames,
        size_t readLengthLimit, 
        double buffersize, double frameWidth, double frameHeight, double autofocus, double fourcc) {
    if (readLengthLimit == 0) throw std::runtime_error{"Read length limit must be a natural number"};
    try {
        return std::unique_ptr<ImagesCapture>{new ImreadWrapper{input, loop}};
    } catch (const InvalidInput &) {}
    try {
        return std::unique_ptr<ImagesCapture>{new DirReader{input, loop, posFrames, readLengthLimit}};
    } catch (const InvalidInput &) {}
    try {
        return std::unique_ptr<ImagesCapture>{new VideoCapWrapper{input, loop, static_cast<double>(posFrames),
            readLengthLimit, buffersize, frameWidth, frameHeight, autofocus, fourcc}};
    } catch (const InvalidInput &) {}
    throw std::runtime_error{"Can't read " + input};
}
