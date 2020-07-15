#include <stdexcept>
#include <string>
#include <memory>

#ifdef _WIN32
#include <os/windows/w_dirent.h>
#else
#include <dirent.h>
#endif

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

class InvalidInput : public std::invalid_argument {
public:
    InvalidInput() : std::invalid_argument{""} {}
};

class ImagesCapture {
public:
    bool loop;

    ImagesCapture(bool loop) : loop{loop} {}
    virtual cv::Mat read() = 0;
    virtual ~ImagesCapture() = default;
};

class ImreadWrapper : public ImagesCapture {
public:
    cv::Mat img;
    bool canRead;

    ImreadWrapper(const std::string &input, bool loop) : ImagesCapture{loop}, canRead{true} {
        img = cv::imread(input);
        if(!img.data) throw InvalidInput{};
    }

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
    size_t imgId;

public:
    const std::string input;

    DirReader(const std::string &input, bool loop): ImagesCapture{loop}, imgId{0}, input{input} {
        DIR *dir = opendir(input.c_str());
        if (!dir) throw InvalidInput{};
        for (struct dirent *ent = readdir(dir); ent != nullptr; ent = readdir(dir))
            if (strcmp(ent->d_name, ".") && strcmp(ent->d_name, ".."))
                names.emplace_back(ent->d_name);
        closedir(dir);
        if (names.empty()) throw InvalidInput{};
        sort(names.begin(), names.end());
        bool foundAtLeastOneImg = false;
        for (const std::string &name : names) {
            cv::Mat img = cv::imread(input + '/' + name);
            if (img.data) {
                foundAtLeastOneImg = true;
                break;
            }
        }
        if (!foundAtLeastOneImg) throw InvalidInput{};
    }

    cv::Mat read() override {
        while (imgId < names.size()) {
            cv::Mat img = cv::imread(input + '/' + names[imgId]);
            ++imgId;
            if (img.data) return img;
        }
        if (loop) {
            imgId = 0;
            while (imgId < names.size()) {
                cv::Mat img = cv::imread(input + '/' + names[imgId]);
                ++imgId;
                if (img.data) return img;
            }
        }
        return cv::Mat{};
    }
};

class VideoCapWrapper : public ImagesCapture {
    cv::VideoCapture cap;
    size_t posFrames;

public:
    VideoCapWrapper(const std::string &input, bool loop) : ImagesCapture{loop}, posFrames{0} {
        try {
            cap.open(std::stoi(input));
        } catch (const std::invalid_argument&) {
            cap.open(input);
        } catch (const std::out_of_range&) {
            cap.open(input);
        }
        if (!cap.isOpened()) throw InvalidInput{};
    }

    cv::Mat read() override {
        cv::Mat img;
        if (!cap.read(img) && loop && cap.set(cv::CAP_PROP_POS_FRAMES, 0.0)) { // TODO first and last pos
            posFrames = 0;
            cap.read(img);
        } else {
            ++posFrames;
        }
        return img;
    }
};

std::unique_ptr<ImagesCapture> openImagesCapture(const std::string &input, bool loop) {
    try {
        return std::unique_ptr<ImagesCapture>{new ImreadWrapper{input, loop}};
    } catch (const InvalidInput &) {}
    try {
        return std::unique_ptr<ImagesCapture>{new DirReader{input, loop}};
    } catch (const InvalidInput &) {}
    try {
        return std::unique_ptr<ImagesCapture>{new VideoCapWrapper{input, loop}};
    } catch (const InvalidInput &) {}
    throw std::runtime_error("Can't read " + input);
}
