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

class ImgCap {
public:
    bool loop;

    ImgCap(bool loop) : loop{loop} {}
    virtual cv::Mat read() = 0;
    virtual cv::Size getSize() = 0;
    virtual ~ImgCap() = default;
};

class ImreadWrapper : public ImgCap {
public:
    cv::Mat img;
    bool canRead;

    ImreadWrapper(const std::string &input, bool loop) : ImgCap{loop}, canRead{true} {
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

    cv::Size getSize() override {return img.size();}
};

class DirReader : public ImgCap {
    std::vector<std::string> names;
    std::vector<cv::Mat> imgs;
    size_t imgId;

public:
    DirReader(const std::string &input, bool loop) : ImgCap{loop}, imgId{0} {
        DIR *dir = opendir(input.c_str());
        if (!dir) throw InvalidInput{};
        for (struct dirent *ent = readdir(dir); ent != nullptr; ent = readdir(dir))
            if (strcmp(ent->d_name, ".") && strcmp(ent->d_name, ".."))
                names.emplace_back(ent->d_name);
        closedir(dir);
        sort(names.begin(), names.end());
        for (size_t i = 0; i < names.size(); ++i) {
            std::string fileName = input + '/' + names[i];
            cv::Mat img = cv::imread(fileName);
            if (img.data) {
                imgs.push_back(img);
            } else {
                std::cerr << "Can't read " << names[i] << '\n';
                names.erase(names.begin() + i);
                --i;
            }
        }
        if (imgs.empty()) throw InvalidInput{};
    }

    std::vector<std::string> getNames() {return names;}

    size_t getImgId() {return imgId;}

    cv::Mat read() override {
        if (imgId < imgs.size()) return imgs[imgId++].clone();
        if (loop) {
            imgId -= imgs.size();
            return imgs[imgId++].clone();
        }
        return cv::Mat{};
    }

    cv::Size getSize() override {
        return imgs.front().size();
    }
};

class VideoCapWrapper : public ImgCap {
    cv::VideoCapture cap;
    size_t posFrames;

public:
    VideoCapWrapper(const std::string &input, bool loop) : ImgCap{loop}, posFrames{0} {
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

    cv::Size getSize() override {
        return {static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
            static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))};
    }
};

std::unique_ptr<ImgCap> openImgCap(const std::string &input, bool loop) {
    try {
        return std::unique_ptr<ImgCap>(new ImreadWrapper{input, loop});
    } catch (const InvalidInput &) {
        try {
            return std::unique_ptr<ImgCap>(new DirReader{input, loop});
        } catch (const InvalidInput &) {
            try {
                return std::unique_ptr<ImgCap>(new VideoCapWrapper{input, loop});
            } catch (const InvalidInput &) {
                throw std::runtime_error("Can't read " + input);
            }
        }
    }
}
