// Copyright (C) 2023 KNS Group LLC (YADRO)
// SPDX-License-Identifier: Apache-2.0
//

#include "reid_gallery.hpp"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#ifdef _WIN32
#    include "w_dirent.hpp"
#else
#    include <dirent.h>  // for closedir, dirent, opendir, readdir, DIR
#endif

#include "utils/kuhn_munkres.hpp"

namespace {
    float computeReidDistance(const cv::Mat& descr1, const cv::Mat& descr2) {
        float xy = static_cast<float>(descr1.dot(descr2));
        float xx = static_cast<float>(descr1.dot(descr1));
        float yy = static_cast<float>(descr2.dot(descr2));
        float norm = sqrt(xx * yy) + 1e-6f;
        return 1.0f - xy / norm;
    }

    std::vector<std::string> file_extensions = {".jpg", ".png"};

}  // namespace

const char EmbeddingsGallery::unknownLabel[] = "Unknown";
const int EmbeddingsGallery::unknownId = -1;
const float EmbeddingsGallery::unknownDistance = 1.0;

EmbeddingsGallery::EmbeddingsGallery(const std::string& fgPath,
                                     double threshold,
                                     bool crop,
                                     const DetectorConfig& detectorConfig,
                                     AsyncModel& landmarksDet,
                                     AsyncModel& imageReid,
                                     bool useGreedyMatcher) :
    reidThreshold(threshold), useGreedyMatcher(useGreedyMatcher), faceGalleryPath(fgPath) {
    if (faceGalleryPath.empty()) {
        return;
    }

    FaceDetector detector(detectorConfig);

    int id = 0;
    DIR* dir = opendir(fgPath.c_str());
    if (!dir) {
        throw std::runtime_error("Can't find the directory " + fgPath);
    }
    while (struct dirent* ent = readdir(dir)) {
        if (strcmp(ent->d_name, ".") && strcmp(ent->d_name, "..")) {
            std::string name = ent->d_name;
            cv::Mat image = cv::imread(fgPath + '/' + name);
            if (image.empty()) {
                throw std::runtime_error("Image is empty");
            }
            std::vector<cv::Mat> embeddings;
            cv::Mat emb;
            RegistrationStatus status = registerIdentity(name, image, crop,  detector, landmarksDet, imageReid, emb);
            if (status == RegistrationStatus::SUCCESS) {
                embeddings.push_back(emb);
                idxToId.push_back(id);
                identities.emplace_back(embeddings, name, id);
                ++id;
            }
        }
    }
    closedir(dir);
    slog::info << identities.size() << " persons to recognize were added from the gallery" << slog::endl;
}

std::vector<std::pair<int, float>>  EmbeddingsGallery::getIDsByEmbeddings(const std::vector<cv::Mat>& embeddings) const {
    if (embeddings.empty() || idxToId.empty()) {
        return std::vector<std::pair<int, float>>(embeddings.size(), {unknownId, unknownDistance});
    }

    cv::Mat distances(static_cast<int>(embeddings.size()), static_cast<int>(idxToId.size()), CV_32F);

    for (int i = 0; i < distances.rows; i++) {
        int k = 0;
        for (size_t j = 0; j < identities.size(); j++) {
            for (const auto& reference_emb : identities[j].embeddings) {
                distances.at<float>(i, k) = computeReidDistance(embeddings[i], reference_emb);
                k++;
            }
        }
    }

    KuhnMunkres matcher(useGreedyMatcher);
    auto matchedIdx = matcher.Solve(distances);
    std::vector<std::pair<int, float>> matches;
    for (auto col_idx : matchedIdx) {
        if (int(col_idx) == -1) {
            matches.push_back({unknownId, unknownDistance});
            continue;
        }
        if (distances.at<float>(matches.size(), col_idx) > reidThreshold) {
            matches.push_back({unknownId, unknownDistance});
        } else {
            matches.push_back({idxToId[col_idx], distances.at<float>(matches.size(), col_idx)});
        }
    }
    return matches;
}

std::string EmbeddingsGallery::getLabelByID(int id) const {
    if (id >= 0 && id < static_cast<int>(identities.size())) {
        return identities[id].label;
    } else {
        return unknownLabel;
    }
}

size_t EmbeddingsGallery::size() const {
    return identities.size();
}

bool EmbeddingsGallery::labelExists(const std::string& label) const {
    return identities.end() != std::find_if(identities.begin(), identities.end(),
                                           [label](const GalleryObject& o) {return o.label == label;});
}

std::string EmbeddingsGallery::tryToSave(cv::Mat newFace){
    std::string winname = "Unknown face";
    size_t height = int(400 * newFace.rows / newFace.cols);
    cv::Mat resized;
    cv::resize(newFace, resized, cv::Size(400, height), 0.0, 0.0, cv::INTER_AREA);
    size_t font = cv::FONT_HERSHEY_PLAIN;
    size_t fontScale = 1;
    cv::Scalar fontColor(255, 255, 255);
    size_t lineType = 1;
    cv::copyMakeBorder(resized, newFace, 5, 5, 5, 5, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
    cv::putText(newFace, "Please, enter person's name and", cv::Point2d(30, 80), font, fontScale, fontColor, lineType);
    cv::putText(newFace, "press \"Enter\" to accept and continue.", cv::Point2d(30, 110), font, fontScale, fontColor, lineType);
    cv::putText(newFace, "Press \"Escape\" to discard.", cv::Point2d(30, 140), font, fontScale, fontColor, lineType);
    cv::putText(newFace, "Name: ", cv::Point2d(30, 170), font, fontScale, fontColor, lineType);
    std::string name;
    bool save = false;
    while (true) {
        cv::Mat cc = newFace.clone();
        cv::putText(cc, name, cv::Point2d(30, 200), font, fontScale, fontColor, lineType);
        cv::imshow(winname, cc);
        int key = (cv::waitKey(0) & 0xFF);
        if (key == 27) {
            break;
        }
        if (key == 13) {
            if (name.size() > 0) {
                if (labelExists(name)) {
                    cv::putText(cc, "The name is already exists! Try another one.", cv::Point2d(30, 200), font, fontScale, fontColor, lineType);
                    continue;
                }
                save = true;
                break;
            } else {
                cv::putText(cc, "Provided name is empty. Please, provide a valid name.", cv::Point2d(30, 200), font, fontScale, fontColor, lineType);
                cv::imshow(winname, cc);
                key = cv::waitKey(0);
                if (key == 27) {
                    break;
                }
                continue;
            }
        }
        if (key == 225) {
            continue;
        }
        if (key == 8) {
            name = name.substr(0, name.size() - 1);
            continue;
        } else {
            name += char(key);
            continue;
        }
    }
    cv::destroyWindow(winname);
    return (save ? name : "");
}

void EmbeddingsGallery::addFace(const cv::Mat newFace, const cv::Mat embedding, std::string label) {
    identities.emplace_back(std::vector<cv::Mat>{embedding}, label, idxToId.size());
    idxToId.push_back(idxToId.size());
    label += ".jpg";
    cv::imwrite(faceGalleryPath + "/" + label, newFace);
}

RegistrationStatus EmbeddingsGallery::registerIdentity(const std::string& identityLabel,
    const cv::Mat& image,
    bool crop,
    FaceDetector& detector,
    AsyncModel& landmarksDet,
    AsyncModel& imageReid,
    cv::Mat& embedding) {
    cv::Mat target = image;
    if (crop) {
        detector.submitData(image);
        std::vector<FaceBox> faces = detector.getResults();
        if (faces.size() == 0) {
            return RegistrationStatus::FAILURE_NOT_DETECTED;
        }
        if (faces.size() != 1) {
            throw std::runtime_error("More than 1 face on the image provided for face gallery");
        }
        cv::Mat faceRoi = image(faces[0].face);
        target = faceRoi;
    }

    cv::Mat landmarks;
    std::vector<cv::Mat> images = { target };
    std::vector<cv::Mat> landmarksVec = landmarksDet.infer(images);
    alignFaces(images, landmarksVec);
    embedding = imageReid.infer(images)[0];
    return RegistrationStatus::SUCCESS;
}
