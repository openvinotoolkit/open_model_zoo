// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "text_detection.hpp"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
void softmax(std::vector<float>* data) {
    auto &rdata = *data;
    const size_t last_dim = 2;
    for (size_t i = 0 ; i < rdata.size(); i+=last_dim) {
       float m = std::max(rdata[i], rdata[i+1]);
       rdata[i] = std::exp(rdata[i] - m);
       rdata[i + 1] = std::exp(rdata[i + 1] - m);
       float s = rdata[i] + rdata[i + 1];
       rdata[i] /= s;
       rdata[i + 1] /= s;
    }
}

std::vector<float> transpose4d(const std::vector<float>& data, const std::vector<size_t> &shape,
                               const std::vector<size_t> &axes) {
    if (shape.size() != axes.size())
        throw std::runtime_error("Shape and axes must have the same dimension.");

    for (size_t a : axes) {
        if (a >= shape.size())
            throw std::runtime_error("Axis must be less than dimension of shape.");
    }

    size_t total_size = shape[0] * shape[1] * shape[2] * shape[3];

    std::vector<size_t> steps{shape[axes[1]] * shape[axes[2]] * shape[axes[3]],
                shape[axes[2]] * shape[axes[3]], shape[axes[3]], 1};

    size_t source_data_idx = 0;
    std::vector<float> new_data(total_size, 0);

    std::vector<size_t> ids(shape.size());
    for (ids[0] = 0; ids[0] < shape[0]; ids[0]++) {
        for (ids[1] = 0; ids[1] < shape[1]; ids[1]++) {
            for (ids[2] = 0; ids[2] < shape[2]; ids[2]++) {
                for (ids[3]= 0; ids[3] < shape[3]; ids[3]++) {
                    size_t new_data_idx = ids[axes[0]] * steps[0] + ids[axes[1]] * steps[1] +
                            ids[axes[2]] * steps[2] + ids[axes[3]] * steps[3];
                    new_data[new_data_idx] = data[source_data_idx++];
                }
            }
        }
    }
    return new_data;
}

std::vector<size_t> ieSizeToVector(const SizeVector& ie_output_dims) {
    std::vector<size_t> blob_sizes(ie_output_dims.size(), 0);
    for (size_t i = 0; i < blob_sizes.size(); ++i) {
        blob_sizes[i] = ie_output_dims[i];
    }
    return blob_sizes;
}

std::vector<float> sliceAndGetSecondChannel(const std::vector<float> &data) {
    std::vector<float> new_data(data.size() / 2, 0);
    for (size_t i = 0; i < data.size() / 2; i++) {
      new_data[i] = data[2 * i + 1];
    }
    return new_data;
}

std::vector<cv::RotatedRect> maskToBoxes(const cv::Mat &mask, float min_area, float min_height,
                                         cv::Size image_size) {
    std::vector<cv::RotatedRect> bboxes;
    double min_val;
    double max_val;
    cv::minMaxLoc(mask, &min_val, &max_val);
    int max_bbox_idx = static_cast<int>(max_val);
    cv::Mat resized_mask;
    cv::resize(mask, resized_mask, image_size, 0, 0, cv::INTER_NEAREST);

    for (int i = 1; i <= max_bbox_idx; i++) {
        cv::Mat bbox_mask = resized_mask == i;
        std::vector<std::vector<cv::Point>> contours;

        cv::findContours(bbox_mask, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty())
            continue;
        cv::RotatedRect r = cv::minAreaRect(contours[0]);
        if (std::min(r.size.width, r.size.height) < min_height)
            continue;
        if (r.size.area() < min_area)
            continue;
        bboxes.emplace_back(r);
    }

    return bboxes;
  }

int findRoot(int point, std::unordered_map<int, int> *group_mask) {
    int root = point;
    bool update_parent = false;
    while (group_mask->at(root) != -1) {
        root = group_mask->at(root);
        update_parent = true;
    }
    if (update_parent) {
       (*group_mask)[point] = root;
    }
    return root;
}

void join(int p1, int p2, std::unordered_map<int, int> *group_mask) {
    int root1 = findRoot(p1, group_mask);
    int root2 = findRoot(p2, group_mask);
    if (root1 != root2) {
        (*group_mask)[root1] = root2;
    }
}

cv::Mat get_all(const std::vector<cv::Point> &points, int w, int h,
                std::unordered_map<int, int> *group_mask) {
    std::unordered_map<int, int> root_map;

    cv::Mat mask(h, w, CV_32S, cv::Scalar(0));
    for (const auto &point : points) {
        int point_root = findRoot(point.x + point.y * w, group_mask);
        if (root_map.find(point_root) == root_map.end()) {
            root_map.emplace(point_root, static_cast<int>(root_map.size() + 1));
        }
        mask.at<int>(point.x + point.y * w) = root_map[point_root];
    }

    return mask;
}

cv::Mat decodeImageByJoin(const std::vector<float> &cls_data, const std::vector<int> & cls_data_shape,
                          const std::vector<float> &link_data, const std::vector<int> & link_data_shape,
                          float cls_conf_threshold, float link_conf_threshold) {
    int h = cls_data_shape[1];
    int w = cls_data_shape[2];

    std::vector<uchar> pixel_mask(h * w, 0);
    std::unordered_map<int, int> group_mask;
    std::vector<cv::Point> points;
    for (size_t i = 0; i < pixel_mask.size(); i++) {
        pixel_mask[i] = cls_data[i] >= cls_conf_threshold;
        if (pixel_mask[i]) {
            points.emplace_back(i % w, i / w);
            group_mask[i] = -1;
        }
    }

    std::vector<uchar> link_mask(link_data.size(), 0);
    for (size_t i = 0; i < link_mask.size(); i++) {
        link_mask[i] = link_data[i] >= link_conf_threshold;
    }

    size_t neighbours = size_t(link_data_shape[3]);
    for (const auto &point : points) {
        size_t neighbour = 0;
        for (int ny = point.y - 1; ny <= point.y + 1; ny++) {
            for (int nx = point.x - 1; nx <= point.x + 1; nx++) {
                if (nx == point.x && ny == point.y)
                    continue;
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    uchar pixel_value = pixel_mask[size_t(ny) * size_t(w) + size_t(nx)];
                    uchar link_value = link_mask[
                        (size_t(point.y) * size_t(w) + size_t(point.x)) * neighbours + neighbour];
                    if (pixel_value && link_value) {
                        join(point.x + point.y * w, nx + ny * w, &group_mask);
                    }
                }
                neighbour++;
            }
        }
    }

    return get_all(points, w, h, &group_mask);
}
}  // namespace

std::vector<cv::RotatedRect> postProcess(const InferenceEngine::BlobMap &blobs, const cv::Size& image_size,
                                         float cls_conf_threshold, float link_conf_threshold) {
    const int kMinArea = 300;
    const int kMinHeight = 10;

    std::string kLocOutputName;
    std::string kClsOutputName;

    for (const auto& blob : blobs) {
        if (blob.second->getTensorDesc().getDims()[1] == 2)
            kClsOutputName = blob.first;
        else if (blob.second->getTensorDesc().getDims()[1] == 16)
            kLocOutputName = blob.first;
    }

    if (kLocOutputName.empty() || kClsOutputName.empty())
        throw std::runtime_error("Failed to determine output blob names");

    auto link_shape = blobs.at(kLocOutputName)->getTensorDesc().getDims();
    size_t link_data_size = link_shape[0] * link_shape[1] * link_shape[2] * link_shape[3];
    float *link_data_pointer =
            blobs.at(kLocOutputName)->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
    std::vector<float> link_data(link_data_pointer, link_data_pointer + link_data_size);
    link_data = transpose4d(link_data, ieSizeToVector(link_shape), {0, 2, 3, 1});
    softmax(&link_data);
    link_data = sliceAndGetSecondChannel(link_data);
    std::vector<int> new_link_data_shape(4);
    new_link_data_shape[0] = static_cast<int>(link_shape[0]);
    new_link_data_shape[1] = static_cast<int>(link_shape[2]);
    new_link_data_shape[2] = static_cast<int>(link_shape[3]);
    new_link_data_shape[3] = static_cast<int>(link_shape[1]) / 2;

    auto cls_shape = blobs.at(kClsOutputName)->getTensorDesc().getDims();
    size_t cls_data_size = cls_shape[0] * cls_shape[1] * cls_shape[2] * cls_shape[3];
    float *cls_data_pointer =
            blobs.at(kClsOutputName)->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
    std::vector<float> cls_data(cls_data_pointer, cls_data_pointer + cls_data_size);
    cls_data = transpose4d(cls_data, ieSizeToVector(cls_shape), {0, 2, 3, 1});
    softmax(&cls_data);
    cls_data = sliceAndGetSecondChannel(cls_data);
    std::vector<int> new_cls_data_shape(4);
    new_cls_data_shape[0] = static_cast<int>(cls_shape[0]);
    new_cls_data_shape[1] = static_cast<int>(cls_shape[2]);
    new_cls_data_shape[2] = static_cast<int>(cls_shape[3]);
    new_cls_data_shape[3] = static_cast<int>(cls_shape[1]) / 2;

    cv::Mat mask = decodeImageByJoin(cls_data, new_cls_data_shape, link_data, new_link_data_shape, cls_conf_threshold, link_conf_threshold);
    std::vector<cv::RotatedRect> rects = maskToBoxes(mask, static_cast<float>(kMinArea),
                                                     static_cast<float>(kMinHeight), image_size);

    return rects;
}
