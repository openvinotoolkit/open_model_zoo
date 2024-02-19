// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/image_utils.h"
#include "utils/ocv_common.hpp"
#include "text_detection.hpp"

namespace {

void softmax(std::vector<float>* data) {
    auto& rdata = *data;
    const size_t last_dim = 2;
    for (size_t i = 0; i < rdata.size(); i += last_dim) {
       float m = std::max(rdata[i], rdata[i+1]);
       rdata[i] = std::exp(rdata[i] - m);
       rdata[i + 1] = std::exp(rdata[i + 1] - m);
       float s = rdata[i] + rdata[i + 1];
       rdata[i] /= s;
       rdata[i + 1] /= s;
    }
}

std::vector<float> transpose4d(
    const std::vector<float>& data, const std::vector<size_t>& shape, const std::vector<size_t>& axes)
{
    if (shape.size() != axes.size())
        throw std::runtime_error("Shape and axes must have the same dimension.");

    for (size_t a : axes) {
        if (a >= shape.size())
            throw std::runtime_error("Axis must be less than dimension of shape.");
    }

    size_t total_size = shape[0] * shape[1] * shape[2] * shape[3];

    std::vector<size_t> steps = {
        shape[axes[1]] * shape[axes[2]] * shape[axes[3]],
        shape[axes[2]] * shape[axes[3]], shape[axes[3]],
        1
    };

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

std::vector<float> sliceAndGetSecondChannel(const std::vector<float>& data) {
    std::vector<float> new_data(data.size() / 2, 0);
    for (size_t i = 0; i < data.size() / 2; i++) {
      new_data[i] = data[2 * i + 1];
    }
    return new_data;
}

std::vector<cv::RotatedRect> maskToBoxes(const cv::Mat& mask, float min_area, float min_height, const cv::Size& image_size)
{
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

std::vector<cv::RotatedRect> coordToBoxes(
    const float* coords, size_t coords_size, float min_area, float min_height,
    const cv::Size& input_shape, const cv::Size& image_size)
{
    std::vector<cv::RotatedRect> bboxes;
    int num_boxes = coords_size / 5;
    float x_scale = image_size.width / float(input_shape.width);
    float y_scale = image_size.height / float(input_shape.height);

    for (int i = 0; i < num_boxes; i++) {
        const float* prediction = &coords[i * 5];
        float confidence = prediction[4];
        if (confidence < std::numeric_limits<float>::epsilon())
            break;

        // predictions are sorted the way that all insignificant boxes are
        // grouped together
        cv::Point2f center = cv::Point2f((prediction[0] + prediction[2]) / 2 * x_scale,
                                         (prediction[1] + prediction[3]) / 2 * y_scale);
        cv::Size2f size = cv::Size2f((prediction[2] - prediction[0]) * x_scale,
                                     (prediction[3] - prediction[1]) * y_scale);
        cv::RotatedRect rect = cv::RotatedRect(center, size, 0);

        if (rect.size.area() < min_area)
            continue;

        bboxes.push_back(rect);
    }

    return bboxes;
}

int findRoot(int point, std::unordered_map<int, int>* group_mask) {
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

void join(int p1, int p2, std::unordered_map<int, int>* group_mask) {
    int root1 = findRoot(p1, group_mask);
    int root2 = findRoot(p2, group_mask);
    if (root1 != root2) {
        (*group_mask)[root1] = root2;
    }
}

cv::Mat get_all(const std::vector<cv::Point>& points, int w, int h, std::unordered_map<int, int>* group_mask)
{
    std::unordered_map<int, int> root_map;

    cv::Mat mask(h, w, CV_32S, cv::Scalar(0));
    for (const auto& point : points) {
        int point_root = findRoot(point.x + point.y * w, group_mask);
        if (root_map.find(point_root) == root_map.end()) {
            root_map.emplace(point_root, static_cast<int>(root_map.size() + 1));
        }
        mask.at<int>(point.x + point.y * w) = root_map[point_root];
    }

    return mask;
}

} // namespace


std::map<std::string, ov::Tensor> TextDetector::Infer(const cv::Mat& frame) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    cv::Mat resizedImg = frame;
    if (!use_auto_resize) {
        resizedImg = resizeImageExt(frame, m_input_size.width, m_input_size.height);
    }

    m_infer_request.set_input_tensor(wrapMat2Tensor(resizedImg));

    m_infer_request.infer();

    // Processing output
    std::map<std::string, ov::Tensor> output_tensors;
    for (const auto& output_name : m_output_names) {
        output_tensors[output_name] = m_infer_request.get_tensor(output_name);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    m_time_elapsed += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    m_ncalls++;

    return output_tensors;
}

std::vector<cv::RotatedRect> TextDetector::postProcess(
    const std::map<std::string, ov::Tensor>& output_tensors,
    const cv::Size& image_size, const cv::Size& input_shape,
    float cls_conf_threshold, float link_conf_threshold) {
    const int kMinArea = 300;
    const int kMinHeight = 10;

    std::string kLocOutputName;
    std::string kClsOutputName;

    for (const auto& output : output_tensors) {
        ov::Shape shape = output.second.get_shape();
        if (shape.size() != 2 && shape.size() != 4)
            continue;

        if (shape[ov::layout::channels_idx(m_modelLayout)] == 2)
            kClsOutputName = output.first;
        else if (shape[ov::layout::channels_idx(m_modelLayout)] == 16)
            kLocOutputName = output.first;
    }

    std::vector<cv::RotatedRect> rects;
    if (!kLocOutputName.empty() && !kClsOutputName.empty()) {
        // PostProcessing for PixelLink Text Detection model
        ov::Shape link_shape = output_tensors.at(kLocOutputName).get_shape();
        size_t link_data_size = link_shape[0] * link_shape[1] * link_shape[2] * link_shape[3];

        float* link_data_pointer = output_tensors.at(kLocOutputName).data<float>();

        std::vector<float> link_data(link_data_pointer, link_data_pointer + link_data_size);;
        if (m_modelLayout == ov::Layout("NCHW")) {
            link_data = transpose4d(link_data, link_shape, { 0, 2, 3, 1 });
        }

        softmax(&link_data);

        link_data = sliceAndGetSecondChannel(link_data);

        ov::Shape new_link_data_shape(link_shape.size());
        new_link_data_shape[0] = static_cast<int>(link_shape[ov::layout::batch_idx(m_modelLayout)]);
        new_link_data_shape[1] = static_cast<int>(link_shape[ov::layout::height_idx(m_modelLayout)]);
        new_link_data_shape[2] = static_cast<int>(link_shape[ov::layout::width_idx(m_modelLayout)]);
        new_link_data_shape[3] = static_cast<int>(link_shape[ov::layout::channels_idx(m_modelLayout)]) / 2;

        ov::Shape cls_shape = output_tensors.at(kClsOutputName).get_shape();
        size_t cls_data_size = cls_shape[0] * cls_shape[1] * cls_shape[2] * cls_shape[3];

        float* cls_data_pointer = output_tensors.at(kClsOutputName).data<float>();

        std::vector<float> cls_data(cls_data_pointer, cls_data_pointer + cls_data_size);
        if (m_modelLayout == ov::Layout("NCHW")) {
            cls_data = transpose4d(cls_data, cls_shape, { 0, 2, 3, 1 });
        }

        softmax(&cls_data);

        cls_data = sliceAndGetSecondChannel(cls_data);

        ov::Shape new_cls_data_shape(cls_shape.size());
        new_cls_data_shape[0] = static_cast<int>(cls_shape[ov::layout::batch_idx(m_modelLayout)]);
        new_cls_data_shape[1] = static_cast<int>(cls_shape[ov::layout::height_idx(m_modelLayout)]);
        new_cls_data_shape[2] = static_cast<int>(cls_shape[ov::layout::width_idx(m_modelLayout)]);
        new_cls_data_shape[3] = static_cast<int>(cls_shape[ov::layout::channels_idx(m_modelLayout)]) / 2;

        cv::Mat mask = decodeImageByJoin(
            cls_data, new_cls_data_shape, link_data, new_link_data_shape, cls_conf_threshold, link_conf_threshold);

        rects = maskToBoxes(
            mask, static_cast<float>(kMinArea), static_cast<float>(kMinHeight), image_size);
    } else {
        // PostProcessing for Horizontal Text Detection model
        for (const auto& output : output_tensors) {
            ov::Shape shape = output.second.get_shape();
            if (shape.size() != 2)
                continue;

            if (shape[1] == 5) {
                kLocOutputName = output.first;
            }
        }

        if (kLocOutputName.empty())
            throw std::runtime_error("Failed to determine output blob names");

        ov::Shape boxes_shape = output_tensors.at(kLocOutputName).get_shape();
        size_t boxes_data_size = boxes_shape[0] * boxes_shape[1];

        const float* boxes_data_pointer = output_tensors.at(kLocOutputName).data<float>();

        rects = coordToBoxes(
            boxes_data_pointer, boxes_data_size,
            static_cast<float>(kMinArea), static_cast<float>(kMinHeight), input_shape, image_size);
    }

    return rects;
}

cv::Mat TextDetector::decodeImageByJoin(
    const std::vector<float>& cls_data, const ov::Shape& cls_data_shape,
    const std::vector<float>& link_data, const ov::Shape& link_data_shape,
    float cls_conf_threshold, float link_conf_threshold) {
    int h = cls_data_shape[ov::layout::height_idx({"NHWC"})];
    int w = cls_data_shape[ov::layout::width_idx({"NHWC"})];

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

    size_t neighbours = size_t(link_data_shape[ov::layout::channels_idx({"NHWC"})]);
    for (const auto& point : points) {
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
