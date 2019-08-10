// This file is based on https://github.com/opencv/open_model_zoo/tree/2019/demos/text_detection_demo
#include "opencv2/open_model_zoo.hpp"
#include "opencv2/open_model_zoo/text_recognition.hpp"
#include "opencv2/open_model_zoo/dnn.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <unordered_map>

namespace cv { namespace open_model_zoo {

static void softmax(std::vector<float>* data) {
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

namespace  {
    void softmax(const std::vector<float>::const_iterator& begin, const std::vector<float>::const_iterator& end, int *argmax, float *prob) {
        auto max_element = std::max_element(begin, end);
        *argmax = static_cast<int>(std::distance(begin, max_element));
        float max_val = *max_element;
        double sum = 0;
        for (auto i = begin; i != end; i++) {
           sum += std::exp((*i) - max_val);
        }
        if (std::fabs(sum) < std::numeric_limits<double>::epsilon()) {
            throw std::logic_error("sum can't be equal to zero");
        }
        *prob = 1.0f / static_cast<float>(sum);
    }
}  // namespace


static std::string CTCGreedyDecoder(const std::vector<float> &data, const std::string& alphabet, char pad_symbol, double *conf) {
    std::string res = "";
    bool prev_pad = false;
    *conf = 1;

    const int num_classes = alphabet.length();
    for (std::vector<float>::const_iterator it = data.begin(); it != data.end(); it += num_classes) {
      int argmax;
      float prob;

      softmax(it, it + num_classes, &argmax, &prob);

      (*conf) *= prob;

      auto symbol = alphabet[argmax];
      if (symbol != pad_symbol) {
          if (res.empty() || prev_pad || (!res.empty() && symbol != res.back())) {
            prev_pad = false;
            res += symbol;
          }
      } else {
        prev_pad = true;
      }
    }
    return res;
}

static std::vector<float> transpose4d(const std::vector<float>& data,
                                      const std::vector<size_t> &shape,
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

static std::vector<float> sliceAndGetSecondChannel(const std::vector<float> &data) {
    std::vector<float> new_data(data.size() / 2, 0);
    for (size_t i = 0; i < data.size() / 2; i++) {
      new_data[i] = data[2 * i + 1];
    }
    return new_data;
}

static std::vector<cv::RotatedRect> maskToBoxes(const cv::Mat &mask, float min_area, float min_height,
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

static int findRoot(int point, std::unordered_map<int, int> *group_mask) {
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

static void join(int p1, int p2, std::unordered_map<int, int> *group_mask) {
    int root1 = findRoot(p1, group_mask);
    int root2 = findRoot(p2, group_mask);
    if (root1 != root2) {
        (*group_mask)[root1] = root2;
    }
}

static cv::Mat get_all(const std::vector<cv::Point> &points, int w, int h,
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

static cv::Mat decodeImageByJoin(const std::vector<float> &cls_data, const std::vector<int> & cls_data_shape,
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

    int neighbours = link_data_shape[3];
    for (const auto &point : points) {
        int neighbour = 0;
        for (int ny = point.y - 1; ny <= point.y + 1; ny++) {
            for (int nx = point.x - 1; nx <= point.x + 1; nx++) {
                if (nx == point.x && ny == point.y)
                    continue;
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    uchar pixel_value = pixel_mask[static_cast<size_t>(ny * w + nx)];
                    uchar link_value = link_mask[static_cast<size_t>(point.y * w * neighbours +
                                                                     point.x * neighbours + neighbour)];
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

static void postProcess(const std::vector<Mat>& blobs, const cv::Size& image_size,
                        float cls_conf_threshold, float link_conf_threshold,
                        std::vector<cv::RotatedRect>& rects) {
    const int kMinArea = 300;
    const int kMinHeight = 10;

    int kLocOutputIdx = -1;
    int kClsOutputIdx = -1;

    for (int i = 0; i < 2; ++i) {
        if (blobs[i].size[1] == 2)
            kClsOutputIdx = i;
        else if (blobs[i].size[1] == 16)
            kLocOutputIdx = i;
    }

    if (kLocOutputIdx == -1 || kClsOutputIdx == -1)
        throw std::runtime_error("Failed to determine output blob names");

    std::vector<size_t> link_shape(&blobs[kLocOutputIdx].size[0], &blobs[kLocOutputIdx].size[0] + 4);
    size_t link_data_size = link_shape[0] * link_shape[1] * link_shape[2] * link_shape[3];
    float *link_data_pointer = (float*)blobs[kLocOutputIdx].data;
    std::vector<float> link_data(link_data_pointer, link_data_pointer + link_data_size);
    link_data = transpose4d(link_data, link_shape, {0, 2, 3, 1});
    softmax(&link_data);
    link_data = sliceAndGetSecondChannel(link_data);
    std::vector<int> new_link_data_shape(4);
    new_link_data_shape[0] = static_cast<int>(link_shape[0]);
    new_link_data_shape[1] = static_cast<int>(link_shape[2]);
    new_link_data_shape[2] = static_cast<int>(link_shape[3]);
    new_link_data_shape[3] = static_cast<int>(link_shape[1]) / 2;

    std::vector<size_t> cls_shape(&blobs[kClsOutputIdx].size[0], &blobs[kClsOutputIdx].size[0] + 4);
    size_t cls_data_size = cls_shape[0] * cls_shape[1] * cls_shape[2] * cls_shape[3];
    float *cls_data_pointer = (float*)blobs[kClsOutputIdx].data;
    std::vector<float> cls_data(cls_data_pointer, cls_data_pointer + cls_data_size);
    cls_data = transpose4d(cls_data, cls_shape, {0, 2, 3, 1});
    softmax(&cls_data);
    cls_data = sliceAndGetSecondChannel(cls_data);
    std::vector<int> new_cls_data_shape(4);
    new_cls_data_shape[0] = static_cast<int>(cls_shape[0]);
    new_cls_data_shape[1] = static_cast<int>(cls_shape[2]);
    new_cls_data_shape[2] = static_cast<int>(cls_shape[3]);
    new_cls_data_shape[3] = static_cast<int>(cls_shape[1]) / 2;

    cv::Mat mask = decodeImageByJoin(cls_data, new_cls_data_shape, link_data, new_link_data_shape, cls_conf_threshold, link_conf_threshold);
    rects = maskToBoxes(mask, static_cast<float>(kMinArea),
                        static_cast<float>(kMinHeight), image_size);
}

static std::vector<cv::Point2f> floatPointsFromRotatedRect(const cv::RotatedRect &rect) {
    cv::Point2f vertices[4];
    rect.points(vertices);

    std::vector<cv::Point2f> points;
    for (int i = 0; i < 4; i++) {
        points.emplace_back(vertices[i].x, vertices[i].y);
    }
    return points;
}

static cv::Point topLeftPoint(const std::vector<cv::Point2f> & points, int *idx) {
    cv::Point2f most_left(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    cv::Point2f almost_most_left(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

    int most_left_idx = -1;
    int almost_most_left_idx = -1;

    for (size_t i = 0; i < points.size() ; i++) {
        if (most_left.x > points[i].x) {
            if (most_left.x < std::numeric_limits<float>::max()) {
                almost_most_left = most_left;
                almost_most_left_idx = most_left_idx;
            }
            most_left = points[i];
            most_left_idx = static_cast<int>(i);
        }
        if (almost_most_left.x > points[i].x && points[i] != most_left) {
            almost_most_left = points[i];
            almost_most_left_idx = static_cast<int>(i);
        }
    }

    if (almost_most_left.y < most_left.y) {
        most_left = almost_most_left;
        most_left_idx = almost_most_left_idx;
    }

    *idx = most_left_idx;
    return most_left;
}

static cv::Mat cropImage(const cv::Mat &image, const std::vector<cv::Point2f> &points, const cv::Size& target_size, int top_left_point_idx) {
    cv::Point2f point0 = points[static_cast<size_t>(top_left_point_idx)];
    cv::Point2f point1 = points[(top_left_point_idx + 1) % 4];
    cv::Point2f point2 = points[(top_left_point_idx + 2) % 4];

    cv::Mat crop(target_size, CV_8UC3, cv::Scalar(0));

    std::vector<cv::Point2f> from{point0, point1, point2};
    std::vector<cv::Point2f> to{cv::Point2f(0.0f, 0.0f), cv::Point2f(static_cast<float>(target_size.width-1), 0.0f),
                                cv::Point2f(static_cast<float>(target_size.width-1), static_cast<float>(target_size.height-1))};

    cv::Mat M = cv::getAffineTransform(from, to);

    cv::warpAffine(image, crop, M, crop.size());

    return crop;
}

struct TextRecognitionPipeline::Impl
{
    Ptr<dnn::Model> detectionNet;
    Ptr<dnn::Model> recognitionNet;
};

TextRecognitionPipeline::TextRecognitionPipelineImpl(const Topology& detection,
                                                     const Topology& recognition)
    : impl(new Impl())
{
    impl->detectionNet = DnnModel(detection);
    impl->recognitionNet = DnnModel(recognition);
}

void TextRecognitionPipeline::process(InputArray frame, std::vector<RotatedRect>& rects,
                                      std::vector<String>& texts)
{
    rects.clear();
    texts.clear();

    std::vector<Mat> outs;
    impl->detectionNet->predict(frame, outs);

    postProcess(outs, frame.size(), 0.5, 0.5, rects);

    // Recognition
    for (const auto &rect : rects) {
        cv::Mat cropped_text;
        std::vector<cv::Point2f> points;
        int top_left_point_idx = 0;

        points = floatPointsFromRotatedRect(rect);
        topLeftPoint(points, &top_left_point_idx);

        cvtColor(cropImage(frame.getMat(), points, Size(120, 32), top_left_point_idx), cropped_text, COLOR_BGR2GRAY);

        std::vector<Mat> recOuts;
        impl->recognitionNet->predict(cropped_text, recOuts);

        std::vector<size_t> output_shape(&recOuts[0].size[0], &recOuts[0].size[0] + recOuts[0].dims);
        std::string kAlphabet = "0123456789abcdefghijklmnopqrstuvwxyz#";
        const char kPadSymbol = '#';

        float *ouput_data_pointer = (float*)recOuts[0].data;
        std::vector<float> output_data(ouput_data_pointer, ouput_data_pointer + output_shape[0] * output_shape[2]);

        double conf;
        texts.push_back(CTCGreedyDecoder(output_data, kAlphabet, kPadSymbol, &conf));
    }
}

}}  // namespace cv::open_model_zoo
