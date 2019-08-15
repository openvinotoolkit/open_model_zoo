// This file is based on https://github.com/opencv/open_model_zoo/tree/2019/demos/text_detection_demo
#include "opencv2/open_model_zoo.hpp"
#include "opencv2/open_model_zoo/text_recognition.hpp"
#include "opencv2/open_model_zoo/dnn.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <unordered_map>

#ifdef HAVE_INF_ENGINE
#include "text_detection.hpp"
#include "text_recognition.hpp"

#include <inference_engine.hpp>
#endif

namespace cv { namespace open_model_zoo {

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

static int strToDnnTarget(std::string device)
{
    std::transform(device.begin(), device.end(), device.begin(), ::tolower);
    if (device == "cpu")          return dnn::DNN_TARGET_CPU;
    else if (device == "gpu")     return dnn::DNN_TARGET_OPENCL;
    else if (device == "myriad")  return dnn::DNN_TARGET_MYRIAD;
    else if (device == "gpu16")   return dnn::DNN_TARGET_OPENCL_FP16;
    else if (device == "fpga")    return dnn::DNN_TARGET_FPGA;
    else
        CV_Error(Error::StsNotImplemented, "Unknown device target: " + device);
}

struct TextRecognitionPipeline::Impl
{
    Impl(const Topology& detection, const Topology& recognition,
         const String& dd, const String& rd)
    {
        detectionNet = DnnModel(detection);
        recognitionNet = DnnModel(recognition);
        detectionNet->setPreferableTarget(strToDnnTarget(dd));
        recognitionNet->setPreferableTarget(strToDnnTarget(rd));
    }

    Ptr<dnn::Model> detectionNet;
    Ptr<dnn::Model> recognitionNet;

    int maxRectNum = -1;
    float pixelClsThr = 0.8, pixelLinkThr = 0.8, recognThr = 0.2;
};

TextRecognitionPipeline::TextRecognitionPipelineImpl(const String& dd, const String& rd)
{
    auto detection = (dd == "GPU16" || dd == "MYRIAD") ? text_detection_fp16() : text_detection();
    auto recognition = (rd == "GPU16" || rd == "MYRIAD") ? text_recognition_fp16() : text_recognition();
    impl.reset(new Impl(detection, recognition, dd, rd));
}

TextRecognitionPipeline::TextRecognitionPipelineImpl(const Topology& detection,
                                                     const Topology& recognition,
                                                     const String& dd,
                                                     const String& rd)
    : impl(new Impl(detection, recognition, dd, rd))
{
}

void TextRecognitionPipeline::setMaxRectNum(int num) { impl->maxRectNum = num; }
void TextRecognitionPipeline::setRecognitionThresh(float thr) { impl->recognThr = thr; }
void TextRecognitionPipeline::setPixelClassificationThresh(float thr) { impl->pixelClsThr = thr; }
void TextRecognitionPipeline::setPixelLinkThresh(float thr) { impl->pixelLinkThr = thr; }

void TextRecognitionPipeline::process(InputArray frame, std::vector<RotatedRect>& rects,
                                      std::vector<String>& texts,
                                      std::vector<float>& confidences)
{
#ifdef HAVE_INF_ENGINE
    rects.clear();
    texts.clear();
    confidences.clear();

    std::vector<Mat> outs;
    impl->detectionNet->predict(frame, outs);

    auto outsNames = impl->detectionNet->getUnconnectedOutLayersNames();
    CV_Assert(outs.size() == outsNames.size());
    InferenceEngine::BlobMap ieBlobs;
    for (size_t i = 0; i < outs.size(); ++i)
    {
        std::vector<size_t> shape(&outs[i].size[0], &outs[i].size[0] + outs[i].dims);
        ieBlobs[outsNames[i]] = InferenceEngine::make_shared_blob<float>({
              InferenceEngine::Precision::FP32, shape, InferenceEngine::Layout::ANY},
              (float*)outs[i].data);
    }
    auto detectedRects = postProcess(ieBlobs, frame.size(), impl->pixelClsThr, impl->pixelLinkThr);

    if (impl->maxRectNum >= 0 && static_cast<int>(detectedRects.size()) > impl->maxRectNum) {
        std::sort(detectedRects.begin(), detectedRects.end(), [](const cv::RotatedRect& a, const cv::RotatedRect& b) {
            return a.size.area() > b.size.area();
        });
        detectedRects.resize(static_cast<size_t>(impl->maxRectNum));
    }

    // Recognition
    for (const auto &rect : detectedRects) {
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
        std::string text = CTCGreedyDecoder(output_data, kAlphabet, kPadSymbol, &conf);
        if (conf >= impl->recognThr)
        {
            texts.push_back(text);
            rects.push_back(rect);
            confidences.push_back(conf);
        }
    }
#else
    CV_UNUSED(frame); CV_UNUSED(rects); CV_UNUSED(texts); CV_UNUSED(confidences);
    CV_Error(Error::StsNotImplemented, "Inference Engine is required");
#endif
}

}}  // namespace cv::open_model_zoo
