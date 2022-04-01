#include "custom_kernels.hpp"

#include <algorithm>

#include <opencv2/gapi/cpu/gcpukernel.hpp>

#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/operators.hpp>

namespace {

struct YOLOv4TinyPostProcessing {

    int region_num = 3;
    int region_classes = 80;
    int region_coords = 4;
    //YOLO v4
    int netInputHeight = 416;
    int netInputWidth = 416;

    double intersectionOverUnion(const custom::DetectedObject& o1, const custom::DetectedObject& o2) {
        double overlappingWidth = fmin(o1.x + o1.width, o2.x + o2.width) - fmax(o1.x, o2.x);
        double overlappingHeight = fmin(o1.y + o1.height, o2.y + o2.height) - fmax(o1.y, o2.y);
        double intersectionArea = (overlappingWidth < 0 || overlappingHeight < 0) ? 0 : overlappingHeight * overlappingWidth;
        double unionArea = o1.width * o1.height + o2.width * o2.height - intersectionArea;
        return intersectionArea / unionArea;
    }

    std::vector<float> defaultAnchors = {
        // YOLOv4_Tiny
        { 10.0f, 14.0f, 23.0f, 27.0f, 37.0f, 58.0f,
          81.0f, 82.0f, 135.0f, 169.0f, 344.0f, 319.0f}
    };

    std::vector<int64_t> defaultMasks = {
        // YOLOv4_Tiny
        {1, 2, 3, 3, 4, 5}
    };

    static inline float sigmoid(float x) {
        return 1.f / (1.f + exp(-x));
    }
    static inline float linear(float x) {
        return x;
    }

    int calculateEntryIndex(int totalCells, int lcoords, int lclasses, int location, int entry) {
        int n = location / totalCells;
        int loc = location % totalCells;
        return (n * (lcoords + lclasses + 1) + entry) * totalCells + loc;
    }

    //std::string getLabelName(int labelID) { return (size_t)labelID < custom::coco_classes.size() ? custom::coco_classes[labelID] : std::string("Label #") + std::to_string(labelID); }
    std::string getLabelName(int labelID, std::vector<std::string> &labels) { return (size_t)labelID < labels.size() ? labels[labelID] : std::string("Label #") + std::to_string(labelID); }

    std::vector<float> blobs_anchor[2]{
        {23.0f, 27.0f, 37.0f, 58.0f, 81.0f, 82.0f},
        {81.0f, 82.0f, 135.0f, 169.0f, 344.0f, 319.0f}
    };

    void parseYOLOOutput(
        const cv::Mat& blob, int blobID, const unsigned long resized_im_h,
        const unsigned long resized_im_w, const unsigned long original_im_h,
        const unsigned long original_im_w, const float confidenceThreshold,
        std::vector<std::string>&labels, std::vector<custom::DetectedObject>& objects) {
        // --------------------------- Extracting layer parameters -------------------------------------
        int sideW = 0;
        int sideH = 0;
        unsigned long scaleH;
        unsigned long scaleW;
        //case YOLO_V3:
        //case YOLO_V4:
        //case YOLO_V4_TINY:
        sideH = static_cast<int>(blob.size[2]);
        sideW = static_cast<int>(blob.size[3]);
        slog::debug << "parseYOLOOutput  sideH " << sideH << " sideW " << sideW << slog::endl;
        scaleW = resized_im_w;
        scaleH = resized_im_h;
        slog::debug << "parseYOLOOutput  scaleH " << scaleH << " scaleW " << scaleW << slog::endl;
        slog::debug << "parseYOLOOutput  original_im_h " << original_im_h << " original_im_w " << original_im_w << slog::endl;


        auto entriesNum = sideW * sideH;
        const float* output_blob = (float*)blob.data;

        auto postprocessRawData = sigmoid;

        // --------------------------- Parsing YOLO Region output -------------------------------------
        for (int i = 0; i < entriesNum; ++i) {
            int row = i / sideW;
            int col = i % sideW;
            for (int n = 0; n < region_num; ++n) {
                //--- Getting region data from blob
                int obj_index = calculateEntryIndex(entriesNum,  region_coords, region_classes, n * entriesNum + i, region_coords);
                int box_index = calculateEntryIndex(entriesNum,  region_coords, region_classes, n * entriesNum + i, 0);
                float scale = postprocessRawData(output_blob[obj_index]);

                //--- Preliminary check for confidence threshold conformance
                if (scale >= confidenceThreshold) {
                    //--- Calculating scaled region's coordinates
                    double x = (col + postprocessRawData(output_blob[box_index + 0 * entriesNum])) / sideW * original_im_w;
                    double y = (row + postprocessRawData(output_blob[box_index + 1 * entriesNum])) / sideH * original_im_h;

                    double height = std::exp(output_blob[box_index + 3 * entriesNum]) * blobs_anchor[blobID][2 * n + 1] * original_im_h / scaleH;
                    double width = std::exp(output_blob[box_index + 2 * entriesNum]) * blobs_anchor[blobID][2 * n] * original_im_w / scaleW;

                    custom::DetectedObject obj;
                    obj.x = static_cast<float>(std::max((x - width / 2), 0.));
                    obj.y = static_cast<float>(std::max((y - height / 2), 0.));
                    obj.width = std::min(static_cast<float>(width), original_im_w - obj.x);
                    obj.height = std::min(static_cast<float>(height), original_im_h - obj.y);

                    for (int j = 0; j < region_classes; ++j) {
                        int class_index = calculateEntryIndex(entriesNum, region_coords, region_classes, n * entriesNum + i, region_coords + 1 + j);
                        float prob = scale * postprocessRawData(output_blob[class_index]);

                        //--- Checking confidence threshold conformance and adding region to the list
                        if (prob >= confidenceThreshold) {
                            obj.confidence = prob;
                            obj.labelID = j;
                            obj.label = getLabelName(obj.labelID, labels);
                            objects.push_back(obj);
                        }
                    }
                }
            }
        }
    }
};

using GPostProc = cv::GOpaque<YOLOv4TinyPostProcessing>;

} // anonymous namespace


GAPI_OCV_KERNEL(OCVYOLOv4TinyPostProcessing, custom::GYOLOv4TinyPostProcessingKernel) {
    static void run(const cv::Mat & image, const cv::Mat & in_blob26x26, const cv::Mat & in_blob13x13, std::vector<std::string> &labels,
        const float confidenceThreshold, const float boxIOUThreshold, const bool useAdvancedPostprocessing, std::vector<custom::DetectedObject> &objects) {
        YOLOv4TinyPostProcessing post_processor;
        int blob_size[2];
        slog::debug << "in_blob26x26 size " << in_blob26x26.size << slog::endl;
        slog::debug << "in_blob13x13 size " << in_blob13x13.size << slog::endl;
        blob_size[0] = in_blob26x26.size[0] * in_blob26x26.size[1] * in_blob26x26.size[2] * in_blob26x26.size[3];
        blob_size[1] = in_blob13x13.size[0] * in_blob13x13.size[1] * in_blob13x13.size[2] * in_blob13x13.size[3];
        int total_size = blob_size[0] + blob_size[1];
        slog::debug << "total_size " << total_size << slog::endl; //should be 215475

        //OMZ Post-processing for Yolo
        std::vector<custom::DetectedObject> initial_objects;
        post_processor.parseYOLOOutput(in_blob26x26, 0, post_processor.netInputHeight, post_processor.netInputWidth,
            image.rows, image.cols, confidenceThreshold, labels, initial_objects);
        slog::debug << "Accumulated DetectedObject size for blobID " << 0 << " is " << initial_objects.size() << slog::endl;
        post_processor.parseYOLOOutput(in_blob13x13, 1, post_processor.netInputHeight, post_processor.netInputWidth,
            image.rows, image.cols, confidenceThreshold, labels, initial_objects);
        slog::debug << "Accumulated DetectedObject size for blobID " << 1 << " is " << initial_objects.size() << slog::endl;

        slog::debug << "Total DetectedObject size " << initial_objects.size() << slog::endl;


        if (useAdvancedPostprocessing) {
            // Advanced postprocessing
            // Checking IOU threshold conformance
            // For every i-th object we're finding all objects it intersects with, and comparing confidence
            // If i-th object has greater confidence than all others, we include it into result
            for (const auto& obj1 : initial_objects) {
                bool isGoodResult = true;
                for (const auto& obj2 : initial_objects) {
                    if (obj1.labelID == obj2.labelID && obj1.confidence < obj2.confidence && post_processor.intersectionOverUnion(obj1, obj2) >= boxIOUThreshold) { // if obj1 is the same as obj2, condition expression will evaluate to false anyway
                        isGoodResult = false;
                        break;
                    }
                }
                if (isGoodResult) {
                    objects.push_back(obj1);
                }
            }
        }
        else {
            // Classic postprocessing
            std::sort(initial_objects.begin(), initial_objects.end(), [](const custom::DetectedObject& x, const custom::DetectedObject& y) { return x.confidence > y.confidence; });
            for (size_t i = 0; i < initial_objects.size(); ++i) {
                if (initial_objects[i].confidence == 0)
                    continue;
                for (size_t j = i + 1; j < initial_objects.size(); ++j)
                    if (post_processor.intersectionOverUnion(initial_objects[i], initial_objects[j]) >= boxIOUThreshold)
                        initial_objects[j].confidence = 0;
                objects.push_back(initial_objects[i]);
            }
        }
        slog::debug << "Filtered DetectedObject size " << objects.size() << slog::endl;
    }
};

GAPI_OCV_KERNEL(OCVSmartFramingKernel, custom::GSmartFramingKernel) {
    static void run(const cv::Mat & image, const std::vector<custom::DetectedObject> &objects, cv::Mat & out) {
        cv::Rect init_rect;
        for (const auto& el : objects) {
            if (el.labelID == 0) {//person ID
                init_rect = init_rect | static_cast<cv::Rect>(el);
            }
        }
        slog::debug << "SF result rect" << init_rect << slog::endl;
        if (!init_rect.empty()) {
            cv::Rect even_rect;
            even_rect.x = (init_rect.x % 2 == 0) ? (init_rect.x) : (init_rect.x - 1);
            even_rect.y = (init_rect.y % 2 == 0) ? (init_rect.y) : (init_rect.y - 1);
            even_rect.width = (init_rect.width % 2 == 0) ? (init_rect.width) : (init_rect.width - 1);
            even_rect.height = (init_rect.height % 2 == 0) ? (init_rect.height) : (init_rect.height - 1);

            cv::Mat SF_ROI;
            image(even_rect).copyTo(SF_ROI);

            cv::Mat SF_resized_ROI;
            cv::Size target_size;
            target_size.height = image.size().height;
            target_size.width = even_rect.width * (static_cast<float>(image.size().height) / static_cast<float>(even_rect.height));
            target_size.width = (target_size.width % 2 == 0) ? (target_size.width) : (target_size.width - 1);
            if (target_size.width > image.size().width) {
                target_size.width = image.size().width;
            }
            cv::resize(SF_ROI, SF_resized_ROI, target_size);
            int left_add = (image.size().width - target_size.width) / 2;
            int right_add = (image.size().width - target_size.width) / 2;
            cv::copyMakeBorder(SF_resized_ROI, out,
                0,
                0,
                left_add,
                right_add,
                cv::BORDER_CONSTANT, cv::Scalar::all(127));
        }
        else {
            image.copyTo(out);
        }
    }
};

GAPI_OCV_KERNEL(OCVSuperResolutionPostProcessingKernel, custom::GSuperResolutionPostProcessingKernel) {
    static void run(const cv::Mat & image, cv::Mat & out) {
        float* outputData = (float*)image.data;
        size_t outChannels = image.size[1];
        size_t outHeight = image.size[2];
        size_t outWidth = image.size[3];
        size_t numOfPixels = outWidth * outHeight;
        std::vector<cv::Mat> imgPlanes;
        if (outChannels == 3) {
            imgPlanes = std::vector<cv::Mat>{
                  cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[0])),
                  cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[numOfPixels])),
                  cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[numOfPixels * 2])) };
        }
        else {
            imgPlanes = std::vector<cv::Mat>{ cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[0])) };
            // Post-processing for text-image-super-resolution models
            cv::threshold(imgPlanes[0], imgPlanes[0], 0.5f, 1.0f, cv::THRESH_BINARY);
        }
        for (auto& img : imgPlanes)
            img.convertTo(img, CV_8UC1, 255);

        cv::merge(imgPlanes, out);
    }
};


// NB: cv::gapi::convertTo + reshape from 4D blob to image
GAPI_OCV_KERNEL(OCVCvt32Fto8U, custom::GCvt32Fto8U) {
    static void run(const cv::Mat & in,
                    cv::Mat & out) {
                    int h = in.size[2];
                    int w = in.size[3];

                    auto* out_p = out.ptr<uint8_t>();
                    auto* in_p = in.ptr<const float>();

                    std::transform(in_p, in_p + h * w, out_p,
                                   [](float v) { return static_cast<uint8_t>(v * 255); });
    }
};

cv::gapi::GKernelPackage custom::kernels() {
    return cv::gapi::kernels<OCVYOLOv4TinyPostProcessing,
                             OCVSmartFramingKernel,
                             OCVSuperResolutionPostProcessingKernel,
                             OCVCvt32Fto8U>();
}
