// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/infer/ie.hpp>

#include <inference_engine.hpp>

#include <utils/slog.hpp>

namespace IE = InferenceEngine;

namespace custom {

const std::vector<std::string> coco_classes = {
"person",         //0
"bicycle",        //1
"car",            //2
"motorcycle",     //3
"airplane",       //4
"bus",            //5
"train",          //6
"truck",          //7
"boat",           //8
"traffic light",  //9
"fire hydrant",   //10
"stop sign",      //11
"parking meter",  //12
"bench",          //13
"bird",           //14
"cat",            //15
"dog",            //16
"horse",          //17
"sheep",          //18
"cow",            //19
"elephant",       //20
"bear",           //21
"zebra",          //22
"giraffe",        //23
"backpack",       //24
"umbrella",       //25
"handbag",        //26
"tie",            //27
"suitcase",       //28
"frisbee",        //29
"skis",           //30
"snowboard",      //31
"sports ball",    //32
"kite",           //33
"baseball bat",   //34
"baseball glove", //35
"skateboard",     //36
"surfboard",      //37
"tennis racket",  //38
"bottle",         //39
"wine glass",     //40
"cup",            //41
"fork",           //42
"knife",          //43
"spoon",          //44
"bowl",           //45
"banana",         //46
"apple",          //47
"sandwich",       //48
"orange",         //49
"broccoli",       //50
"carrot",         //51
"hot dog",        //52
"pizza",          //53
"donut",          //54
"cake",           //55
"chair",          //56
"couch",          //57
"potted plant",   //58
"bed",            //59
"dining table",   //60
"toilet",         //61
"tv",             //62
"laptop",         //63
"mouse",          //64
"remote",         //65
"keyboard",       //66
"cell phone",     //67
"microwave",      //68
"oven",           //69
"toaster",        //70
"sink",           //71
"refrigerator",   //72
"book",           //73
"clock",          //74
"vase",           //75
"scissors",       //76
"teddy bear",     //77
"hair drier",     //78
"toothbrush"      //79
};

std::vector<std::string> coco_labels;

struct DetectedObject : public cv::Rect2f
{
    unsigned int labelID;
    std::string label;
    float confidence;
};

using GDetections = cv::GArray<DetectedObject>;

G_API_OP(GYOLOv4TinyPostProcessingKernel, < GDetections(cv::GMat, cv::GMat, cv::GMat, float, float, bool) >, "custom.yolov4_tiny_post_processing") {
        static cv::GArrayDesc outMeta(const cv::GMatDesc&, const cv::GMatDesc&, const cv::GMatDesc&, const float, const float, const bool) {
            return cv::empty_array_desc();
        }
};

G_API_OP(GSmartFramingKernel, <cv::GMat(cv::GMat, GDetections)>, "custom.smart_framing") {
        static cv::GMatDesc outMeta(const cv::GMatDesc & in, const cv::GArrayDesc&) {
            return in;
        }
};

G_API_OP(GSuperResolutionPostProcessingKernel, < cv::GMat(cv::GMat) >, "custom.super_resolution_post_processing") {
        static cv::GMatDesc outMeta(const cv::GMatDesc & in) {
            cv::GMatDesc out_desc(CV_8U /* depth */, in.dims[1] /* channels */, cv::Size(in.dims[3], in.dims[2]), false /* planar */);
            return out_desc;
        }
};

G_API_OP(GCvt32Fto8U, <cv::GMat(cv::GMat)>, "custom.convertFP32ToU8") {
    static cv::GMatDesc outMeta(const cv::GMatDesc & in) {
        // NB: Input is ND mat.
        return cv::GMatDesc{ CV_8U, in.dims[1], cv::Size(in.dims[3], in.dims[2]) };
    }
};

cv::gapi::GKernelPackage kernels();

} // namespace custom
