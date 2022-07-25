// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

//#include <gflags/gflags.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include <models/detection_model.h>
#include <models/detection_model_centernet.h>
#include <models/detection_model_ssd.h>
#include <models/detection_model_yolo.h>
#include <models/input_data.h>
#include <models/internal_model_data.h>
#include <models/model_base.h>
#include <models/results.h>
#include <monitors/presenter.h>
#include <pipelines/metadata.h>
#include <utils/common.hpp>
#include <utils/config_factory.h>
#include <utils/images_capture.h>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>

#include <pipelines/async_pipeline.h>

#include "cnn.hpp"
#include "core.hpp"
#include "descriptor.hpp"
#include "distance.hpp"
#include "pedestrian_tracker_demo.hpp"
#include "tracker.hpp"
#include "utils.hpp"

#include <atomic>
#include <stdio.h>
#include <queue>
#include <thread>
#include <condition_variable>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>

//#include <libnlab-ctrl.hpp>
#include <VimbaCPP/Include/VimbaCPP.h>

#include <ncam/MJPEGStreamer.hpp>
#include <ncam/BufferedChannel.hpp>
#include <ncam/Camera.hpp>

using cv::Mat;
using std::vector;

using ImageWithFrameIndex = std::pair<cv::Mat, int>;

#define INFERENCE_CHANNEL_SIZE 3
#define JPEG_ENCODING_CHANNEL_SIZE 3
#define MAX_FRAME_BUFFERS 5

// The local port for the MJPEG server.
#define MJPEG_PORT 8080

ncam::BufferedChannel<Mat>           jpegEncodeChan(JPEG_ENCODING_CHANNEL_SIZE);
//ncam::BufferedChannel<vector<uchar>> videoChan(VIDEO_CHANNEL_SIZE);
ncam::BufferedChannel<Mat>           infChan(INFERENCE_CHANNEL_SIZE);
//ncam::BufferedChannel<vector<Rect>>  infResChan(INFERENCE_RESULT_CHANNEL_SIZE);

//DEFINE_INPUT_FLAGS
//DEFINE_OUTPUT_FLAGS


static const char help_message[] = "Print a usage message.";
static const char at_message[] = "Required. Type of the model, either 'ae' for Associative Embedding, 'higherhrnet' "
                                 "for HigherHRNet models based on ae "
                                 "or 'openpose' for OpenPose.";
static const char model_message[] = "Required. Path to an .xml file with a trained model.";
static const char layout_message[] = "Optional. Specify inputs layouts."
                                     " Ex. NCHW or input0:NCHW,input1:NC in case of more than one input.";
static const char target_size_message[] = "Optional. Target input size.";
static const char target_device_message[] =
    "Optional. Specify the target device to infer on (the list of available devices is shown below). "
    "Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
    "The demo will look for a suitable plugin for a specified device.";
static const char thresh_output_message[] = "Optional. Probability threshold for poses filtering.";
static const char nireq_message[] = "Optional. Number of infer requests. If this option is omitted, number of infer "
                                    "requests is determined automatically.";
static const char num_threads_message[] = "Optional. Number of threads.";
static const char num_streams_message[] = "Optional. Number of streams to use for inference on the CPU or/and GPU in "
                                          "throughput mode (for HETERO and MULTI device cases use format "
                                          "<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)";
static const char no_show_message[] = "Optional. Don't show output.";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";
static const char output_resolution_message[] =
    "Optional. Specify the maximum output window resolution "
    "in (width x height) format. Example: 1280x720. Input frame size used by default.";

DEFINE_bool(h, false, help_message);
DEFINE_string(at, "", at_message);
DEFINE_string(m, "", model_message);
DEFINE_string(layout, "", layout_message);
DEFINE_uint32(tsize, 0, target_size_message);
DEFINE_string(d, "CPU", target_device_message);
DEFINE_double(t, 0.1, thresh_output_message);
DEFINE_uint32(nireq, 0, nireq_message);
DEFINE_uint32(nthreads, 0, num_threads_message);
DEFINE_string(nstreams, "", num_streams_message);
DEFINE_bool(no_show, false, no_show_message);
DEFINE_string(u, "", utilization_monitors_message);
DEFINE_string(output_resolution, "", output_resolution_message);

 * \brief This function shows a help message

static void showUsage() {
    std::cout << std::endl;
    std::cout << "pedestrian_tracker_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -at \"<type>\"              " << at_message << std::endl;
    std::cout << "    -i                        " << input_message << std::endl;
    std::cout << "    -m \"<path>\"               " << model_message << std::endl;
    std::cout << "    -layout \"<string>\"        " << layout_message << std::endl;
    std::cout << "    -o \"<path>\"               " << output_message << std::endl;
    std::cout << "    -limit \"<num>\"            " << limit_message << std::endl;
    std::cout << "    -tsize                    " << target_size_message << std::endl;
    std::cout << "    -d \"<device>\"             " << target_device_message << std::endl;
    std::cout << "    -t                        " << thresh_output_message << std::endl;
    std::cout << "    -nireq \"<integer>\"        " << nireq_message << std::endl;
    std::cout << "    -nthreads \"<integer>\"     " << num_threads_message << std::endl;
    std::cout << "    -nstreams                 " << num_streams_message << std::endl;
    std::cout << "    -loop                     " << loop_message << std::endl;
    std::cout << "    -no_show                  " << no_show_message << std::endl;
    std::cout << "    -output_resolution        " << output_resolution_message << std::endl;
    std::cout << "    -u                        " << utilization_monitors_message << std::endl;
}


std::unique_ptr<PedestrianTracker> CreatePedestrianTracker(const std::string& reid_model,
                                                           const ov::Core& core,
                                                           const std::string& deviceName,
                                                           bool should_keep_tracking_info) {
    TrackerParams params;

    if (should_keep_tracking_info) {
        params.drop_forgotten_tracks = false;
        params.max_num_objects_in_track = -1;
    }

    std::unique_ptr<PedestrianTracker> tracker(new PedestrianTracker(params));

    // Load reid-model.
    std::shared_ptr<IImageDescriptor> descriptor_fast =
        std::make_shared<ResizedImageDescriptor>(cv::Size(16, 32), cv::InterpolationFlags::INTER_LINEAR);
    std::shared_ptr<IDescriptorDistance> distance_fast = std::make_shared<MatchTemplateDistance>();

    tracker->set_descriptor_fast(descriptor_fast);
    tracker->set_distance_fast(distance_fast);

    if (!reid_model.empty()) {
        ModelConfigTracker reid_config(reid_model);
        reid_config.max_batch_size = 16;  // defaulting to 16
        std::shared_ptr<IImageDescriptor> descriptor_strong =
            std::make_shared<Descriptor>(reid_config, core, deviceName);

        if (descriptor_strong == nullptr) {
            throw std::runtime_error("[SAMPLES] internal error - invalid descriptor");
        }
        std::shared_ptr<IDescriptorDistance> distance_strong = std::make_shared<CosDistance>(descriptor_strong->size());

        tracker->set_descriptor_strong(descriptor_strong);
        tracker->set_distance_strong(distance_strong);
    } else {
        slog::warn << "Reid model "
                   << "was not specified. "
                   << "Only fast reidentification approach will be used." << slog::endl;
    }

    return tracker;
}

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

    //gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m_det.empty()) {
        throw std::logic_error("Parameter -m_det is not set");
    }

    if (FLAGS_m_reid.empty()) {
        throw std::logic_error("Parameter -m_reid is not set");
    }

    if (FLAGS_at.empty()) {
        throw std::logic_error("Parameter -at is not set");
    }

    return true;
}

int main(int argc, char** argv) {
    try {
        PerformanceMetrics metrics;

        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        // Create camera.
        ncam::Camera cam = ncam::Camera();
        cam.printSystemVersion();

        // Start the camera.
        bool ok = cam.start(MAX_FRAME_BUFFERS);
        if (!ok) {
            return 1;
        }

        //auto startTime = std::chrono::steady_clock::now();
        cv::Mat curr_frame;
        if (!cam.read(curr_frame)) {
            return 1;
        }

        //OutputTransform outputTransform = OutputTransform();
        //cv::Size outputResolution = curr_frame.size();
        //size_t found = FLAGS_output_resolution.find("x");
        //if (found != std::string::npos) {
        //    outputResolution =
        //        cv::Size{std::stoi(FLAGS_output_resolution.substr(0, found)),
        //                 std::stoi(FLAGS_output_resolution.substr(found + 1, FLAGS_output_resolution.length()))};
        //    outputTransform = OutputTransform(curr_frame.size(), outputResolution);
        //    outputResolution = outputTransform.computeResolution();
        //}

        // Our motion jpeg server.
        ncam::MJPEGStreamer streamer;
        streamer.start(MJPEG_PORT, 1);
        std::cout << "MJPEG server listening on port " << std::to_string(MJPEG_PORT) << std::endl;

        //------------------------------ Running Human Pose Estimation routines
        //----------------------------------------------

        //double aspectRatio = curr_frame.cols / static_cast<double>(curr_frame.rows);
        std::unique_ptr<ModelBase> model;
        //if (FLAGS_at == "openpose") {
        //    model.reset(new HPEOpenPose(FLAGS_m, aspectRatio, FLAGS_tsize, static_cast<float>(FLAGS_t), FLAGS_layout));
        //} else if (FLAGS_at == "ae") {
        //    model.reset(new HpeAssociativeEmbedding(FLAGS_m,
        //                                            aspectRatio,
        //                                            FLAGS_tsize,
        //                                            static_cast<float>(FLAGS_t),
        //                                            FLAGS_layout));
        if (FLAGS_at == "higherhrnet") {
        //    float delta = 0.5f;
        //    model.reset(new HpeAssociativeEmbedding(FLAGS_m,
        //                                            aspectRatio,
        //                                            FLAGS_tsize,
        //                                            static_cast<float>(FLAGS_t),
        //                                            FLAGS_layout,
        //                                            delta,
        //                                            RESIZE_KEEP_ASPECT_LETTERBOX));
        } else {
            slog::err << "No model type or invalid model type (-at) provided: " + FLAGS_at << slog::endl;
            return -1;
        }

        slog::info << ov::get_openvino_version() << slog::endl;
        ov::Core core;

        //auto AsyncPipeline = AsyncPipeline;

        //AsyncPipeline pipeline(std::move(model),
        //                       ConfigFactory::getUserConfig(FLAGS_d, FLAGS_nireq, FLAGS_nstreams, FLAGS_nthreads),
        //                       core);
        Presenter presenter(FLAGS_u);

        //int64_t frameNum; 
        //int64_t frameNum =
        //    pipeline.submitData(ImageInputData(curr_frame), std::make_shared<ImageMetaData>(curr_frame, startTime));

        uint32_t framesProcessed = 0;
        bool keepRunning = true;
        std::unique_ptr<ResultBase> result;

        while (keepRunning) {
            {
        //    if (pipeline.isReadyToProcess()) {
                //--- Capturing frame
//                startTime = std::chrono::steady_clock::now();
                if (!cam.read(curr_frame) || curr_frame.empty()) {
                    std::cout << "no frame received" << std::endl;
                    // Input stream is over
                    break;
                }
        //        frameNum = pipeline.submitData(ImageInputData(curr_frame),
        //                                       std::make_shared<ImageMetaData>(curr_frame, startTime));
        //    }

            //--- Waiting for free input slot or output data available. Function will return immediately if any of them
            // are available.
            //pipeline.waitForData();

            //--- Checking for results and rendering data if it's ready
            //--- If you need just plain data without rendering - cast result's underlying pointer to HumanPoseResult*
            //    and use your own processing instead of calling renderHumanPose().
            //while (keepRunning && (result = pipeline.getResult())) {
            //    auto renderingStart = std::chrono::steady_clock::now();
                //cv::Mat outFrame = renderHumanPose(result->asRef<HumanPoseResult>(), outputTransform);
                //--- Showing results and device information

                cv::Mat outFrame = curr_frame;

                presenter.drawGraphs(outFrame);
                //renderMetrics.update(renderingStart);
                metrics.update(result->metaData->asRef<ImageMetaData>().timeStamp,
                               outFrame,
                               {10, 22},
                               cv::FONT_HERSHEY_COMPLEX,
                               0.65);
                streamer.publish("/stream", outFrame);
                framesProcessed++;
                if (!FLAGS_no_show) {
                    cv::imshow("Pedistrian Tracker Results", outFrame);
                    //--- Processing keyboard events
                    int key = cv::waitKey(1);
                    if (27 == key || 'q' == key || 'Q' == key) {  // Esc
                        keepRunning = false;
                    } else {
                        presenter.handleKey(key);
                    }
                }
            }
        }

        // ------------ Waiting for completion of data processing and rendering the rest of results ---------
        //pipeline.waitForTotalCompletion();
        //for (; framesProcessed <= frameNum; framesProcessed++) {
        //    while (!(result = pipeline.getResult())) {}
        //    auto renderingStart = std::chrono::steady_clock::now();
            //cv::Mat outFrame = renderHumanPose(result->asRef<HumanPoseResult>(), outputTransform);
            //--- Showing results and device information
            //presenter.drawGraphs(outFrame);

        //    cv::Mat outFrame = curr_frame;
        //
            //renderMetrics.update(renderingStart);
        //    metrics.update(result->metaData->asRef<ImageMetaData>().timeStamp,
        //                   outFrame,
        //                   {10, 22},
        //                   cv::FONT_HERSHEY_COMPLEX,
        //                   0.65);
        //    streamer.publish("/stream", outFrame);
        //    if (!FLAGS_no_show) {
        //        cv::imshow("P Results", outFrame);
                //--- Updating output window
        //        cv::waitKey(1);
        //    }
        //}

        // Reading command line parameters.
        auto catch det_model = FLAGS_m_det;
        auto reid_model = FLAGS_m_reid;

        auto detlog_out = FLAGS_out;

        auto detector_mode = FLAGS_d_det;
        auto reid_mode = FLAGS_d_reid;

        bool should_print_out = FLAGS_r;

        bool should_show = !FLAGS_no_show;
        int delay = FLAGS_delay;
        if (!should_show)
            delay = -1;
        should_show = (delay >= 0);

        bool should_save_det_log = !detlog_out.empty();

        std::vector<std::string> labels;
        if (!FLAGS_labels.empty())
            labels = DetectionModel::loadLabels(FLAGS_labels);

        std::unique_ptr<ModelBase> detectionModel;
        if (FLAGS_at == "centernet") {
            detectionModel.reset(new ModelCenterNet(det_model, static_cast<float>(FLAGS_t), labels, FLAGS_layout_det));
        } else if (FLAGS_at == "ssd") {
            detectionModel.reset(
                new ModelSSD(det_model, static_cast<float>(FLAGS_t), FLAGS_auto_resize, labels, FLAGS_layout_det));
        } else if (FLAGS_at == "yolo") {
            detectionModel.reset(new ModelYolo(det_model,
                                               static_cast<float>(FLAGS_t),
                                               FLAGS_auto_resize,
                                               FLAGS_yolo_af,
                                               static_cast<float>(FLAGS_iou_t),
                                               labels,
                                               {},
                                               {},
                                               FLAGS_layout_det));
        } else {
            slog::err << "No model type or invalid model type (-at) provided: " + FLAGS_at << slog::endl;
            return -1;
        }

        std::vector<std::string> devices{detector_mode, reid_mode};

        slog::info << ov::get_openvino_version() << slog::endl;
        
        //auto model = detectionModel->compileModel(
        //    ConfigFactory::getUserConfig(FLAGS_d_det, FLAGS_nireq, FLAGS_nstreams, FLAGS_nthreads),
        //    core);
        //auto req = model.create_infer_request();
        //auto req;

        bool should_keep_tracking_info = should_save_det_log || should_print_out;
        std::unique_ptr<PedestrianTracker> tracker =
            CreatePedestrianTracker(reid_model, core, reid_mode, should_keep_tracking_info);

        std::unique_ptr<ImagesCapture> cap =
            openImagesCapture(FLAGS_i,
                              FLAGS_loop,
                              FLAGS_nireq == 1 ? read_type::efficient : read_type::safe,
                              FLAGS_first,
                              FLAGS_read_limit);
        double video_fps = cap->fps();
        if (0.0 == video_fps) {
            // the default frame rate for DukeMTMC dataset
            video_fps = 60.0;
        }

        auto frame = curr_frame;

        for (unsigned frameIdx = 0;; ++frameIdx) {
            //detectionModel->preprocess(ImageInputData(frame), req);

            //req.infer();

            InferenceResult res;

            auto result = (detectionModel->postprocess(res))->asRef<DetectionResult>();

            TrackedObjects detections;

            for (size_t i = 0; i < result.objects.size(); i++) {
                TrackedObject object;
                object.confidence = result.objects[i].confidence;

                const float frame_width_ = static_cast<float>(frame.cols);
                const float frame_height_ = static_cast<float>(frame.rows);
                object.frame_idx = result.frameId;

                const float x0 = std::min(std::max(0.0f, result.objects[i].x / frame_width_), 1.0f) * frame_width_;
                const float y0 = std::min(std::max(0.0f, result.objects[i].y / frame_height_), 1.0f) * frame_height_;
                const float x1 =
                    std::min(std::max(0.0f, (result.objects[i].x + result.objects[i].width) / frame_width_), 1.0f) *
                    frame_width_;
                const float y1 =
                    std::min(std::max(0.0f, (result.objects[i].y + result.objects[i].height) / frame_height_), 1.0f) *
                    frame_height_;

                object.rect = cv::Rect2f(cv::Point(static_cast<int>(round(static_cast<double>(x0))),
                                                   static_cast<int>(round(static_cast<double>(y0)))),
                                         cv::Point(static_cast<int>(round(static_cast<double>(x1))),
                                                   static_cast<int>(round(static_cast<double>(y1)))));

                if (object.rect.area() > 0 &&
                    (static_cast<int>(result.objects[i].labelID) == FLAGS_person_label || FLAGS_person_label == -1)) {
                    detections.emplace_back(object);
                }
            }

            // timestamp in milliseconds
            uint64_t cur_timestamp = static_cast<uint64_t>(1000.0 / video_fps * frameIdx);
            tracker->Process(frame, detections, cur_timestamp);

            // Drawing colored "worms" (tracks).
            frame = tracker->DrawActiveTracks(frame);

            // Drawing all detected objects on a frame by BLUE COLOR
            for (const auto& detection : detections) {
                cv::rectangle(frame, detection.rect, cv::Scalar(255, 0, 0), 3);
            }

            // Drawing tracked detections only by RED color and print ID and detection
            // confidence level.
            for (const auto& detection : tracker->TrackedDetections()) {
                cv::rectangle(frame, detection.rect, cv::Scalar(0, 0, 255), 3);
                std::string text =
                    std::to_string(detection.object_id) + " conf: " + std::to_string(detection.confidence);
                putHighlightedText(frame,
                                   text,
                                   detection.rect.tl() - cv::Point{10, 10},
                                   cv::FONT_HERSHEY_COMPLEX,
                                   0.65,
                                   cv::Scalar(0, 0, 255),
                                   2);
            }
            presenter.drawGraphs(frame);
            //metrics.update(startTime, frame, {10, 22}, cv::FONT_HERSHEY_COMPLEX, 0.65);

            if (should_save_det_log && (frameIdx % 100 == 0)) {
                DetectionLog log = tracker->GetDetectionLog(true);
                SaveDetectionLogToTrajFile(detlog_out, log);
            }

        }

        if (should_keep_tracking_info) {
            DetectionLog log = tracker->GetDetectionLog(true);

            if (should_save_det_log)
                SaveDetectionLogToTrajFile(detlog_out, log);
            if (should_print_out)
                PrintDetectionLog(log);
        }

        slog::info << "Metrics report:" << slog::endl;
        metrics.logTotal();
        slog::info << presenter.reportMeans() << slog::endl;
    } catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    } catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    return 0;
}
