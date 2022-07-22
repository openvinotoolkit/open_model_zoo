// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/garray.hpp>
#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/gcomputation.hpp>
#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/gopaque.hpp>
#include <opencv2/gapi/gproto.hpp>
#include <opencv2/gapi/gstreaming.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/render/render.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include <monitors/presenter.h>
#include <utils/args_helper.hpp>
#include <utils/common.hpp>
#include <utils/images_capture.h>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>
#include <utils_gapi/stream_source.hpp>

#include "custom_kernels.hpp"
#include "face_detection_mtcnn_demo.hpp"
#include "gflags/gflags.h"
#include "utils.hpp"

#include <gflags/gflags.h>
#include <models/hpe_model_associative_embedding.h>
#include <models/hpe_model_openpose.h>
#include <models/input_data.h>
#include <models/model_base.h>
#include <models/results.h>
#include <monitors/presenter.h>
#include <pipelines/async_pipeline.h>
#include <pipelines/metadata.h>
#include <utils/common.hpp>
#include <utils/config_factory.h>
#include <utils/default_flags.hpp>
#include <utils/image_utils.h>
#include <utils/images_capture.h>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>

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

using namespace nlab;
using namespace std::chrono;

#define INFERENCE_CHANNEL_SIZE 3
#define JPEG_ENCODING_CHANNEL_SIZE 3
#define MAX_FRAME_BUFFERS 5

// The local port for the MJPEG server.
#define MJPEG_PORT 8080

ncam::BufferedChannel<Mat>           jpegEncodeChan(JPEG_ENCODING_CHANNEL_SIZE);
//ncam::BufferedChannel<vector<uchar>> videoChan(VIDEO_CHANNEL_SIZE);
ncam::BufferedChannel<Mat>           infChan(INFERENCE_CHANNEL_SIZE);
//ncam::BufferedChannel<vector<Rect>>  infResChan(INFERENCE_RESULT_CHANNEL_SIZE);

DEFINE_INPUT_FLAGS
DEFINE_OUTPUT_FLAGS

const int MAX_PYRAMID_LEVELS = 13;

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

/**
 * \brief This function shows a help message
 */
static void showUsage() {
    std::cout << std::endl;
    std::cout << "face_detection_mtcnn_demo [OPTION]" << std::endl;
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

namespace util {
bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    /** ---------- Parsing and validating input arguments ----------**/
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    if (FLAGS_i.empty())
        throw std::logic_error("Parameter -i is not set");
    if (FLAGS_m_p.empty())
        throw std::logic_error("Parameter -m_p is not set");
    if (FLAGS_m_r.empty())
        throw std::logic_error("Parameter -m_r is not set");
    if (FLAGS_m_o.empty())
        throw std::logic_error("Parameter -m_o is not set");
    return true;
}
}  // namespace util

namespace nets {
G_API_NET(MTCNNRefinement, <custom::GMat2(cv::GMat)>, "custom.mtcnn_refinement");
G_API_NET(MTCNNOutput, <custom::GMat3(cv::GMat)>, "custom.mtcnn_output");
}  // namespace nets

int main(int argc, char* argv[]) {
    try {
        PerformanceMetrics metrics;
        /** Get OpenVINO runtime version **/
        slog::info << ov::get_openvino_version() << slog::endl;

        if (!util::ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        /** Get information about frame **/
        std::shared_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i,
                                                               FLAGS_loop,
                                                               read_type::safe,
                                                               0,
                                                               std::numeric_limits<size_t>::max(),
                                                               stringToSize(FLAGS_res));
        // Create camera.
        ncam::Camera cam = ncam::Camera();
        cam.printSystemVersion();

        // Start the camera.
        bool ok = cam.start(MAX_FRAME_BUFFERS);
        if (!ok) {
            return 1;
        }

        auto startTime = std::chrono::steady_clock::now();
        cv::Mat curr_frame;
        if (!cam.read(curr_frame)) {
            return 1;
        }

                OutputTransform outputTransform = OutputTransform();
        cv::Size outputResolution = curr_frame.size();
        size_t found = FLAGS_output_resolution.find("x");
        if (found != std::string::npos) {
            outputResolution =
                cv::Size{std::stoi(FLAGS_output_resolution.substr(0, found)),
                         std::stoi(FLAGS_output_resolution.substr(found + 1, FLAGS_output_resolution.length()))};
            outputTransform = OutputTransform(curr_frame.size(), outputResolution);
            outputResolution = outputTransform.computeResolution();
        }

        // Our motion jpeg server.
        ncam::MJPEGStreamer streamer;
        streamer.start(MJPEG_PORT, 1);
        std::cout << "MJPEG server listening on port " << std::to_string(MJPEG_PORT) << std::endl;
        

        //------------------------------ Running Face Detection routines
        //----------------------------------------------

        double aspectRatio = curr_frame.cols / static_cast<double>(curr_frame.rows);
        std::unique_ptr<ModelBase> model;
        if (FLAGS_at == "openpose") {
            model.reset(new HPEOpenPose(FLAGS_m, aspectRatio, FLAGS_tsize, static_cast<float>(FLAGS_t), FLAGS_layout));
        } else if (FLAGS_at == "ae") {
            model.reset(new HpeAssociativeEmbedding(FLAGS_m,
                                                    aspectRatio,
                                                    FLAGS_tsize,
                                                    static_cast<float>(FLAGS_t),
                                                    FLAGS_layout));
        } else if (FLAGS_at == "higherhrnet") {
            float delta = 0.5f;
            model.reset(new HpeAssociativeEmbedding(FLAGS_m,
                                                    aspectRatio,
                                                    FLAGS_tsize,
                                                    static_cast<float>(FLAGS_t),
                                                    FLAGS_layout,
                                                    delta,
                                                    RESIZE_KEEP_ASPECT_LETTERBOX));
        } else {
            slog::err << "No model type or invalid model type (-at) provided: " + FLAGS_at << slog::endl;
            return -1;
        }

        slog::info << ov::get_openvino_version() << slog::endl;
        ov::Core core;

        AsyncPipeline pipeline(std::move(model),
                               ConfigFactory::getUserConfig(FLAGS_d, FLAGS_nireq, FLAGS_nstreams, FLAGS_nthreads),
                               core);
        Presenter presenter(FLAGS_u);

        int64_t frameNum =
            pipeline.submitData(ImageInputData(curr_frame), std::make_shared<ImageMetaData>(curr_frame, startTime));

        uint32_t framesProcessed = 0;
        bool keepRunning = true;
        std::unique_ptr<ResultBase> result;

        while (keepRunning) {
            if (pipeline.isReadyToProcess()) {
                //--- Capturing frame
                startTime = std::chrono::steady_clock::now();
                if (!cam.read(curr_frame) || curr_frame.empty()) {
                    std::cout << "no frame received" << std::endl;
                    // Input stream is over
                    break;
                }
                frameNum = pipeline.submitData(ImageInputData(curr_frame),
                                               std::make_shared<ImageMetaData>(curr_frame, startTime));
            }

            //--- Waiting for free input slot or output data available. Function will return immediately if any of them
            // are available.
            pipeline.waitForData();

            //--- Checking for results and rendering data if it's ready
            //--- If you need just plain data without rendering - cast result's underlying pointer to HumanPoseResult*
            //    and use your own processing instead of calling renderHumanPose().
            while (keepRunning && (result = pipeline.getResult())) {
                auto renderingStart = std::chrono::steady_clock::now();
            //    cv::Mat outFrame = renderHumanPose(result->asRef<HumanPoseResult>(), outputTransform);
                //--- Showing results and device information
                presenter.drawGraphs(outFrame);
                renderMetrics.update(renderingStart);
                metrics.update(result->metaData->asRef<ImageMetaData>().timeStamp,
                               outFrame,
                               {10, 22},
                               cv::FONT_HERSHEY_COMPLEX,
                               0.65);
                streamer.publish("/stream", outFrame);
                framesProcessed++;
                if (!FLAGS_no_show) {
                    cv::imshow("Face Detection Results", outFrame);
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

        /** Calculate scales, number of pyramid levels and sizes for PNet pyramid **/
        std::vector<cv::Size> level_size;
        std::vector<double> scales;

        const auto pyramid_levels = FLAGS_hs ? calculate_half_scales(frame_size, scales, level_size)
                                             : calculate_scales(frame_size, scales, level_size);
        CV_Assert(pyramid_levels <= MAX_PYRAMID_LEVELS);

        /** ---------------- Main graph of demo ---------------- **/
        /** Graph input
         * Proposal part of MTCNN graph
         * Preprocessing BGR2RGB + transpose (NCWH is expected instead of NCHW)
         **/
        cv::GMat in_original;
        cv::GMat in_originalRGB = cv::gapi::BGR2RGB(in_original);
        cv::GMat in_transposedRGB = cv::gapi::transpose(in_originalRGB);
        cv::GOpaque<cv::Size> in_sz = cv::gapi::streaming::size(in_original);
        cv::GMat regressions[MAX_PYRAMID_LEVELS];
        cv::GMat scores[MAX_PYRAMID_LEVELS];
        cv::GArray<custom::Face> nms_p_faces[MAX_PYRAMID_LEVELS];
        cv::GArray<custom::Face> total_faces[MAX_PYRAMID_LEVELS];

        /** The very first PNet pyramid layer to init total_faces[0] **/
        std::tie(regressions[0], scores[0]) = run_mtcnn_p(in_transposedRGB, get_pnet_level_name(level_size[0]));
        cv::GArray<custom::Face> faces0 = custom::BuildFaces::on(scores[0],
                                                                 regressions[0],
                                                                 static_cast<float>(scales[0]),
                                                                 static_cast<float>(FLAGS_th));
        cv::GArray<custom::Face> final_p_faces_for_bb2squares = custom::ApplyRegression::on(faces0, true);
        cv::GArray<custom::Face> final_faces_pnet0 = custom::BBoxesToSquares::on(final_p_faces_for_bb2squares);
        total_faces[0] = custom::RunNMS::on(final_faces_pnet0, 0.5f, false);

        /** The rest PNet pyramid layers to accumlate all layers result in total_faces[PYRAMID_LEVELS - 1]] **/
        for (int i = 1; i < pyramid_levels; ++i) {
            std::tie(regressions[i], scores[i]) = run_mtcnn_p(in_transposedRGB, get_pnet_level_name(level_size[i]));
            cv::GArray<custom::Face> faces = custom::BuildFaces::on(scores[i],
                                                                    regressions[i],
                                                                    static_cast<float>(scales[i]),
                                                                    static_cast<float>(FLAGS_th));
            cv::GArray<custom::Face> final_p_faces_for_bb2squares_i = custom::ApplyRegression::on(faces, true);
            cv::GArray<custom::Face> final_faces_pnet_i = custom::BBoxesToSquares::on(final_p_faces_for_bb2squares_i);
            nms_p_faces[i] = custom::RunNMS::on(final_faces_pnet_i, 0.5f, false);
            total_faces[i] = custom::AccumulatePyramidOutputs::on(total_faces[i - 1], nms_p_faces[i]);
        }

        /** Proposal post-processing **/
        cv::GArray<custom::Face> final_faces_pnet = custom::RunNMS::on(total_faces[pyramid_levels - 1], 0.7f, true);

        /** Refinement part of MTCNN graph **/
        cv::GArray<cv::Rect> faces_roi_pnet = custom::R_O_NetPreProcGetROIs::on(final_faces_pnet, in_sz);
        cv::GArray<cv::GMat> regressionsRNet, scoresRNet;
        std::tie(regressionsRNet, scoresRNet) =
            cv::gapi::infer<nets::MTCNNRefinement>(faces_roi_pnet, in_transposedRGB);

        /** Refinement post-processing **/
        cv::GArray<custom::Face> rnet_post_proc_faces =
            custom::RNetPostProc::on(final_faces_pnet, scoresRNet, regressionsRNet, static_cast<float>(FLAGS_th));
        cv::GArray<custom::Face> nms07_r_faces_total = custom::RunNMS::on(rnet_post_proc_faces, 0.7f, false);
        cv::GArray<custom::Face> final_r_faces_for_bb2squares = custom::ApplyRegression::on(nms07_r_faces_total, true);
        cv::GArray<custom::Face> final_faces_rnet = custom::BBoxesToSquares::on(final_r_faces_for_bb2squares);

        /** Output part of MTCNN graph **/
        cv::GArray<cv::Rect> faces_roi_rnet = custom::R_O_NetPreProcGetROIs::on(final_faces_rnet, in_sz);
        cv::GArray<cv::GMat> regressionsONet, scoresONet, landmarksONet;
        std::tie(regressionsONet, landmarksONet, scoresONet) =
            cv::gapi::infer<nets::MTCNNOutput>(faces_roi_rnet, in_transposedRGB);

        /** Output post-processing **/
        cv::GArray<custom::Face> onet_post_proc_faces = custom::ONetPostProc::on(final_faces_rnet,
                                                                                 scoresONet,
                                                                                 regressionsONet,
                                                                                 landmarksONet,
                                                                                 static_cast<float>(FLAGS_th));
        cv::GArray<custom::Face> final_o_faces_for_nms07 = custom::ApplyRegression::on(onet_post_proc_faces, true);
        cv::GArray<custom::Face> nms07_o_faces_total = custom::RunNMS::on(final_o_faces_for_nms07, 0.7f, true);
        cv::GArray<custom::Face> final_faces_onet = custom::SwapFaces::on(nms07_o_faces_total);

        /** Draw ROI and marks **/
        auto rendered =
            cv::gapi::wip::draw::render3ch(in_original, custom::BoxesAndMarks::on(in_original, final_faces_onet));

  ?      cv::GComputation graph_mtcnn(cv::GIn(in_original), cv::GOut(rendered));
        /** ---------------- End of graph ---------------- **/
        /** Configure networks **/

        // MTCNN Refinement detection network
        // clang-format off
        auto mtcnnr_net =
            cv::gapi::ie::Params<nets::MTCNNRefinement>{
                FLAGS_m_r,  // path to topology IR
                fileNameNoExt(FLAGS_m_r) + ".bin",  // path to weights
                FLAGS_d_r,  // device specifier
            }.cfgOutputLayers({"conv5-2", "prob1"})
             .cfgInputLayers({"data"});

        // MTCNN Output detection network
        auto mtcnno_net =
            cv::gapi::ie::Params<nets::MTCNNOutput>{
                FLAGS_m_o,  // path to topology IR
                fileNameNoExt(FLAGS_m_o) + ".bin",  // path to weights
                FLAGS_d_o,  // device specifier
            }.cfgOutputLayers({"conv6-2", "conv6-3", "prob1"})
             .cfgInputLayers({"data"});
        // clang-format on
        auto networks_mtcnn = cv::gapi::networks(mtcnnr_net, mtcnno_net);

        // MTCNN Proposal detection network
        for (int i = 0; i < pyramid_levels; ++i) {
            std::string net_id = get_pnet_level_name(level_size[i]);
            std::vector<size_t> reshape_dims = {1, 3, size_t(level_size[i].width), size_t(level_size[i].height)};
            cv::gapi::ie::Params<cv::gapi::Generic> mtcnnp_net{
                net_id,  // tag
                FLAGS_m_p,  // path to topology IR
                fileNameNoExt(FLAGS_m_p) + ".bin",  // path to weights
                FLAGS_d_p,  // device specifier
            };
            mtcnnp_net.cfgInputReshape("data", reshape_dims);
            networks_mtcnn += cv::gapi::networks(mtcnnp_net);
        }

        /** Custom kernels **/
        auto kernels_mtcnn = custom::kernels();
        auto mtcnn_args = cv::compile_args(networks_mtcnn, kernels_mtcnn);
        if (FLAGS_qc != 0) {
            mtcnn_args += cv::compile_args(cv::gapi::streaming::queue_capacity{FLAGS_qc});
        }
        auto pipeline_mtcnn = graph_mtcnn.compileStreaming(std::move(mtcnn_args));

        /** ---------------- The execution part ---------------- **/
        pipeline_mtcnn.setSource<custom::CommonCapSrc>(cap);

        cv::Size graphSize{static_cast<int>(frame_size.width / 4), 60};
        Presenter presenter(FLAGS_u, frame_size.height - graphSize.height - 10, graphSize);

        LazyVideoWriter videoWriter{FLAGS_o, cap->fps(), FLAGS_limit};

        /** Output Mat for result **/
        cv::Mat out_image;
        bool isStart = true;
        const auto startTime = std::chrono::steady_clock::now();
        pipeline_mtcnn.start();
        while (pipeline_mtcnn.pull(cv::gout(out_image))) {
            if (isStart) {
                metrics.update(startTime,
                               out_image,
                               {10, 22},
                               cv::FONT_HERSHEY_COMPLEX,
                               0.65,
                               {200, 10, 10},
                               2,
                               PerformanceMetrics::MetricTypes::FPS);
                isStart = false;
            } else {
                metrics.update({},
                               out_image,
                               {10, 22},
                               cv::FONT_HERSHEY_COMPLEX,
                               0.65,
                               {200, 10, 10},
                               2,
                               PerformanceMetrics::MetricTypes::FPS);
            }
            videoWriter.write(out_image);
            if (!FLAGS_no_show) {
                cv::imshow("Face detection mtcnn demo G-API", out_image);
                int key = cv::waitKey(1);
                /** Press 'Esc' or 'Q' to quit **/
                if (key == 27)
                    break;
                if (key == 81)  // Q
                    break;
                else
                    presenter.handleKey(key);
            }
        }

        slog::info << "Metrics report:" << slog::endl;
        slog::info << "\tFPS: " << std::fixed << std::setprecision(1) << metrics.getTotal().fps << slog::endl;
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
