// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <monitors/presenter.h>
#include <utils/args_helper.hpp>
#include <utils/slog.hpp>

#include "gaze_estimation_demo_gapi.hpp"
#include "face_inference_results.hpp"
#include "results_marker.hpp"
#include "exponential_averager.hpp"
#include "utils.hpp"
#include "custom_kernels.hpp"
#include "kernel_packages.hpp"
#include "stream_source.hpp"

#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/core.hpp>

namespace util {
bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    /** ---------- Parsing and validating input arguments ----------**/
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;
    if (FLAGS_i.empty())
        throw std::logic_error("Parameter -i is not set");
    if (FLAGS_m.empty())
        throw std::logic_error("Parameter -m is not set");
    if (FLAGS_m_fd.empty())
        throw std::logic_error("Parameter -m_fd is not set");
    if (FLAGS_m_hp.empty())
        throw std::logic_error("Parameter -m_hp is not set");
    if (FLAGS_m_lm.empty())
        throw std::logic_error("Parameter -m_lm is not set");
    if (FLAGS_m_es.empty())
        throw std::logic_error("Parameter -m_es is not set");
    return true;
}
} // namespace util

namespace nets {
G_API_NET(Faces,     <cv::GMat(cv::GMat)>, "face-detector"   );
G_API_NET(Landmarks, <cv::GMat(cv::GMat)>, "facial-landmarks");
G_API_NET(HeadPose,  <custom::GMat3(cv::GMat)>, "head-pose");
G_API_NET(Gaze,      <cv::GMat(cv::GMat,cv::GMat,cv::GMat)>, "gaze-vector");
G_API_NET(Eyes,      <cv::GMat(cv::GMat)>, "l-open-closed-eyes");
} // namespace nets

int main(int argc, char *argv[]) {
    try {
        using namespace gaze_estimation;
        /** Print info about Inference Engine **/
        std::cout << "InferenceEngine: " << printable(*InferenceEngine::GetInferenceEngineVersion()) << std::endl;
        // ---------- Parsing and validating of input arguments ----------
        if (!util::ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        /** ---------------- Main graph of demo ---------------- **/
        /** Graph input **/
        cv::GMat in;

        /** Face detection **/
        cv::GMat faces = cv::gapi::infer<nets::Faces>(in);
        /** Get size of frame **/
        cv::GOpaque<cv::Size> sz = cv::gapi::streaming::size(in);
        cv::GArray<cv::Rect> faces_rc;
        cv::GArray<float> faces_conf;
        /** Get ROI for each face and its confidence **/
        std::tie(faces_rc, faces_conf) = custom::ParseSSD::on(faces, sz, float(FLAGS_t));

        /** Head pose recognition **/
        cv::GArray<cv::GMat> angles_y, angles_p, angles_r;
        std::tie(angles_y, // yaw
                 angles_p, // pitch
                 angles_r) // roll
            = cv::gapi::infer<nets::HeadPose>(faces_rc, in);

        cv::GArray<cv::GMat> heads_pos_without_roll;
        cv::GArray<cv::Point3f> heads_pos;
        /** Get heads poses and poses without roll part **/
        std::tie(heads_pos,
                 heads_pos_without_roll) = custom::ProcessPoses::on(angles_y,
                                                                    angles_p,
                                                                    angles_r);

        /** Landmarks detector **/
        cv::GArray<cv::GMat> landmarks = cv::gapi::infer<nets::Landmarks>(faces_rc, in);
        cv::GArray<cv::Rect> left_eyes_rc, right_eyes_rc;
        cv::GArray<cv::Point2f> leftEyeMidpoint, rightEyeMidpoint;
        cv::GArray<std::vector<cv::Point>> faces_landmarks;
        /** Get information about eyes by landmarks **/
        std::tie(left_eyes_rc,     // ROIs for left eyes
                 right_eyes_rc,    // ROIs for right eyes
                 leftEyeMidpoint,  // left eyes midpoints
                 rightEyeMidpoint, // right eyes midpoints
                 faces_landmarks)  // processed landmarks
            = custom::ProcessLandmarks::on(in,
                                           landmarks,
                                           faces_rc);

        /** Prepare eyes for open-closed-eye network **/
        cv::GArray<cv::GMat> left_processed_eyes, right_processed_eyes;
        std::tie(left_processed_eyes,
                 right_processed_eyes) = custom::PrepareEyes::on(in,
                                                                 left_eyes_rc,
                                                                 right_eyes_rc,
                                                                 angles_r,
                                                                 cv::Size{32, 32});

        /** Detect states of left eyes **/
        cv::GArray<cv::GMat> left_state_eyes =
            cv::gapi::infer2<nets::Eyes>(in, left_processed_eyes);
        /** Detect states of right eyes **/
        cv::GArray<cv::GMat> right_state_eyes =
            cv::gapi::infer2<nets::Eyes>(in, right_processed_eyes);
        /** Recognize states of eyes **/
        cv::GArray<int> state_left_eyes, state_right_eyes;
        std::tie(state_left_eyes,  // open/closed
                 state_right_eyes) // 1   /0
            = custom::ProcessEyes::on(in, left_state_eyes, right_state_eyes);

        /** Prepare eyes for gaze-estimation network **/
        std::tie(left_processed_eyes,
                 right_processed_eyes) = custom::PrepareEyes::on(in,
                                                                 left_eyes_rc,
                                                                 right_eyes_rc,
                                                                 angles_r,
                                                                 cv::Size{60, 60});

        /** Gaze estimation **/
        cv::GArray<cv::GMat> gaze_vectors = cv::gapi::infer2<nets::Gaze>(in,
                                                                         left_processed_eyes,
                                                                         right_processed_eyes,
                                                                         heads_pos_without_roll);
        /** Processing gaze estimation results **/
        cv::GArray<cv::Point3f> processed_gaze_vectors =
            custom::ProcessGazes::on(gaze_vectors, angles_r);

        /** Inputs and outputs of graph **/
        cv::GComputation graph(cv::GIn(in),
                               cv::GOut(cv::gapi::copy(in),
                                        faces_conf,
                                        faces_rc,
                                        faces_landmarks,
                                        heads_pos,
                                        left_eyes_rc,
                                        right_eyes_rc,
                                        leftEyeMidpoint,
                                        rightEyeMidpoint,
                                        state_left_eyes,
                                        state_right_eyes,
                                        processed_gaze_vectors));
        /** ---------------- End of graph ---------------- **/
        /** Configure networks **/
        auto face_net = cv::gapi::ie::Params<nets::Faces> {
            FLAGS_m_fd,                          // path to topology IR
            fileNameNoExt(FLAGS_m_fd) + ".bin",  // path to weights
            FLAGS_d_fd,                          // device specifier
        };
        /** Get information about frame from cv::VideoCapture **/
        std::shared_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop, 0,
            std::numeric_limits<size_t>::max(), stringToSize(FLAGS_res));
        const auto tmp = cap->read();
        cap.reset();
        if (!tmp.data) {
            throw std::runtime_error("Couldn't grab first frame");
        }
        cv::Size frame_size = cv::Size{tmp.cols, tmp.rows};
        cap = openImagesCapture(FLAGS_i, FLAGS_loop, 0,
            std::numeric_limits<size_t>::max(), stringToSize(FLAGS_res));
        if (FLAGS_fd_reshape) {
            InferenceEngine::Core ie;
            const auto network = ie.ReadNetwork(FLAGS_m_fd);
            const auto layerName = network.getInputsInfo().begin()->first;
            const auto layerData = network.getInputsInfo().begin()->second;
                  auto layerDims = layerData->getTensorDesc().getDims();

            const double imageAspectRatio = std::round(100. * frame_size.width / frame_size.height) / 100.;
            const double networkAspectRatio = std::round(100. * layerDims[3] / layerDims[2]) / 100.;
            const double aspectRatioThreshold = 0.01;

            if (std::fabs(imageAspectRatio - networkAspectRatio) > aspectRatioThreshold) {
                std::cout << "Face Detection network is reshaped" << std::endl;
                layerDims[3] = static_cast<unsigned long>(layerDims[2] * imageAspectRatio);
                face_net.cfgInputReshape(layerName, layerDims);
            }
        }
        auto head_net = cv::gapi::ie::Params<nets::HeadPose> {
            FLAGS_m_hp,                               // path to topology IR
            fileNameNoExt(FLAGS_m_hp) + ".bin",       // path to weights
            FLAGS_d_hp,                               // device specifier
        }.cfgOutputLayers({"angle_y_fc", "angle_p_fc", "angle_r_fc"});
        auto landmarks_net = cv::gapi::ie::Params<nets::Landmarks> {
            FLAGS_m_lm,                               // path to topology IR
            fileNameNoExt(FLAGS_m_lm) + ".bin",       // path to weights
            FLAGS_d_lm,                               // device specifier
        };
        auto gaze_net = cv::gapi::ie::Params<nets::Gaze> {
            FLAGS_m,                                  // path to topology IR
            fileNameNoExt(FLAGS_m) + ".bin",          // path to weights
            FLAGS_d,                                  // device specifier
        }.cfgInputLayers({"left_eye_image", "right_eye_image", "head_pose_angles"});
        auto eyes_net = cv::gapi::ie::Params<nets::Eyes> {
            FLAGS_m_es,                               // path to topology IR
            fileNameNoExt(FLAGS_m_es) + ".bin",       // path to weights
            FLAGS_d_es,                               // device specifier
        };

        /** Custom kernels **/
        auto kernels = custom::kernels();
        auto networks = cv::gapi::networks(face_net, head_net, landmarks_net, gaze_net, eyes_net);
        auto pipeline = graph.compileStreaming(cv::compile_args(networks, kernels));

        /** Output containers for results **/
        cv::Mat frame;
        std::vector<float> out_cofidence;
        std::vector<cv::Rect> out_faces, out_right_eyes, out_left_eyes;
        std::vector<cv::Point2f> out_right_midpoint, out_left_midpoint;
        std::vector<std::vector<cv::Point>> out_landmarks;
        std::vector<cv::Point3f> out_poses;
        std::vector<int> out_left_state, out_right_state;
        std::vector<cv::Point3f> out_gazes;

        /** ---------------- The execution part ---------------- **/
        pipeline.setSource<custom::CustomCapSource>(cap);
        ResultsMarker resultsMarker(false, false, false, true, true);
        int delay = 1;
        bool flipImage = false;
        std::string windowName = "Gaze estimation demo G-API";

        cv::Size graphSize{static_cast<int>(frame_size.width / 4), 60};
        Presenter presenter(FLAGS_u, frame_size.height - graphSize.height - 10, graphSize);

        /** Exponential averagers for times **/
        double smoothingFactor = 0.1;
        ExponentialAverager overallTimeAverager(smoothingFactor, 30.);
        auto tIterationBegins = cv::getTickCount();

        /** Save output result **/
        cv::VideoWriter videoWriter;
        if (!FLAGS_o.empty() && !videoWriter.open(FLAGS_o, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                                  cap->fps(), frame_size)) {
            throw std::runtime_error("Can't open video writer");
        }

        pipeline.start();
        while (pipeline.pull(cv::gout(frame,
                                      out_cofidence,
                                      out_faces,
                                      out_landmarks,
                                      out_poses,
                                      out_left_eyes,
                                      out_right_eyes,
                                      out_left_midpoint,
                                      out_right_midpoint,
                                      out_left_state,
                                      out_right_state,
                                      out_gazes))) {
            /** Results **/
            std::vector<FaceInferenceResults> inferenceResults;
            /** Pack results from graph for universal drawing **/
            for (size_t i = 0; i < out_faces.size(); ++i) {
                FaceInferenceResults inferenceResult;
                inferenceResult.faceDetectionConfidence = out_cofidence[i];
                inferenceResult.faceBoundingBox = out_faces[i];
                inferenceResult.faceLandmarks = out_landmarks[i];
                inferenceResult.headPoseAngles = out_poses[i];
                inferenceResult.leftEyeBoundingBox = out_left_eyes[i];
                inferenceResult.rightEyeBoundingBox = out_right_eyes[i];
                inferenceResult.leftEyeMidpoint = out_left_midpoint[i];
                inferenceResult.rightEyeMidpoint = out_right_midpoint[i];
                inferenceResult.leftEyeState = out_left_state[i];
                inferenceResult.rightEyeState = out_right_state[i];
                inferenceResult.gazeVector = out_gazes[i];
                inferenceResults.push_back(inferenceResult);
            }

            /** Measure FPS **/
            auto tIterationEnds = cv::getTickCount();
            double overallTime = (tIterationEnds - tIterationBegins) * 1000. / cv::getTickFrequency();
            overallTimeAverager.updateValue(overallTime);
            tIterationBegins = tIterationEnds;

            /** Print logs **/
            if (FLAGS_r) {
                for (auto& inferenceResult : inferenceResults) {
                    std::cout << inferenceResult << std::endl;
                }
            }

            /** Display system parameters **/
            presenter.drawGraphs(frame);

            /** Display the results **/
            for (auto const& inferenceResult : inferenceResults) {
                resultsMarker.mark(frame, inferenceResult);
            }

            /** FlipImage **/
            if (flipImage) {
                cv::flip(frame, frame, 1);
            }

            putTimingInfoOnFrame(frame, overallTimeAverager.getAveragedValue());
            if (videoWriter.isOpened()) {
                videoWriter.write(frame);
            }
            if (!FLAGS_no_show) {
                cv::imshow(windowName, frame);
                /** Controls the information being displayed while demo runs **/
                int key = cv::waitKey(delay);
                resultsMarker.toggle(key);

                /** Press 'Esc' to quit, 'f' to flip the video horizontally **/
                if (key == 27)
                    break;
                if (key == 'f')
                    flipImage = !flipImage;
                else
                    presenter.handleKey(key);
            }
        }

        std::cout << presenter.reportMeans() << '\n';
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }
    slog::info << "Execution successful" << slog::endl;
    return 0;
}
