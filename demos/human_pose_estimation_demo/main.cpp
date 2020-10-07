// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine Human Pose Estimation demo application
* \file human_pose_estimation_demo/main.cpp
* \example human_pose_estimation_demo/main.cpp
*/

#include <vector>
#include <chrono>

#include <inference_engine.hpp>

#include <monitors/presenter.h>
#include <samples/images_capture.h>
#include <samples/ocv_common.hpp>

#include "human_pose_estimation_demo.hpp"
#include "human_pose_estimator.hpp"
#include "render_human_pose.hpp"

using namespace InferenceEngine;
using namespace human_pose_estimation;

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    std::cout << "Parsing input parameters" << std::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "InferenceEngine: " << *GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return EXIT_SUCCESS;
        }

        HumanPoseEstimator estimator(FLAGS_m, FLAGS_d, FLAGS_pc);

        std::unique_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop);
        cv::Mat curr_frame = cap->read();
        if (!curr_frame.data) {
            throw std::runtime_error("Can't read an image from the input");
        }

        cv::Size graphSize{curr_frame.cols / 4, 60};
        Presenter presenter(FLAGS_u, curr_frame.rows - graphSize.height - 10, graphSize);

        estimator.reshape(curr_frame);  // Do not measure network reshape, if it happened

        std::cout << "To close the application, press 'CTRL+C' here";
        if (!FLAGS_no_show) {
            std::cout << " or switch to the output window and press ESC key" << std::endl;
            std::cout << "To pause execution, switch to the output window and press 'p' key" << std::endl;
        }
        std::cout << std::endl;

        int delay = 1;
        bool isAsyncMode = false; // execution is always started in SYNC mode
        bool isModeChanged = false; // set to true when execution mode is changed (SYNC<->ASYNC)
        bool blackBackground = FLAGS_black;

        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        auto total_t0 = std::chrono::high_resolution_clock::now();
        auto wallclock = std::chrono::high_resolution_clock::now();
        double render_time = 0;
        do {
            auto t0 = std::chrono::high_resolution_clock::now();
            //here is the first asynchronus point:
            //in the async mode we capture frame to populate the NEXT infer request
            //in the regular mode we capture frame to the current infer request

            cv::Mat next_frame = cap->read();
            if (isAsyncMode) {
                if (isModeChanged) {
                    estimator.frameToBlobCurr(curr_frame);
                }
                if (next_frame.data) {
                    estimator.frameToBlobNext(next_frame);
                }
            } else if (!isModeChanged) {
                estimator.frameToBlobCurr(curr_frame);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            t0 = std::chrono::high_resolution_clock::now();
            // Main sync point:
            // in the trully Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
            // in the regular mode we start the CURRENT request and immediately wait for it's completion
            if (isAsyncMode) {
                if (isModeChanged) {
                    estimator.startCurr();
                }
                if (next_frame.data) {
                    estimator.startNext();
                }
            } else if (!isModeChanged) {
                estimator.startCurr();
            }

            estimator.waitCurr();
            t1 = std::chrono::high_resolution_clock::now();
            ms detection = std::chrono::duration_cast<ms>(t1 - t0);
            t0 = std::chrono::high_resolution_clock::now();
            ms wall = std::chrono::duration_cast<ms>(t0 - wallclock);
            wallclock = t0;

            t0 = std::chrono::high_resolution_clock::now();

            if (blackBackground) {
                curr_frame = cv::Mat::zeros(curr_frame.size(), curr_frame.type());
            }
            std::ostringstream out;
            out << "OpenCV cap/render time: " << std::fixed << std::setprecision(1)
                << (decode_time + render_time) << " ms";
            cv::putText(curr_frame, out.str(), cv::Point2f(0, 25),
                        cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 255, 0));
            out.str("");
            out << "Wallclock time " << (isAsyncMode ? "(TRUE ASYNC):      " : "(SYNC, press Tab): ");
            out << std::fixed << std::setprecision(2) << wall.count()
                << " ms (" << 1000.0 / wall.count() << " fps)";
            cv::putText(curr_frame, out.str(), cv::Point2f(0, 50),
                        cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 0, 255));
            if (!isAsyncMode) {  // In the true async mode, there is no way to measure detection time directly
                out.str("");
                out << "Detection time  : " << std::fixed << std::setprecision(1) << detection.count()
                << " ms ("
                << 1000.0 / detection.count() << " fps)";
                cv::putText(curr_frame, out.str(), cv::Point2f(0, 75), cv::FONT_HERSHEY_TRIPLEX, 0.6,
                    cv::Scalar(255, 0, 0));
            }

            std::vector<HumanPose> poses = estimator.postprocessCurr();

            if (FLAGS_r) {
                if (!poses.empty()) {
                    std::time_t result = std::time(nullptr);
                    char timeString[sizeof("2020-01-01 00:00:00: ")];
                    std::strftime(timeString, sizeof(timeString), "%Y-%m-%d %H:%M:%S: ", std::localtime(&result));
                    std::cout << timeString;
                    }

                for (HumanPose const& pose : poses) {
                    std::stringstream rawPose;
                    rawPose << std::fixed << std::setprecision(0);
                    for (auto const& keypoint : pose.keypoints) {
                        rawPose << keypoint.x << "," << keypoint.y << " ";
                    }
                    rawPose << pose.score;
                    std::cout << rawPose.str() << std::endl;
                }
            }

            presenter.drawGraphs(curr_frame);
            renderHumanPose(poses, curr_frame);

            if (!FLAGS_no_show) {
                cv::imshow("Human Pose Estimation on " + FLAGS_d, curr_frame);
            }

            t1 = std::chrono::high_resolution_clock::now();
            render_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            if (isModeChanged) {
                isModeChanged = false;
            }

            // Final point:
            // in the truly Async mode we swap the NEXT and CURRENT requests for the next iteration
            curr_frame = std::move(next_frame);
            if (isAsyncMode) {
                estimator.swapRequest();
            }

            if (!FLAGS_no_show) {
                const int key = cv::waitKey(delay) & 255;
                if (key == 'p') {
                    delay = (delay == 0) ? 1 : 0;
                } else if (27 == key) { // Esc
                    break;
                } else if (9 == key) { // Tab
                    isAsyncMode = !isAsyncMode;
                    isModeChanged = true;
                } else if (32 == key) { // Space
                    blackBackground = !blackBackground;
                }
                presenter.handleKey(key);
            }
        } while (curr_frame.data);

        auto total_t1 = std::chrono::high_resolution_clock::now();
        ms total = std::chrono::duration_cast<ms>(total_t1 - total_t0);
        std::cout << "Total Inference time: " << total.count() << std::endl;
        std::cout << presenter.reportMeans() << '\n';
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Execution successful" << std::endl;
    return EXIT_SUCCESS;
}
