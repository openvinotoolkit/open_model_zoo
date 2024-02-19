// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <chrono>
#include <algorithm>
#include <iterator>
#include <map>
#include <string>
#include <vector>
#include <set>

#include "openvino/openvino.hpp"

#include "gflags/gflags.h"
#include "monitors/presenter.h"
#include "utils/images_capture.h"
#include "utils/ocv_common.hpp"
#include "utils/performance_metrics.hpp"
#include "utils/slog.hpp"

#include "detection_base.hpp"
#include "detection_person.hpp"
#include "detection_person_attr.hpp"
#include "detection_person_reid.hpp"
#include "crossroad_camera_demo.hpp"

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // Parsing and validation of input args

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

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
        PerformanceMetrics metrics;

        // This demo covers 3 certain topologies and cannot be generalized
        // Parsing and validation of input args
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        std::unique_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop);

        // 1. Load OpenVINO runtime
        slog::info << ov::get_openvino_version() << slog::endl;

        ov::Core core;

        PersonDetection personDetection;
        PersonAttribsDetection personAttribs;
        PersonReIdentification personReId;

        // 2. Read IR models and load them to devices
        Load(personDetection).into(core, FLAGS_d);
        Load(personAttribs).into(core, FLAGS_d_pa);
        Load(personReId).into(core, FLAGS_d_reid);

        // 3. Do inference
        cv::Rect cropRoi; // cropped image coordinates
        ov::Tensor frameTensor;
        ov::Tensor roiTensor;
        cv::Mat person; // Mat object containing person data cropped by openCV

        // Start inference & calc performance
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

        auto startTime = std::chrono::steady_clock::now();
        cv::Mat frame = cap->read();

        LazyVideoWriter videoWriter{FLAGS_o, cap->fps(), FLAGS_limit};
        cv::Size graphSize{frame.cols / 4, 60};
        Presenter presenter(FLAGS_u, frame.rows - graphSize.height - 10, graphSize);

        bool shouldHandleTopBottomColors = personAttribs.HasTopBottomColor();

        do {
            if (FLAGS_auto_resize) {
                // just wrap Mat object with Tensor without additional memory allocation
                frameTensor = wrapMat2Tensor(frame);
                personDetection.setRoiTensor(frameTensor);
            } else {
                // resize Mat and copy data into OpenVINO allocated Tensor
                personDetection.enqueue(frame);
            }
            // Run Person detection inference
            auto t0 = std::chrono::high_resolution_clock::now();
            personDetection.submitRequest();
            personDetection.wait();
            auto t1 = std::chrono::high_resolution_clock::now();
            ms detection = std::chrono::duration_cast<ms>(t1 - t0);
            // parse inference results internally (e.g. apply a threshold, etc)
            personDetection.fetchResults();

            // Process the results down to the pipeline
            ms personAttribsNetworkTime(0), personReIdNetworktime(0);
            int personAttribsInferred = 0,  personReIdInferred = 0;
            for (PersonDetection::Result& result : personDetection.results) {
                if (result.label == FLAGS_person_label) {
                    // person
                    if (FLAGS_auto_resize) {
                        cropRoi.x = (result.location.x < 0) ? 0 : result.location.x;
                        cropRoi.y = (result.location.y < 0) ? 0 : result.location.y;
                        cropRoi.width = std::min(result.location.width, frame.cols - cropRoi.x);
                        cropRoi.height = std::min(result.location.height, frame.rows - cropRoi.y);
                        ov::Coordinate p00({ 0, (size_t)cropRoi.y, (size_t)cropRoi.x, 0 });
                        ov::Coordinate p01({ 1, (size_t)(cropRoi.y + cropRoi.height), (size_t)(cropRoi.x + cropRoi.width), 3 });
                        roiTensor = ov::Tensor(frameTensor, p00, p01);
                    } else {
                        // To crop ROI manually and allocate required memory (cv::Mat) again
                        auto clippedRect = result.location & cv::Rect(0, 0, frame.cols, frame.rows);
                        person = frame(clippedRect);
                    }

                    PersonAttribsDetection::AttributesAndColorPoints resPersAttrAndColor;
                    if (personAttribs.enabled()) {
                        // Run Person Attributes Recognition
                        if (FLAGS_auto_resize) {
                            personAttribs.setRoiTensor(roiTensor);
                        } else {
                            personAttribs.enqueue(person);
                        }

                        t0 = std::chrono::high_resolution_clock::now();
                        personAttribs.submitRequest();
                        personAttribs.wait();
                        t1 = std::chrono::high_resolution_clock::now();
                        personAttribsNetworkTime += std::chrono::duration_cast<ms>(t1 - t0);
                        personAttribsInferred++;
                        // Process outputs

                        resPersAttrAndColor = personAttribs.GetPersonAttributes();

                        if (shouldHandleTopBottomColors) {
                            cv::Point top_color_p;
                            cv::Point bottom_color_p;

                            top_color_p.x = static_cast<int>(resPersAttrAndColor.top_color_point.x) * person.cols;
                            top_color_p.y = static_cast<int>(resPersAttrAndColor.top_color_point.y) * person.rows;

                            bottom_color_p.x = static_cast<int>(resPersAttrAndColor.bottom_color_point.x) * person.cols;
                            bottom_color_p.y = static_cast<int>(resPersAttrAndColor.bottom_color_point.y) * person.rows;


                            cv::Rect person_rect(0, 0, person.cols, person.rows);

                            // Define area around top color's location
                            cv::Rect tc_rect;
                            tc_rect.x = top_color_p.x - person.cols / 6;
                            tc_rect.y = top_color_p.y - person.rows / 10;
                            tc_rect.height = 2 * person.rows / 8;
                            tc_rect.width = 2 * person.cols / 6;

                            tc_rect = tc_rect & person_rect;

                            // Define area around bottom color's location
                            cv::Rect bc_rect;
                            bc_rect.x = bottom_color_p.x - person.cols / 6;
                            bc_rect.y = bottom_color_p.y - person.rows / 10;
                            bc_rect.height =  2 * person.rows / 8;
                            bc_rect.width = 2 * person.cols / 6;

                            bc_rect = bc_rect & person_rect;

                            if (!isRectEmpty(tc_rect))
                                resPersAttrAndColor.top_color = PersonAttribsDetection::GetAvgColor(person(tc_rect));
                            if (!isRectEmpty(bc_rect))
                                resPersAttrAndColor.bottom_color = PersonAttribsDetection::GetAvgColor(person(bc_rect));
                        }
                    }

                    std::string resPersReid = "";
                    if (personReId.enabled()) {
                        // Run Person Reidentification
                        if (FLAGS_auto_resize) {
                            personReId.setRoiTensor(roiTensor);
                        } else {
                            personReId.enqueue(person);
                        }

                        t0 = std::chrono::high_resolution_clock::now();
                        personReId.submitRequest();
                        personReId.wait();
                        t1 = std::chrono::high_resolution_clock::now();

                        personReIdNetworktime += std::chrono::duration_cast<ms>(t1 - t0);
                        personReIdInferred++;

                        auto reIdVector = personReId.getReidVec();

                        // Check cosine similarity with all previously detected persons.
                        //   If it's new person it is added to the global Reid vector and
                        //   new global ID is assigned to the person. Otherwise, ID of
                        //   matched person is assigned to it.
                        auto foundId = personReId.findMatchingPerson(reIdVector);
                        resPersReid = "REID: " + std::to_string(foundId);
                    }

                    // Process outputs
                    if (!resPersAttrAndColor.attributes_strings.empty()) {
                        cv::Rect image_area(0, 0, frame.cols, frame.rows);
                        cv::Rect tc_label(result.location.x + result.location.width, result.location.y,
                                          result.location.width / 4, result.location.height / 2);
                        cv::Rect bc_label(result.location.x + result.location.width, result.location.y + result.location.height / 2,
                                            result.location.width / 4, result.location.height / 2);

                        if (shouldHandleTopBottomColors) {
                            frame(tc_label & image_area) = resPersAttrAndColor.top_color;
                            frame(bc_label & image_area) = resPersAttrAndColor.bottom_color;
                        }

                        for (size_t i = 0; i < resPersAttrAndColor.attributes_strings.size(); ++i) {
                            cv::Scalar color;
                            if (resPersAttrAndColor.attributes_indicators[i]) {
                                color = cv::Scalar(0, 200, 0); // has attribute
                            } else {
                                color = cv::Scalar(0, 0, 255); // doesn't have attribute
                            }
                            putHighlightedText(frame,
                                    resPersAttrAndColor.attributes_strings[i],
                                    cv::Point2f(static_cast<float>(result.location.x + 5 * result.location.width / 4),
                                                static_cast<float>(result.location.y + 15 + 15 * i)),
                                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                                    0.5,
                                    color, 1);
                        }

                        if (FLAGS_r) {
                            std::string output_attribute_string;
                            for (size_t i = 0; i < resPersAttrAndColor.attributes_strings.size(); ++i)
                                if (resPersAttrAndColor.attributes_indicators[i])
                                    output_attribute_string += resPersAttrAndColor.attributes_strings[i] + ",";
                            slog::debug << "Person Attributes results: " << output_attribute_string << slog::endl;
                            if (shouldHandleTopBottomColors) {
                                slog::debug << "Person top color: " << resPersAttrAndColor.top_color << slog::endl;
                                slog::debug << "Person bottom color: " << resPersAttrAndColor.bottom_color << slog::endl;
                            }
                        }
                    }
                    if (!resPersReid.empty()) {
                        putHighlightedText(frame,
                                    resPersReid,
                                    cv::Point2f(static_cast<float>(result.location.x), static_cast<float>(result.location.y + 30)),
                                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                                    0.55,
                                    cv::Scalar(250, 10, 10), 1);

                        if (FLAGS_r) {
                            slog::debug << "Person Re-Identification results: " << resPersReid << slog::endl;
                        }
                    }
                    cv::rectangle(frame, result.location, cv::Scalar(0, 255, 0), 1);
                }
            }

            presenter.drawGraphs(frame);
            metrics.update(startTime);

            // Execution statistics
            std::ostringstream out;
            out << "Detection time : " << std::fixed << std::setprecision(2) << detection.count()
                << " ms (" << 1000.f / detection.count() << " fps)";

            putHighlightedText(frame, out.str(), cv::Point2f(0, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, { 200, 10, 10 }, 2);

            if (personDetection.results.size()) {
                if (personAttribs.enabled() && personAttribsInferred) {
                    float average_time = static_cast<float>(personAttribsNetworkTime.count() / personAttribsInferred);
                    out.str("");
                    out << "Attributes Recognition time: " << std::fixed << std::setprecision(2) << average_time
                        << " ms (" << 1000.f / average_time << " fps)";
                    putHighlightedText(frame, out.str(), cv::Point2f(0, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, { 200, 10, 10 }, 2);
                    if (FLAGS_r) {
                        slog::debug << out.str() << slog::endl;
                    }
                }
                if (personReId.enabled() && personReIdInferred) {
                    float average_time = static_cast<float>(personReIdNetworktime.count() / personReIdInferred);
                    out.str("");
                    out << "Re-Identification time: " << std::fixed << std::setprecision(2) << average_time
                        << " ms (" << 1000.f / average_time << " fps)";
                    putHighlightedText(frame, out.str(), cv::Point2f(0, 60), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, { 200, 10, 10 }, 2);
                    if (FLAGS_r) {
                        slog::debug << out.str() << slog::endl;
                    }
                }
            }
            videoWriter.write(frame);
            if (!FLAGS_no_show) {
                cv::imshow("Detection results", frame);
                const int key = cv::waitKey(1);
                if (27 == key) // Esc
                    break;
                presenter.handleKey(key);
            }
            startTime = std::chrono::steady_clock::now();

            // get next frame
            frame = cap->read();
        } while (frame.data);

        slog::info << "Metrics report:" << slog::endl;
        metrics.logTotal();
        slog::info << presenter.reportMeans() << slog::endl;
    }
    catch (const std::exception& error) {
        slog ::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    return 0;
}
