// Copyright (C) 2018-2022 Intel Corporation
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

#include <gflags/gflags.h>
#include <monitors/presenter.h>
#include <utils/images_capture.h>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>

#include <openvino/openvino.hpp>

#include "detection_base.hpp"

namespace {
constexpr char h_msg[] = "show the help message and exit";
DEFINE_bool(h, false, h_msg);

constexpr char m_msg[] = "path to the Person/Vehicle/Bike Detection Crossroad model (.xml) file";
DEFINE_string(m, "", m_msg);

constexpr char i_msg[] = "an input to process. The input must be a single image, a folder of images, video file or camera id. Default is 0";
DEFINE_string(i, "0", i_msg);

constexpr char auto_resize_msg[] = "enables resizable input with support of ROI crop & auto resize";
DEFINE_bool(auto_resize, false, auto_resize_msg);

constexpr char d_msg[] =
    "specify a device to infer on (the list of available devices is shown below). "
    "Use '-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. "
    "Use '-d MULTI:<comma-separated_devices_list>' format to specify MULTI plugin. "
    "Default is CPU";
DEFINE_string(d, "CPU", d_msg);

constexpr char dpa_msg[] =
    "specify the target device for Person Attributes Recognition. "
    "Use '-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. "
    "Default is CPU";
DEFINE_string(dpa, "CPU", dpa_msg);

constexpr char dpr_msg[] =
    "specify the target device for Person Reidentification Retail. "
    "Use '-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. "
    "Default is CPU";
DEFINE_string(dpr, "CPU", dpr_msg);

constexpr char lim_msg[] = "number of frames to store in output. If 0 is set, all frames are stored. Default is 1000";
DEFINE_uint32(lim, 1000, lim_msg);

constexpr char loop_msg[] = "enable reading the input in a loop";
DEFINE_bool(loop, false, loop_msg);

constexpr char mpa_msg[] = "path to the Person Attributes Recognition Crossroad model (.xml) file";
DEFINE_string(mpa, "", mpa_msg);

constexpr char mpr_msg[] = "path to the Person Reidentification Retail model (.xml) file";
DEFINE_string(mpr, "", mpr_msg);

constexpr char o_msg[] = "name of the output file(s) to save";
DEFINE_string(o, "", o_msg);

constexpr char person_label_msg[] = "the integer index of the objects' category corresponding to persons "
                          "(as it is returned from the detection network, may vary from one network to another). "
                          "Default is 1";
DEFINE_int32(person_label, 1, person_label_msg);

constexpr char r_msg[] = "output inference results as raw values";
DEFINE_bool(r, false, r_msg);

constexpr char show_msg[] = "(don't) show output";
DEFINE_bool(show, true, show_msg);

constexpr char t_msg[] = "probability threshold for detections. Default is 0.5";
DEFINE_double(t, 0.5, t_msg);

constexpr char tpr_msg[] = "cosine similarity threshold between two vectors for person reidentification. Default is 0.7";
DEFINE_double(tpr, 0.7, tpr_msg);

constexpr char u_msg[] = "resource utilization graphs. "
                         "c - average CPU load, d - load distribution over cores, m - memory usage, h - hide";
DEFINE_string(u, "", u_msg);

void parse(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    if (FLAGS_h || 1 == argc) {
        std::cout <<   "\t[ -h]                                         " << h_msg
                  << "\n\t[--help]                                           print help on all arguments"
                  << "\n\t  -m <MODEL FILE>                             " << m_msg
                  << "\n\t[ -i <INPUT>]                                 " << i_msg
                  << "\n\t[--auto_resize]                               " << auto_resize_msg
                  << "\n\t[ -d <DEVICE>]                                " << d_msg
                  << "\n\t[--dpa <DEVICE>]                              " << dpa_msg
                  << "\n\t[--dpr <DEVICE>]                              " << dpr_msg
                  << "\n\t[--lim <NUMBER>]                              " << lim_msg
                  << "\n\t[--loop]                                      " << loop_msg
                  << "\n\t[--mpa <MODEL FILE>]                          " << mpa_msg
                  << "\n\t[--mpr <MODEL FILE>]                          " << mpr_msg
                  << "\n\t[ -o <OUTPUT>]                                " << o_msg
                  << "\n\t[--person_label <NUMBER>]                     " << person_label_msg
                  << "\n\t[ -r]                                         " << r_msg
                  << "\n\t[--show] ([--noshow])                         " << show_msg
                  << "\n\t[ -t <NUMBER>]                                " << t_msg
                  << "\n\t[--tpr <NUMBER>]                              " << tpr_msg
                  << "\n\t[ -u <DEVICE>]                                " << u_msg
                  << "\n\tKey bindings:"
                     "\n\t\tQ, q, Esc - Quit"
                     "\n\t\tP, p, 0, spacebar - Pause"
                     "\n\t\tC - average CPU load, D - load distribution over cores, M - memory usage, H - hide\n";
        showAvailableDevices();
        exit(0);
    } if (FLAGS_i.empty()) {
        throw std::invalid_argument{"-i <INPUT> can't be empty"};
    } if (FLAGS_m.empty()) {
        throw std::invalid_argument{"-m <MODEL FILE> can't be empty"};
    }
    slog::info << ov::get_openvino_version() << slog::endl;
}
} // namespace

#include "detection_person.hpp"
#include "detection_person_attr.hpp"
#include "detection_person_reid.hpp"

int main(int argc, char* argv[]) {
    std::set_terminate(catcher);
    parse(argc, argv);
    PerformanceMetrics metrics;

    // This demo covers 3 certain topologies and cannot be generalized

    std::unique_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop);

    // 1. Load OpenVINO runtime

    ov::Core core;

    PersonDetection personDetection;
    PersonAttribsDetection personAttribs;
    PersonReIdentification personReId;

    // 2. Read IR models and load them to devices
    Load(personDetection).into(core, FLAGS_d);
    Load(personAttribs).into(core, FLAGS_dpa);
    Load(personReId).into(core, FLAGS_dpr);

    // 3. Do inference
    cv::Rect cropRoi; // cropped image coordinates
    ov::Tensor frameTensor;
    ov::Tensor roiTensor;
    cv::Mat person; // Mat object containing person data cropped by openCV

    // Start inference & calc performance
    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

    auto startTime = std::chrono::steady_clock::now();
    cv::Mat frame = cap->read();

    LazyVideoWriter videoWriter{FLAGS_o, cap->fps(), FLAGS_lim};
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

                        if (!tc_rect.empty())
                            resPersAttrAndColor.top_color = PersonAttribsDetection::GetAvgColor(person(tc_rect));
                        if (!bc_rect.empty())
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
        if (FLAGS_show) {
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

    return 0;
}
