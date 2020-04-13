// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>

static const char help_message[] = "Print a usage message.";
static const char video_message[] = "Required for video or image files input. Path to video or image files.";
static const char detection_model_message[] = "Required. Path to the Vehicle and License Plate Detection model .xml file.";
static const char vehicle_attribs_model_message[] = "Optional. Path to the Vehicle Attributes model .xml file.";
static const char lpr_model_message[] = "Optional. Path to the License Plate Recognition model .xml file.";
static const char target_device_message[] = "Optional. Specify the target device for Vehicle Detection "
                                            "(the list of available devices is shown below). Default value is CPU. "
                                            "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                            "The application looks for a suitable plugin for the specified device.";
static const char target_device_message_vehicle_attribs[] = "Optional. Specify the target device for Vehicle Attributes "
                                                            "(the list of available devices is shown below). Default value is CPU. "
                                                            "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                                            "The application looks for a suitable plugin for the specified device.";
static const char target_device_message_lpr[] = "Optional. Specify the target device for License Plate Recognition "
                                                "(the list of available devices is shown below). Default value is CPU. "
                                                "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
                                                "The application looks for a suitable plugin for the specified device.";
static const char performance_counter_message[] = "Optional. Enables per-layer performance statistics.";
static const char raw_output_message[] = "Optional. Output inference results as raw values.";
static const char thresh_output_message[] = "Optional. Probability threshold for vehicle and license plate detections.";
static const char custom_cldnn_message[] = "Required for GPU custom kernels. "
                                           "Absolute path to an .xml file with the kernels description.";
static const char custom_cpu_library_message[] = "Required for CPU custom layers. "
                                                 "Absolute path to a shared library with the kernels implementation.";
static const char no_show_processed_video[] = "Optional. Do not show processed video.";
static const char input_resizable_message[] = "Optional. Enable resizable input with support of ROI crop and auto resize.";
static const char ninfer_request_message[] = "Optional. Number of infer requests. 0 sets the number of infer requests equal to the number of inputs.";
static const char num_cameras[] = "Required for web camera input. Maximum number of processed camera inputs (web cameras).";
static const char fpga_device_ids_message[] = "Optional. Specify FPGA device IDs (0,1,n).";
static const char loop_video_output_message[] = "Optional. Enable playing video on a loop.";
static const char input_queue_size[] = "Optional. Number of allocated frames. It is a multiplier of the number of inputs.";
static const char ninputs_message[] = "Optional. Specify the number of channels generated from provided inputs (with -i and -nc keys). "
                                      "For example, if only one camera is provided, but -ni is set to 2, the demo will process frames as if they are captured from two cameras. "
                                      "0 sets the number of input channels equal to the number of provided inputs.";
static const char fps[] = "Optional. Set the playback speed not faster than the specified FPS. 0 removes the upper bound.";
static const char worker_threads[] = "Optional. Set the number of threads including the main thread a Worker class will use.";
static const char display_resolution_message[] = "Optional. Specify the maximum output window resolution.";
static const char use_tag_scheduler_message[] = "Required for HDDL plugin only. "
                                                "If not set, the performance on Intel(R) Movidius(TM) X VPUs will not be optimal. "
                                                "Running each network on a set of Intel(R) Movidius(TM) X VPUs with a specific tag. "
                                                "You must specify the number of VPUs for each network in the hddl_service.config file. "
                                                "Refer to the corresponding README file for more information.";
static const char infer_num_threads_message[] = "Optional. Number of threads to use for inference on the CPU "
                                                "(including HETERO and MULTI cases).";
static const char infer_num_streams_message[] = "Optional. Number of streams to use for inference on the CPU or/and GPU in throughput mode "
                                                "(for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";

DEFINE_bool(h, false, help_message);
DEFINE_string(i, "", video_message);
DEFINE_string(m, "", detection_model_message);
DEFINE_string(m_va, "", vehicle_attribs_model_message);
DEFINE_string(m_lpr, "", lpr_model_message);
DEFINE_string(d, "CPU", target_device_message);
DEFINE_string(d_va, "CPU", target_device_message_vehicle_attribs);
DEFINE_string(d_lpr, "CPU", target_device_message_lpr);
DEFINE_bool(pc, false, performance_counter_message);
DEFINE_bool(r, false, raw_output_message);
DEFINE_double(t, 0.5, thresh_output_message);
DEFINE_string(c, "", custom_cldnn_message);
DEFINE_string(l, "", custom_cpu_library_message);
DEFINE_bool(no_show, false, no_show_processed_video);
DEFINE_bool(auto_resize, false, input_resizable_message);
DEFINE_uint32(nireq, 0, ninfer_request_message);
DEFINE_uint32(nc, 0, num_cameras);
DEFINE_string(fpga_device_ids, "", fpga_device_ids_message);
DEFINE_bool(loop_video, false, loop_video_output_message);
DEFINE_uint32(n_iqs, 3, input_queue_size);
DEFINE_uint32(ni, 0, ninputs_message);
DEFINE_uint32(fps, 0, fps);
DEFINE_uint32(n_wt, 1, worker_threads);
DEFINE_string(display_resolution, "1920x1080", display_resolution_message);
DEFINE_bool(tag, false, use_tag_scheduler_message);
DEFINE_uint32(nthreads, 0, infer_num_threads_message);
DEFINE_string(nstreams, "", infer_num_streams_message);
DEFINE_string(u, "", utilization_monitors_message);

/**
* \brief This function show a help message
*/
void showUsage() {
    std::cout << std::endl;
    std::cout << "interactive_vehicle_detection [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                         " << help_message << std::endl;
    std::cout << "    -i \"<path1>\" \"<path2>\"     " << video_message << std::endl;
    std::cout << "    -m \"<path>\"                " << detection_model_message << std::endl;
    std::cout << "    -m_va \"<path>\"             " << vehicle_attribs_model_message << std::endl;
    std::cout << "    -m_lpr \"<path>\"            " << lpr_model_message << std::endl;
    std::cout << "      -l \"<absolute_path>\"     " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"     " << custom_cldnn_message << std::endl;
    std::cout << "    -d \"<device>\"              " << target_device_message << std::endl;
    std::cout << "    -d_va \"<device>\"           " << target_device_message_vehicle_attribs << std::endl;
    std::cout << "    -d_lpr \"<device>\"          " << target_device_message_lpr << std::endl;
    std::cout << "    -pc                        " << performance_counter_message << std::endl;
    std::cout << "    -r                         " << raw_output_message << std::endl;
    std::cout << "    -t                         " << thresh_output_message << std::endl;
    std::cout << "    -no_show                   " << no_show_processed_video << std::endl;
    std::cout << "    -auto_resize               " << input_resizable_message << std::endl;
    std::cout << "    -nireq                     " << ninfer_request_message << std::endl;
    std::cout << "    -nc                        " << num_cameras << std::endl;
    std::cout << "    -fpga_device_ids           " << fpga_device_ids_message << std::endl;
    std::cout << "    -loop_video                " << loop_video_output_message << std::endl;
    std::cout << "    -n_iqs                     " << input_queue_size << std::endl;
    std::cout << "    -ni                        " << ninputs_message << std::endl;
    std::cout << "    -fps                       " << fps << std::endl;
    std::cout << "    -n_wt                      " << worker_threads << std::endl;
    std::cout << "    -display_resolution        " << display_resolution_message << std::endl;
    std::cout << "    -tag                       " << use_tag_scheduler_message << std::endl;
    std::cout << "    -nstreams \"<integer>\"      " << infer_num_streams_message << std::endl;
    std::cout << "    -nthreads \"<integer>\"      " << infer_num_threads_message << std::endl;
    std::cout << "    -u                         " << utilization_monitors_message << std::endl;
}
