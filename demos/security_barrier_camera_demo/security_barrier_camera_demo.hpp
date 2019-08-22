// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>

/// @brief message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief message for enabling input video
static const char video_message[] = "Required for video or image files input. Path to video or image files.";

/// @brief message for model argument
static const char detection_model_message[] = "Required. Path to the Vehicle and License Plate Detection model .xml file.";
static const char vehicle_attribs_model_message[] = "Optional. Path to the Vehicle Attributes model .xml file.";
static const char lpr_model_message[] = "Optional. Path to the License Plate Recognition model .xml file.";

/// @brief message for assigning vehicle detection inference to device
static const char target_device_message[] = "Optional. Specify the target device for Vehicle Detection "\
                                            "(the list of available devices is shown below). Default value is CPU. " \
                                            "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
                                            "The application looks for a suitable plugin for the specified device.";

/// @brief message for assigning vehicle attributes to device
static const char target_device_message_vehicle_attribs[] = "Optional. Specify the target device for Vehicle Attributes "\
                                                            "(the list of available devices is shown below). Default value is CPU. " \
                                                            "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
                                                            "The application looks for a suitable plugin for the specified device.";

/// @brief message for assigning LPR inference to device
static const char target_device_message_lpr[] = "Optional. Specify the target device for License Plate Recognition "\
                                                "(the list of available devices is shown below). Default value is CPU. " \
                                                "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. " \
                                                "The application looks for a suitable plugin for the specified device.";

/// @brief message for performance counters
static const char performance_counter_message[] = "Optional. Enables per-layer performance statistics.";

/// @brief message raw output flag
static const char raw_output_message[] = "Optional. Output inference results as raw values.";

/// @brief message for probability threshold argument
static const char thresh_output_message[] = "Optional. Probability threshold for vehicle and license plate detections.";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Required for GPU custom kernels. "\
"Absolute path to an .xml file with the kernels description.";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Required for CPU custom layers. " \
"Absolute path to a shared library with the kernels implementation.";

/// @brief message no show processed video
static const char no_show_processed_video[] = "Optional. Do not show processed video.";

/// @brief message resizable input flag
static const char input_resizable_message[] = "Optional. Enable resizable input with support of ROI crop and auto resize.";

/// @brief message for number of infer requests
static const char ninfer_request_message[] = "Optional. Number of infer requests. 0 sets the number of infer requests equal to the number of inputs.";

/// @brief message for number of camera inputs
static const char num_cameras[] = "Required for web camera input. Maximum number of processed camera inputs (web cameras).";

/// @brief message for FPGA device IDs
static const char fpga_device_ids_message[] = "Optional. Specify FPGA device IDs (0,1,n).";

/// @brief Message for looping video argument
static const char loop_video_output_message[] = "Optional. Enable playing video on a loop.";

/// @brief message for inputs queue size
static const char input_queue_size[] = "Optional. Number of allocated frames. It is a multiplier of the number of inputs.";

/// @brief message for enabling channel duplication
static const char ninputs_message[] = "Optional. Specify the number of channels generated from provided inputs (with -i and -nc keys). "\
"For example, if only one camera is provided, but -ni is set to 2, the demo will process frames as if they are captured from two cameras. "\
"0 sets the number of input channels equal to the number of provided inputs.";

/// @brief message for setting playing fps
static const char fps[] = "Optional. Set the playback speed not faster than the specified FPS. 0 removes the upper bound.";

/// @brief message for setting the number of threads in Worker
static const char worker_threads[] = "Optional. Set the number of threads including the main thread a Worker class will use.";

/// @brief Message for display resolution argument
static const char display_resolution_message[] = "Optional. Specify the maximum output window resolution.";

/// @brief Message for using tag scheduler
static const char use_tag_scheduler_message[] = "Required for HDDL plugin only. "
                                                "If not set, the performance on Intel(R) Movidius(TM) X VPUs will not be optimal. "
                                                "Running each network on a set of Intel(R) Movidius(TM) X VPUs with a specific tag. "
                                                "You must specify the number of VPUs for each network in the hddl_service.config file. "
                                                "Refer to the corresponding README file for more information.";

/// @brief message for #threads for CPU inference
static const char infer_num_threads_message[] = "Optional. Number of threads to use for inference on the CPU "
                                                "(including HETERO and MULTI cases).";

/// @brief message for #streams for CPU inference
static const char infer_num_streams_message[] = "Optional. Number of streams to use for inference on the CPU or/and GPU in throughput mode "
                                                "(for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)";

/// \brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// \brief Define parameter for input video files <br>
/// It is a optional parameter
DEFINE_string(i, "", video_message);

/// \brief Define parameter for vehicle detection  model file <br>
/// It is a required parameter
DEFINE_string(m, "", detection_model_message);

/// \brief Define parameter for vehicle attributes model file <br>
/// It is a required parameter
DEFINE_string(m_va, "", vehicle_attribs_model_message);

/// \brief Define parameter for vehicle detection  model file <br>
/// It is a required parameter
DEFINE_string(m_lpr, "", lpr_model_message);

/// \brief device the target device for vehicle detection infer on <br>
DEFINE_string(d, "CPU", target_device_message);

/// \brief device the target device for age gender detection on <br>
DEFINE_string(d_va, "CPU", target_device_message_vehicle_attribs);

/// \brief device the target device for head pose detection on <br>
DEFINE_string(d_lpr, "CPU", target_device_message_lpr);

/// \brief enable per-layer performance report <br>
DEFINE_bool(pc, false, performance_counter_message);

/// \brief Flag to output raw scoring results<br>
/// It is an optional parameter
DEFINE_bool(r, false, raw_output_message);

/// \brief Flag to output raw scoring results<br>
/// It is an optional parameter
DEFINE_double(t, 0.5, thresh_output_message);

/// @brief clDNN custom kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// \brief Flag to disable processed video showing<br>
/// It is an optional parameter
DEFINE_bool(no_show, false, no_show_processed_video);

/// \brief Enables resizable input<br>
/// It is an optional parameter
DEFINE_bool(auto_resize, false, input_resizable_message);

/// \brief Flag to specify number of infer requests<br>
/// It is an optional parameter
DEFINE_uint32(nireq, 0, ninfer_request_message);

/// \brief Flag to specify number of expected input channels<br>
/// It is an optional parameter
DEFINE_uint32(nc, 0, num_cameras);

/// \brief Flag to specify FPGA device IDs
/// It is an optional parameter
DEFINE_string(fpga_device_ids, "", fpga_device_ids_message);

/// \brief Define a flag to loop video<br>
/// It is an optional parameter
DEFINE_bool(loop_video, false, loop_video_output_message);

/// \brief Flag to specify number of allocated frames. It is a multiplyir of inputs number.<br>
/// It is an optional parameter
DEFINE_uint32(n_iqs, 3, input_queue_size);

/// \brief Flag to specify number of input channels. It will multiply channels by reusing provided ones if there is lack of inputs<br>
/// It is an optional parameter
DEFINE_uint32(ni, 0, ninputs_message);

/// \brief Define parameter for playing FPS <br>
/// It is a optional parameter
DEFINE_uint32(fps, 0, fps);

/// \brief Define parameter for the number of threads including the main theread a Worker will use<br>
/// It is a optional parameter
DEFINE_uint32(n_wt, 1, worker_threads);

/// \brief Flag to specify the maximum output window resolution<br>
/// It is an optional parameter
DEFINE_string(display_resolution, "1920x1080", display_resolution_message);

/// \brief Message for using tag scheduler<br>
/// It is a optional parameter
DEFINE_bool(tag, false, use_tag_scheduler_message);

/// @brief Number of threads to use for inference on the CPU in throughput mode (also affects Hetero cases)
DEFINE_uint32(nthreads, 0, infer_num_threads_message);

/// @brief Number of streams to use for inference on the CPU (also affects Hetero cases)
DEFINE_string(nstreams, "", infer_num_streams_message);

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
}
