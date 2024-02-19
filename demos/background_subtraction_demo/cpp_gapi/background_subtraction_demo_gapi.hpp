// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <iostream>

#include <gflags/gflags.h>

#include <utils/default_flags.hpp>

DEFINE_INPUT_FLAGS
DEFINE_OUTPUT_FLAGS

static const char help_message[] = "Print a usage message.";
static const char camera_resolution_message[] = "Optional. Set camera resolution in format WxH.";
static const char at_message[] = "Required. Architecture type: maskrcnn, background-matting.";
static const char model_message[] = "Required. Path to an .xml file with a trained model.";
static const char kernel_package_message[] =
    "Optional. G-API kernel package type: opencv, fluid (by default opencv is used).";
static const char device_message[] =
    "Optional. Target device for network (the list of available devices is shown below). "
    "The demo will look for a suitable plugin for a specified device. Default value is \"CPU\".";
static const char nireq_message[] = "Optional. Number of infer requests. If this option is omitted, number of infer "
                                    "requests is determined automatically.";
static const char num_threads_message[] = "Optional. Number of threads.";
static const char num_streams_message[] = "Optional. Number of streams to use for inference on the CPU or/and GPU in "
                                          "throughput mode (for HETERO and MULTI device cases use format "
                                          "<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)";
static const char no_show_message[] = "Optional. Don't show output.";
static const char blur_bgr_message[] = "Optional. Blur background.";
static const char target_bgr_message[] =
    "Optional. Background onto which to composite the output (by default to green field).";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";
static const char use_onevpl_message[] = "Optional. Use onevpl video decoding.";
static const char onevpl_params_message[] = "Optional. Parameters for onevpl video decoding. "
                                            "OneVPL source can be fine-grained by providing configuration parameters. "
                                            "Format: <prop name>:<value>,<prop name>:<value> "
                                            "Several important configuration parameters: "
                                            "-mfxImplDescription.mfxDecoderDescription.decoder.CodecID "
                                            "values: https://spec.oneapi.io/onevpl/2.7.0/API_ref/VPL_enums.html?highlight=mfx_codec_hevc#codecformatfourcc"
                                            "-mfxImplDescription.AccelerationMode "
                                            "values: https://spec.oneapi.io/onevpl/2.7.0/API_ref/VPL_disp_api_enum.html?highlight=d3d11#mfxaccelerationmode"
                                            "(see `MFXSetConfigFilterProperty` by https://spec.oneapi.io/versions/latest/elements/oneVPL/source/index.html)";
static const char onevpl_pool_size_message[] = "OneVPL source applies this parameter as preallocated frames pool size. 0 leaves frames pool size default for your system. "
                                               "This parameter doesn't have a god default value. It must be adjusted for specific execution (video, model, system, ...).";

DEFINE_bool(h, false, help_message);
DEFINE_string(res, "1280x720", camera_resolution_message);
DEFINE_string(at, "", at_message);
DEFINE_string(m, "", model_message);
DEFINE_string(kernel_package, "opencv", kernel_package_message);
DEFINE_string(d, "CPU", device_message);
DEFINE_uint32(nireq, 1, nireq_message);
DEFINE_uint32(nthreads, 0, num_threads_message);
DEFINE_string(nstreams, "", num_streams_message);
DEFINE_bool(no_show, false, no_show_message);
DEFINE_string(target_bgr, "", target_bgr_message);
DEFINE_uint32(blur_bgr, 0, blur_bgr_message);
DEFINE_string(u, "", utilization_monitors_message);
DEFINE_bool(use_onevpl, false, use_onevpl_message);
DEFINE_string(onevpl_params, "", onevpl_params_message);
DEFINE_uint32(onevpl_pool_size, 0, onevpl_pool_size_message);

/**
 * \brief This function shows a help message
 */

static void showUsage() {
    std::cout << std::endl;
    std::cout << "background_subtraction_demo_gapi [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                         " << help_message << std::endl;
    std::cout << "    -i                         " << input_message << std::endl;
    std::cout << "    -loop                      " << loop_message << std::endl;
    std::cout << "    -o \"<path>\"                " << output_message << std::endl;
    std::cout << "    -limit \"<num>\"             " << limit_message << std::endl;
    std::cout << "    -res \"<WxH>\"               " << camera_resolution_message << std::endl;
    std::cout << "    -at \"<type>\"               " << at_message << std::endl;
    std::cout << "    -m \"<path>\"                " << model_message << std::endl;
    std::cout << "    -kernel_package \"<string>\" " << kernel_package_message << std::endl;
    std::cout << "    -d \"<device>\"              " << device_message << std::endl;
    std::cout << "    -nireq \"<integer>\"         " << nireq_message << std::endl;
    std::cout << "    -nthreads \"<integer>\"      " << num_threads_message << std::endl;
    std::cout << "    -nstreams                  " << num_streams_message << std::endl;
    std::cout << "    -no_show                   " << no_show_message << std::endl;
    std::cout << "    -blur_bgr \"<integer>\"      " << blur_bgr_message << std::endl;
    std::cout << "    -target_bgr                " << target_bgr_message << std::endl;
    std::cout << "    -u                         " << utilization_monitors_message << std::endl;
    std::cout << "    -use_onevpl                " << use_onevpl_message << std::endl;
    std::cout << "    -onevpl_params             " << onevpl_params_message << std::endl;
    std::cout << "    -onevpl_pool_size          " << onevpl_pool_size_message << std::endl;
}
