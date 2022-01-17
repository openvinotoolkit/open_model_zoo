// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine noise_suppression_demo demo
* \file noise_suppression_demo/main.cpp
* \example noise_suppression_demo/main.cpp
*/
#include <climits>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>

#include "openvino/openvino.hpp"

#include "gflags/gflags.h"
#include "utils/common.hpp"
#include "utils/slog.hpp"

typedef std::chrono::steady_clock Time;

static const char help_message[] = "Print a usage message.";
static const char inp_wav_message[] = "Required. Path to a input WAV file.";
static const char out_wav_message[] = "Optional. Path to a output WAV file.";
static const char model_message[] = "Required. Path to an .xml file with a trained model.";
static const char target_device_message[] = "Optional. Specify the target device to infer on (the list of available "
                                            "devices is shown below). Default value is CPU. "
                                            "The demo will look for a suitable plugin for device specified.";
DEFINE_bool(h, false, help_message);
DEFINE_string(i, "", inp_wav_message);
DEFINE_string(o, "noise_suppression_demo_out.wav", out_wav_message);
DEFINE_string(m, "", model_message);
DEFINE_string(d, "CPU", target_device_message);

static void showUsage() {
    std::cout << std::endl;
    std::cout << "noise_suppression_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h           " << help_message << std::endl;
    std::cout << "    -i INPUT     " << inp_wav_message << std::endl;
    std::cout << "    -o OUTPUT    " << out_wav_message << std::endl;
    std::cout << "    -m MODEL     " << model_message << std::endl;
    std::cout << "    -d DEVICE    " << target_device_message << std::endl;
}

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // ---------------------------Parsing and validating input arguments--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }

    if (FLAGS_i.empty())
        throw std::logic_error("Parameter -i is not set");
    if (FLAGS_o.empty())
        throw std::logic_error("Parameter -o is not set");
    if (FLAGS_m.empty())
        throw std::logic_error("Parameter -m is not set");

    return true;
}

struct RiffWaveHeader {
    unsigned int riff_tag; // "RIFF" string
    int riff_length;       // Total length
    unsigned int wave_tag; // "WAVE"
    unsigned int fmt_tag;  // "fmt " string (note space after 't')
    int fmt_length;        // Remaining length
    short data_format;     // Data format tag, 1 = PCM
    short num_of_channels; // Number of channels in file
    int sampling_freq;     // Sampling frequency
    int bytes_per_sec;     // Average bytes/sec
    short block_align;     // Block align
    short bits_per_sample;
    unsigned int data_tag; // "data" string
    int data_length;       // Raw data length
};


const unsigned int fourcc(const char c[4]) {
    return (c[3] << 24) | (c[2] << 16) | (c[1] << 8) | (c[0]);
}

void read_wav(const std::string& file_name, RiffWaveHeader& wave_header, std::vector<int16_t>& wave) {
    std::ifstream inp_wave(file_name, std::ios::in|std::ios::binary);
    if(!inp_wave.is_open())
        throw std::logic_error("fail to open " + file_name);

    inp_wave.read((char*)&wave_header, sizeof(RiffWaveHeader));

    std::string error_msg = "";
    #define CHECK_IF(cond) if(cond){ error_msg = error_msg + #cond + ", "; }

    // make sure it is actually a RIFF file with WAVE
    CHECK_IF(wave_header.riff_tag != fourcc("RIFF"));
    CHECK_IF(wave_header.wave_tag != fourcc("WAVE"));
    CHECK_IF(wave_header.fmt_tag != fourcc("fmt "));
    // only PCM
    CHECK_IF(wave_header.data_format != 1);
    // only mono
    CHECK_IF(wave_header.num_of_channels != 1);
    // only 16 bit
    CHECK_IF(wave_header.bits_per_sample != 16);
    // only 16KHz
    CHECK_IF(wave_header.sampling_freq != 16000);
    // make sure that data chunk follows file header
    CHECK_IF(wave_header.data_tag != fourcc("data"));
    #undef CHECK_IF

    if (!error_msg.empty()) {
        throw std::logic_error(error_msg + "for '" + file_name + "' file.");
    }

    size_t wave_size = wave_header.data_length / sizeof(int16_t);
    wave.resize(wave_size);

    inp_wave.read((char*)&(wave.front()), wave_size * sizeof(int16_t));
}

void write_wav(const std::string& file_name, const RiffWaveHeader& wave_header, const std::vector<int16_t>& wave) {
    std::ofstream out_wave(file_name, std::ios::out|std::ios::binary);
    if(!out_wave.is_open())
        throw std::logic_error("fail to open " + file_name);

    out_wave.write((char*)&wave_header, sizeof(RiffWaveHeader));
    out_wave.write((char*)&(wave.front()), wave.size() * sizeof(int16_t));
}

int main(int argc, char* argv[]) {
    try {
        // ------------------------------ Parsing and validating of input arguments --------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return EXIT_FAILURE;
        }

        slog::info << ov::get_openvino_version() << slog::endl;

        // Loading Inference Engine
        ov::runtime::Core core;

        std::shared_ptr<ov::Model> model = core.read_model(FLAGS_m);
        slog::info << "model file: " << FLAGS_m << slog::endl;
        log_model_info(model);

        ov::OutputVector inputs = model->inputs();
        ov::OutputVector outputs = model->outputs();

        // get state names pairs (inp,out) and compute overall states size
        size_t state_size = 0;
        std::vector<std::pair<std::string, std::string>> state_names;
        for (size_t i = 0; i < inputs.size(); i++) {
            std::string inp_state_name = inputs[i].get_any_name();
            if (inp_state_name.find("inp_state_") == std::string::npos)
                continue;

            std::string out_state_name(inp_state_name);
            out_state_name.replace(0, 3, "out");

            // find corresponding output state
            auto scmp = [&](ov::Output<ov::Node> output) { return output.get_any_name() == out_state_name; };
            if (std::end(outputs) == std::find_if(outputs.begin(), outputs.end(), scmp))
                throw std::logic_error("model output state name not corresponf input state name");

            state_names.emplace_back(inp_state_name, out_state_name);

            ov::Shape shape = inputs[i].get_shape();
            size_t tensor_size = 1;
            for (size_t s : shape)
                tensor_size *= s;

            state_size += tensor_size;
        }

        if (state_size == 0)
            throw std::logic_error("no expected model state inputs found");

        std::cout << "State_param_num = " << state_size << " (" << state_size * 4e-6f << "Mb)" << std::endl;

        // sort names
        auto scmp = [](const std::pair<std::string, std::string>& a, const std::pair<std::string,std::string>& b) { return a.first < b.first; };
        std::sort(state_names.begin(), state_names.end(), scmp);

        ov::runtime::CompiledModel compiled_model = core.compile_model(model, FLAGS_d);
        log_compiled_model_info(compiled_model, FLAGS_m, FLAGS_d);

        ov::runtime::InferRequest infer_request = compiled_model.create_infer_request();

        // Prepare input
        // get size of network input (pacth_size)
        std::string input_name("input");
        ov::Shape inp_shape = model->input(input_name).get_shape();
        size_t patch_size = inp_shape[1];

        // read input wav file
        RiffWaveHeader wave_header;
        std::vector<int16_t> inp_wave_s16;
        read_wav(FLAGS_i, wave_header, inp_wave_s16);

        std::vector<int16_t> out_wave_s16;
        out_wave_s16.resize(inp_wave_s16.size());

        std::vector<float> inp_wave_fp32;
        std::vector<float> out_wave_fp32;
        // fp32 input wave will be expanded to be divisible by patch_size
        size_t iter = 1 + (inp_wave_s16.size() / patch_size);
        size_t inp_size = patch_size * iter;
        inp_wave_fp32.resize(inp_size, 0);
        out_wave_fp32.resize(inp_size, 0);

        // convert sint16_t  to float
        float scale = 1.0f / std::numeric_limits<int16_t>::max();
        for(size_t i = 0; i < inp_wave_s16.size(); ++i) {
            inp_wave_fp32[i] = (float)inp_wave_s16[i] * scale;
        }

        auto start_time = Time::now();
        for(size_t i = 0; i < iter; ++i) {
            ov::runtime::Tensor inputBlob(ov::element::f32, inp_shape, &inp_wave_fp32[i * patch_size]);
            infer_request.set_tensor(input_name, inputBlob);

            for (auto& state_name: state_names) {
                const std::string& inp_state_name = state_name.first;
                const std::string& out_state_name = state_name.second;

                if (i > 0) {
                    // set input state by coresponding output state from prev infer
                    ov::runtime::Tensor blob_ptr = infer_request.get_tensor(out_state_name);
                    infer_request.set_tensor(inp_state_name, blob_ptr);
                } else {
                    // first iteration. set input state to zero tensor.
                    ov::Shape state_shape = model->input(inp_state_name).get_shape();
                    ov::runtime::Tensor blob_ptr(ov::element::f32, state_shape);
                    memset(blob_ptr.data<float>(), 0, blob_ptr.get_byte_size());
                    infer_request.set_tensor(inp_state_name, blob_ptr);
                }
            }

            // make infer
            infer_request.infer();

            {
                // process output
                float* src = infer_request.get_tensor("output").data<float>();
                float* dst = &out_wave_fp32[i * patch_size];
                memcpy(dst, src, patch_size * sizeof(float));
            }
        } // for iter

        using ms = std::chrono::duration<double, std::ratio<1, 1000>>;
        double total_latency = std::chrono::duration_cast<ms>(Time::now() - start_time).count();
        slog::info << "Metrics report:" << slog::endl;
        slog::info << "\tLatency: " << std::fixed << std::setprecision(1) << total_latency << " ms" << slog::endl;
        slog::info << "\tSample length: " << std::fixed << std::setprecision(1) << patch_size * iter / 16.0f << " ms" << slog::endl;

        // convert fp32 to int16_t
        for(size_t i = 0; i < out_wave_s16.size(); ++i) {
            out_wave_s16[i] = (int16_t)(out_wave_fp32[i] * std::numeric_limits<int16_t>::max());
        }
        write_wav(FLAGS_o, wave_header, out_wave_s16);
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cerr << "Unknown/internal exception happened." << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
