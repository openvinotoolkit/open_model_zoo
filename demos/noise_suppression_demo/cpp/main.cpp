// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <iomanip>
#include <openvino/openvino.hpp>
#include <gflags/gflags.h>
#include <utils/common.hpp>
#include <utils/slog.hpp>

namespace {
constexpr char h_msg[] = "show this help message and exit";
DEFINE_bool(h, false, h_msg);

constexpr char m_msg[] = "path to an .xml file with a trained model";
DEFINE_string(m, "", m_msg);

constexpr char i_msg[] = "path to an input WAV file";
DEFINE_string(i, "", i_msg);

constexpr char d_msg[] = "specify a device to infer on (the list of available devices is shown below). Default is CPU";
DEFINE_string(d, "CPU", d_msg);

constexpr char o_msg[] = "path to an output WAV file. Default is noise_suppression_demo_out.wav";
DEFINE_string(o, "noise_suppression_demo_out.wav", o_msg);

void parse(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    if (FLAGS_h || 1 == argc) {
        std::cout <<   "\t[-h]               " << h_msg
                  << "\n\t[--help]           print help on all arguments"
                  << "\n\t -m <MODEL FILE>   " << m_msg
                  << "\n\t -i <WAV>          " << i_msg
                  << "\n\t[-d <DEVICE>]      " << d_msg
                  << "\n\t[-o <WAV>]         " << o_msg << '\n';
        showAvailableDevices();
        slog::info << ov::get_openvino_version() << slog::endl;
        exit(0);
    } if (FLAGS_m.empty()) {
        throw std::invalid_argument{"-m <MODEL FILE> can't be empty"};
    } if (FLAGS_i.empty()) {
        throw std::invalid_argument{"-i <WAV> can't be empty"};
    } if (FLAGS_o.empty()) {
        throw std::invalid_argument{"-o <WAV> can't be empty"};
    }
    slog::info << ov::get_openvino_version() << slog::endl;
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
        throw std::runtime_error("fail to open " + file_name);

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
    // make sure that data chunk follows file header
    CHECK_IF(wave_header.data_tag != fourcc("data"));
    #undef CHECK_IF

    if (!error_msg.empty()) {
        throw std::runtime_error(error_msg + "for '" + file_name + "' file.");
    }

    size_t wave_size = wave_header.data_length / sizeof(int16_t);
    wave.resize(wave_size);

    inp_wave.read((char*)&(wave.front()), wave_size * sizeof(int16_t));
}

void write_wav(const std::string& file_name, const RiffWaveHeader& wave_header, const std::vector<int16_t>& wave) {
    std::ofstream out_wave(file_name, std::ios::out|std::ios::binary);
    if(!out_wave.is_open())
        throw std::runtime_error("fail to open " + file_name);

    out_wave.write((char*)&wave_header, sizeof(RiffWaveHeader));
    out_wave.write((char*)&(wave.front()), wave.size() * sizeof(int16_t));
}
}  // namespace

int main(int argc, char* argv[]) {
    std::set_terminate(catcher);
    parse(argc, argv);
    ov::Core core;
    slog::info << "Reading model: " << FLAGS_m << slog::endl;
    std::shared_ptr<ov::Model> model = core.read_model(FLAGS_m);
    logBasicModelInfo(model);

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
        if (outputs.end() == std::find_if(outputs.begin(), outputs.end(), [&out_state_name](const ov::Output<ov::Node>& output) {
            return output.get_any_name() == out_state_name;
        }))
            throw std::runtime_error("model output state name does not correspond input state name");

        state_names.emplace_back(inp_state_name, out_state_name);

        ov::Shape shape = inputs[i].get_shape();
        size_t tensor_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

        state_size += tensor_size;
    }

    if (state_size == 0)
        throw std::runtime_error("no expected model state inputs found");

    slog::info << "State_param_num = " << state_size << " (" << std::setprecision(4) << state_size * sizeof(float) * 1e-6f << "Mb)" << slog::endl;

    ov::CompiledModel compiled_model = core.compile_model(model, FLAGS_d, {});
    logCompiledModelInfo(compiled_model, FLAGS_m, FLAGS_d);

    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // Prepare input
    // get size of network input (patch_size)
    std::string input_name("input");
    ov::Shape inp_shape = model->input(input_name).get_shape();
    size_t patch_size = inp_shape[1];

    // try to get delay output and freq output for model
    int delay = 0;
    int freq_model = 16000; // default sampling rate for model
    infer_request.infer();
    for (size_t i = 0; i < outputs.size(); i++) {
        std::string out_name = outputs[i].get_any_name();
        if (out_name == "delay") {
            delay = infer_request.get_tensor("delay").data<int>()[0];
        }
        if (out_name == "freq") {
            freq_model = infer_request.get_tensor("freq").data<int>()[0];
        }
    }
    slog::info << "\tDelay: " << delay << " samples" << slog::endl;
    slog::info << "\tFreq: " << freq_model << " Hz" << slog::endl;

    // read input wav file
    RiffWaveHeader wave_header;
    std::vector<int16_t> inp_wave_s16;
    read_wav(FLAGS_i, wave_header, inp_wave_s16);
    int freq_data = wave_header.sampling_freq;

    if (freq_data != freq_model) {
        slog::err << "Wav file " << FLAGS_i << " sampling rate " << freq_data << " does not match model sampling rate " << freq_model << slog::endl;
        throw std::runtime_error("data sampling rate does not match model sampling rate");
    }

    std::vector<int16_t> out_wave_s16;
    out_wave_s16.resize(inp_wave_s16.size());

    std::vector<float> inp_wave_fp32;
    std::vector<float> out_wave_fp32;

    // fp32 input wave will be expanded to be divisible by patch_size
    size_t iter = 1 + ((inp_wave_s16.size() + delay) / patch_size);
    size_t inp_size = patch_size * iter;
    inp_wave_fp32.resize(inp_size, 0);
    out_wave_fp32.resize(inp_size, 0);

    // convert sint16_t to float
    float scale = 1.0f / std::numeric_limits<int16_t>::max();
    for(size_t i = 0; i < inp_wave_s16.size(); ++i) {
        inp_wave_fp32[i] = (float)inp_wave_s16[i] * scale;
    }

    auto start_time = std::chrono::steady_clock::now();
    for(size_t i = 0; i < iter; ++i) {
        ov::Tensor input_tensor(ov::element::f32, inp_shape, &inp_wave_fp32[i * patch_size]);
        infer_request.set_tensor(input_name, input_tensor);

        for (auto& state_name: state_names) {
            const std::string& inp_state_name = state_name.first;
            const std::string& out_state_name = state_name.second;
            ov::Tensor state_tensor;
            if (i > 0) {
                // set input state by coresponding output state from prev infer
                state_tensor = infer_request.get_tensor(out_state_name);
            } else {
                // first iteration. set input state to zero tensor.
                ov::Shape state_shape = model->input(inp_state_name).get_shape();
                state_tensor = ov::Tensor(ov::element::f32, state_shape);
                std::memset(state_tensor.data<float>(), 0, state_tensor.get_byte_size());
            }
            infer_request.set_tensor(inp_state_name, state_tensor);
        }

        infer_request.infer();

        {
            // process output
            float* src = infer_request.get_tensor("output").data<float>();
            float* dst = &out_wave_fp32[i * patch_size];
            std::memcpy(dst, src, patch_size * sizeof(float));
        }
    } // for iter

    using ms = std::chrono::duration<double, std::ratio<1, 1000>>;
    double total_latency = std::chrono::duration_cast<ms>(std::chrono::steady_clock::now() - start_time).count();
    slog::info << "Metrics report:" << slog::endl;
    slog::info << "\tLatency: " << std::fixed << std::setprecision(1) << total_latency << " ms" << slog::endl;
    slog::info << "\tSample length: " << std::fixed << std::setprecision(1) << patch_size * iter / (freq_data * 1e-3) << " ms" << slog::endl;
    slog::info << "\tSampling freq: " << freq_data << " Hz" << slog::endl;

    // convert fp32 to int16_t and save to wav
    for(size_t i = 0; i < out_wave_s16.size(); ++i) {
        float v = out_wave_fp32[i+delay];
        v = clamp(v, -1.0f, +1.0f);
        out_wave_s16[i] = (int16_t)(v * std::numeric_limits<int16_t>::max());
    }
    write_wav(FLAGS_o, wave_header, out_wave_s16);
    return 0;
}
