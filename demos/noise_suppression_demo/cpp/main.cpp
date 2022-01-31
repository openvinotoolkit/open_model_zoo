// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utils/common.hpp>
#include <utils/slog.hpp>

#include <gflags/gflags.h>
#include <openvino/openvino.hpp>

using namespace std;

namespace {
constexpr char h_msg[] = "show the [H]elp message and exit";
DEFINE_bool(h, false, h_msg);

constexpr char m_msg[] = "path to an .xml file with a trained [M]odel";
DEFINE_string(m, "", m_msg);

constexpr char i_msg[] = "path to an [I]nput 16kHz WAV file";
DEFINE_string(i, "", i_msg);

constexpr char d_msg[] = "specify a [D]evice to infer on (the list of available devices is shown below)";
DEFINE_string(d, "CPU", d_msg);

constexpr char o_msg[] = "path to an [O]utput WAV file";
DEFINE_string(o, "noise_suppression_demo_out.wav", o_msg);

void parse(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    slog::info << ov::get_openvino_version() << slog::endl;
    if (FLAGS_h || 1 == argc) {
        cout <<   "\t[-h]                                  " << h_msg
             << "\n\t[--help]                              print [HELP] on all arguments"
             << "\n\t -m <[M]odel file>                    " << m_msg
             << "\n\t -i  <[I]nput file>                   " << i_msg
             << "\n\t[-d] <CPU>                            " << d_msg
             << "\n\t[-o] <noise_suppression_demo_out.wav> " << o_msg;
        showAvailableDevices();
        exit(0);
    } if (FLAGS_m.empty()) {
        throw invalid_argument{"-m <[M]odel file> can't be empty"};
    } if (FLAGS_i.empty()) {
        throw invalid_argument{"-i <[I]nput file> can't be empty"};
    } if (FLAGS_o.empty()) {
        throw invalid_argument{"-o <[O]utput file> can't be empty"};
    }
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

void read_wav(const string& file_name, RiffWaveHeader& wave_header, vector<int16_t>& wave) {
    ifstream inp_wave(file_name, ios::in|ios::binary);
    if(!inp_wave.is_open())
        throw logic_error("fail to open " + file_name);

    inp_wave.read((char*)&wave_header, sizeof(RiffWaveHeader));

    string error_msg = "";
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
        throw logic_error(error_msg + "for '" + file_name + "' file.");
    }

    size_t wave_size = wave_header.data_length / sizeof(int16_t);
    wave.resize(wave_size);

    inp_wave.read((char*)&(wave.front()), wave_size * sizeof(int16_t));
}

void write_wav(const string& file_name, const RiffWaveHeader& wave_header, const vector<int16_t>& wave) {
    ofstream out_wave(file_name, ios::out|ios::binary);
    if(!out_wave.is_open())
        throw logic_error("fail to open " + file_name);

    out_wave.write((char*)&wave_header, sizeof(RiffWaveHeader));
    out_wave.write((char*)&(wave.front()), wave.size() * sizeof(int16_t));
}
}  // namespace

int main(int argc, char* argv[]) {
    set_terminate(catcher);
    parse(argc, argv);

    ov::Core core;
    slog::info << "Reading model: " << FLAGS_m << slog::endl;
    shared_ptr<ov::Model> model = core.read_model(FLAGS_m);
    logBasicModelInfo(model);

    ov::OutputVector inputs = model->inputs();
    ov::OutputVector outputs = model->outputs();

    // get state names pairs (inp,out) and compute overall states size
    size_t state_size = 0;
    vector<pair<string, string>> state_names;
    for (size_t i = 0; i < inputs.size(); i++) {
        string inp_state_name = inputs[i].get_any_name();
        if (inp_state_name.find("inp_state_") == string::npos)
            continue;

        string out_state_name(inp_state_name);
        out_state_name.replace(0, 3, "out");

        // find corresponding output state
        auto name_equal = [&](ov::Output<ov::Node> output) { return output.get_any_name() == out_state_name; };
        if (end(outputs) == find_if(outputs.begin(), outputs.end(), name_equal))
            throw logic_error("model output state name does not correspond input state name");

        state_names.emplace_back(inp_state_name, out_state_name);

        ov::Shape shape = inputs[i].get_shape();
        size_t tensor_size = accumulate(shape.begin(), shape.end(), 1, multiplies<size_t>());

        state_size += tensor_size;
    }

    if (state_size == 0)
        throw logic_error("no expected model state inputs found");

    slog::info << "State_param_num = " << state_size << " (" << setprecision(4) << state_size * sizeof(float) * 1e-6f << "Mb)" << slog::endl;

    ov::CompiledModel compiled_model = core.compile_model(model, FLAGS_d, {});
    logCompiledModelInfo(compiled_model, FLAGS_m, FLAGS_d);

    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // Prepare input
    // get size of network input (patch_size)
    string input_name("input");
    ov::Shape inp_shape = model->input(input_name).get_shape();
    size_t patch_size = inp_shape[1];

    // read input wav file
    RiffWaveHeader wave_header;
    vector<int16_t> inp_wave_s16;
    read_wav(FLAGS_i, wave_header, inp_wave_s16);

    vector<int16_t> out_wave_s16;
    out_wave_s16.resize(inp_wave_s16.size());

    vector<float> inp_wave_fp32;
    vector<float> out_wave_fp32;
    // fp32 input wave will be expanded to be divisible by patch_size
    size_t iter = 1 + (inp_wave_s16.size() / patch_size);
    size_t inp_size = patch_size * iter;
    inp_wave_fp32.resize(inp_size, 0);
    out_wave_fp32.resize(inp_size, 0);

    // convert sint16_t to float
    float scale = 1.0f / numeric_limits<int16_t>::max();
    for(size_t i = 0; i < inp_wave_s16.size(); ++i) {
        inp_wave_fp32[i] = (float)inp_wave_s16[i] * scale;
    }

    auto start_time = chrono::steady_clock::now();
    for(size_t i = 0; i < iter; ++i) {
        ov::Tensor input_tensor(ov::element::f32, inp_shape, &inp_wave_fp32[i * patch_size]);
        infer_request.set_tensor(input_name, input_tensor);

        for (auto& state_name: state_names) {
            const string& inp_state_name = state_name.first;
            const string& out_state_name = state_name.second;
            ov::Tensor state_tensor;
            if (i > 0) {
                // set input state by coresponding output state from prev infer
                state_tensor = infer_request.get_tensor(out_state_name);
            } else {
                // first iteration. set input state to zero tensor.
                ov::Shape state_shape = model->input(inp_state_name).get_shape();
                state_tensor = ov::Tensor(ov::element::f32, state_shape);
                memset(state_tensor.data<float>(), 0, state_tensor.get_byte_size());
            }
            infer_request.set_tensor(inp_state_name, state_tensor);
        }

        infer_request.infer();

        {
            // process output
            float* src = infer_request.get_tensor("output").data<float>();
            float* dst = &out_wave_fp32[i * patch_size];
            memcpy(dst, src, patch_size * sizeof(float));
        }
    } // for iter

    using ms = chrono::duration<double, ratio<1, 1000>>;
    double total_latency = chrono::duration_cast<ms>(chrono::steady_clock::now() - start_time).count();
    slog::info << "Metrics report:" << slog::endl;
    slog::info << "\tLatency: " << fixed << setprecision(1) << total_latency << " ms" << slog::endl;
    slog::info << "\tSample length: " << fixed << setprecision(1) << patch_size * iter / 16.0f << " ms" << slog::endl;

    // convert fp32 to int16_t
    for(size_t i = 0; i < out_wave_s16.size(); ++i) {
        out_wave_s16[i] = (int16_t)(out_wave_fp32[i] * numeric_limits<int16_t>::max());
    }
    write_wav(FLAGS_o, wave_header, out_wave_s16);
    return 0;
}
