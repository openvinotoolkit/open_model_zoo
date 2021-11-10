// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine noise_suppression_demo demo
* \file noise_suppression_demo/main.cpp
* \example noise_suppression_demo/main.cpp
*/
#include <gflags/gflags.h>
#include <climits>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

#include <inference_engine.hpp>

typedef std::chrono::steady_clock Time;
typedef std::chrono::nanoseconds ns;

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

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
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
    unsigned int riff_tag;       /* "RIFF" string */
    int riff_length;             /* Total length */
    unsigned int wave_tag;       /* "WAVE" */
    unsigned int fmt_tag;        /* "fmt " string (note space after 't') */
    int fmt_length;              /* Remaining length */
    short data_format;           /* Data format tag, 1 = PCM */
    short num_of_channels;       /* Number of channels in file */
    int sampling_freq;           /* Sampling frequency */
    int bytes_per_sec;           /* Average bytes/sec */
    short block_align;           /* Block align */
    short bits_per_sample;
    unsigned int data_tag;       /* "data" string */
    int data_length;             /* Raw data length */
};


void read_wav(const std::string& file_name, std::vector<int16_t>& wave, RiffWaveHeader& wave_header){
    FILE *inp_wave = fopen(file_name.c_str(), "rb");
    if (inp_wave == NULL) {
        throw std::logic_error("fail to read " + file_name);
    }

    if (1 != fread(&wave_header, sizeof(RiffWaveHeader), 1, inp_wave)) {
        fclose(inp_wave);
        throw std::logic_error("fail to read header for " + file_name);
    }

    bool read_ok = true;
    // make sure it is actually a RIFF file
    if (0 != memcmp(&wave_header.riff_tag, "RIFF", 4)) {
        std::cerr << "riff_tag != 'RIFF' for " << file_name << std::endl;
        read_ok = false;
    }
    if (0 != memcmp(&wave_header.wave_tag, "WAVE", 4)) {
        std::cerr << "wave_tag != 'WAVE' for " << file_name << std::endl;
        read_ok = false;
    }
    if (0 != memcmp(&wave_header.fmt_tag, "fmt ", 4)) {
        std::cerr << "fmt_tag != 'fmt' for " << file_name << std::endl;
        read_ok = false;
    }

    // only PCM
    if (wave_header.data_format != 1) {
        std::cerr << "data_format != kPCMFormat(1) for " << file_name << std::endl;
        read_ok = false;
    }
    // only mono
    if (wave_header.num_of_channels != 1) {
        std::cerr << "num_of_channels != 1 for " << file_name << std::endl;
        read_ok = false;
    }
    // only 16 bit
    if (wave_header.bits_per_sample != 16) {
        std::cerr << "bits_per_sample != 16 for " << file_name << std::endl;
        read_ok = false;
    }
    // only 16KHz
    if (wave_header.sampling_freq != 16000) {
        std::cerr << "sampling_freq != 16000 for " << file_name << std::endl;
        read_ok = false;
    }
    // make sure that data chunk follows file header
    if (0 != memcmp(&wave_header.data_tag, "data", 4)) {
        std::cerr << "data_tag != 'data' for " << file_name << std::endl;
        read_ok = false;
    }

    if (!read_ok) {
        fclose(inp_wave);
        throw std::logic_error("bad header for " + file_name);
    }

    size_t wave_size = wave_header.data_length / 2;
    wave.resize(wave_size);

    if (1 != fread(&(wave.front()), wave_size*2, 1, inp_wave)) {
        fclose(inp_wave);
        throw std::logic_error("fail to read data for " + file_name);
    }
    fclose(inp_wave);
}

void write_wav(const std::string& file_name, const std::vector<int16_t>& wave, const RiffWaveHeader& wave_header) {
    FILE *out_wave = fopen(file_name.c_str(), "wb");
    if (out_wave == NULL) {
        throw std::logic_error("fail to write into " + file_name);
    }

    if (1 != fwrite(&wave_header, sizeof(RiffWaveHeader), 1, out_wave)) {
        fclose(out_wave);
        throw std::logic_error("fail to write header into " + file_name);
    }
    if (1 != fwrite(&wave.front(), wave.size() * 2, 1, out_wave)) {
        fclose(out_wave);
        throw std::logic_error("fail to write data into " + file_name);
    }
    fclose(out_wave);
}


int main(int argc, char *argv[]) {
    try {
        // ------------------------------ Parsing and validating of input arguments --------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return EXIT_FAILURE;
        }

        // Loading Inference Engine
        InferenceEngine::Core ie;

        InferenceEngine::CNNNetwork network = ie.ReadNetwork(FLAGS_m);
        InferenceEngine::InputsDataMap inputs = network.getInputsInfo();

        //get state names pairs (inp,out)
        std::vector<std::pair<std::string, std::string>> state_names;
        size_t state_size = 0;
        for(auto& inp: inputs) {
            std::string inp_state_name = inp.first;
            if (inp_state_name.find("inp_state") == std::string::npos)
                continue;
            std::string out_state_name(inp_state_name);
            out_state_name.replace(0, 3, "out");
            state_names.emplace_back(inp_state_name, out_state_name);
            const InferenceEngine::SizeVector& size = inputs[inp_state_name]->getInputData()->getTensorDesc().getDims();
            size_t tensor_size = 1;
            for(size_t s: size) {
                tensor_size *= s;
            }
            std::cout << inp_state_name << "<-" << out_state_name << " " << tensor_size << " params" << std::endl;
            state_size += tensor_size;
        }
        std::cout << state_size*1e-6 << "M params in all states" << std::endl;
        
        InferenceEngine::ExecutableNetwork executable_network = ie.LoadNetwork(network, FLAGS_d);
        InferenceEngine::InferRequest infer_request = executable_network.CreateInferRequest();

        // Prepare input
        // get size of network input (pacth_size)
        std::string input_name("input");
        InferenceEngine::InputInfo::Ptr input = inputs[input_name];
        const InferenceEngine::TensorDesc& inp_desc = inputs[input_name]->getInputData()->getTensorDesc();
        const InferenceEngine::SizeVector& inp_shape = inp_desc.getDims();
        size_t patch_size = inp_shape[1];
        std::cout << "patch_size " << patch_size << std::endl;

        //read input wav file
        std::vector<int16_t> inp_wave_s16, out_wave_s16;
        std::vector<float> inp_wave_fp32, out_wave_fp32;
        RiffWaveHeader wave_header;
        read_wav(FLAGS_i, inp_wave_s16, wave_header);
        out_wave_s16.resize(inp_wave_s16.size());

        //fp32 input wave will be expanded to be divisible by patch_size
        size_t iter = 1 + (inp_wave_s16.size() / patch_size);
        std::cout << "iter " << iter << std::endl;
        size_t inp_size = patch_size * iter;
        inp_wave_fp32.resize(inp_size, 0);
        out_wave_fp32.resize(inp_size);

        //convert sint16_t  to float
        float scale = 1.0f/std::numeric_limits<int16_t>::max();
        size_t i=0;
        for(; i < inp_wave_s16.size(); ++i) {
            inp_wave_fp32[i] = (float)inp_wave_s16[i] * scale;
        }

        auto start_time = Time::now();
        for(size_t i=0; i<iter; ++i) {
            auto inputBlob = InferenceEngine::make_shared_blob<float>(inp_desc, &inp_wave_fp32[i * patch_size]);
            infer_request.SetBlob(input_name, inputBlob);  

            for (auto &state_name: state_names) {
                const std::string& inp_state_name = state_name.first;
                const std::string& out_state_name = state_name.second;

                if (i > 0) {
                    //set input state by coresponding output state from prev infer
                    auto blob_ptr = infer_request.GetBlob(out_state_name);
                    infer_request.SetBlob(inp_state_name,blob_ptr);  // infer_request accepts input blob of any size
                } else {
                    // first iteration. set input state to zero tensor.
                    const InferenceEngine::TensorDesc &state_desc = inputs[inp_state_name]->getInputData()->getTensorDesc();
                    auto blob_ptr = InferenceEngine::make_shared_blob<float>(state_desc);
                    blob_ptr->allocate();
                    for (auto &val: *blob_ptr) {
                        val = 0;
                    }
                    infer_request.SetBlob(inp_state_name,blob_ptr);  // infer_request accepts input blob of any size
                }
            }

            // make infer
            infer_request.Infer();
            InferenceEngine::Blob::Ptr output = infer_request.GetBlob("output");

            InferenceEngine::MemoryBlob::CPtr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(output);
            {
                // locked memory holder should be alive all time while access to its buffer happens
                auto moutputHolder = moutput->rmap();
                float* src = moutputHolder.as<float*>();
                float* dst = &out_wave_fp32[i * patch_size];
                memcpy(dst, src, patch_size*sizeof(float));
            }
        }

        using ms = std::chrono::duration<double, std::ratio<1, 1000>>;
        double total_latency = std::chrono::duration_cast<ms>(Time::now() - start_time).count();
        std::cout << "Metrics report:" << std::endl;
        std::cout << "\tLatency: " << std::fixed << std::setprecision(1) << total_latency << " ms" << std::endl;

        //convert fp32 to int16_t
        for(size_t i=0; i < out_wave_s16.size(); ++i) {
            out_wave_s16[i] = (int16_t)(out_wave_fp32[i] * std::numeric_limits<int16_t>::max());
        }
        write_wav(FLAGS_o, out_wave_s16, wave_header);
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
