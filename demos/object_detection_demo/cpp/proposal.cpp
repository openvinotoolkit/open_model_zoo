// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "proposal_common.hpp"
#include <models/detection_model_ssd.h>
#include <monitors/presenter.h>
#include <utils/images_capture.h>

#include <gflags/gflags.h>

using namespace std;
using SSD = ModelSSD;

namespace {
constexpr char m_msg[] = "Path to an .xml file with a trained [M]odel";
DEFINE_string(m, "", m_msg);

constexpr char p_msg[] = "[P]rocessor type: centernet, faceboxes, retinaface, retinaface-pytorch, ssd or yolo";
DEFINE_string(p, "", p_msg);

constexpr char d_msg = "Specify a [D]evice to infer on (the list of available devices is shown below). "
    "Use '-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. "
    "Use '-d MULTI:<comma-separated_devices_list>' format to specify MULTI plugin. "
    "The application looks for a suitable plugin for the specified device";
DEFINE_string(d, "CPU", d_msg);

constexpr char h_msg[] = "Print a [H]elp message";
DEFINE_bool(h, false, h_msg);

constexpr char i_message[] = "[I]nput id or file to process. The input "
    "must be a single image, a folder of images, video file or camera id";
DEFINE_string(i, "0", i_msg);

constexpr char l_msg[] = "Path to a file with [L]abels mapping";
DEFINE_string(l, "", l_msg);

constexpr char lim_msg[] = "[LIM]it the umber of frames to store in output. If 0 is set, all frames are stored";
DEFINE_uint32(lim, 1000, lim_msg);

constexpr char loop_msg[] = "Enable reading the input in a [LOOP]";
DEFINE_bool(loop, false, loop_msg);

constexpr mean_values_msg[] = "Normalize input by subtracting the [MEAN VALUES] per channel. Example: '255.0 255.0 255.0'";
DEFINE_string(mean_values, "", mean_values_msg);

constexpr char nireq_msg[] = "[N]umber of [I]nfer [REQ]uests. Default value is determined automatically per a device";
DEFINE_uint32(nireq, 0, nireq_msg);
constexpr char nstreams_msg[] = "[N]umber of [STREAMS] to use for inference on the CPU, GPU or MYRIAD devices "
    "(for HETERO and MULTI device cases use format <dev1>:<nstreams1>,<dev2>:<nstreams2> or just <nstreams>). "
    "Default value is determined automatically per a device. Please note that although the automatic selection "
    "usually provides a reasonable performance, it still may be non-optimal for some cases, especially for "
    "very small networks. See sample's README for more details. "
    "Also, using nstreams>1 is inherently throughput-oriented option, "
    "while for the best-latency estimations the number of streams should be set to 1";
DEFINE_string(nstreams, "", nstreams_msg);
constexpr char nthreads_msg = "[N]umber of [THREADS] to use for inference on the CPU (including HETERO and MULTI cases)";
DEFINE_uint32(nthreads, 0, nthreads_msg);

constexpr char o_msg[] = "Pattern for [O]utput file(s) to save";
DEFINE_string(o, "", o_msg);

constexpr char output_resolution_msg[] = "Specify the maximum output window resolution "
    "in 'width'x'height' format. Example: 1280x720. Input frame size used by default.";
DEFINE_string(output_resolution, "", output_resolution_msg)

constexpr char r_msg[] = "Inference results as [R]aw values";
DEFINE_bool(r, false, r_msg);

constexpr reverse_input_channels_msg[] = "[REVERSE INPUT CHANNELS] order from BGR to RGB";
DEFINE_bool(reverse_input_channels, false, reverse_input_channels_msg);

constexpr char s_msg[] = "(Do [NO]t) [S]how output";
DEFINE_bool(s, false, s_msg);

constexpr char scale_values_msg[] = "Divide input by [SCALE VALUES] per channel. "
    "Division is applied after mean values subtraction. Example: '255.0 255.0 255.0'";
DEFINE_string(scale_values, "", scale_values_msg);

constexpr char t_msg[] = "Confidence [T]hreshold for detections";
DEFINE_double(t, 0.5, t_msg);

constexpr char u_msg[] = "Resource [U]tilization graphs: -u cdm. "
    "c - average [C]PU load, d - load [D]istrobution over cores, m - [M]emory usage";
DEFINE_string(u, "cdm", u_msg);

void parse(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    slog::info << ov::get_openvino_version() << slog::endl;
    if (FLAGS_h || 1 == argc) {
        cout << "\n\t  -m <[M]odel file>                    " << m_msg
             << "\n\t  -p <[P]rocessor type>                " << p_msg
             << "\n\t[ -d] <CPU>                            " << d_msg
             << "\n\t[ -h]                                  " << h_msg
             << "\n\t[--help]                               print [HELP] on all arguments"
             << "\n\t[ -i] <0>                              " << i_msg
             << "\n\t[ -l] <[L]abels file>                  " << l_msg
             << "\n\t[--lim] <1000>                         " << lim_msg
             << "\n\t[--loop]                               " << loop_msg
             << "\n\t[--mean_values] <0.0 0.0 0.0>          " << mean_values_msg
             << "\n\t[--nireq] <0>                          " << nireq_msg
             << "\n\t[--nstreams] <str>                     " << nstreams_msg
             << "\n\t[--nthreads] <0>                       " << nthreads_msg
             << "\n\t[ -o] <[O]utput file(s) pattern>       " << o_msg
             << "\n\t[--output_resolution] <width>x<height> " << output_resolution_msg
             << "\n\t[ -r]                                  " << r_msg
             << "\n\t[--reverse_input_channels]             " << reverse_input_channels_msg
             << "\n\t[ -s] ([--nos])                        " << s_msg
             << "\n\t[--scale_values] <1.0 1.0 1.0>         " << scale_values_msg
             << "\n\t[ -t] <0.5>                            " << t_msg
             << "\n\t[ -u] <cdm>                            " << u_msg
             << "\n\tKey bindings:"
                "\n\t\tQ, Esc - [Q]uit"
                "\n\t\tP, 0, spacebar - [P]ause"
                "\n\t\tC - average [C]PU load, D - load [D]istrobution over cores, M - [M]emory usage, H - [H]ide";
        showAvailableDevices();
        exit(0);
    } if (FLAGS_m.empty()) {
        throw invalid_argument{"-m <[M]odel file> is required"};
    } if (FLAGS_at.empty()) {
        throw invalid_argument{"-p <[P]rocessor type> is required"};
    } if (!FLAGS_output_resolution.empty() && FLAGS_output_resolution.find("x") == string::npos) {
        throw invalid_argument{"Correct format of --output_resolution is 'width'x'height'"};
    }
}

unique_ptr<Processor> create_processor(const string& aarchitecture_type, const string& weights,
        double confidence_threshold, const vector<string>& labels,
        bool reverse_nput_channels, const string &mean_values, const string &scale_values) {
    return unique_ptr<Processor>(new SSD(weights, float(confidence_threshold), true, labels));
}

cv::Mat render_detection_data(DetectionPredictions& predictions, const ColorPalette& palette, OutputTransform& output_transform) {
    return cv::Mat{};
}
}  // namespace

int main(int argc, char *argv[]) {
    set_terminate(catcher);
    parse(argc, argv);
    shared_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop);
    ColorPalette palette;
    OutputTransform output_transform;
    LazyVideoWriter video_writer;
    Presenter presenter(FLAGS_u);
    PerformanceMetrics render_metrics;
    PerformanceMetrics metrics;
    vector<string> labels;
    if (!FLAGS_labels.empty()) {
        labels = DetectionModel::loadLabels(FLAGS_labels);
    }
    SingleProcessorInferer inferer(
        create_processor(FLAGS_at, FLAGS_m, FLAGS_t, labels,
            FLAGS_reverse_input_channels, FLAGS_mean_values, FLAGS_scale_values),
        ConfigFactory::getUserConfig(FLAGS_d, "", "", FLAGS_nireq, FLAGS_nstreams, FLAGS_nthreads));

    for (const shared_ptr<State>& state : inferer) {
        if (State::StateEnum::has_free_input == state->state) {
            auto start = chrono::steady_clock::now();
            // cap's first read() call must throw if it would return empty mat
            inferer.submit(cap->read(), start);
            continue;
        }
        auto render_start = chrono::steady_clock::now();
        shared_ptr<DetectionPredictions> predictions = static_pointer_cast<DetectionPredictions>(state);
        const cv::Mat& out_im = render_detection_data(*predictions, palette, output_transform);
        presenter.drawGraphs(out_im);
        render_metrics.update(render_start);
        metrics.update(predictions->metaData->asRef<ImageMetaData>().timeStamp,
            out_im, {10, 22}, cv::FONT_HERSHEY_COMPLEX, 0.65);
        video_writer.write(out_im);
        if (FLAGS_s) {
            cv::imshow(argv[0], out_im);
            int key = cv::pollKey();
            if ('P' == key || 'p' == key || '0' == key || ' ' == key) {
                key = cv::waitKey(0);
            }
            if ('Q' == key || 'q' == key || 27 == key) {  // Esc
                break;
            }
            presenter.handle(key);
        }
    }
    logLatencyPerStage(cap->getMetrics().getTotal().latency, inferer.getPreprocessMetrics().getTotal().latency,
        inferer.getInferenceMetircs().getTotal().latency, inferer.getPostprocessMetrics().getTotal().latency,
        render_metrics.getTotal().latency);
    return 0;
}
