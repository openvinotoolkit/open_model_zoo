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
DEFINE_string(at, "", p_msg);  // TODO -p

constexpr char d_msg[] = "Specify a [D]evice to infer on (the list of available devices is shown below). "
    "Use '-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. "
    "Use '-d MULTI:<comma-separated_devices_list>' format to specify MULTI plugin. "
    "The application looks for a suitable plugin for the specified device";
DEFINE_string(d, "CPU", d_msg);

constexpr char h_msg[] = "Print a [H]elp message";
DEFINE_bool(h, false, h_msg);

constexpr char i_msg[] = "[I]nput id or file to process. The input "
    "must be a single image, a folder of images, video file or camera id";
DEFINE_string(i, "0", i_msg);

constexpr char l_msg[] = "Path to a file with [L]abels mapping";
DEFINE_string(l, "", l_msg);

constexpr char lim_msg[] = "[LIM]it the umber of frames to store in output. If 0 is set, all frames are stored";
DEFINE_uint32(lim, 1000, lim_msg);

constexpr char loop_msg[] = "Enable reading the input in a [LOOP]";
DEFINE_bool(loop, false, loop_msg);

constexpr char mean_values_msg[] = "Normalize input by subtracting the [MEAN VALUES] per channel. Example: '255.0 255.0 255.0'";
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
constexpr char nthreads_msg[] = "[N]umber of [THREADS] to use for inference on the CPU (including HETERO and MULTI cases)";
DEFINE_uint32(nthreads, 0, nthreads_msg);

constexpr char o_msg[] = "Pattern for [O]utput file(s) to save";
DEFINE_string(o, "", o_msg);

constexpr char output_resolution_msg[] = "Specify the maximum output window resolution "
    "in 'width'x'height' format. Example: 1280x720. Input frame size used by default.";
DEFINE_string(output_resolution, "", output_resolution_msg);

constexpr char r_msg[] = "Inference results as [R]aw values";
DEFINE_bool(r, false, r_msg);

constexpr char reverse_input_channels_msg[] = "[REVERSE INPUT CHANNELS] order from BGR to RGB";
DEFINE_bool(reverse_input_channels, false, reverse_input_channels_msg);

constexpr char s_msg[] = "(Do [NO]t) [S]how output";
DEFINE_bool(s, true, s_msg);

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

unique_ptr<Processor> create(const string& aarchitecture_type, const string& weights,
        double confidence_threshold, const vector<string>& labels,
        bool reverse_nput_channels, const string &mean_values, const string &scale_values) {
    return unique_ptr<Processor>(new SSD(weights, float(confidence_threshold), labels));
}

// Input image is stored inside metadata, as we put it there during submission stage
cv::Mat render_detection_data(DetectionResult& result, const ColorPalette& palette, OutputTransform& outputTransform) {
    if (!result.metaData) {
        throw std::invalid_argument("Renderer: metadata is null");
    }

    auto outputImg = result.metaData->asRef<ImageMetaData>().img;

    if (outputImg.empty()) {
        throw std::invalid_argument("Renderer: image provided in metadata is empty");
    }
    outputTransform.resize(outputImg);
    // Visualizing result data over source image
    if (FLAGS_r) {
        slog::debug << " -------------------- Frame # " << result.frameId << "--------------------" << slog::endl;
        slog::debug << " Class ID  | Confidence | XMIN | YMIN | XMAX | YMAX " << slog::endl;
    }

    for (auto& obj : result.objects) {
        if (FLAGS_r) {
            slog::debug << " "
                << std::left << std::setw(9) << obj.label << " | "
                << std::setw(10) << obj.confidence << " | "
                << std::setw(4) << int(obj.x) << " | "
                << std::setw(4) << int(obj.y) << " | "
                << std::setw(4) << int(obj.x + obj.width) << " | "
                << std::setw(4) << int(obj.y + obj.height)
                << slog::endl;
        }
        outputTransform.scaleRect(obj);
        std::ostringstream conf;
        conf << ":" << std::fixed << std::setprecision(1) << obj.confidence * 100 << '%';
        const auto& color = palette[obj.labelID];
        putHighlightedText(outputImg, obj.label + conf.str(),
            cv::Point2f(obj.x, obj.y - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2);
        cv::rectangle(outputImg, obj, color, 2);
    }

    try {
        for (auto& lmark : result.asRef<RetinaFaceDetectionResult>().landmarks) {
            outputTransform.scaleCoord(lmark);
            cv::circle(outputImg, lmark, 2, cv::Scalar(0, 255, 255), -1);
        }
    }
    catch (const std::bad_cast&) {}

    return outputImg;
}
}  // namespace

struct TimedMat {
    cv::Mat mat;
    chrono::steady_clock::time_point start;
};

int main(int argc, char *argv[]) {
    set_terminate(catcher);
    parse(argc, argv);
    OutputTransform output_transform;
    LazyVideoWriter video_writer;
    Presenter presenter(FLAGS_u);
    PerformanceMetrics render_metrics;
    PerformanceMetrics metrics;
    vector<string> labels;
    if (!FLAGS_l.empty()) {
        labels = DetectionModel::loadLabels(FLAGS_l);
    }
    ColorPalette palette(labels.size() > 0 ? labels.size() : 100);
    unique_ptr<Processor> ssd = create(FLAGS_at, FLAGS_m, FLAGS_t, labels,
            FLAGS_reverse_input_channels, FLAGS_mean_values, FLAGS_scale_values);
    ssd->setInputsPreprocessing(FLAGS_reverse_input_channels, FLAGS_mean_values, FLAGS_scale_values);
    ov::Core core;
    auto cnnConfig = ConfigFactory::getUserConfig(FLAGS_d, FLAGS_nireq, FLAGS_nstreams, FLAGS_nthreads);
    auto cml = ssd->compileModel(cnnConfig, core);
    unsigned int nireq = cnnConfig.maxAsyncRequests;
    if (nireq == 0) {
        try {
            // +1 to use it as a buffer of the pipeline
            nireq = cml.get_property(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>() + 1;
        } catch (const ov::Exception& ex) {
            throw std::runtime_error(std::string("Every device used with the demo should support "
                "OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. Failed to query the metric with error: ") + ex.what());
        }
    }
    vector<ov::InferRequest> ireqs;
    while (ireqs.size() < nireq) {
        ireqs.push_back(cml.create_infer_request());
    }
    OVAdapter<TimedMat> inferer(ireqs);
    shared_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop, nireq > 1);

    for (auto state : inferer.iterate) {
        TimedMat* timed_mat = state->data();
        if (nullptr == timed_mat) {
            auto start = chrono::steady_clock::now();
            const cv::Mat& mat = cap->read();
            if (mat.data) {
                ssd->preprocess(ImageInputData{mat}, state->ireq);
                inferer.submit(std::move(state->ireq), {std::move(mat), start});
            } else {
                inferer.end();
            }
            continue;
        }
        auto post_start = chrono::steady_clock::now();
        InferenceResult inf_res;
        inf_res.metaData = std::make_shared<ImageMetaData>(timed_mat->mat, timed_mat->start);  // TODO align order time and mat args in my meta and theirs
        inf_res.internalModelData = make_shared<InternalImageModelData>(timed_mat->mat.cols, timed_mat->mat.rows);
        inf_res.outputsData = {{"", state->ireq.get_output_tensor()}};
        std::unique_ptr<ResultBase> result = ssd->postprocess(inf_res);  // TODO process req and timed_mat inhere
        auto render_start = chrono::steady_clock::now();
        const cv::Mat& out_im = render_detection_data(result->asRef<DetectionResult>(), palette, output_transform);
        presenter.drawGraphs(out_im);
        render_metrics.update(render_start);
        metrics.update(timed_mat->start,
            timed_mat->mat, {10, 22}, cv::FONT_HERSHEY_COMPLEX, 0.65);
        video_writer.write(out_im);
        if (FLAGS_s) {
            cv::imshow(argv[0], timed_mat->mat);
            int key = cv::pollKey();
            if ('P' == key || 'p' == key || '0' == key || ' ' == key) {
                key = cv::waitKey(0);
            }
            if ('Q' == key || 'q' == key || 27 == key) {  // Esc
                break;
            }
            presenter.handleKey(key);
        }
    }
    // logLatencyPerStage(cap->getMetrics().getTotal().latency, inferer.getPreprocessMetrics().getTotal().latency,
    //     inferer.getInferenceMetircs().getTotal().latency, inferer.getPostprocessMetrics().getTotal().latency,
    //     render_metrics.getTotal().latency);
    return 0;
}
