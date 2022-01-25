#include "proposal_common.hpp"

#include <utils/default_flags.hpp>
#include <models/detection_model_ssd.h>
#include <monitors/presenter.h>
#include <utils/images_capture.h>

#include <gflags/gflags.h>

using namespace std;
using SSD = ModelSSD;

namespace {
DEFINE_INPUT_FLAGS
DEFINE_OUTPUT_FLAGS

DEFINE_bool(h, false, "Print a usage message");
DEFINE_string(at, "", "Architecture type: centernet, faceboxes, retinaface, retinaface-pytorch, ssd or yolo");
DEFINE_string(m, "", "Path to an .xml file with a trained model");
DEFINE_string(d, "CPU",
    "Specify a target device to infer on (the list of available devices is shown below). "
    "Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
    "Use \"-d MULTI:<comma-separated_devices_list>\" format to specify MULTI plugin. "
    "The application looks for a suitable plugin for the specified device.");
DEFINE_string(labels, "", "Path to a file with labels mapping");
DEFINE_bool(r, false, "Inference results as raw values");
DEFINE_double(t, 0.5, "Probability threshold for detections.");
DEFINE_uint32(nireq, 0, "Number of infer requests. Default value is determined automatically for device");
DEFINE_uint32(nthreads, 0, "Number of threads to use for inference on the CPU (including HETERO and MULTI cases)");
DEFINE_string(nstreams, "",
    "Number of streams to use for inference on the CPU, GPU or MYRIAD devices "
    "(for HETERO and MULTI device cases use format <dev1>:<nstreams1>,<dev2>:<nstreams2> or just <nstreams>). "
    "Default value is determined automatically for a device.Please note that although the automatic selection "
    "usually provides a reasonable performance, it still may be non - optimal for some cases, especially for "
    "very small networks. See sample's README for more details. "
    "Also, using nstreams>1 is inherently throughput-oriented option, "
    "while for the best-latency estimations the number of streams should be set to 1");
DEFINE_bool(show, false, "-noshow: don't show output");
DEFINE_string(u, "", "List of monitors to show initially");  // TODO: more clear wording
DEFINE_bool(reverse_input_channels, false, "Switch the input channels order from BGR to RGB");
DEFINE_string(mean_values, "", "Normalize input by subtracting the mean values per channel. Example: '255.0 255.0 255.0'");
DEFINE_string(scale_values, "",
    "Divide input by scale values per channel. Division is applied "
    "after mean values subtraction. Example: '255.0 255.0 255.0'");

void parse(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    slog::info << ov::get_openvino_version() << slog::endl;
    if (FLAGS_h || argc == 1) {
        exit(0);
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
        } else {
            auto render_start = chrono::steady_clock::now();
            shared_ptr<DetectionPredictions> predictions = static_pointer_cast<DetectionPredictions>(state);
            const cv::Mat& out_im = render_detection_data(*predictions, palette, output_transform);
            presenter.drawGraphs(out_im);
            render_metrics.update(render_start);
            metrics.update(predictions->metaData->asRef<ImageMetaData>().timeStamp,
                out_im, {10, 22}, cv::FONT_HERSHEY_COMPLEX, 0.65);
            video_writer.write(out_im);
            if (FLAGS_show) {
                cv::imshow("Detection Results", out_im);
                int key = cv::waitKey(1);
                if ('q' == key || 'Q' == key || 27 == key) {  // Esc
                    break;
                } else {
                    presenter.handleKey(key);
                }
            }
        }
    }
    logLatencyPerStage(cap->getMetrics().getTotal().latency, inferer.getPreprocessMetrics().getTotal().latency,
        inferer.getInferenceMetircs().getTotal().latency, inferer.getPostprocessMetrics().getTotal().latency,
        render_metrics.getTotal().latency);
    return 0;
}
