#include <iostream>
#include <fstream>

#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/parsers.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/render.hpp>
#include <opencv2/gapi/streaming/meta.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/cpu/core.hpp>

#include "cascade_detection_demo_gapi.hpp"

#include <monitors/presenter.h>
#include <utils_gapi/stream_source.hpp>
#include <utils/args_helper.hpp>
#include <utils/config_factory.h>
#include <utils/ocv_common.hpp>

using GDetections = cv::GArray<cv::Rect>;
using GLabelsIds  = cv::GArray<int>;
using GSize       = cv::GOpaque<cv::Size>;
using GProbs      = cv::GArray<cv::GMat>;
using GPrims      = cv::GArray<cv::gapi::wip::draw::Prim>;
using Labels      = std::vector<std::string>;

static std::string getLabel(const int id, const std::vector<std::string>& labels) {
	std::string out_label;
	if (id > -1) {
		if (!labels.empty()) {
			out_label = labels[id];
		} else {
			out_label = std::to_string(id);
		}
	}
	return out_label;
};

G_API_OP(FilterOutOfBounds, <std::tuple<GDetections,GLabelsIds>(GDetections,GLabelsIds,GSize)>,
        "sample.custom.filter_out_of_bounds") {
    static std::tuple<cv::GArrayDesc, cv::GArrayDesc> outMeta(cv::GArrayDesc,
            cv::GArrayDesc,
            cv::GOpaqueDesc) {
        return std::make_tuple(cv::empty_array_desc(), cv::empty_array_desc());
    }
};

GAPI_OCV_KERNEL(OCVFilterOutOfBounds, FilterOutOfBounds) {
    static void run(const std::vector<cv::Rect>& in_rcts,
            const std::vector<int>& in_det_ids,
            const cv::Size& in_size,
            std::vector<cv::Rect>& out_rcts,
            std::vector<int>& out_det_ids) {
        cv::Rect surface({0, 0}, in_size);
        for (uint32_t i = 0; i < in_rcts.size(); ++i) {
            const auto rc = in_rcts[i];
            // NOTE: IE adds one more row or column to the ROI in case if ROI
            //       has odd height or width and original image has NV12 format.
            auto adj_rc = rc;
            adj_rc.width += adj_rc.width % 2;
            adj_rc.height += adj_rc.height % 2;

            auto clipped_rc = adj_rc & surface;

            if (clipped_rc.area() != adj_rc.area())
            {
                continue;
            }

            out_rcts.push_back(rc);
            out_det_ids.push_back(in_det_ids[i]);
        }
    }
};

G_API_OP(ParseProbs, <GLabelsIds(GProbs,int,float)>,
         "sample.custom.parse_probs") {
    static cv::GArrayDesc outMeta(cv::GArrayDesc, const int, const float) {
        return cv::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVParseProbs, ParseProbs) {
    static void run(const std::vector<cv::Mat>& samples_probs, const int num_classes,
                    const float threshold, std::vector<int>& out_classes) {
        out_classes.resize(samples_probs.size());
        for (uint32_t i = 0; i < samples_probs.size(); ++i) {
            const auto probs = samples_probs[i];
            const float* probs_data = probs.ptr<float>();
            const float* id = std::max_element(probs_data, probs_data + num_classes);

            out_classes[i] = (*id) >= threshold ? id - probs_data : -1;
        }
    }
};


G_API_OP(LabeledBoxes, <GPrims(GDetections,GLabelsIds,GLabelsIds,Labels,Labels)>,
         "sample.custom.labeled_boxes") {
    static cv::GArrayDesc outMeta(cv::GArrayDesc,
                                  cv::GArrayDesc,
                                  cv::GArrayDesc,
                                  Labels,
                                  Labels) {
        return cv::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVLabeledBoxes, LabeledBoxes) {
    // Converts rectangles, labels into G-API's rendering primitives
    static void run(const std::vector<cv::Rect>& in_rcs,
                    const std::vector<int>& in_det_ids,
                    const std::vector<int>& in_cls_ids,
                    const Labels& det_labels,
                    const Labels& cls_labels,
                    std::vector<cv::gapi::wip::draw::Prim>& out_prims) {
        out_prims.clear();

        for (uint32_t i = 0; i < in_rcs.size(); ++i) {
            out_prims.emplace_back(cv::gapi::wip::draw::Rect(in_rcs[i], CV_RGB(0, 255, 0), 2));

            const auto detection_str = getLabel(in_det_ids[i], det_labels);
            out_prims.emplace_back(cv::gapi::wip::draw::Text
                                   { detection_str,
                                     in_rcs[i].tl() + cv::Point(3, 20), cv::FONT_HERSHEY_SIMPLEX,
                                     0.7, CV_RGB(0, 255, 0), 2, 8, false });

            const auto detection_str_h = getTextSize(detection_str, cv::FONT_HERSHEY_SIMPLEX,
                                                     0.7, 2, nullptr).height;
            out_prims.emplace_back(cv::gapi::wip::draw::Text
                                   { getLabel(in_cls_ids[i], cls_labels),
                                     in_rcs[i].tl() + cv::Point(3, 25 + detection_str_h),
                                     cv::FONT_HERSHEY_SIMPLEX,
                                     0.7, CV_RGB(230, 216, 173), 2, 8, false });
        }
    }
};

G_API_NET(Detector, <cv::GMat(cv::GMat)>, "sample.detector");
G_API_NET(Classifier, <cv::GMat(cv::GMat)>, "sample.classifier");

namespace util {
bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    /** ---------- Parsing and validating input arguments ----------**/
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    if (FLAGS_i.empty())
        throw std::logic_error("Parameter -i is not set");
    if (FLAGS_dm.empty())
        throw std::logic_error("Parameter -dm is not set");
    if (FLAGS_cm.empty()) {
        throw std::logic_error("Parameter -cm is not set");
    }
    return true;
}
static std::vector<std::string> readLabelsFromFile(const std::string& file_path) {
    std::vector<std::string> out_labels;

    std::ifstream is;
    is.open(file_path, std::ios::in);

    if (!is.is_open()) {
        throw std::logic_error(std::string("Could not open ") + file_path);
    }

    std::string label;
    while (std::getline(is, label)) {
        // simple trimming from the end:
        label.erase(std::find_if(label.rbegin(), label.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), label.end());

        out_labels.push_back(label);
    }
    return out_labels;
}

} // namespace util

int main(int argc, char* argv[]) {
    PerformanceMetrics metrics;

    /** Get OpenVINO runtime version **/
    slog::info << ov::get_openvino_version() << slog::endl;
    // ---------- Parsing and validating of input arguments ----------
    if (!util::ParseAndCheckCommandLine(argc, argv)) {
        return 0;
    }

    std::vector<std::string> detector_labels;
    std::vector<std::string> classifier_labels;
    if (!FLAGS_det_labels.empty()) {
        detector_labels = util::readLabelsFromFile(FLAGS_det_labels);
    }
    auto num_classes = FLAGS_num_classes;
    if (!FLAGS_cls_labels.empty()) {
        classifier_labels = util::readLabelsFromFile(FLAGS_cls_labels);
        num_classes = classifier_labels.size();
    }

    /** Get information about frame **/
    std::shared_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop, read_type::safe, 0,
        std::numeric_limits<size_t>::max());
    const auto tmp = cap->read();
    cv::Size frame_size = cv::Size{tmp.cols, tmp.rows};

    /** ---------------- Main graph of demo ---------------- **/
    cv::GComputation comp([&]{
        cv::GMat in;
        cv::GMat detections = cv::gapi::infer<Detector>(in);

        auto im_size = cv::gapi::streaming::size(in);

        cv::GArray<cv::Rect> objs;
        cv::GArray<int> det_ids;
        if (FLAGS_parser == "yolo") {
            std::tie(objs, det_ids) = cv::gapi::streaming::parseYolo(detections, im_size, 0.5f, 0.5f);
        }
        else {
            std::tie(objs, det_ids) = cv::gapi::streaming::parseSSD(detections, im_size, 0.6f, -1);
        }

        // Filter out of bounds projections
        cv::GArray<cv::Rect> filtered_objs;
        cv::GArray<int> filtered_det_ids;
        std::tie(filtered_objs, filtered_det_ids) = FilterOutOfBounds::on(objs,
                det_ids,
                im_size);

        // Run Inference for classifier on the passed ROIs of the frame
        cv::GArray<cv::GMat> filtered_probs = cv::gapi::infer<Classifier>(filtered_objs, in);
        // Run custom operation to project probabilities to the labels identifiers
        cv::GArray<int> filtered_cls_ids = ParseProbs::on(filtered_probs, num_classes, 0.5f);

        auto prims = LabeledBoxes::on(filtered_objs,
                filtered_det_ids,
                filtered_cls_ids,
                detector_labels,
                classifier_labels);

        auto rendered = cv::gapi::wip::draw::render3ch(in, prims);
        return cv::GComputation(cv::GIn(in), cv::GOut(rendered));
    });

    auto det_config = ConfigFactory::getUserConfig(FLAGS_ddm, FLAGS_det_nireq,
                                                   FLAGS_det_nstreams, FLAGS_det_nthreads);
    const auto detector = cv::gapi::ie::Params<Detector> {
        FLAGS_dm,                           // path to topology IR
        fileNameNoExt(FLAGS_dm) + ".bin",   // path to weights
        FLAGS_ddm                           // device specifier
    }.cfgNumRequests(det_config.maxAsyncRequests)
     .pluginConfig(det_config.getLegacyConfig());
    slog::info << "The detection model " << FLAGS_dm << " is loaded to " << FLAGS_ddm << " device." << slog::endl;

    auto cls_config = ConfigFactory::getUserConfig(FLAGS_dcm, FLAGS_cls_nireq,
                                                   FLAGS_cls_nstreams, FLAGS_cls_nthreads);
    const auto classifier = cv::gapi::ie::Params<Classifier> {
        FLAGS_cm,                           // path to topology IR
        fileNameNoExt(FLAGS_cm) + ".bin",   // path to weights
        FLAGS_dcm                           // device specifier
    }.cfgNumRequests(cls_config.maxAsyncRequests)
     .pluginConfig(cls_config.getLegacyConfig());
    slog::info << "The classification model " << FLAGS_cm << " is loaded to " << FLAGS_dcm << " device." << slog::endl;

    auto pipeline = comp.compileStreaming(
            cv::compile_args(cv::gapi::kernels<OCVFilterOutOfBounds, OCVParseProbs, OCVLabeledBoxes>(),
                             cv::gapi::networks(detector, classifier)));

    /** Output container for result **/
    cv::Mat output;

    /** ---------------- The execution part ---------------- **/
    cap = openImagesCapture(FLAGS_i, FLAGS_loop, read_type::safe, 0,
        std::numeric_limits<size_t>::max());

    pipeline.setSource<custom::CommonCapSrc>(cap);
    std::string windowName = "Cascade detection demo G-API";
    int delay = 1;

    cv::Size graphSize{static_cast<int>(frame_size.width / 4), 60};
    Presenter presenter(FLAGS_u, frame_size.height - graphSize.height - 10, graphSize);

    LazyVideoWriter videoWriter{FLAGS_o, cap->fps(), FLAGS_limit};

    bool isStart = true;
    const auto startTime = std::chrono::steady_clock::now();
    pipeline.start();

    while(pipeline.pull(cv::gout(output))) {
        presenter.drawGraphs(output);
        if (isStart) {
            metrics.update(startTime, output, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX,
                0.65, { 200, 10, 10 }, 2, PerformanceMetrics::MetricTypes::FPS);
            isStart = false;
        }
        else {
            metrics.update({}, output, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX,
                0.65, { 200, 10, 10 }, 2, PerformanceMetrics::MetricTypes::FPS);
        }

        videoWriter.write(output);

        if (!FLAGS_no_show) {
            cv::imshow(windowName, output);
            int key = cv::waitKey(delay);
            /** Press 'Esc' to quit **/
            if (key == 27) {
                break;
            } else {
                presenter.handleKey(key);
            }
        }
    }
    slog::info << "Metrics report:" << slog::endl;
    slog::info << "\tFPS: " << std::fixed << std::setprecision(1) << metrics.getTotal().fps << slog::endl;
    slog::info << presenter.reportMeans() << slog::endl;

    return 0;
}
