#include <iostream>

#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/parsers.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/render.hpp>
#include <opencv2/gapi/streaming/meta.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/cpu/core.hpp>

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

G_API_NET(ObjectDetector, <cv::GMat(cv::GMat)>, "sample.vpu.object-detector");
G_API_NET(Classifier, <cv::GMat(cv::GMat)>, "sample.vpu.classifier");

int main() {
    std::vector<std::string> detector_labels;
    std::vector<std::string> classifier_labels;

	const std::string opt_parser = "yolo";
    const int opt_num_classes = 1;
	// Declare the pipeline inputs
	cv::GFrame in;
	// Run Inference for detector on the full frame
	cv::GMat detections = cv::gapi::infer<ObjectDetector>(in);
	// Parse detections, project them to the original image frame
	cv::GArray<cv::Rect> objs;
	cv::GArray<int> det_ids;

	auto im_size = cv::gapi::streaming::size(in);

	if (opt_parser == "yolo") {
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
	cv::GArray<int> filtered_cls_ids = ParseProbs::on(filtered_probs, opt_num_classes, 0.5f);

    auto prims = LabeledBoxes::on(filtered_objs,
                                  filtered_det_ids,
                                  filtered_cls_ids,
                                  detector_labels,
                                  classifier_labels);
    auto rendered = cv::gapi::wip::draw::renderFrame(in, prims);
    cv::GComputation comp(cv::GIn(in), cv::GOut(rendered));
}
