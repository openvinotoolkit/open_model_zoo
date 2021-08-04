#include <monitors/presenter.h>
#include <utils/args_helper.hpp>
#include <utils/slog.hpp>
#include <utils/ocv_common.hpp>

#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/render.hpp>

#include "face_detection_mtcnn_demo.hpp"
#include "custom_kernels.hpp"
#include "utils.hpp"
#include "stream_source.hpp"

const int MAX_PYRAMID_LEVELS = 13;

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
    if (FLAGS_m_p.empty())
        throw std::logic_error("Parameter -m_p is not set");
    if (FLAGS_m_r.empty())
        throw std::logic_error("Parameter -m_r is not set");
    if (FLAGS_m_o.empty())
        throw std::logic_error("Parameter -m_o is not set");
    return true;
}
} // namespace util

namespace nets {
G_API_NET(MTCNNRefinement, <custom::GMat2(cv::GMat)>, "custom.mtcnn_refinement");
G_API_NET(MTCNNOutput, <custom::GMat3(cv::GMat)>, "custom.mtcnn_output");
}

int main(int argc, char* argv[]) {
    try {
        /** Print info about Inference Engine **/
        slog::info << *InferenceEngine::GetInferenceEngineVersion() << slog::endl;

        if (!util::ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        /** Get information about frame from cv::VideoCapture **/
        std::shared_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop, 0,
            std::numeric_limits<size_t>::max(), stringToSize(FLAGS_res));
        const auto tmp = cap->read();
        cap.reset();
        if (!tmp.data) {
            throw std::runtime_error("Couldn't grab first frame");
        }
        cv::Size frame_size = cv::Size{tmp.cols, tmp.rows};
        cap = openImagesCapture(FLAGS_i, FLAGS_loop, 0,
            std::numeric_limits<size_t>::max(), stringToSize(FLAGS_res));

        /** Calculate scales, number of pyramid levels and sizes for PNet pyramid **/
        std::vector<cv::Size> level_size;
        std::vector<double> scales;

        const auto pyramid_levels = FLAGS_hs ? calculate_half_scales(frame_size, scales, level_size) :
                                               calculate_scales(frame_size, scales, level_size);
        CV_Assert(pyramid_levels <= MAX_PYRAMID_LEVELS);

        /** ---------------- Main graph of demo ---------------- **/
        /** Graph input
          * Proposal part of MTCNN graph
          * Preprocessing BGR2RGB + transpose (NCWH is expected instead of NCHW)
          **/
        cv::GMat in_original;
        cv::GMat in_originalRGB = cv::gapi::BGR2RGB(in_original);
        cv::GMat in_transposedRGB = cv::gapi::transpose(in_originalRGB);
        cv::GOpaque<cv::Size> in_sz = cv::gapi::streaming::size(in_original);
        cv::GMat regressions[MAX_PYRAMID_LEVELS];
        cv::GMat scores[MAX_PYRAMID_LEVELS];
        cv::GArray<custom::Face> nms_p_faces[MAX_PYRAMID_LEVELS];
        cv::GArray<custom::Face> total_faces[MAX_PYRAMID_LEVELS];

        /** The very first PNet pyramid layer to init total_faces[0] **/
        std::tie(regressions[0], scores[0]) = run_mtcnn_p(in_transposedRGB, get_pnet_level_name(level_size[0]));
        cv::GArray<custom::Face> faces0 = custom::BuildFaces::on(scores[0], regressions[0], float(scales[0]), float(FLAGS_th));
        cv::GArray<custom::Face> final_p_faces_for_bb2squares = custom::ApplyRegression::on(faces0, true);
        cv::GArray<custom::Face> final_faces_pnet0 = custom::BBoxesToSquares::on(final_p_faces_for_bb2squares);
        total_faces[0] = custom::RunNMS::on(final_faces_pnet0, 0.5f, false);

        /** The rest PNet pyramid layers to accumlate all layers result in total_faces[PYRAMID_LEVELS - 1]] **/
        for (int i = 1; i < pyramid_levels; ++i) {
            std::tie(regressions[i], scores[i]) = run_mtcnn_p(in_transposedRGB, get_pnet_level_name(level_size[i]));
            cv::GArray<custom::Face> faces = custom::BuildFaces::on(scores[i], regressions[i], float(scales[i]), float(FLAGS_th));
            cv::GArray<custom::Face> final_p_faces_for_bb2squares_i = custom::ApplyRegression::on(faces, true);
            cv::GArray<custom::Face> final_faces_pnet_i = custom::BBoxesToSquares::on(final_p_faces_for_bb2squares_i);
            nms_p_faces[i] = custom::RunNMS::on(final_faces_pnet_i, 0.5f, false);
            total_faces[i] = custom::AccumulatePyramidOutputs::on(total_faces[i - 1], nms_p_faces[i]);
        }

        /** Proposal post-processing **/
        cv::GArray<custom::Face> final_faces_pnet = custom::RunNMS::on(total_faces[pyramid_levels - 1], 0.7f, true);

        /** Refinement part of MTCNN graph **/
        cv::GArray<cv::Rect> faces_roi_pnet = custom::R_O_NetPreProcGetROIs::on(final_faces_pnet, in_sz);
        cv::GArray<cv::GMat> regressionsRNet, scoresRNet;
        std::tie(regressionsRNet, scoresRNet) = cv::gapi::infer<nets::MTCNNRefinement>(faces_roi_pnet, in_transposedRGB);

        /** Refinement post-processing **/
        cv::GArray<custom::Face> rnet_post_proc_faces = custom::RNetPostProc::on(final_faces_pnet, scoresRNet, regressionsRNet, float(FLAGS_th));
        cv::GArray<custom::Face> nms07_r_faces_total = custom::RunNMS::on(rnet_post_proc_faces, 0.7f, false);
        cv::GArray<custom::Face> final_r_faces_for_bb2squares = custom::ApplyRegression::on(nms07_r_faces_total, true);
        cv::GArray<custom::Face> final_faces_rnet = custom::BBoxesToSquares::on(final_r_faces_for_bb2squares);

        /** Output part of MTCNN graph **/
        cv::GArray<cv::Rect> faces_roi_rnet = custom::R_O_NetPreProcGetROIs::on(final_faces_rnet, in_sz);
        cv::GArray<cv::GMat> regressionsONet, scoresONet, landmarksONet;
        std::tie(regressionsONet, landmarksONet, scoresONet) = cv::gapi::infer<nets::MTCNNOutput>(faces_roi_rnet, in_transposedRGB);

        /** Output post-processing **/
        cv::GArray<custom::Face> onet_post_proc_faces = custom::ONetPostProc::on(final_faces_rnet, scoresONet, regressionsONet, landmarksONet, float(FLAGS_th));
        cv::GArray<custom::Face> final_o_faces_for_nms07 = custom::ApplyRegression::on(onet_post_proc_faces, true);
        cv::GArray<custom::Face> nms07_o_faces_total = custom::RunNMS::on(final_o_faces_for_nms07, 0.7f, true);
        cv::GArray<custom::Face> final_faces_onet = custom::SwapFaces::on(nms07_o_faces_total);

        /** Draw ROI and marks **/
        auto rendered = cv::gapi::wip::draw::render3ch(in_original,
                                                       custom::BoxesAndMarks::on(in_original, final_faces_onet));

        cv::GComputation graph_mtcnn(cv::GIn(in_original), cv::GOut(rendered));
        /** ---------------- End of graph ---------------- **/
        /** Configure networks **/

        // MTCNN Refinement detection network
        auto mtcnnr_net = cv::gapi::ie::Params<nets::MTCNNRefinement> {
            FLAGS_m_r,                         // path to topology IR
            fileNameNoExt(FLAGS_m_r) + ".bin", // path to weights
            FLAGS_d_r,                         // device specifier
        }.cfgOutputLayers({ "conv5-2", "prob1" }).cfgInputLayers({ "data" });

        // MTCNN Output detection network
        auto mtcnno_net = cv::gapi::ie::Params<nets::MTCNNOutput> {
            FLAGS_m_o,                         // path to topology IR
            fileNameNoExt(FLAGS_m_o) + ".bin", // path to weights
            FLAGS_d_o,                         // device specifier
        }.cfgOutputLayers({ "conv6-2", "conv6-3", "prob1" }).cfgInputLayers({ "data" });

        auto networks_mtcnn = cv::gapi::networks(mtcnnr_net, mtcnno_net);

        // MTCNN Proposal detection network
        for (int i = 0; i < pyramid_levels; ++i) {
            std::string net_id = get_pnet_level_name(level_size[i]);
            std::vector<size_t> reshape_dims = { 1, 3, size_t(level_size[i].width), size_t(level_size[i].height) };
            cv::gapi::ie::Params<cv::gapi::Generic> mtcnnp_net {
                net_id,                            // tag
                FLAGS_m_p,                         // path to topology IR
                fileNameNoExt(FLAGS_m_p) + ".bin", // path to weights
                FLAGS_d_p,                         // device specifier
            };
            mtcnnp_net.cfgInputReshape("data", reshape_dims);
            networks_mtcnn += cv::gapi::networks(mtcnnp_net);
        }

        /** Custom kernels **/
        auto kernels_mtcnn = custom::kernels();
        auto mtcnn_args = cv::compile_args(networks_mtcnn, kernels_mtcnn);
        if (FLAGS_qc != 0) {
            mtcnn_args += cv::compile_args(cv::gapi::streaming::queue_capacity{ FLAGS_qc });
        }
        auto pipeline_mtcnn = graph_mtcnn.compileStreaming(std::move(mtcnn_args));

        /** ---------------- The execution part ---------------- **/
        pipeline_mtcnn.setSource<custom::CustomCapSource>(cap);

        cv::Size graphSize{static_cast<int>(frame_size.width / 4), 60};
            Presenter presenter(FLAGS_u, frame_size.height - graphSize.height - 10, graphSize);

        int frames = 0;

        /** Save output result **/
        cv::VideoWriter videoWriter;
        if (!FLAGS_o.empty() && !videoWriter.open(FLAGS_o, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                                  cap->fps(), frame_size)) {
            throw std::runtime_error("Can't open video writer");
        }

        /** Output Mat for result **/
        cv::Mat out_image;
        PerformanceMetrics metrics;
        bool isStart = true;
        const auto startTime = std::chrono::steady_clock::now();
        pipeline_mtcnn.start();
        while (pipeline_mtcnn.pull(cv::gout(out_image))) {
            if (isStart) {
                metrics.update(startTime, out_image, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX,
                    0.65, { 200, 10, 10 }, 2, PerformanceMetrics::MetricTypes::FPS);
                isStart = false;
            }
            else {
                metrics.update({}, out_image, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX,
                    0.65, { 200, 10, 10 }, 2, PerformanceMetrics::MetricTypes::FPS);
            }
            if (videoWriter.isOpened()) {
                videoWriter.write(out_image);
            }
            if (!FLAGS_no_show) {
                cv::imshow("Face detection mtcnn demo G-API", out_image);
                int key = cv::waitKey(1);
                presenter.handleKey(key);
            }
        }

        slog::info << "Metrics report:" << slog::endl;
        slog::info << "\tFPS: " << std::fixed << std::setprecision(1) << metrics.getTotal().fps << slog::endl;
        slog::info << presenter.reportMeans() << slog::endl;
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    return 0;
}
