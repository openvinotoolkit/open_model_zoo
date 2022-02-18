// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include <models/classification_model.h>
#include <models/results.h>
#include <pipelines/async_pipeline.h>
#include <pipelines/metadata.h>

#include <utils/args_helper.hpp>
#include <utils/common.hpp>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>

#include "grid_mat.hpp"

namespace{
constexpr char h_msg[] = "show the help message and exit";
DEFINE_bool(h, false, h_msg);

constexpr char i_msg[] = "an input to process. The input must be a single image, a folder of images";
DEFINE_string(i, "", i_msg);

constexpr char labels_msg[] = "path to .txt file with labels";
DEFINE_string(labels, "", labels_msg);

constexpr char m_msg[] = "path to an .xml file with a trained model";
DEFINE_string(m, "", m_msg);

constexpr char auto_resize_msg[] = "enables resizable input";
DEFINE_bool(auto_resize, false, auto_resize_msg);

constexpr char d_msg[] = "specify the target device to infer on. "
    "The list of available devices is shown below. The demo will look for a suitable plugin for device specified. "
    "Default value is CPU";
DEFINE_string(d, "CPU", d_msg);

constexpr char gt_msg[] = "path to ground truth .txt file";
DEFINE_string(gt, "", gt_msg);

constexpr char layout_msg[] = "specify inputs layouts."
    "Ex. \"[NCHW]\" or \"input1[NCHW],input2[NC]\" in case of more than one input";
DEFINE_string(layout, "", layout_msg);

constexpr char nireq_msg[] = "number of infer requests";
DEFINE_uint32(nireq, 0, nireq_msg);

constexpr char nstreams_msg[] = "specify count of streams";
DEFINE_string(nstreams, "", nstreams_msg);

constexpr char nt_msg[] = "number of top results. Must be >= 1. Default value is 5";
DEFINE_uint32(nt, 5, nt_msg);

constexpr char nthreads_msg[] = "specify count of threads";
DEFINE_uint32(nthreads, 0, nthreads_msg);

constexpr char res_msg[] = "set image grid resolution in format WxH. Default value is 1280x720";
DEFINE_string(res, "1280x720", res_msg);

constexpr char show_msg[] = "disable showing of processed images";
DEFINE_bool(show, true, show_msg);

constexpr char time_msg[] = "time in seconds to execute program. Default is -1 (infinite time)";
DEFINE_uint32(time, std::numeric_limits<gflags::uint32>::max(), time_msg);

constexpr char u_msg[] = "list of monitors to show initially";
DEFINE_string(u, "", u_msg);

void parse(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    if (FLAGS_h || 1 == argc) {
        std::cout <<   "\t[ -h]                 " << h_msg
                << "\n\t  -i <INPUT>            " << i_msg
                << "\n\t --labels <LABELS>     " << labels_msg
                << "\n\t  -m <MODEL FILE>       " << m_msg
                << "\n\t[--auto_resize]         " << res_msg
                << "\n\t[ -d <DEVICE>]          " << d_msg
                << "\n\t[--gt <STRING>]       " << gt_msg
                << "\n\t[--layout <STRING>] " << layout_msg
                << "\n\t[--nireq <NUMBER>]      " << nireq_msg
                << "\n\t[--nstreams <NUMBER>]   " << nstreams_msg
                << "\n\t[--nt <NUMBER>]         " << nt_msg
                << "\n\t[--nthreads <NUMBER>]   " << nthreads_msg
                << "\n\t[--res <STRING>]       " << res_msg
                << "\n\t[--show] ([--noshow])   " << show_msg
                << "\n\t[--time <NUMBER>]       " << time_msg
                << "\n\t[ -u]                   " << u_msg
                << "\n\tKey bindings:"
                    "\n\t\tQ, q, Esc - Quit"
                    "\n\t\tR, r, SpaceBar - Restart testing"
                    "\n\t\tC - average CPU load, D - load distrobution over cores, M - memory usage, H - hide\n";
        showAvailableDevices();
        slog::info << ov::get_openvino_version() << slog::endl;
        exit(0);
    } if (FLAGS_i.empty()) {
        throw std::invalid_argument("-i <INPUT> can't be empty");
    } if (FLAGS_m.empty()) {
        throw std::invalid_argument("-m <MODEL FILE> can't be empty");
    } if (FLAGS_labels.empty()) {
        throw std::logic_error("--labels <LABELS> can't be empty");
    }
    slog::info << ov::get_openvino_version() << slog::endl;
}

cv::Mat centerSquareCrop(const cv::Mat& image) {
    if (image.cols >= image.rows) {
        return image(cv::Rect((image.cols - image.rows) / 2, 0, image.rows, image.rows));
    }
    return image(cv::Rect(0, (image.rows - image.cols) / 2, image.cols, image.cols));
}
} // namespace

int main(int argc, char *argv[]) {
    std::set_terminate(catcher);
    parse(argc, argv);
    PerformanceMetrics metrics, readerMetrics, renderMetrics;

    //------------------------------- Preparing Input ------------------------------------------------------
    std::vector<std::string> imageNames;
    std::vector<cv::Mat> inputImages;
    parseInputFilesArguments(imageNames);
    if (imageNames.empty()) throw std::runtime_error("No images provided");
    std::sort(imageNames.begin(), imageNames.end());
    for (size_t i = 0; i < imageNames.size(); i++) {
        const std::string& name = imageNames[i];
        auto readingStart = std::chrono::steady_clock::now();
        const cv::Mat& tmpImage = cv::imread(name);
        if (tmpImage.data == nullptr) {
            slog::err << "Could not read image " << name << slog::endl;
            imageNames.erase(imageNames.begin() + i);
            i--;
        } else {
            readerMetrics.update(readingStart);
            // Clone cropped image to keep memory layout dense to enable -auto_resize
            inputImages.push_back(centerSquareCrop(tmpImage).clone());
            size_t lastSlashIdx = name.find_last_of("/\\");
            if (lastSlashIdx != std::string::npos) {
                imageNames[i] = name.substr(lastSlashIdx + 1);
            } else {
                imageNames[i] = name;
            }
        }
    }

    // ----------------------------------------Read image classes-----------------------------------------
    std::vector<unsigned> classIndices;
    if (!FLAGS_gt.empty()) {
        std::map<std::string, unsigned> classIndicesMap;
        std::ifstream inputGtFile(FLAGS_gt);
        if (!inputGtFile.is_open()) {
            throw std::runtime_error("Can't open the ground truth file.");
        }

        std::string line;
        while (std::getline(inputGtFile, line)) {
            size_t separatorIdx = line.find(' ');
            if (separatorIdx == std::string::npos) {
                throw std::runtime_error("The ground truth file has incorrect format.");
            }
            std::string imagePath = line.substr(0, separatorIdx);
            size_t imagePathEndIdx = imagePath.rfind('/');
            unsigned classIndex = static_cast<unsigned>(std::stoul(line.substr(separatorIdx + 1)));
            if ((imagePathEndIdx != 1 || imagePath[0] != '.') && imagePathEndIdx != std::string::npos) {
                throw std::runtime_error("The ground truth file has incorrect format.");
            }
            classIndicesMap.insert({imagePath.substr(imagePathEndIdx + 1), classIndex});
        }

        for (size_t i = 0; i < imageNames.size(); i++) {
            auto imageSearchResult = classIndicesMap.find(imageNames[i]);
            if (imageSearchResult != classIndicesMap.end()) {
                classIndices.push_back(imageSearchResult->second);
            } else {
                throw std::runtime_error("No class specified for image " + imageNames[i]);
            }
        }
    } else {
        classIndices.resize(inputImages.size());
        std::fill(classIndices.begin(), classIndices.end(), 0);
    }

    //------------------------------ Running routines ----------------------------------------------
    std::vector<std::string> labels = ClassificationModel::loadLabels(FLAGS_labels);
    for (const auto & classIndex : classIndices) {
        if (classIndex >= labels.size()) {
            throw std::runtime_error("Class index " + std::to_string(classIndex)
                                        + " is outside the range supported by the model.");
            }
    }

    slog::info << ov::get_openvino_version() << slog::endl;
    ov::Core core;

    AsyncPipeline pipeline(std::unique_ptr<ModelBase>(new ClassificationModel(FLAGS_m, FLAGS_nt, FLAGS_auto_resize, labels, FLAGS_layout)),
        ConfigFactory::getUserConfig(FLAGS_d, FLAGS_nireq, FLAGS_nstreams, FLAGS_nthreads), core);

    Presenter presenter(FLAGS_u, 0);
    int width;
    int height;
    std::vector<std::string> gridMatRowsCols = split(FLAGS_res, 'x');
    if (gridMatRowsCols.size() != 2) {
        throw std::runtime_error("The value of GridMat resolution flag is not valid.");
    } else {
        width = std::stoi(gridMatRowsCols[0]);
        height = std::stoi(gridMatRowsCols[1]);
    }
    GridMat gridMat(presenter, cv::Size(width, height));
    bool keepRunning = true;
    std::unique_ptr<ResultBase> result;
    double accuracy = 0;
    bool isTestMode = true;
    std::chrono::steady_clock::duration elapsedSeconds = std::chrono::steady_clock::duration(0);
    std::chrono::seconds testDuration = std::chrono::seconds(3);
    std::chrono::seconds fpsCalculationDuration = std::chrono::seconds(1);
    unsigned int framesNum = 0;
    long long correctPredictionsCount = 0;
    unsigned int framesNumOnCalculationStart = 0;
    std::size_t nextImageIndex = 0;
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    while (keepRunning && elapsedSeconds < std::chrono::seconds(FLAGS_time)) {
        if (elapsedSeconds >= testDuration - fpsCalculationDuration && framesNumOnCalculationStart == 0) {
            framesNumOnCalculationStart = framesNum;
        }
        if (isTestMode && elapsedSeconds >= testDuration) {
            isTestMode = false;
            typedef std::chrono::duration<double, std::chrono::seconds::period> Sec;
            gridMat = GridMat(presenter, cv::Size(width, height), cv::Size(16, 9),
                                (framesNum - framesNumOnCalculationStart) / std::chrono::duration_cast<Sec>(
                                fpsCalculationDuration).count());
            metrics = PerformanceMetrics();
            startTime = std::chrono::steady_clock::now();
            framesNum = 0;
            correctPredictionsCount = 0;
            accuracy = 0;
        }

        if (pipeline.isReadyToProcess()) {
            auto imageStartTime = std::chrono::steady_clock::now();

            pipeline.submitData(ImageInputData(inputImages[nextImageIndex]),
                std::make_shared<ClassificationImageMetaData>(inputImages[nextImageIndex], imageStartTime, classIndices[nextImageIndex]));
            nextImageIndex++;
            if (nextImageIndex == imageNames.size()) {
                nextImageIndex = 0;
            }
        }

        //--- Waiting for free input slot or output data available. Function will return immediately if any of them are available.
        pipeline.waitForData(false);

        //--- Checking for results and rendering data if it's ready
        while ((result = pipeline.getResult(false)) && keepRunning) {
            auto renderingStart = std::chrono::steady_clock::now();
            const ClassificationResult& classificationResult = result->asRef<ClassificationResult>();
            if (!classificationResult.metaData) {
                throw std::invalid_argument("Renderer: metadata is null");
            }
            const ClassificationImageMetaData& classificationImageMetaData
                = classificationResult.metaData->asRef<const ClassificationImageMetaData>();

            auto outputImg = classificationImageMetaData.img;

            if (outputImg.empty()) {
                throw std::invalid_argument("Renderer: image provided in metadata is empty");
            }
            PredictionResult predictionResult = PredictionResult::Incorrect;
            std::string label = classificationResult.topLabels.front().label;
            if (!FLAGS_gt.empty()) {
                for (size_t i = 0; i < FLAGS_nt; i++) {
                    unsigned predictedClass = classificationResult.topLabels[i].id;
                    if (predictedClass == classificationImageMetaData.groundTruthId) {
                        predictionResult = PredictionResult::Correct;
                        correctPredictionsCount++;
                        label = classificationResult.topLabels[i].label;
                        break;
                    }
                }
            } else {
                predictionResult = PredictionResult::Unknown;
            }
            framesNum++;
            gridMat.updateMat(outputImg, label, predictionResult);
            accuracy = static_cast<double>(correctPredictionsCount) / framesNum;
            gridMat.textUpdate(metrics, classificationResult.metaData->asRef<ImageMetaData>().timeStamp, accuracy, FLAGS_nt, isTestMode,
                                !FLAGS_gt.empty(), presenter);
            renderMetrics.update(renderingStart);
            elapsedSeconds = std::chrono::steady_clock::now() - startTime;
            if (FLAGS_show) {
                cv::imshow("classification_demo", gridMat.outImg);
                //--- Processing keyboard events
                int key = cv::waitKey(1);
                if (27 == key || 'q' == key || 'Q' == key) {  // Esc
                    keepRunning = false;
                }
                else if (32 == key || 'r' == key || 'R' == key) {  // press space or r to restart testing if needed
                    isTestMode = true;
                    framesNum = 0;
                    framesNumOnCalculationStart = 0;
                    correctPredictionsCount = 0;
                    accuracy = 0;
                    elapsedSeconds = std::chrono::steady_clock::duration(0);
                    startTime = std::chrono::steady_clock::now();
                }
                else {
                    presenter.handleKey(key);
                }
            }
        }
    }

    if (!FLAGS_gt.empty()) {
        slog::info << "Accuracy (top " << FLAGS_nt << "): " << accuracy << slog::endl;
    }

    slog::info << "Metrics report:" << slog::endl;
    metrics.logTotal();
    logLatencyPerStage(readerMetrics.getTotal().latency, pipeline.getPreprocessMetrics().getTotal().latency,
        pipeline.getInferenceMetircs().getTotal().latency, pipeline.getPostprocessMetrics().getTotal().latency,
        renderMetrics.getTotal().latency);
    slog::info << presenter.reportMeans() << slog::endl;

    return 0;
}
