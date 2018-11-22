/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

/*
// brief This tutorial replaces OpenCV DNN with IE
*/

#include "main.h"
#include <string>
#include <chrono>
#include <memory>
#include <vector>
#include <gflags/gflags.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

#include <inference_engine.hpp>
#include <cpp/ie_plugin_cpp.hpp>

#include <ext_list.hpp>

#include <common.hpp>
#include <common_labelinfo.hpp>

using namespace std;
using namespace cv;
using namespace InferenceEngine::details;
using namespace InferenceEngine;

/**
 * \brief shows command line parameter usage
 */
void showUsage();

/**
 * \brief The main function of inference engine demo application
 * @param argc - The number of arguments
 * @param argv - Arguments
 * @return 0 if all good
 */
int main(int argc, char *argv[]) {
    // ----------------
    // Parse command line parameters
    // ----------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);

    if (FLAGS_h) {
        showUsage();
        return 1;
    }

    if (FLAGS_l.empty()) {
        cout << "ERROR: labels file path not set" << endl;
        showUsage();
        return 1;
    }

    bool bshow_less = FLAGS_s ? true : false;

    // ----------------
    // Check inputs
    // ----------------
    bool noPluginAndBadDevice = FLAGS_p.empty() && FLAGS_d.compare("CPU") && FLAGS_d.compare("GPU") && FLAGS_d.compare("MYRIAD");

    if (FLAGS_i.empty() || FLAGS_m.empty() || noPluginAndBadDevice) {
        if (noPluginAndBadDevice)
            cout << "ERROR: device is not supported" << endl;
        if (FLAGS_m.empty())
            cout << "ERROR: file with model - not set" << endl;
        if (FLAGS_i.empty())
            cout << "ERROR: image(s) for inference - not set" << endl;
        showUsage();
        return 2;
    }

    //----------------------------------------------------------------------------
    // inference engine initialization
    //----------------------------------------------------------------------------

    // -----------------
    // Load plugin
    // -----------------

    InferenceEngine::PluginDispatcher dispatcher({"../../../lib/intel64", ""});
    InferencePlugin plugin(dispatcher.getPluginByDevice(FLAGS_d));

    /** Loading default extensions **/
    if (FLAGS_d.find("CPU") != std::string::npos) {
        /**
         * cpu_extensions library is compiled from "extension" folder containing
         * custom MKLDNNPlugin layer implementations. These layers are not supported
         * by mkldnn, but they can be useful for inferring custom topologies.
        **/
        plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
    }

    // loading layer extension plugin - from Gold release, not supported in R3
     if (!FLAGS_e.empty()) {
        if (ifstream(FLAGS_e.c_str())) {
            cout << "\nFound layer extension plugin: " << FLAGS_e.c_str() << "\n";
            auto extension_ptr = make_so_pointer<InferenceEngine::IExtension>(FLAGS_e.c_str());
            plugin.AddExtension(extension_ptr);
        } else {
            cout << "\nCan't find a layer extension plugin, so operation may be failure or ok with legacy layer support." << endl;
        }
    }

    cout << endl << "\t== Inference Engine Plugin Information ==" << endl;
    printPluginVersion(plugin, cout);
    cout << endl;

    // ----------------
    // Enable performance counters
    // ----------------
    if (FLAGS_pc) {
        plugin.SetConfig({{ PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } });
    }

    // ----------------
    // Read network
    // ----------------
    InferenceEngine::CNNNetReader networkReader;
    string modelfile(FLAGS_m);
    try {
        networkReader.ReadNetwork(modelfile);
    }
    catch (InferenceEngineException ex) {
        cerr << "Failed to load network: "  << endl;
        return 1;
    }

    cout << "Network loaded." << endl;
    auto pos = modelfile.rfind('.');
    if (pos != string::npos) {
        string binFileName = modelfile.substr(0, pos)+".bin";
        networkReader.ReadWeights(binFileName.c_str());
    } else {
        cerr << "Failed to load weights: " << endl;
        return 1;
    }

    auto network = networkReader.getNetwork();

    // --------------------
    // Set batch size
    // --------------------
    network.setBatchSize(FLAGS_batch);
    size_t batchSize = network.getBatchSize();

    cout << "Batch size = " << batchSize << endl;

    //----------------------------------------------------------------------------
    //  Inference engine input setup
    //----------------------------------------------------------------------------

    cout << "Setting-up input, output blobs..." << endl;

    // ---------------
    // set input configuration
    // ---------------

    InferenceEngine::InputsDataMap input_info(network.getInputsInfo());
    InferenceEngine::SizeVector inputDims;

    if (input_info.size() != 1) {
        cout << "This demo accepts networks having only one input." << endl;
        return 1;
    }

    for (auto &item : input_info) {
        auto input_data = item.second;
        input_data->setPrecision(Precision::FP32);
        input_data->setLayout(Layout::NCHW);
        inputDims = input_data->getDims();
    }
    cout << "inputDims=";
    for (int i = 0; i < inputDims.size(); i++) {
        cout << static_cast<int>(inputDims[i]) << " ";
    }
    cout << endl;

    const int infer_width = inputDims[0];
    const int infer_height = inputDims[1];
    const int num_channels = inputDims[2];
    const int channel_size = infer_width*infer_height;
    const int full_image_size = channel_size*num_channels;

    /** Get information about topology outputs **/
    InferenceEngine::OutputsDataMap output_info(network.getOutputsInfo());
    InferenceEngine::SizeVector outputDims;
    for (auto &item : output_info) {
        auto output_data = item.second;
        output_data->setPrecision(Precision::FP32);
        output_data->setLayout(Layout::NCHW);
        outputDims = output_data->getDims();
    }
    cout << "outputDims=";
    for (int i = 0; i < outputDims.size(); i++) {
        cout << static_cast<int>(outputDims[i]) << " ";
    }
    cout << endl;

    const int output_data_size = outputDims[1]*outputDims[2]*outputDims[3];

    // --------------------------------------------------------------------------
    // Load model into plugin
    // --------------------------------------------------------------------------
    cout << "Loading model to plugin..." << endl;

    auto executable_network = plugin.LoadNetwork(network, {});

    // --------------------------------------------------------------------------
    // Create infer request
    // --------------------------------------------------------------------------
    cout << "Create infer request..." << endl;

    auto async_infer_request = executable_network.CreateInferRequest();

    //----------------------------------------------------------------------------
    //  Inference engine output setup
    //----------------------------------------------------------------------------

    Mat frame, frameInfer;

    // get the input blob buffer pointer location
    float *input_buffer = NULL;
    float *input_buffer_current_image = NULL;
    for (auto &item : input_info) {
        auto input_name = item.first;
        auto input_data = item.second;
        auto input = async_infer_request.GetBlob(input_name);
        input_buffer = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
    }

    // get the output blob pointer location
    float *output_buffer = NULL;
    float *output_buffer_current_image = NULL;
    for (auto &item : output_info) {
        auto output_name = item.first;
        auto output = async_infer_request.GetBlob(output_name);
        output_buffer = output->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
    }


    //----------------------------------------------------------------------------
    // General Initialization
    // video inputs/outputs, etc.
    //----------------------------------------------------------------------------

    string input_filename = FLAGS_i;

    bool bSingleImageMode = false;

    vector<string> imgExt = { ".bmp", ".jpg" };

    for (size_t i = 0; i < imgExt.size(); i++) {
        if (input_filename.rfind(imgExt[i]) != string::npos) {
            bSingleImageMode = true;
            network.setBatchSize(1);
            break;
        }
    }

    // open video capture
    VideoCapture cap(FLAGS_i.c_str());
    if (!cap.isOpened()) {  // check if VideoCapture init successful
        cout << "Could not open input file" << endl;
        return 1;
    }

    const size_t input_width   = cap.get(CAP_PROP_FRAME_WIDTH);
    const size_t input_height  = cap.get(CAP_PROP_FRAME_HEIGHT);
    const size_t output_width  = FLAGS_output_w;
    const size_t output_height = FLAGS_output_h;

    int pipeline_type = 3;  // default.  Input frame is resized to output dims as well as infer dims
    if ((output_width == infer_width) && (output_height == infer_height)) pipeline_type = 1;  // Input frame is resized to infer resolution
    else if ((output_width == input_width) && (output_height == input_height)) pipeline_type = 2;  // output is input resolution

    // open video output
    Size outputFrameSize(output_width, output_height);

    VideoWriter video;
    if (!bSingleImageMode) {
        video.open("out.h264", VideoWriter::fourcc('H', '2', '6', '4'), 24, outputFrameSize, true);
        cout << "video mode: output written to out.h264" << endl;
    }

    // ----------------
    // Read class names
    // ----------------
    std::vector<labelinfo> labeldata = readlabels(FLAGS_l, FLAGS_useclasses);
    int numlabels = labeldata.size();
    for (int i = 0; i < labeldata.size(); i++) {
        cout << labeldata[i].label.substr(0, 7) << "\t\t";
        if (labeldata[i].useclass == 1)
            cout << "used ";
        else
            cout << "not used ";

        cout << endl;
    }

    //----------------------------------------------------------------------------
    //---------------------------
    // main loop starts here
    //---------------------------
    //----------------------------------------------------------------------------
    Mat* output_frames = new Mat[batchSize];

    auto input_channels = inputDims[2];  // channels for color format.  RGB=4
    auto input_size = channel_size * input_channels;
    bool no_more_data = false;

    int totalFrames = 0;

    cout << "Running inference...\n\n";

    chrono::high_resolution_clock::time_point tmStart, tmEnd;
    chrono::high_resolution_clock::time_point time1, time2;
    chrono::duration<double> diff;

    vector<double> pre_stage_times;
    vector<double> infer_times;
    vector<double> post_stage_times;

    tmStart = chrono::high_resolution_clock::now();

    int nBatch = 0;
    int frames_in_output = batchSize;

    for (;;) {
        double pre_stagesum = 0;
        double post_stagesum = 0;

        for (size_t mb = 0; mb < batchSize; mb++) {
            input_buffer_current_image = input_buffer+(full_image_size*mb);

            //---------------------------
            // get a new frame
            //---------------------------
            // [note] It might take long time to obtain the first frame, especially video, because of the first decoding operation.
            // It might effect to performance analysis accuracy. In this case, you can provide a long video or you can skip the first frame from measurement.
            //---------------------------
            time1 = chrono::high_resolution_clock::now();
            cap.read(frame);

            totalFrames++;

            if (!frame.data) {
                no_more_data = true;
                frames_in_output = mb;
                break;  // frame input ended
            }

            //---------------------------
            // resize to expected size (in model .xml file)
            //---------------------------

            switch (pipeline_type) {
            case 1:  // Input frame is resized to infer resolution
                resize(frame, output_frames[mb], Size(infer_width, infer_height));
                frameInfer = output_frames[mb];
                break;
            case 2:  // output is input resolution
                output_frames[mb] = frame;
                resize(frame, frameInfer, Size(infer_width, infer_height));
                break;
            default:  // other cases -- resize for output and infer
                resize(frame, output_frames[mb], Size(output_width, output_height));
                resize(frame, frameInfer, Size(infer_width, infer_height));
            }

            //---------------------------
            // PREPROCESS STAGE:
            // convert image to format expected by inference engine
            // IE expects planar, convert from packed
            //---------------------------
            size_t framesize = frameInfer.rows * frameInfer.step1();

            if (framesize != input_size) {
                cout << "input pixels mismatch, expecting " << input_size
                          << " bytes, got: " << framesize << endl;
                return 1;
            }

            /** Fill input tensor with planes. First b channel, then g and r channels **/
            for (size_t pixelnum = 0, imgIdx = 0; pixelnum < channel_size; ++pixelnum) {
                for (size_t ch = 0; ch < num_channels; ++ch) {
                    input_buffer_current_image[(ch * channel_size) + pixelnum] = frameInfer.at<cv::Vec3b>(pixelnum)[ch];
                }
            }

            time2 = chrono::high_resolution_clock::now();
            diff = time2-time1;
            pre_stagesum += diff.count()*1000.0;
        }

        if (bshow_less) {
            if (FLAGS_fr > 0)
                cout << "Batch: " << nBatch << "/" << FLAGS_fr/frames_in_output << "\r";
            else
                cout << "Batch: " << nBatch << "\r";
        } else {
            if (FLAGS_fr > 0)
                cout << "Batch: " << nBatch+1 << "/" << FLAGS_fr/frames_in_output << endl;
            else
                cout << "Batch: " << nBatch+1 << endl;

            cout << "\tpre-stage:" << fixed << "\t" << setprecision(2) << setfill('0') << pre_stagesum/frames_in_output << " ms/frame" << endl;
        }

        pre_stage_times.push_back(pre_stagesum/frames_in_output);
        fflush(stdout);

        if (no_more_data) {
            break;
        }

        if (FLAGS_fr > 0 && totalFrames > FLAGS_fr) {
            break;
        }

        //---------------------------
        // INFER STAGE
        //---------------------------
        time1 = chrono::high_resolution_clock::now();

        async_infer_request.StartAsync();
        async_infer_request.Wait(IInferRequest::WaitMode::RESULT_READY);

        time2 = chrono::high_resolution_clock::now();
        diff = time2-time1;
        infer_times.push_back(diff.count()*1000.0/frames_in_output);

        if (!bshow_less)
            cout << "\tinfer:" << fixed << "\t\t" << setprecision(2) << setfill('0') << diff.count()*1000.0/frames_in_output << " ms/frame" << endl;

        fflush(stdout);

        //---------------------------
        // Read performance counters
        //---------------------------

        if (FLAGS_pc) {
            printPerformanceCounts(plugin, cout);
        }

        //---------------------------
        // POSTPROCESS STAGE:
        // parse output
        //---------------------------
        for (size_t mb = 0; mb < frames_in_output; mb++) {
            time1 = chrono::high_resolution_clock::now();
            output_buffer_current_image = output_buffer+(output_data_size*mb);

            int maxProposalCount = outputDims[2];

            //---------------------------
            // parse SSD output
            //---------------------------
            for (int c = 0; c < maxProposalCount; c++) {
                float *localbox = &output_buffer_current_image[c * 7];
                float image_id   = localbox[0];
                float locallabel = localbox[1] - 1;
                float confidence = localbox[2];

                int labelnum = static_cast<int>(locallabel);
                labelnum = (labelnum < 0)?0:(labelnum > numlabels)?numlabels:labelnum;
                if ((confidence > FLAGS_thresh) && labeldata[labelnum].useclass) {
                    int xmin = static_cast<int>(localbox[3] * output_width);
                    int ymin = static_cast<int>(localbox[4] * output_height);
                    int xmax = static_cast<int>(localbox[5] * output_width);
                    int ymax = static_cast<int>(localbox[6] * output_height);

                    char tmplabel[32] = {0};
                    snprintf(tmplabel, sizeof(tmplabel), "%s %d%%", labeldata[labelnum].label.c_str(), static_cast<int>(confidence*100.0));

                    rectangle(output_frames[mb], Point(xmin, ymin-16), Point(xmin + strlen(tmplabel)*8, ymin), Scalar(155, 155, 155), FILLED, LINE_8, 0);
                    putText(output_frames[mb], tmplabel, Point(xmin, ymin - 4), FONT_HERSHEY_PLAIN, .7, Scalar(0, 0, 0));
                    rectangle(output_frames[mb], Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 255, 0), 4, LINE_AA, 0);
                }
            }
        }

        //---------------------------
        // output/render
        //---------------------------

        for (int mb = 0; mb < batchSize; mb++) {
            if (bSingleImageMode) {
                imwrite("out.jpg", output_frames[mb]);
            } else {
                video.write(output_frames[mb]);
            }

            time2 = chrono::high_resolution_clock::now();
            diff = time2-time1;
            post_stagesum+=diff.count()*1000.0;
        }
        post_stage_times.push_back(post_stagesum/batchSize);

        if (!bshow_less)
            cout << "\tpost-stage:" << fixed << "\t" << setprecision(2) << setfill('0') << post_stagesum/batchSize << " ms/frame" << endl << endl;

        fflush(stdout);

        nBatch++;
        if (bSingleImageMode) break;
    }

    tmEnd = chrono::high_resolution_clock::now();
    diff = tmEnd - tmStart;
    double total_elapsed_time = (diff.count()*1000.0)/1000.0;

    double pre_stage;   // reading bitstream file, decoding, resizing, color converting
    double infer_stage;
    double post_stage;  // drawing box, encoding, writing

    if (bSingleImageMode) {
        pre_stage = accumulate(pre_stage_times.begin(), pre_stage_times.end(), 0.0) / pre_stage_times.size();
        infer_stage = accumulate(infer_times.begin(), infer_times.end(), 0.0) / infer_times.size();
    } else {
        if (pre_stage_times.size() > 1) {
            // to prevent mis-analysis from first loading time, remove maximum time from calculation
            // most likely, first time measurement is relatively huge.
            // substract max from sum and divide with list size - 1
            vector<double>::iterator it_max;
            it_max = max_element(pre_stage_times.begin(), pre_stage_times.end());  // find max

            pre_stage = (accumulate(pre_stage_times.begin(), pre_stage_times.end(), 0.0) - *it_max) / (pre_stage_times.size() - 1);
        } else {
            pre_stage = accumulate(pre_stage_times.begin(), pre_stage_times.end(), 0.0) / pre_stage_times.size();
        }

        if (infer_times.size() > 1) {
            // to prevent mis-analysis from first model loading time, remove maximum time from calculation
            // most likely, first time measurement is relatively huge.
            // substract max from sum and divide with list size - 1
            vector<double>::iterator it_max;
            it_max = max_element(infer_times.begin(), infer_times.end());  // find max

            infer_stage = (accumulate(infer_times.begin(), infer_times.end(), 0.0) - *it_max) / (infer_times.size() - 1);
        } else {
            infer_stage = accumulate(infer_times.begin(), infer_times.end(), 0.0) / infer_times.size();
        }
    }

    post_stage = accumulate(post_stage_times.begin(), post_stage_times.end(), 0.0) / post_stage_times.size();

    totalFrames -= 1;

    cout << "\n\n";
    cout << "> Pre-stage average:" << fixed << "\t" << setprecision(2) << setfill('0') << pre_stage
        << " ms/frame (decoding, color converting, resizing)" << endl;
    cout << "> Infer average:" << fixed << "\t" << setprecision(2) << infer_stage << " ms/frame (inferencing)" << endl;
    cout << "> Post-stage average:" << fixed << "\t" << setprecision(2) << post_stage
        << " ms/frame (drawing bounding box, encoding, saving)" << endl;

    cout << endl << "> Total elapsed execution time: " << total_elapsed_time << " sec\n\n";

    delete [] output_frames;

    cout << "Done!" << endl << endl;

    return 0;
}

void showUsage() {
    cout << endl;
    cout << "[usage]" << endl;
    cout << "\tend2end_video_analytics_ie [option]" << endl;
    cout << "\toptions:" << endl;
    cout << endl;
    cout << "\t\t-h                  " << help_message << endl;
    cout << "\t\t-i <path/filename>  " << image_message << endl;
    cout << "\t\t-fr <val>           " << frames_message << endl;
    cout << "\t\t-m <path/filename>  " << model_message << endl;
    cout << "\t\t-l <path/filename>  " << labels_message << endl;
    cout << "\t\t-d <device>         " << target_device_message << endl;
    cout << "\t\t-pc                 " << performance_counter_message << endl;
    cout << "\t\t-thresh <val>       " << threshold_message << endl;
    cout << "\t\t-batch <val>        " << batch_message << endl;
    cout << "\t\t-s                  " << simple_output_message << endl;
    cout << "\t\t-e <path/filename>  " << layer_extension_plugin_message << endl;
    cout << "\t\t-mean               " << mean_message << endl;
    cout << "\t\t-scale              " << scale_message << endl << endl;
}
