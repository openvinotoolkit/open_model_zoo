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
// brief This tutorial shows a baseline with OpenCV DNN 
*/

#include "main.h"
#include <string>
#include <chrono>
#include <numeric>
#include <vector>
#include <gflags/gflags.h>

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <common_labelinfo.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;



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

    if (FLAGS_i.empty() || FLAGS_m.empty() || FLAGS_weights.empty() || FLAGS_l.empty()) {
        if (FLAGS_l.empty())
            cout << "ERROR: labels file path not set" << endl;
        if (FLAGS_m.empty())
            cout << "ERROR: file with model - not set" << endl;
        if (FLAGS_weights.empty())
            cout << "ERROR: file with weights - not set" << endl;
        if (FLAGS_i.empty())
            cout << "ERROR: image(s) for inference - not set" << endl;

        showUsage();
        return 2;
    }

    bool bshow_less = FLAGS_s ? true : false;

    // this example only uses batch=1
    const int batchSize = 1;

    //----------------------------------------------------------------------------
    // Inference Initialization
    //----------------------------------------------------------------------------
    String modelConfiguration = FLAGS_m;
    String modelBinary = FLAGS_weights;

    cout << "reading prototxt:" << modelConfiguration << endl;
    cout << "reading weights:" << modelBinary << endl;
    dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);

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

    // SSD accepts 300x300 RGB-images
    const size_t infer_width   = 300;
    const size_t infer_height  = 300;

    int pipeline_type = 3;  // default. Input frame is resized to output dims as well as infer dims
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

    Mat frame, frameInfer;
    Mat* output_frames = new Mat[batchSize];

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
    for (;;) {
        double pre_stagesum = 0;
        double post_stagesum = 0;

        for (size_t mb = 0; mb < batchSize; mb++) {
            //---------------------------
            // get a new frame
            //---------------------------
            // [note] It might take long time to obtain the first frame, especially video, because of the first decoding operation.
            // It can effect to performance analysis accuracy. In this case, you can provide a long video or you can just skip the first frame from measurement.
            //---------------------------
            time1 = chrono::high_resolution_clock::now();
            cap.read(frame);

            totalFrames++;

            if (!frame.data) {
                no_more_data = true;
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

            Mat inputBlob = blobFromImage(frameInfer, 1.0f, Size(infer_width, infer_height), Scalar(), false);  // Convert Mat to batch of images
            net.setInput(inputBlob, "data");  // set the network input

            time2 = chrono::high_resolution_clock::now();
            diff = time2-time1;
            pre_stagesum += diff.count()*1000.0;
        }

        if (bshow_less) {
            if (FLAGS_fr > 0)
                cout << "Batch: " << nBatch << "/" << FLAGS_fr/batchSize << "\r";
            else
                cout << "Batch: " << nBatch << "\r";
        } else {
            if (FLAGS_fr > 0)
                cout << "Batch: " << nBatch+1 << "/" << FLAGS_fr/batchSize << endl;
            else
                cout << "Batch: " << nBatch+1 << endl;

            cout << "\tpre-stage:" << fixed << "\t" << setprecision(2) << setfill('0') << pre_stagesum/batchSize << " ms/frame" << endl;
        }

        pre_stage_times.push_back(pre_stagesum/batchSize);

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
        Mat detection = net.forward("detection_out");  // compute output
        time2 = chrono::high_resolution_clock::now();
        diff = time2-time1;
        infer_times.push_back(diff.count()*1000.0/batchSize);

        if (!bshow_less)
            cout << "\tinfer:" << fixed << "\t\t" << setprecision(2) << setfill('0') << diff.count()*1000.0/batchSize << " ms/frame" << endl;

        fflush(stdout);

        //---------------------------
        // POSTPROCESS STAGE:
        // parse output
        //---------------------------

        for (size_t mb = 0; mb < batchSize; mb++) {
            time1 = chrono::high_resolution_clock::now();
            Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

            //---------------------------
            // parse SSD output
            //---------------------------
            for (int c = 0; c < numlabels; c++) {
                int labelnum = (size_t)(detectionMat.at<float>(c, 1))-1;
                labelnum = (labelnum < 0) ? 0 : (labelnum > numlabels) ? numlabels : labelnum;

                float confidence = detectionMat.at<float>(c, 2);

                if ((confidence > FLAGS_thresh) && labeldata[labelnum].useclass) {
                    int xmin = static_cast<int>(detectionMat.at<float>(c, 3) * output_width);
                    int ymin = static_cast<int>(detectionMat.at<float>(c, 4) * output_height);
                    int xmax = static_cast<int>(detectionMat.at<float>(c, 5) * output_width);
                    int ymax = static_cast<int>(detectionMat.at<float>(c, 6) * output_height);

                    char tmplabel[32] = {0};
                    snprintf(tmplabel, sizeof(tmplabel), "%s %d%%", labeldata[labelnum].label.c_str(), static_cast<int>(confidence*100.0));

                    rectangle(output_frames[mb], Point(xmin, ymin-16), Point(xmin+strlen(tmplabel)*8, ymin), Scalar(155, 155, 155), FILLED, LINE_8, 0);
                    putText(output_frames[mb], tmplabel, Point(xmin, ymin-4), FONT_HERSHEY_PLAIN, .7, Scalar(0, 0, 0));
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
            post_stagesum += diff.count()*1000.0;
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

    pre_stage = accumulate(pre_stage_times.begin(), pre_stage_times.end(), 0.0)/pre_stage_times.size();
    infer_stage = accumulate(infer_times.begin(), infer_times.end(), 0.0)/infer_times.size();
    post_stage = accumulate(post_stage_times.begin(), post_stage_times.end(), 0.0)/post_stage_times.size();

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
    cout << "\tend2end_video_analytics_opencv [option]" << endl;
    cout << "\toptions:" << endl;
    cout << endl;
    cout << "\t\t-h              " << help_message << endl;
    cout << "\t\t-i <path>       " << image_message << endl;
    cout << "\t\t-fr <path>      " << frames_message << endl;
    cout << "\t\t-m <path>       " << model_message << endl;
    cout << "\t\t-weights <path> " << weights_message << endl;
    cout << "\t\t-l <path>       " << labels_message << endl;
    cout << "\t\t-thresh <val>   " << threshold_message << endl;
    cout << "\t\t-s              " << simple_output_message << endl << endl;
}
