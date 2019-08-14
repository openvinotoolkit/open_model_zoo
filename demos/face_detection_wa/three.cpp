#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <ie_extension.h>
#include <ie_plugin_dispatcher.hpp>
#include <string>

using namespace cv;
using namespace cv::dnn;
using namespace InferenceEngine;

float confThreshold = 0.9;//*100%
int NUM_THREAD = 12; //
bool READ = false;// READ - true, WA - false
bool WA = !READ;

enum {
    REQ_READY_TO_START = 0,
    REQ_WORK_FIN = 1,
    REQ_WORK = 2
};

static char* get_cameras_list()
{
    return getenv("OPENCV_TEST_CAMERA_LIST");
}

void postprocess(Mat& frame, const Mat& outs)
{
    // Network produces output blob with a shape 1x1xNx7 where N is a number of
    // detections and an every detection is a vector of values
    // [batchId, classId, confidence, left, top, right, bottom]
    float* data = (float*)outs.data;
    for (size_t i = 0; i < outs.total(); i += 7)
    {
        float confidence = data[i + 2];
        if (confidence > confThreshold)
        {
            int left   = (int)(data[i + 3] * frame.cols);
            int top    = (int)(data[i + 4] * frame.rows);
            int right  = (int)(data[i + 5] * frame.cols);
            int bottom = (int)(data[i + 6] * frame.rows);
            rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));
        }
    }
}

int main(int argc, char* argv[])
{
    char* datapath_dir = get_cameras_list(); //export OPENCV_TEST_CAMERA_LIST=...
    std::vector<VideoCapture> cameras;
    int step = 0; std::string path;
    while(true)
    {
        if(datapath_dir[step] == ':' || datapath_dir[step] == '\0')
        {
            cameras.emplace_back(VideoCapture(path, CAP_V4L));
            cameras.back().set(CAP_PROP_FRAME_WIDTH, 640);
            cameras.back().set(CAP_PROP_FRAME_HEIGHT, 480);
            path.clear();
            if(datapath_dir[step] != '\0')
                ++step;
        }
        if(datapath_dir[step] == '\0')
            break;
        path += datapath_dir[step];
        ++step;
    }
    std::vector<InferRequest> vRequest;
    std::vector<int> state(cameras.size(), 0);
    std::vector<int> threadState(NUM_THREAD, 0);// 0 - ready to start, 1 - not ready, 2 - work
    std::vector<int> numCam(NUM_THREAD, -1);
    std::vector<std::string>cam_names;
    std::vector<BlobMap> inpBlobMap(NUM_THREAD);
    std::vector<BlobMap> outBlobMap(NUM_THREAD);
    std::vector<Mat> forImg(NUM_THREAD);
    std::vector<Mat> inpTen, outTen;

    std::string xmlPath = "/home/volskig/src/weights/face-detection-retail-0004/FP32/face-detection-retail-0004.xml";
    std::string binPath = "/home/volskig/src/weights/face-detection-retail-0004/FP32/face-detection-retail-0004.bin";
    CNNNetReader reader;
    reader.ReadNetwork(xmlPath);
    reader.ReadWeights(binPath);

    CNNNetwork net = reader.getNetwork();
    InferenceEnginePluginPtr enginePtr;
    InferencePlugin plugin;
    ExecutableNetwork netExec;
    try
    {
        auto dispatcher = InferenceEngine::PluginDispatcher({""});
        enginePtr = dispatcher.getPluginByDevice("CPU");
        IExtensionPtr extension = make_so_pointer<IExtension>("libcpu_extension.so");
        enginePtr->AddExtension(extension, 0);
        plugin = InferencePlugin(enginePtr);
        netExec = plugin.LoadNetwork(net, {});
    }
    catch (const std::exception& ex)
    {
        CV_Error(Error::StsAssert, format("Failed to initialize Inference Engine backend: %s", ex.what()));
    }

    for(int nt = 0; nt < NUM_THREAD; ++nt)
    {
        inpTen.emplace_back(Mat({1,3,300,300}, CV_32F));
        outTen.emplace_back(Mat({1,1,200,7}, CV_32F));
        cam_names.emplace_back("Camera " + std::to_string(nt + 1));
        inpBlobMap[nt]["data"] = make_shared_blob<float>({Precision::FP32,  {1,3,300,300}, Layout::ANY}, (float*)inpTen[nt].data);
        outBlobMap[nt]["detection_out"] = make_shared_blob<float>({Precision::FP32, {1,1,200,7}, Layout::ANY}, (float*)outTen[nt].data);

        vRequest.emplace_back(netExec.CreateInferRequest());
        vRequest[nt].SetInput(inpBlobMap[nt]);
        vRequest[nt].SetOutput(outBlobMap[nt]);
        InferenceEngine::IInferRequest::Ptr infRequestPtr = vRequest[nt];
        int* pmsg = &threadState[nt];
        infRequestPtr->SetUserData(pmsg, 0);
        infRequestPtr->SetCompletionCallback(
            [](InferenceEngine::IInferRequest::Ptr reqst, InferenceEngine::StatusCode code)
            {
                int* ptr;
                reqst->GetUserData((void**)&ptr, 0);
                *ptr = REQ_WORK_FIN;
            });
    }    
    int start_frame = 100;
    bool flag = false;
    int NUM_CAM = 0;
    TickMeter tm; int frame_count = 0;
    std::vector<int> next_frame(cameras.size(), 0);
    std::vector<int> thread_frame(NUM_THREAD, 0);
    std::vector<int> launch_frame(cameras.size(), 0);
    if(READ)//.read() method
    {         
        while(true)
        {
            if(frame_count > start_frame && !flag)
            {
                tm.start();
                flag = true;
            }

            for(int nt = 0; nt < NUM_THREAD; ++nt)
            {
                if(threadState[nt] == REQ_WORK_FIN && thread_frame[nt] == next_frame[numCam[nt]] + 1)
                {
                     postprocess(forImg[nt], outTen[nt]);
                     imshow(cam_names[numCam[nt]], forImg[nt]);                     
                     threadState[nt] = REQ_READY_TO_START;
                     ++next_frame[numCam[nt]];
                     ++frame_count;
                }
                if(threadState[nt] == REQ_READY_TO_START)
                {
                    std::cout << nt + 1 << std::endl;
                    cameras[NUM_CAM].read(forImg[nt]);
                    blobFromImage(forImg[nt], inpTen[nt], 1, Size(300, 300));
                    numCam[nt] = NUM_CAM;
                    thread_frame[nt] = ++launch_frame[NUM_CAM];
                    threadState[nt] = REQ_WORK;
                    vRequest[nt].StartAsync();
                    break;
                }
            }
            NUM_CAM = (NUM_CAM + 1) % cameras.size();
            if((int)waitKey(1) == 27)
            {
                break;
            }
        }
        tm.stop();
        std::cout << (frame_count - start_frame) / tm.getTimeSec() << std::endl;
    }

    if(WA)//waitAny() method
    {        
        VideoCapture::waitAny(cameras, state, -1);
        while(true)
        {
            if(frame_count > start_frame && !flag)
            {
                tm.start();
                flag = true;
            }
            for(int nt = 0; nt < NUM_THREAD; ++nt)
            {
                if(threadState[nt] == REQ_WORK_FIN  && thread_frame[nt] == next_frame[numCam[nt]] + 1)
                {
                    postprocess(forImg[nt], outTen[nt]);
                    imshow(cam_names[numCam[nt]], forImg[nt]);
                    threadState[nt] = REQ_READY_TO_START;
                    ++next_frame[numCam[nt]];
                    ++frame_count;
                }
                if(threadState[nt] == REQ_READY_TO_START)
                {
                    for(unsigned int i = 0; i < state.size(); ++i)
                    {
                        if(state[i] == CAP_CAM_READY)
                        {                            
                            state[i] = CAP_CAM_NOT_READY;
                            cameras[i].retrieve(forImg[nt]);
                            numCam[nt] = i;
                            thread_frame[nt] = ++launch_frame[i];
                            blobFromImage(forImg[nt], inpTen[nt], 1, Size(300, 300));
                            threadState[nt] = REQ_WORK;
                            vRequest[nt].StartAsync();
                            break;
                        }
                    }
                }
            }

            VideoCapture::waitAny(cameras, state, -1);
            if((int)waitKey(1) == 27)
            {
                break;
            }
        }
        tm.stop();
        std::cout << (frame_count - start_frame) / tm.getTimeSec() << std::endl;
    }

    for(int nt = 0; nt < NUM_THREAD; ++nt)
    {
        vRequest[nt].Wait(InferenceEngine::IInferRequest::RESULT_READY);
    }
    return 0;
}

