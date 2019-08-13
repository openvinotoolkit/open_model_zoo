#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <ie_extension.h>
#include <ie_plugin_dispatcher.hpp>
#include <string>

using namespace cv;
using namespace cv::dnn;
using namespace InferenceEngine;

float confThreshold = 0.5;//*100%
int NUM_THREAD = 2; // >=2 for SYNC == true
bool SYNC = false;// SYNC - true, ASYNC - false
bool ASYNC = !SYNC;

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

std::vector<Mat> GetVector(int sizev)
{
    std::vector<Mat> vectorr;
    for (int i =0; i < sizev; ++i)
    {
        vectorr.push_back(Mat({1,3,300,300}, CV_32F));
    }
    return vectorr;
}

template<typename T>
std::vector<T> GetVector(int sizev)
{
    std::vector<T> vectorr;
    for (int i = 0; i < sizev; ++i)
    {
        vectorr.push_back(T());
    }
    return vectorr;
}

int main(int argc, char* argv[])
{
    char* datapath_dir = get_cameras_list(); //export OPENCV_TEST_CAMERA_LIST=...
    std::vector<VideoCapture> VCM;
    int step = 0; std::string path;
    while(true)
    {
        if(datapath_dir[step] == ':' || datapath_dir[step] == '\0')
        {
            VCM.push_back(VideoCapture(path, CAP_V4L));
            path.clear();
            if(datapath_dir[step] != '\0')
                ++step;
        }
        if(datapath_dir[step] == '\0')
            break;
        path += datapath_dir[step];
        ++step;
    }
    std::vector<int> state(VCM.size(), 0);
    std::vector<InferRequest> vRequest;    
    std::vector<int> threadState(NUM_THREAD, 0);// 0 - ready to start, 1 - not ready
    std::vector<std::string>cam_names = {"cam1", "cam2", "cam3", "cam4", "cam5", "cam6", "cam7", "cam8", "cam9", "cam10"};
    std::vector<int> numCam(NUM_THREAD, -1);
    std::vector<BlobMap> IblobMap = GetVector<BlobMap>(NUM_THREAD);
    std::vector<BlobMap> OblobMap = GetVector<BlobMap>(NUM_THREAD);
    std::vector<Mat> forTen = GetVector(NUM_THREAD);
    std::vector<Mat> forImg = GetVector(NUM_THREAD);

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
        vRequest.push_back(netExec.CreateInferRequest());
        for (auto& it : net.getInputsInfo())
        {
                IblobMap[nt][it.first] = make_shared_blob<float>({Precision::FP32,  it.second->getTensorDesc().getDims(), Layout::ANY}, (float*)forTen[nt].data);
        }

        for (auto& it : net.getOutputsInfo())
        {
                OblobMap[nt][it.first] = make_shared_blob<float>({Precision::FP32, it.second->getTensorDesc().getDims(), Layout::ANY}, (float*)forTen[nt].data);
        }
        vRequest[nt].SetInput(IblobMap[nt]);
        vRequest[nt].SetOutput(OblobMap[nt]);
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
    VideoCapture::waitAny(VCM, state, -1);
    while(true)
    {
        if(SYNC)
        {
            for(unsigned int i = 0; i < state.size(); ++i)
            {
                if(state[i] == CAP_CAM_READY)
                {
                    VCM[i].retrieve(forImg[i]);
                    blobFromImage(forImg[i], forTen[i], 1, Size(300, 300));
                    vRequest[i].Infer();
                    postprocess(forImg[i], forTen[i]);
                    imshow(cam_names[i], forImg[i]);
                }
            }
        }
        if(ASYNC)
        {
            for(int nt = 0; nt < NUM_THREAD; ++nt)
            {
                if(threadState[nt] == REQ_WORK_FIN)
                {
                    postprocess(forImg[nt], forTen[nt]);
                    imshow(cam_names[numCam[nt]], forImg[nt]);
                    threadState[nt] = REQ_READY_TO_START;
                }
                if(threadState[nt] == REQ_READY_TO_START)
                {
                    for(unsigned int i = 0; i < state.size(); ++i)
                    {
                        if(state[i] == CAP_CAM_READY)
                        {
                            state[i] = CAP_CAM_NOT_READY;
                            VCM[i].retrieve(forImg[nt]);
                            numCam[nt] = i;
                            blobFromImage(forImg[nt], forTen[nt], 1, Size(300, 300));
                            threadState[nt] = REQ_WORK;
                            vRequest[nt].StartAsync();
                            break;
                        }
                    }
                }
            }
        }
        VideoCapture::waitAny(VCM, state, -1);

        if((int)waitKey(1) == 27)
        {           
            break;
        }
    }
    return 0;
}

