#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>


#include <ie_extension.h>
#include <ie_plugin_dispatcher.hpp>
#include <string>

using namespace cv;
using namespace cv::dnn;
using namespace InferenceEngine;

float confThreshold = 0.5;//*100%
int NUM_THREAD = 2;
bool SYNC = false;// SYNC - true, ASYNC - false

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
    std::vector<InferRequest> vRequest;
    std::vector<int> threadState(NUM_THREAD, 0);// 0 - ready to start, 1 - not ready
    std::vector<std::string>cam_names = {"cam1", "cam2"};
    std::string xmlPath = "/home/volskig/src/weights/face-detection-retail-0004/FP32/face-detection-retail-0004.xml";
    std::string binPath = "/home/volskig/src/weights/face-detection-retail-0004/FP32/face-detection-retail-0004.bin";

    std::map<std::string, cv::Mat> inputsMap1, inputsMap2, inputsMap3, inputsMap4, inputsMap5, inputsMap6,
                                                                        ieOutputsMap1, ieOutputsMap2, ieOutputsMap3, ieOutputsMap4, ieOutputsMap5, ieOutputsMap6;

    BlobMap inputBlobs1, inputBlobs2, inputBlobs3, inputBlobs4, inputBlobs5, inputBlobs6,
                        outBlobs1, outBlobs2, outBlobs3, outBlobs4, outBlobs5, outBlobs6;
    std::vector<BlobMap> IblobMap = {inputBlobs1, inputBlobs2, inputBlobs3, inputBlobs4, inputBlobs5, inputBlobs6};
    std::vector<BlobMap> OblobMap = {outBlobs1, outBlobs2, outBlobs3, outBlobs4, outBlobs5, outBlobs6};

    std::vector<VideoCapture> VCM;
    VideoCapture cap1(0);
    VideoCapture cap2(2);

    Mat ten1({1,3,300,300}, CV_32F),
        ten2({1,3,300,300}, CV_32F),
        ten3({1,3,300,300}, CV_32F),
        ten4({1,3,300,300}, CV_32F),
        ten5({1,3,300,300}, CV_32F),
        ten6({1,3,300,300}, CV_32F),
        ten7({1,3,300,300}, CV_32F),
        ten8({1,3,300,300}, CV_32F),
        ten9({1,3,300,300}, CV_32F),
        ten10({1,3,300,300}, CV_32F);
    std::vector<Mat> forTen = {ten1, ten2, ten3, ten4, ten5, ten6, ten7, ten8, ten9, ten10};

    Mat img1, img2, img3, img4, img5, img6, img7, img8, img9, img10;
    std::vector<Mat> forImg = {img1, img2, img3, img4, img5, img6, img7, img8, img9, img10};

    VCM.push_back(cap1);
    VCM.push_back(cap2);

    std::vector<int> state(VCM.size(), 0);

    CNNNetReader reader;
    reader.ReadNetwork(xmlPath);
    reader.ReadWeights(binPath);

    CNNNetwork net = reader.getNetwork();
    InferenceEnginePluginPtr enginePtr;
    InferencePlugin plugin;	
    ExecutableNetwork netExec;
				
    for(int nt = 0; nt < NUM_THREAD; ++nt)
    {
        try
        {
                auto dispatcher = InferenceEngine::PluginDispatcher({""});
                enginePtr = dispatcher.getPluginByDevice("CPU");

                IExtensionPtr extension = make_so_pointer<IExtension>("libcpu_extension.so");
                enginePtr->AddExtension(extension, 0);
                plugin = InferencePlugin(enginePtr);
                netExec = plugin.LoadNetwork(net, {});

                vRequest.push_back(netExec.CreateInferRequest());
        }
        catch (const std::exception& ex)
        {
                CV_Error(Error::StsAssert, format("Failed to initialize Inference Engine backend: %s", ex.what()));
        }

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
                    *ptr = 1;
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
        else
            for(int nt = 0; nt < NUM_THREAD; ++nt)
            {
                if(threadState[nt] == 0)
                {
                    for(unsigned int i = 0; i < state.size(); ++i)
                    {
                        if(state[i] == CAP_CAM_READY && nt == i)
                        {
                            VCM[nt].retrieve(forImg[nt]);
                            blobFromImage(forImg[nt], forTen[nt], 1, Size(300, 300));
                            threadState[nt] = 0;
                            vRequest[nt].StartAsync();
                        }
                    }
                }
            }
        VideoCapture::waitAny(VCM, state, -1);
        if(!SYNC)
        {
            for(int nt = 0; nt < NUM_THREAD; ++nt)
            {
                if(threadState[nt] == 1)
                {
                    threadState[nt] = 0;
                    postprocess(forImg[nt], forTen[nt]);
                    imshow(cam_names[nt], forImg[nt]);
                }
            }
        }

        if ((int)waitKey(10) == 27)
        {
            break;
        }
    }
    return 0;
}

