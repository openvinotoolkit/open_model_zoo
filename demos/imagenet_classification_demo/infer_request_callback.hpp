#include <cstdio>
#include <string>
#include <functional>
#include <atomic>

#include <inference_engine.hpp>

#include <ie_iextension.h>

void resizeImage(cv::Mat& image, int modelInputResolution) {
    double scale = static_cast<double>(modelInputResolution) / std::min(image.cols, image.rows);
    cv::resize(image, image, cv::Size(), scale, scale);

    cv::Rect imgROI;
    if (image.cols >= image.rows) {
        int fromWidth = image.cols/2 - modelInputResolution/2;
        imgROI = cv::Rect(fromWidth,
                          0,
                          std::min(modelInputResolution, image.cols - fromWidth),
                          modelInputResolution);
    }
    else {
        int fromHeight = image.rows/2 - modelInputResolution/2;
        imgROI = cv::Rect(0,
                          fromHeight,
                          modelInputResolution,
                          std::min(modelInputResolution, image.rows - fromHeight));
    }
    image(imgROI).copyTo(image);
}

class InferRequestCallback
{
public:
    InferenceEngine::InferRequest& ir;
    unsigned modelInputResolution;
    mutable int firstIndex;

    std::mutex&  mutex;
    std::condition_variable& condVar;
    std::vector<cv::Mat>& inputImgs;
    std::string inputBlobName;
    unsigned batchSize;
    std::list<cv::Mat>& showMats;
    unsigned& curPos;
    unsigned& framesNum;
    std::atomic<bool>& quitFlag;

    InferRequestCallback(InferenceEngine::InferRequest& ir,
                         unsigned modelInputResolution,
                         int index, 
                         std::mutex& mutex, std::condition_variable& condVar,
                         std::vector<cv::Mat>& inputImgs,
                         const std::string& inputBlobName,
                         unsigned batchSize,
                         std::list<cv::Mat>& showMats,
                         unsigned& curPos,
                         unsigned& framesNum,
                         std::atomic<bool>& quitFlag
                         ):
                         ir(ir), modelInputResolution(modelInputResolution),
                         firstIndex(index),
                         mutex(mutex), condVar(condVar),
                         inputImgs(inputImgs),
                         inputBlobName(inputBlobName),
                         batchSize(batchSize),
                         showMats(showMats),
                         curPos(curPos),
                         framesNum(framesNum),
                         quitFlag(quitFlag) {}


    // curPos and framesNum are stored in ieWrapper
    void operator()() const {
        if (!quitFlag) {
            int inputDataSize = inputImgs.size();
            if (inputDataSize > 0) {
                std::lock_guard<std::mutex> lock(mutex);

                for (unsigned j = 0; j < batchSize; j++) {
                    cv::Mat& inputImg = inputImgs[(firstIndex + j) % inputDataSize];
                    resizeImage(inputImg, modelInputResolution);
                    showMats.push_back(inputImgs[(firstIndex + j) % inputDataSize]);
                }

                framesNum += batchSize;
                firstIndex = curPos;
                curPos = (curPos + batchSize) % inputDataSize;
            }
            condVar.notify_one();

            auto inputBlob = ir.GetBlob(inputBlobName);
        
            for (unsigned i = 0; i < batchSize; i++) {        
                cv::Mat& inputImg = inputImgs[(firstIndex + i) % inputDataSize];
                resizeImage(inputImg, modelInputResolution);
                matU8ToBlob<uint8_t>(inputImg, inputBlob, i);
            }

            ir.StartAsync();
        }
    }
};
