#include <cstdio>
#include <string>
#include <functional>
#include <atomic>

#include <inference_engine.hpp>

#include <ie_iextension.h>

class  InferRequestCallback
{
public:
    InferenceEngine::InferRequest& ir;
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
                ir(ir), firstIndex(index),
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
                    showMats.push_back(inputImgs[(firstIndex + j) % inputDataSize]);
                }

                framesNum += batchSize;
                firstIndex = curPos;
                curPos = (curPos + batchSize) % inputDataSize;
            }
            condVar.notify_one();

            auto inputBlob = ir.GetBlob(inputBlobName);
        
            for (unsigned i = 0; i < batchSize; i++) {        
                cv::Mat inputImg = inputImgs.at((firstIndex + i) % inputDataSize);
                matU8ToBlob<uint8_t>(inputImg, inputBlob, i);
            }

            ir.StartAsync(); // start async for actual ir
        }
    }
};
