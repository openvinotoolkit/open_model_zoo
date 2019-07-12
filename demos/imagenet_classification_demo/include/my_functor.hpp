#include <cstdio>
#include <string>
#include <functional>

#include <inference_engine.hpp>

#include <ie_iextension.h>
#include <ext_list.hpp>

#include "captures.hpp"

class  infReqCallback
{
    public:
    InferenceEngine::InferRequest& ir;
    mutable int firstIndex;

    std::mutex&  mutex;
    std::condition_variable& condVar;
    Captures& captures;
    std::queue<cv::Mat>& showMats;

    infReqCallback(InferenceEngine::InferRequest& ir, 
                std::mutex& mutex, std::condition_variable& condVar,
                Captures& captures,
                std::queue<cv::Mat>& showMats,
                int index
                ):
                ir(ir), firstIndex(index),
                mutex(mutex), condVar(condVar),
                captures(captures),
                showMats(showMats) {}


    //curPos and framesNum is stored in ieWrapper
    void operator()() const{
        if(!captures.quitFlag) {
            int batchSize = captures.batchSize;
            std::vector<cv::Mat>& inputImgs = captures.inputImgs;
            int inputDataSize = captures.inputImgs.size();
            {
                std::lock_guard<std::mutex> lock(mutex);

                for(int j = 0; j < batchSize; j++)
                    showMats.push(inputImgs[(firstIndex+j)%inputDataSize]);

                //sumTime += lastInferTime = cv::getTickCount() - startTime; // >:-/
                captures.framesNum += batchSize;
                firstIndex = captures.curPos;
                captures.curPos = (captures.curPos + batchSize) % inputDataSize;
            }
            condVar.notify_one();

            auto inputBlob = ir.GetBlob(captures.inputBlobName);
        
            for(int i = 0; i < batchSize; i++) {        
                cv::Mat inputImg = inputImgs.at((firstIndex + i) % inputDataSize);
                matU8ToBlob<uint8_t>(inputImg, inputBlob, i);
            }
            
            //startTime = cv::getTickCount();

            //start async for actual ir
            ir.StartAsync();
        }
    }
};
