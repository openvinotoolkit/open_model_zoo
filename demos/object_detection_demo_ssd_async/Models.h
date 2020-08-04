#pragma once
#include <string>
#include <vector>
#include <map>
#include <ie_core.hpp>
#include <ie_cnn_network.h>
#include "object_detection_demo_ssd_async.hpp"
class model
{
private:
	InferenceEngine::CNNNetwork cnnNetwork;
	std::map<std::string, InferenceEngine::SizeVector > inputs;
	//std::map<const std::string&, Blob::Ptr> outputs;
	std vector<std::string> outputs;// возможно outputs надо переделать 
	//сделать lables полем
public:
	model(const Core& ie, std::string FLAGS_m){

		this->cnnNetwork = ie.ReadNetwork(FLAGS_m);
	}
	InferenceEngine::CNNNetwork get_cnnNetwork(){
	   
		return this->cnnNetwork;
	
	}
	

	
	virtual void PreparingInputBlobs() = 0;
	virtual void PreparingOutputBlobs() = 0;
	
	

	
};
