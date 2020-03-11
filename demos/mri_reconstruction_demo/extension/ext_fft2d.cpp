/*
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

// ===============================================================================
// Generated file for Inference Engine extension for CPU plugin
//
// IMPLEMENT YOUR KERNEL HERE.
//
// You need to edit this file in order to:
//  1. initialize parameters (in constructor)
//  2. implement inference logic (in execute() method)
//
// Refer to the section "Adding Your Own Kernels to the Inference Engine" in
// OpenVINO* documentation (either online or offline in
// <INSTALL_DIR>/deployment_tools/documentation/docs/index.html an then navigate
// to the corresponding section).
// ===============================================================================

#include "ext_list.hpp"
#include "ext_base.hpp"
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

static cv::Mat infEngineBlobToMat(const InferenceEngine::Blob::Ptr& blob)
{
    // NOTE: Inference Engine sizes are reversed.
    std::vector<size_t> dims = blob->getTensorDesc().getDims();
    std::vector<int> size(dims.begin(), dims.end());
    auto precision = blob->getTensorDesc().getPrecision();
    CV_Assert(precision == InferenceEngine::Precision::FP32);
    return cv::Mat(size, CV_32F, (void*)blob->buffer());
}

class FFT2DImpl: public ExtLayerBase {
public:
    explicit FFT2DImpl(const CNNLayer* layer) {
        inverse = layer->type == "IFFT2D";
        addConfig(layer, { { ConfLayout::PLN, false, 0 } }, { { ConfLayout::PLN, false, 0 } });
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {

        cv::Mat inp = infEngineBlobToMat(inputs[0]);
        cv::Mat out = infEngineBlobToMat(outputs[0]);

        const int n = inp.size[0];
        const int h = inp.size[2];
        const int w = inp.size[3];
        cv::Mat complex(h, w, CV_32FC2), interleavedOut(h, w, CV_32FC2);
        for (int i = 0; i < n; ++i)
        {
            std::vector<cv::Mat> components = {
                cv::Mat(h, w, CV_32F, inp.ptr<float>(i, 0)),
                cv::Mat(h, w, CV_32F, inp.ptr<float>(i, 1))
            };
            cv::merge(components, complex);

            if (!inverse)
                cv::dft(complex, interleavedOut);
            else
                cv::idft(complex, interleavedOut, cv::DFT_SCALE);

            components = {
                cv::Mat(h, w, CV_32F, out.ptr<float>(i, 0)),
                cv::Mat(h, w, CV_32F, out.ptr<float>(i, 1))
            };
            cv::split(interleavedOut, components);
        }
        return OK;
    }
private:
    bool inverse;
};

REG_FACTORY_FOR(ImplFactory<FFT2DImpl>, FFT2D);
REG_FACTORY_FOR(ImplFactory<FFT2DImpl>, IFFT2D);

// TODO: The following code is a workaround for a bug with networks reshaping from Python.
// Replece it to nGraph nodes management with the release after 2020.1

class FFTShapeInferImpl : public IShapeInferImpl {
    StatusCode inferShapes(const std::vector<Blob::CPtr>& inBlobs,
                           const std::map<std::string, std::string>& params,
                           const std::map<std::string, Blob::Ptr>& blobs,
                           std::vector<SizeVector>& outShapes,
                           ResponseDesc* desc) noexcept override
    {
        outShapes.resize(inBlobs.size());
        for (size_t i = 0; i < inBlobs.size(); ++i)
            outShapes[i] = inBlobs[i]->getTensorDesc().getDims();
        return StatusCode::OK;
    }
};

class FFTShapeInferFactory : public IShapeInferExtension {
    virtual void SetLogCallback(IErrorListener& listener) noexcept {}
    virtual void GetVersion(const Version*& versionInfo) const noexcept {}
    virtual void Unload() noexcept {}
    virtual void Release() noexcept {}

    virtual StatusCode getShapeInferTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept {
        size = 2;
        types = new char*[size];

        const char* fftType = "FFT2D";
        types[0] = new char[/*sizeof(fftType)*/ 5];
        memcpy(types[0], fftType, 5);

        const char* ifftType = "IFFT2D";
        types[1] = new char[/*sizeof(ifftType)*/ 6];
        memcpy(types[1], ifftType, 6);

        return StatusCode::OK;
    }

    virtual StatusCode getShapeInferImpl(IShapeInferImpl::Ptr& impl, const char* type, ResponseDesc* resp) noexcept {
        impl = std::make_shared<FFTShapeInferImpl>();
        return StatusCode::OK;
    }
};

class FFT2DShapeInferFactory {
public:
    FFT2DShapeInferFactory() {

      // We need to do it to trigger custom shape inference registration.
      const std::string ir = R"V0G0N(
      <?xml version="1.0" ?>
      <net name="" version="10">
        <layers>
          <layer id="0" name="Placeholder_2" type="Parameter" version="opset1">
            <data element_type="f32" shape="1"/>
            <output>
              <port id="0" precision="FP32">
                <dim>1</dim>
              </port>
            </output>
          </layer>
        </layers>
      </net>
      )V0G0N";

      InferenceEngine::Core ie;
      InferenceEngine::CNNNetwork net = ie.ReadNetwork(ir, Blob::CPtr());
      net.AddExtension(std::make_shared<FFTShapeInferFactory>());
    }
};

static FFT2DShapeInferFactory factory;

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
