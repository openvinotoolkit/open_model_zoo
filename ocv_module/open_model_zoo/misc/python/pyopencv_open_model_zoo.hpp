#ifdef HAVE_OPENCV_OPEN_MODEL_ZOO

#include <iostream>

#include <Python.h>

namespace cv { namespace open_model_zoo {

static void downloadFile(const std::string& url, const std::string& sha,
                         const std::string& path)
{
    // std::cout << "/* message */" << std::endl;
    std::cout << url << std::endl;
    std::cout << sha << std::endl;
    std::cout << path << std::endl;
    // std::cout << "Hello!" << std::endl;

    PyGILState_STATE gstate = PyGILState_Ensure();
    std::cout << 1 << std::endl;
    PyRun_SimpleString("print(1)");
    std::cout << 2 << std::endl;
    PyGILState_Release(gstate);
    std::cout << 3 << std::endl;
}

void Topology::download()
{
    std::string url, sha, path;
    getModelInfo(url, sha, path);
    downloadFile(url, sha, path);
}

Ptr<Topology> densenet_161()
{
    Ptr<Topology> t(new Topology({{"model_name", "densenet-161.prototxt"}, {"model_url", "https://raw.githubusercontent.com/shicai/DenseNet-Caffe/a68651c0b91d8dcb7c0ecd39d1fc76da523baf8a/DenseNet_161.prototxt"}, {"description", "The `densenet-161` model is one of the DenseNet <https://arxiv.org/pdf/1608.06993> group of models designed to perform image classification. The main difference with the `densenet-121` model is the size and accuracy of the model. The `densenet-161` is much larger at 100MB in size vs the `densenet-121` model's roughly 31MB size. Originally trained on Torch, the authors converted them into Caffe* format. All the DenseNet models have been pretrained on the ImageNet image database. For details about this family of models, check out the repository <https://github.com/shicai/DenseNet-Caffe>. The model input is a blob that consists of a single image of 1x3x224x224 in BGR order. The BGR mean values need to be subtracted as follows: [103.94, 116.78, 123.68] before passing the image blob into the network. In addition, values must be divided by 0.017. The model output for `densenet-161` is the typical object classifier output for the 1000 different classifications matching those in the ImageNet database."}, {"license", "https://raw.githubusercontent.com/liuzhuang13/DenseNet/master/LICENSE"}, {"model_sha256", "a193e029d66112b077ed29e8b8d36d0bae0593a7f3c64125a433937b5f035b69"}}));
    t->download();
    return t;
}


}}  // namespace open_model_zoo

#endif
