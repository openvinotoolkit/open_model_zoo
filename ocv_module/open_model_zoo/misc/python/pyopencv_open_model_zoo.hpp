#ifdef HAVE_OPENCV_OPEN_MODEL_ZOO

#include <iostream>

#include <Python.h>

namespace cv { namespace open_model_zoo {

static void downloadFile(const std::string& url, const std::string& sha,
                         const std::string& path)
{
    std::string urlOpen = "r = urlopen('" + url + "')";
    std::string fileOpen = "f = open('" + path + "', 'wb')";

    PyGILState_STATE gstate = PyGILState_Ensure();
    PyRun_SimpleString("from urllib2 import urlopen, Request");
    PyRun_SimpleString("import re");
    PyRun_SimpleString(fileOpen.c_str());
    PyRun_SimpleString("BUFSIZE = 10*1024*1024");

    PyRun_SimpleString(urlOpen.c_str());
    PyRun_SimpleString("buf = r.read(BUFSIZE)");

    if (url.find("drive.google.com") != std::string::npos)
    {
        // For Google Drive we need to add confirmation code for large files
        PyRun_SimpleString("matches = re.search('confirm=(\\w+)&', buf)");

        std::string cmd = "if matches: " \
                          "cookie = r.headers.get('Set-Cookie'); " \
                          "req = Request('" + url + "' + '&confirm=' + matches.group(1)); " \
                          "req.add_header('cookie', cookie); " \
                          "r = urlopen(req); " \
                          "buf = r.read(BUFSIZE)";  // Reread first chunk
        PyRun_SimpleString(cmd.c_str());
    }
    PyRun_SimpleString("while buf: f.write(buf); buf = r.read(BUFSIZE)");
    PyRun_SimpleString("f.close()");
    PyGILState_Release(gstate);
}

void Topology::download()
{
    std::string url, sha, path;
    getModelInfo(url, sha, path);
    downloadFile(url, sha, path);
}

Ptr<Topology> densenet_121()
{
    Ptr<Topology> t(new Topology({{"config_sha256", "baeed2a423794c2c8dc1a80ad96e961112224fa1d319d535735ba93a2b535170"}, {"config_name", "densenet-121.prototxt"}, {"model_url", "https://drive.google.com/uc?export=download&id=0B7ubpZO7HnlCcHlfNmJkU2VPelE"}, {"description", "The `densenet-121` model is one of the DenseNet <https://arxiv.org/pdf/1608.06993> group of models designed to perform image classification. Originally trained on Torch, the authors converted them into Caffe\\* format. All the DenseNet models have been pretrained on the ImageNet image database. For details about this family of models, check out the repository <https://github.com/shicai/DenseNet-Caffe>. The model input is a blob that consists of a single image of 1x3x224x224 in BGR order. The BGR mean values need to be subtracted as follows: [103.94, 116.78, 123.68] before passing the image blob into the network. In addition, values must be divided by 0.017. The model output for `densenet-121` is the typical object classifier output for the 1000 different classifications matching those in the ImageNet database."}, {"config_url", "https://raw.githubusercontent.com/shicai/DenseNet-Caffe/a68651c0b91d8dcb7c0ecd39d1fc76da523baf8a/DenseNet_121.prototxt"}, {"model_sha256", "c6a6ec988d76c468c3f67501a23a39ec7bf6ebe6729fd99496a15d0e845478b2"}, {"license", "https://raw.githubusercontent.com/liuzhuang13/DenseNet/master/LICENSE"}, {"model_name", "densenet-121.caffemodel"}}));
    t->download();
    return t;
}


Ptr<Topology> densenet_161()
{
    Ptr<Topology> t(new Topology({{"config_sha256", "a193e029d66112b077ed29e8b8d36d0bae0593a7f3c64125a433937b5f035b69"}, {"config_name", "densenet-161.prototxt"}, {"model_url", "https://drive.google.com/uc?export=download&id=0B7ubpZO7HnlCa0phRGJIRERoTXM"}, {"description", "The `densenet-161` model is one of the DenseNet <https://arxiv.org/pdf/1608.06993> group of models designed to perform image classification. The main difference with the `densenet-121` model is the size and accuracy of the model. The `densenet-161` is much larger at 100MB in size vs the `densenet-121` model's roughly 31MB size. Originally trained on Torch, the authors converted them into Caffe* format. All the DenseNet models have been pretrained on the ImageNet image database. For details about this family of models, check out the repository <https://github.com/shicai/DenseNet-Caffe>. The model input is a blob that consists of a single image of 1x3x224x224 in BGR order. The BGR mean values need to be subtracted as follows: [103.94, 116.78, 123.68] before passing the image blob into the network. In addition, values must be divided by 0.017. The model output for `densenet-161` is the typical object classifier output for the 1000 different classifications matching those in the ImageNet database."}, {"config_url", "https://raw.githubusercontent.com/shicai/DenseNet-Caffe/a68651c0b91d8dcb7c0ecd39d1fc76da523baf8a/DenseNet_161.prototxt"}, {"model_sha256", "e124d9a8f2284f4ab160569139217f709f21be6fc132c865b6a55cb8cae7d6b5"}, {"license", "https://raw.githubusercontent.com/liuzhuang13/DenseNet/master/LICENSE"}, {"model_name", "densenet-161.caffemodel"}}));
    t->download();
    return t;
}

}}  // namespace open_model_zoo

#endif
