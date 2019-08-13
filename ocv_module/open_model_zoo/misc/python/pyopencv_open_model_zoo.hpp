#ifdef HAVE_OPENCV_OPEN_MODEL_ZOO

typedef std::vector<HumanPose> vector_HumanPose;

#include <iostream>
#include <string>

#include "opencv2/core/utils/logger.hpp"
#include "opencv2/core/utils/filesystem.hpp"

namespace cv { namespace open_model_zoo {

static std::string getSHA(const std::string& path)
{
    PyGILState_STATE gstate = PyGILState_Ensure();

    std::string fileOpen = "f = open('" + path + "', 'rb')";
    PyObject* pModule = PyImport_AddModule("__main__");

    PyRun_SimpleString("import hashlib");
    PyRun_SimpleString("sha = hashlib.sha256()");
    PyRun_SimpleString(fileOpen.c_str());
    PyRun_SimpleString("sha.update(f.read())");
    PyRun_SimpleString("f.close()");
    PyRun_SimpleString("sha256 = sha.hexdigest()");

    PyObject* sha256 = PyObject_GetAttrString(pModule, "sha256");
    std::string sha;
    getUnicodeString(sha256, sha);

    Py_DECREF(sha256);
    PyGILState_Release(gstate);
    return sha;
}

static void extractAndRemove(const std::string& archive)
{
    PyGILState_STATE gstate = PyGILState_Ensure();

    std::string cmd = "archive = '" + archive + "'";
    PyRun_SimpleString(cmd.c_str());

    std::string archiveOpen;
    if (archive.size() >= 7 && archive.substr(archive.size() - 7) == ".tar.gz")
    {
        PyRun_SimpleString("import tarfile");
        PyRun_SimpleString("f = tarfile.open(archive)");
    }
    else if (archive.size() >= 4 && archive.substr(archive.size() - 4) == ".zip")
    {
        PyRun_SimpleString("from zipfile import ZipFile");
        PyRun_SimpleString("f = ZipFile.open(archive, 'r')");
    }
    else
        CV_Error(Error::StsNotImplemented, "Unexpected archive extension: " + archive);

    PyRun_SimpleString("import os");
    PyRun_SimpleString("f.extractall(path=os.path.dirname(archive))");
    PyRun_SimpleString("f.close()");
    PyRun_SimpleString("os.remove(archive)");

    PyGILState_Release(gstate);
}

static void downloadFile(const std::string& url, const std::string& sha,
                         const std::string& path)
{
    if (utils::fs::exists(path))
    {
        std::string currSHA = getSHA(path);
        if (sha != currSHA)
        {
            // We won't download this file because in case of outdated SHA all
            // the applications will download it and still have hash mismatch.
            CV_LOG_WARNING(NULL, "Hash mismatch for " + path + "\n" + "expected: " + sha + "\ngot:      " + currSHA);
        }
        return;
    }
    PyGILState_STATE gstate = PyGILState_Ensure();

    utils::fs::createDirectories(utils::fs::getParent(path));

    std::string urlOpen = "r = urlopen('" + url + "')";
    std::string fileOpen = "f = open('" + path + "', 'wb')";
    std::string printSize = "d = dict(r.info()); size = '<unknown>'\n" \
                            "if 'content-length' in d: size = int(d['content-length']) // MB\n" \
                            "elif 'Content-Length' in d: size = int(d['Content-Length']) // MB\n" \
                            "print('  %s %s [%s MB]' % (r.getcode(), r.msg, size))";

#if PY_MAJOR_VERSION >= 3
    PyRun_SimpleString("from urllib.request import urlopen, Request");
#else
    PyRun_SimpleString("from urllib2 import urlopen, Request");
#endif
    PyRun_SimpleString(fileOpen.c_str());
    PyRun_SimpleString("MB = 1024*1024");
    PyRun_SimpleString("BUFSIZE = 10*MB");

    std::string info = "print('get ' + '" + url + "')";
    PyRun_SimpleString(info.c_str());
    if (PyRun_SimpleString(urlOpen.c_str()) == -1)
    {
        PyGILState_Release(gstate);
        CV_Error(Error::StsError, "Failed to download a file");
    }
    PyRun_SimpleString(printSize.c_str());
    PyRun_SimpleString("import sys; sys.stdout.write('  progress '); sys.stdout.flush()");
    PyRun_SimpleString("buf = r.read(BUFSIZE)");

    if (url.find("drive.google.com") != std::string::npos)
    {
        // For Google Drive we need to add confirmation code for large files
        PyRun_SimpleString("import re");
        PyRun_SimpleString("matches = re.search(b'confirm=(\\w+)&', buf)");

        std::string cmd = "if matches: " \
                          "cookie = r.headers.get('Set-Cookie'); " \
                          "req = Request('" + url + "' + '&confirm=' + matches.group(1).decode('utf-8')); " \
                          "req.add_header('cookie', cookie); " \
                          "r = urlopen(req); " \
                          "buf = r.read(BUFSIZE)";  // Reread first chunk
        PyRun_SimpleString(cmd.c_str());
    }
    PyRun_SimpleString("while buf: sys.stdout.write('>'); sys.stdout.flush(); " \
                       "f.write(buf); buf = r.read(BUFSIZE)");
    PyRun_SimpleString("sys.stdout.write('\\n'); f.close()");
    PyGILState_Release(gstate);

    std::string resSHA = getSHA(path);
    if (sha != resSHA)
        CV_LOG_WARNING(NULL, "Hash mismatch for " + path + "\n" + "expected: " + sha + "\ngot:      " + resSHA);
}

void Topology::download() const
{
    std::string archiveURL, archiveSHA, archivePath;
    std::vector<std::string> paths(2), shas(2), urls(2);
    getModelInfo(urls[0], shas[0], paths[0]);
    getConfigInfo(urls[1], shas[1], paths[1]);
    getArchiveInfo(archiveURL, archiveSHA, archivePath);
    if ((!paths[0].empty() && !utils::fs::exists(paths[0])) ||
        (!paths[1].empty() && !utils::fs::exists(paths[1])))
    {
        if (!archiveURL.empty())
        {
            downloadFile(archiveURL, archiveSHA, archivePath);
            extractAndRemove(archivePath);
        }
        else
        {
            for (int i = 0; i < 2; ++i)
            {
                if (!urls[i].empty())
                    downloadFile(urls[i], shas[i], paths[i]);
            }
        }
    }
    else if (archiveURL.empty())  // There is no SHA sums for files from archives.
    {
        for (int i = 0; i < 2; ++i)
        {
            if (!paths[i].empty())
            {
                std::string sha = getSHA(paths[i]);
                if (sha != shas[i])
                    CV_LOG_WARNING(NULL, "Hash mismatch for " + paths[i] + "\n" + "expected: " + shas[i] + "\ngot:      " + sha);
            }
        }
    }
}

void Topology::convertToIR(String& xmlPath, String& binPath,
                           const std::vector<String>& extraArgs,
                           const std::vector<String>& excludeArgs) const
{
    if (getOriginFramework() == "dldt")
    {
        xmlPath = getConfigPath();
        binPath = getModelPath();
        return;
    }

    std::string outDir = utils::fs::getParent(getModelPath());
    std::string topologyName = getName();

    // Precision suffix
    for (const auto& arg : extraArgs)
    {
        if (arg.size() >= 11 && arg.substr(0, 11) == "--data_type")
        {
            std::string suffix = arg.substr(arg.find_first_not_of(" =", 11));
            std::transform(suffix.begin(), suffix.end(), suffix.begin(), ::tolower);
            topologyName += "_" + suffix;
            break;
        }
    }
    xmlPath = utils::fs::join(outDir, topologyName + ".xml");
    binPath = utils::fs::join(outDir, topologyName + ".bin");

    if (utils::fs::exists(xmlPath) && utils::fs::exists(binPath))
        return;

    // Create a list of args
    std::string args = "";
    for (const auto& it : getModelOptimizerArgs())
    {
        const std::string& key = it.first;
        std::string value = it.second;

        if (std::find(excludeArgs.begin(), excludeArgs.end(), key) != excludeArgs.end())
            continue;

        if (key == "--input_model")
            value = getModelPath();
        else if (key == "--input_symbol" || key == "--input_proto")
            value = getConfigPath();
        args += format("'%s=%s', ", key.c_str(), value.c_str());
    }
    args += "'--output_dir=" + outDir + "', ";
    args += "'--model_name=" + topologyName + "', ";
    for (const auto& arg : extraArgs)
        args += "'" + arg + "', ";

    // We aren't able to run mo.py directly because there is also module names "mo"
    // in the same location. As a workaround we import mo_tf and detect mo.py location
    // by mo_tf.py
    PyGILState_STATE gstate = PyGILState_Ensure();
    auto cmd = "import mo_tf; import os; import sys; " \
               "path = os.path.join(os.path.dirname(mo_tf.__file__), 'mo.py'); " \
               "sys.argv = [path, " + args + "]";
    // There is sys.exit(0) inside MO so wrap it to try-except
    auto run = "try: exec(open(path).read())\nexcept SystemExit as e: assert(e.code == 0)";
    bool failed = PyRun_SimpleString(cmd.c_str()) == -1 || PyRun_SimpleString(run) == -1;

    // Prevent AttributeError: protobuf
    PyRun_SimpleString("if 'google.protobuf' in sys.modules: del sys.modules['google.protobuf']");
    PyGILState_Release(gstate);

    if (failed)
        CV_Error(Error::StsError, "Failed to run Model Optimizer");
}

static Ptr<TextRecognitionPipeline> createTextRecognitionPipeline(const Topology& detection, const Topology& recognition)
{
    return new TextRecognitionPipeline(detection, recognition);
}

static Ptr<HumanPoseEstimation> createHumanPoseEstimation(const std::string& device)
{
    Topology t;
    if (device == "GPU16" || device == "MYRIAD")
        t = topologies::human_pose_estimation_fp16();
    else
        t = topologies::human_pose_estimation();
    return new HumanPoseEstimation(t, device);
}

static Ptr<HumanPoseEstimation> createHumanPoseEstimation(const Topology& t, const std::string& device)
{
    return new HumanPoseEstimation(t, device);
}

}}  // namespace open_model_zoo

template<> struct pyopencvVecConverter<HumanPose>
{
    static bool to(PyObject* obj, std::vector<HumanPose>& value, const ArgInfo info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<HumanPose>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

#endif
