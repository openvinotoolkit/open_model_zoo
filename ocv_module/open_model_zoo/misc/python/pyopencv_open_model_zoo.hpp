#ifdef HAVE_OPENCV_OPEN_MODEL_ZOO

#include <iostream>
#include <string>

#include "opencv2/core/utils/logger.hpp"
#include "opencv2/core/utils/filesystem.hpp"

namespace cv { namespace open_model_zoo {

static std::string getSHA(const std::string& path)
{
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
    return sha;
}

static void extract(const std::string& archive)
{
    PyGILState_STATE gstate = PyGILState_Ensure();

    std::string archiveOpen;
    if (archive.size() >= 7 && archive.substr(archive.size() - 7) == ".tar.gz")
    {
        PyRun_SimpleString("import tarfile");
        archiveOpen = "f = tarfile.open('" + archive + "')";
    }
    else if (archive.size() >= 4 && archive.substr(archive.size() - 4) == ".zip")
    {
        PyRun_SimpleString("from zipfile import ZipFile");
        archiveOpen = "f = ZipFile('" + archive + "', 'r')";
    }
    else
        CV_Error(Error::StsNotImplemented, "Unexpected archive extension: " + archive);

    std::string cmd = "f.extractall(path=os.path.dirname('" + archive + "'))";
    PyRun_SimpleString("import os");
    PyRun_SimpleString(archiveOpen.c_str());
    PyRun_SimpleString(cmd.c_str());
    PyRun_SimpleString("f.close()");

    PyGILState_Release(gstate);
}

static void downloadFile(const std::string& url, const std::string& sha,
                         const std::string& path)
{
    PyGILState_STATE gstate = PyGILState_Ensure();

    if (utils::fs::exists(path))
    {
        std::string currSHA = getSHA(path);
        if (sha != currSHA)
        {
            // We won't download this file because in case of outdated SHA all
            // the applications will download it and still have hash mismatch.
            CV_LOG_WARNING(NULL, "Hash mismatch for " + path + "\n" + "expected: " + sha + "\ngot:      " + currSHA);
        }
        PyGILState_Release(gstate);
        return;
    }

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

    std::string resSHA = getSHA(path);
    if (sha != resSHA)
        CV_LOG_WARNING(NULL, "Hash mismatch for " + path + "\n" + "expected: " + sha + "\ngot:      " + resSHA);

    PyGILState_Release(gstate);
}

void Topology::download() const
{
    std::string url, sha, path;
    getArchiveInfo(url, sha, path);
    if (!url.empty())
    {
        downloadFile(url, sha, path);

        std::string modelPath = getModelPath();
        std::string configPath = getConfigPath();
        if ((!modelPath.empty() && !utils::fs::exists(modelPath)) ||
            (!configPath.empty() && !utils::fs::exists(configPath)))
            extract(path);
    }
    else
    {
        getModelInfo(url, sha, path);
        if (!url.empty())
            downloadFile(url, sha, path);

        getConfigInfo(url, sha, path);
        if (!url.empty())
            downloadFile(url, sha, path);
    }
}

void Topology::convertToIR(String& xmlPath, String& binPath) const
{
    if (getOriginFramework() == "dldt")
    {
        xmlPath = getConfigPath();
        binPath = getModelPath();
        return;
    }

    std::string outDir = utils::fs::getParent(getModelPath());
    std::string topologyName = getName();

    xmlPath = utils::fs::join(outDir, topologyName + ".xml");
    binPath = utils::fs::join(outDir, topologyName + ".bin");

    if (utils::fs::exists(xmlPath) && utils::fs::exists(binPath))
        return;

    // Create a list of args
    std::string args = "";
    for (const auto& it : getModelOptimizerArgs())
    {
        std::string value = it.second;
        if (it.first == "--input_model")
            value = getModelPath();
        else if (it.first == "--input_symbol" || it.first == "--input_proto")
            value = getConfigPath();
        args += format("'%s=%s', ", it.first.c_str(), value.c_str());
    }
    args += "'--output_dir=" + outDir + "', ";
    args += "'--model_name=" + topologyName + "', ";

    // We aren't able to run mo.py directly because there is also module names "mo"
    // in the same location. As a workaround we import mo_tf and detect mo.py location
    // by mo_tf.py
    PyGILState_STATE gstate = PyGILState_Ensure();
    auto cmd = "import mo_tf; import os; import sys; " \
               "path = os.path.join(os.path.dirname(mo_tf.__file__), 'mo.py'); " \
               "sys.argv = [path, " + args + "]";
    // There is sys.exit() inside MO so wrap it to try-except
    auto run = "try: exec(open(path).read())\nexcept SystemExit as e: assert(e.code == 0)";
    if (PyRun_SimpleString(cmd.c_str()) == -1 || PyRun_SimpleString(run) == -1)
    {
        PyGILState_Release(gstate);
        CV_Error(Error::StsError, "Failed to run Model Optimizer");
    }
    PyGILState_Release(gstate);
}

static Ptr<TextRecognitionPipeline> createTextRecognitionPipeline(const Topology& detection, const Topology& recognition)
{
    return new TextRecognitionPipeline(detection, recognition);
}

}}  // namespace open_model_zoo

#endif
