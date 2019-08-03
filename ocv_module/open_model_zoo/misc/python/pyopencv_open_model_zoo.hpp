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
    std::string sha = PyString_AsString(sha256);

    Py_DECREF(sha256);
    return sha;
}

static void downloadFile(const std::string& url, const std::string& sha,
                         const std::string& path)
{
    PyGILState_STATE gstate = PyGILState_Ensure();

    if (utils::fs::exists(path) && sha == getSHA(path))
    {
        std::cout << "File " + path + " exists" << std::endl;
        PyGILState_Release(gstate);
        return;
    }

    std::string urlOpen = "r = urlopen('" + url + "')";
    std::string fileOpen = "f = open('" + path + "', 'wb')";

    PyRun_SimpleString("from urllib2 import urlopen, Request");  // TODO: add python3 support
    PyRun_SimpleString(fileOpen.c_str());
    PyRun_SimpleString("BUFSIZE = 10*1024*1024");  // 10MB

    PyRun_SimpleString(urlOpen.c_str());
    PyRun_SimpleString("buf = r.read(BUFSIZE)");

    if (url.find("drive.google.com") != std::string::npos)
    {
        // For Google Drive we need to add confirmation code for large files
        PyRun_SimpleString("import re");
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

    std::string resSHA = getSHA(path);
    if (sha != resSHA)
        CV_LOG_WARNING(NULL, "Hash mismatch for " + path + "\n" + "expected: " + sha + "\ngot:      " + resSHA);

    PyGILState_Release(gstate);
}

void Topology::download()
{
    std::string url, sha, path;
    getModelInfo(url, sha, path);
    downloadFile(url, sha, path);
}

}}  // namespace open_model_zoo

#endif
