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

}}  // namespace open_model_zoo

#endif
