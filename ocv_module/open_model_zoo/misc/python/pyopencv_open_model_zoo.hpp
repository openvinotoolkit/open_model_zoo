#ifdef HAVE_OPENCV_OPEN_MODEL_ZOO

#include <iostream>

#include <Python.h>

namespace cv { namespace open_model_zoo {

void Topology::download()
{
    std::cout << "/* message */" << std::endl;
    std::cout << getModelURL() << std::endl;
    std::cout << "Hello!" << std::endl;

    PyGILState_STATE gstate = PyGILState_Ensure();
    std::cout << 1 << std::endl;
    PyRun_SimpleString("print(1)");
    std::cout << 2 << std::endl;
    PyGILState_Release(gstate);
    std::cout << 3 << std::endl;
}

Ptr<Topology> face_detection_retail()
{
    Ptr<Topology> t(new Topology("myurl", ""));
    t->download();
    return t;
}

}}  // namespace open_model_zoo

#endif
