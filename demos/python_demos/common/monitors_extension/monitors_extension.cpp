// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include <exception>
#include <monitors/presenter.h>

struct PresenterObject {
    PyObject_HEAD
    Presenter *_presenter;
};

namespace {
void presenter_dealloc(PresenterObject *self) {
    delete self->_presenter;
    PyTypeObject *tp = Py_TYPE(self);
    tp->tp_free(self);
    Py_DECREF(tp);
}

char yPosName[] = "yPos", graphSizeName[] = "graphSize";

int presenter_init(PresenterObject *self, PyObject *args, PyObject *kwds) {
    static char keysName[] = "keys", historySizeName[] = "graphSize";
    static char *kwlist[] = {keysName, yPosName, graphSizeName, historySizeName, nullptr};
    const char *keys;
    int yPos = 20, graphSizeWidth = 150, graphSizeHeight = 60;
    unsigned long long historySize = 20;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|i(ii)K", kwlist, &keys, &yPos, &graphSizeWidth, &graphSizeHeight,
        &historySize)) return -1;
    try {
        self->_presenter = new Presenter(keys, yPos, {graphSizeWidth, graphSizeHeight}, historySize);
        return 0;
    } catch (std::exception &exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
        return -1;
    }
}

PyObject *presenter_handleKey(PresenterObject *self, PyObject *args, PyObject *kwds) {
    if (!self->_presenter) {
        PyErr_SetString(PyExc_AssertionError, "Underlying C++ presenter is nullptr");
        return nullptr;
    }
    static char keyName[] = "key";
    static char *kwlist[] = {keyName, nullptr};
    int key;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i", kwlist, &key)) return nullptr;
    try {
        self->_presenter->handleKey(key);
        Py_RETURN_NONE;
    } catch (std::exception &exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
        return nullptr;
    }
}

PyObject *presenter_drawGraphs(PresenterObject *self, PyObject *args, PyObject *kwds) {
    if (!self->_presenter) {
        PyErr_SetString(PyExc_AssertionError, "Underlying C++ presenter is nullptr");
        return nullptr;
    }
    static char frameName[] = "frame";
    static char *kwlist[] = {frameName, nullptr};
    PyArrayObject *npFrame;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &npFrame)) return nullptr;
    if (PyArray_Check(npFrame)
            && PyArray_TYPE(npFrame) != NPY_UINT8
            && PyArray_NDIM(npFrame) != 3
            && PyArray_SHAPE(npFrame)[2] != 3) {
        PyErr_SetString(PyExc_TypeError, "frame must be an array of type uint8 with 3 dimensions with 3 elements in the"
            " last dimension");
        return nullptr;
    }
    int height = static_cast<int>(PyArray_SHAPE(npFrame)[0]);
    int width = static_cast<int>(PyArray_SHAPE(npFrame)[1]);
    try {
        cv::Mat frame(height, width, CV_8UC3, PyArray_DATA(npFrame), PyArray_STRIDE(npFrame, 0));
        self->_presenter->drawGraphs(frame);
        Py_RETURN_NONE;
    } catch (std::exception &exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
        return nullptr;
    }
}

PyObject *presenter_reportMeans(PresenterObject *self, PyObject *Py_UNUSED(ignored)) {
    if (!self->_presenter) {
        PyErr_SetString(PyExc_AssertionError, "Underlying C++ presenter is nullptr");
        return nullptr;
    }
    try {
        return PyUnicode_FromFormat(self->_presenter->reportMeans().c_str());
    } catch (std::exception &exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
        return nullptr;
    }
}

PyMethodDef presenter_methods[] = {
        {"handleKey", reinterpret_cast<PyCFunction>(presenter_handleKey), METH_VARARGS | METH_KEYWORDS},
        {"drawGraphs", reinterpret_cast<PyCFunction>(presenter_drawGraphs), METH_VARARGS | METH_KEYWORDS},
        {"reportMeans", reinterpret_cast<PyCFunction>(presenter_reportMeans), METH_NOARGS},
        {}};  // Sentinel

PyObject *presenter_getYPos(PresenterObject *self, void *closure) {
    if (!self->_presenter) {
        PyErr_SetString(PyExc_AssertionError, "Underlying C++ presenter is nullptr");
        return nullptr;
    }
    return PyLong_FromLong(self->_presenter->yPos);
}

PyObject *presenter_getGraphSize(PresenterObject *self, void *closure) {
        if (!self->_presenter) {
            PyErr_SetString(PyExc_AssertionError , "Underlying C++ presenter is nullptr");
            return nullptr;
        }
        return Py_BuildValue("ii", self->_presenter->graphSize.width, self->_presenter->graphSize.height);
}

PyObject *presenter_getGraphPadding(PresenterObject *self, void *closure) {
    if (!self->_presenter) {
        PyErr_SetString(PyExc_AssertionError, "Underlying C++ presenter is nullptr");
        return nullptr;
    }
    return PyLong_FromLong(self->_presenter->graphPadding);
}

char graphPaddingName[] = "graphPadding";
PyGetSetDef presenter_getsetters[] = {
    {yPosName, reinterpret_cast<getter>(presenter_getYPos)},
    {graphSizeName, reinterpret_cast<getter>(presenter_getGraphSize)},
    {graphPaddingName, reinterpret_cast<getter>(presenter_getGraphPadding)},
    {}};  // Sentinel

char monitors_extension_doc[] = "The module is a wrapper over C++ monitors. It guarantees that C++ and Python "
    "monitors are consistent.";

PyType_Slot presenterSlots[] = {
    {Py_tp_dealloc, reinterpret_cast<void*>(presenter_dealloc)},
    {Py_tp_doc, monitors_extension_doc},
    {Py_tp_methods, presenter_methods},
    {Py_tp_getset, presenter_getsetters},
    {Py_tp_init, reinterpret_cast<void*>(presenter_init)},
    {Py_tp_new, reinterpret_cast<void*>(PyType_GenericNew)},
    {}};  // Sentinel

PyType_Spec presenterSpec{"monitors_extension.Presenter", sizeof(PresenterObject), 0, 0, presenterSlots};

PyModuleDef monitors_extension{PyModuleDef_HEAD_INIT, "monitors_extension", monitors_extension_doc, 0};
}

PyMODINIT_FUNC PyInit_monitors_extension() {
    import_array();
    if (PyErr_Occurred()) return nullptr;

    PyObject *presenterType = PyType_FromSpec(&presenterSpec);
    if (!presenterType) return nullptr;

    PyObject *m = PyModule_Create(&monitors_extension);
    if (m == nullptr) {
        Py_DECREF(presenterType);
        return nullptr;
    }

    if (PyModule_AddObject(m, "Presenter", presenterType) < 0) {
        Py_DECREF(presenterType);
        Py_DECREF(m);
        return nullptr;
    }
    return m;
}
