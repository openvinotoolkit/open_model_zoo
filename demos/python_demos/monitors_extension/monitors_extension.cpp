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
    std::unique_ptr<Presenter> _presenter;
};

namespace {
void presenter_dealloc(PresenterObject *self) {
    Py_TYPE(self)->tp_free(self);
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
        self->_presenter.reset(new Presenter(keys, yPos, {graphSizeWidth, graphSizeHeight}, historySize));
        return 0;
    } catch (std::exception &exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
        return -1;
    }
}

PyObject *presenter_handleKey(PresenterObject *self, PyObject *args, PyObject *kwds) {
    if (!self->_presenter) {
        PyErr_SetString(PyExc_AttributeError, "Underlying C++ presenter is nullptr");
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
        PyErr_SetString(PyExc_AttributeError, "Underlying C++ presenter is nullptr");
        return nullptr;
    }
    static char frameName[] = "frame";
    static char *kwlist[] = {frameName, nullptr};
    PyArrayObject *npFrame;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &npFrame)) return nullptr;
    if (PyArray_TYPE(npFrame) != NPY_UINT8
            && PyArray_NDIM(npFrame) != 3
            && PyArray_SHAPE(npFrame)[2] != 3) {
        PyErr_SetString(PyExc_TypeError, "An array must be of type uint8 with 3 dimentions with 3 elements in the last "
                "dimension");
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
        PyErr_SetString(PyExc_AttributeError, "Underlying C++ presenter is nullptr");
        return nullptr;
    }
    try {
        return PyUnicode_FromFormat(self->_presenter->reportMeans().c_str());
    } catch (std::exception &exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
        return nullptr;
    }
}

PyMethodDef presneter_methods[] = {
        {"reportMeans", reinterpret_cast<PyCFunction>(presenter_reportMeans), METH_NOARGS},
        {"handleKey", reinterpret_cast<PyCFunction>(presenter_handleKey), METH_VARARGS | METH_KEYWORDS},
        {"drawGraphs", reinterpret_cast<PyCFunction>(presenter_drawGraphs), METH_VARARGS | METH_KEYWORDS},
        {nullptr}};  // Sentinel

PyObject *presenter_getYPos(PresenterObject *self, void *closure) {
    if (!self->_presenter) {
        PyErr_SetString(PyExc_AttributeError, "Underlying C++ presenter is nullptr");
        return nullptr;
    }
    PyObject * yPos = PyLong_FromLong(self->_presenter->yPos);
    if (PyErr_Occurred())
        return nullptr;
    return yPos;
}

int presenter_setYPos(PresenterObject *self, PyObject *value, void *closure) {
    PyErr_Format(PyExc_RuntimeError, "Read-only attribute: yPos");
    return -1;
}

PyObject *presenter_getGraphSize(PresenterObject *self, void *closure) {
        if (!self->_presenter) {
            PyErr_SetString(PyExc_AttributeError, "Underlying C++ presenter is nullptr");
            return nullptr;
        }
        PyObject * width, * height;
        width = PyLong_FromLong(self->_presenter->graphSize.width);
        if (PyErr_Occurred())
            return nullptr;
        height = PyLong_FromLong(self->_presenter->graphSize.height);
        if (PyErr_Occurred())
            return nullptr;
        return PyTuple_Pack(2, width, height);
}

int presenter_setGraphSize(PresenterObject *self, PyObject *value, void *closure) {
    PyErr_Format(PyExc_RuntimeError, "Read-only attribute: graphSize");
    return -1;
}

PyObject *presenter_getGraphPadding(PresenterObject *self, void *closure) {
    if (!self->_presenter) {
        PyErr_SetString(PyExc_AttributeError, "Underlying C++ presenter is nullptr");
        return nullptr;
    }
    PyObject * graphPadding = PyLong_FromLong(self->_presenter->graphPadding);
    if (PyErr_Occurred())
        return nullptr;
    return graphPadding;
}

int presenter_setGraphPadding(PresenterObject *self, PyObject *value, void *closure) {
    PyErr_Format(PyExc_RuntimeError, "Read-only attribute: graphPadding");
    return -1;
}

char graphPaddingName[] = "graphPadding";
PyGetSetDef presenter_getsetters[] = {
    {yPosName, (getter)presenter_getYPos, (setter)presenter_setYPos,},
    {graphSizeName, (getter)presenter_getGraphSize, (setter)presenter_setGraphSize},
    {graphPaddingName, (getter)presenter_getGraphPadding, (setter)presenter_setGraphPadding},
    {nullptr}};  // Sentinel

char monitors_extension_doc[] = "The module is a wrapper over C++ monitors. It guarantees that C++ and Python "
    "monitors are consistent.";

PyTypeObject presenterType{PyVarObject_HEAD_INIT(nullptr, 0)
    "monitors_extension.Presenter",  // tp_name
    sizeof(PresenterObject),  // tp_basicsize
    0,  // tp_itemsize
    reinterpret_cast<destructor>(presenter_dealloc),  // tp_dealloc
    0,  // tp_vectorcall_offset
    0,  // tp_getattr
    0,  // tp_setattr
    0,  // tp_as_async
    0,  // tp_repr
    0,  // tp_as_number
    0,  // tp_as_sequence
    0,  // tp_as_mapping
    0,  // tp_hash
    0,  // tp_call
    0,  // tp_str
    0,  // tp_getattro
    0,  // tp_setattro
    0,  // tp_as_buffer
    0,  // tp_flags
    monitors_extension_doc,  // tp_doc
    0,  // tp_traverse
    0,  // tp_clear
    0,  // tp_richcompare
    0,  // tp_weaklistoffset
    0,  // tp_iter
    0,  // tp_iternext
    presneter_methods,  // tp_methods
    0,  // tp_members
    presenter_getsetters,  // tp_getset
    0,  // tp_base
    0,  // tp_dict
    0,  // tp_descr_get
    0,  // tp_descr_set
    0,  // tp_dictoffset
    reinterpret_cast<initproc>(presenter_init),  // tp_init
    0,  // tp_alloc
    PyType_GenericNew};  // tp_new

PyModuleDef monitors_extension{PyModuleDef_HEAD_INIT, "monitors_extension", monitors_extension_doc, 0};
}

PyMODINIT_FUNC PyInit_monitors_extension() {
    if (PyType_Ready(&presenterType) < 0) return nullptr;

    PyObject *m = PyModule_Create(&monitors_extension);
    if (m == nullptr) return nullptr;

    Py_INCREF(&presenterType);
    if (PyModule_AddObject(m, "Presenter", reinterpret_cast<PyObject *>(&presenterType)) < 0) {
        Py_DECREF(&presenterType);
        Py_DECREF(m);
        return nullptr;
    }

    import_array();
    if (PyErr_Occurred()) return nullptr;
    return m;
}
