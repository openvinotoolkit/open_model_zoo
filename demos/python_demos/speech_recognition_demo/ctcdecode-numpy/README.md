# ctcdecode-numpy Python package

ctcdecode-numpy is an implementation of CTC (Connectionist Temporal Classification) beam search decoding for NumPy with optional n-gram language modeling.
C++ code is based on [Parlance](https://github.com/parlance/ctcdecode), which in turn borrowed liberally from Paddle Paddles' [DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech).
It includes standard beam search with swappable scorer support enabling KenLM-based n-gram scoring powered by yoklm library.
KenLM dependency was removed due to licensing concerns, but can be restored manually using Parlance code.

yoklm subcomponent is a library for reading KenLM binary format.  It supports KenLM binary format version 5, with quantization and trie with Bhiksha array representation.

## Installation
To build ctcdecode-numpy, please refer to [Open Model Zoo demos](../../../README.md) for instructions
on how to build the extension module and prepare the environment for running the demo.

Alternatively, you can [install ctcdecode-numpy from OpenVINO 2021.1 Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.1/demos/python_demos/speech_recognition_demo/ctcdecode-numpy#installation).
