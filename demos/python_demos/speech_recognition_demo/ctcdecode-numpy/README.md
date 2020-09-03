# ctcdecode-numpy Python package

ctcdecode-numpy is an implementation of CTC (Connectionist Temporal Classification) beam search decoding for NumPy with optional n-gram language modeling.
C++ code is based on [Parlance](https://github.com/parlance/ctcdecode), which in turn borrowed liberally from Paddle Paddles' [DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech).
It includes standard beam search with swappable scorer support enabling KenLM-based n-gram scoring powered by yoklm library.
KenLM dependency was removed due to licensing concerns, but can be restored manually using Parlance code.

yoklm subcomponent is a library for reading KenLM binary format.  It supports KenLM binary format version 5, with quantization and trie with Bhiksha array representation.

## Installation
The library needs SWIG for building Python interface. For Ubuntu 16.04 or 18.04:

```shell
sudo apt install swig
```

For Windows [download](http://www.swig.org/download.html) and unzip a recent version of `swigwin-*.zip`, and add the directory containing `swig.exe` to your PATH environment valiable.

To build ctcdecode-numpy run:

```shell
pip install -r requirements.txt
python setup.py build_ext install
```
