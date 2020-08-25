# ctcdecode

ctcdecode is an implementation of CTC (Connectionist Temporal Classification) beam search decoding for NumPy.
C++ code borrowed liberally from Paddle Paddles' [DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech) and [Parlance](https://github.com/parlance/ctcdecode).
It includes swappable scorer support enabling standard beam search, and KenLM-based decoding powered by yoklm library.

## Installation
The library is largely self-contained and requires only kenlm compiled as a shared library. Building the C++ library requires gcc or clang. KenLM language modeling support is also optionally included powered by yoklm library.

```bash
# get the code
git clone --recursive git@gitlab-icv.inn.intel.com:alexeykr/ctcdecode-numpy.git
cd ctcdecode-numpy
pip install -r requirements.txt
# pip install .
python setup.py build_ext install  # !!!TODO: get rid of build_ext and replace with pip ^^^
```
