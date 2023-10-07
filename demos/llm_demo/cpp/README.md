# LLM C++ Demo

This application showcases inference of large language model (LLM). Unlike most of other demos, this application doesn't have a rich set of command line arguments to encourage the reader to explore and modify the source code.

## How it works

The demo loads a provided vocab (`.gguf`) to use for tokinezation and a model (`.xml` and `.bin`) to OpenVINO™. The model is reshaped to batch 1 and variable prompt size. A prompt is tokenized and passed to the model. The model greedily generates token by token until the special end of sentence token is obtained. The predicted tokens are converted to chars and printed in a streaming fashion.

## Supported models

### LLaMA 2

Follow https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/254-llm-chatbot to convert the model to intermediate representation (IR) (`.xml` and `.bin`) format. Use [../../thirdparty/llama.cpp/models/ggml-vocab-llama.gguf](../../thirdparty/llama.cpp/models/ggml-vocab-llama.gguf) as a vocab.

### TODO: The one I use in test

## Running

Usage: `llm_demo <model_path> <vocab_path> '<prompt>'`

Example: TODO: use from test

Follow https://github.com/ggerganov/llama.cpp/discussions/366#discussioncomment-5384744 to enable non ASCII characters for Windows cmd.
