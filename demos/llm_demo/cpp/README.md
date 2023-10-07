# LLM C++ Demo

This application showcases inference of large language model (LLM). Unlike most of other demos, this application doesn't have a rich set of command line arguments to encourage the reader to explore and modify the source code.

## How it works

The demo loads a model (`.xml` and `.bin`) to OpenVINO™ and a provided vocab (`.gguf`) to use for tokinezation. The model is reshaped to batch 1 and variable prompt size. A prompt is tokenized and passed to the model. The model greedily generates token by token until the special end of sentence token is obtained. The predicted tokens are converted to chars and printed in a streaming fashion.

## Supported models

### LLaMA 2

Follow https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/254-llm-chatbot to convert the model to intermediate representation (IR) (`.xml` and `.bin`) format. Use [../../thirdparty/llama.cpp/models/ggml-vocab-llama.gguf](../../thirdparty/llama.cpp/models/ggml-vocab-llama.gguf) as a vocab.

### OpenLLaMA

1. Install dependencies

   ```sh
   git lfs install
   python -m pip install git+https://github.com/huggingface/optimum-intel.git
   ```

2. Download and convert the model

   ```sh
   git clone https://huggingface.co/openlm-research/open_llama_3b_v2
   python -c "from optimum.intel.openvino import OVModelForCausalLM; model = OVModelForCausalLM.from_pretrained('open_llama_3b_v2', export=True); model.save_pretrained('.')"
   ```

3. Convert the vocab

   `python demos/thirdparty/llama.cpp/convert.py open_llama_3b_v2/ --vocab-only --outfile open_llama_3b_v2/vocab.gguf`

## Running

Usage: `llm_demo <model_path> <vocab_path> "<prompt>"`

Example: `llm_demo openvino_model.xml vocab.gguf "1+1="`

Follow https://github.com/ggerganov/llama.cpp/discussions/366#discussioncomment-5384744 to enable non ASCII characters for Windows cmd.
