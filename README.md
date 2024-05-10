# WangchanX Fine-tuning Pipeline

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![Python 3.10.12](https://img.shields.io/static/v1?label=Python&message=3.10.12&color=blue&logo=python&logoColor=white)

This repository contains fine-tuning scripts for both supervised fine-tuning (SFT) and alignment scripts.
Our goal is to create a model-agnostic fine-tuning pipeline and evaluation scripts focusing on the usability of the Thai language.
The repository consists of three training scripts: (i) supervised fine-tuning (SFT), (ii) [direct preference optimization (DPO)](https://arxiv.org/abs/2305.18290), and (iii) [odds ratio preference optimization (ORPO)](https://arxiv.org/abs/2403.07691).

## Content

- [Supported base LLMs](#-supported-base-llms)
- [Released Models](#-released-models)
- [Evaluation (0-shot)](#-evaluation-0-shot)
- [Installation](#-installation)
- [Prepare Dataset (Optional)](#-prepare-dataset-optional)
- [Fine-tuning](#-fine-tuning)
- [Inference](#-inference)
- [Deployment](#-deployment)
- [Retrieval Augmented Generation (RAG)](#-retrieval-augmented-generation-rag)
- [Acknowledgements](#-acknowledgements)
- [Future Plans](#-future-plans)
- [Citation](#-citation)

## 💡 Supported base LLMs

Here is the list of supported base LLMs that we have tested on our scripts.

- LLaMa3
- SeaLLMs
- PolyLM
- Typhoon
- SEA-LION (Please refer to GitHub: [vistec-AI/WangchanLion](https://github.com/vistec-AI/WangchanLion) for the full detail)

## 🤖 Released Models

We apply our fine-tuning pipeline to various open-source models and publish their weights as follows:

### Demo models

The models that trained on small instruction datasets

- [LLaMa3-8b-WangchanX-sft-Demo](https://huggingface.co/airesearch/LLaMa3-8b-WangchanX-sft-Demo)
- [PolyLM-13b-WangchanX-sft-Demo](https://huggingface.co/airesearch/PolyLM-13b-WangchanX-sft-Demo)
- [typhoon-7b-WangchanX-sft-Demo](https://huggingface.co/airesearch/typhoon-7b-WangchanX-sft-Demo)

### Full models

The models that trained on large instruction datasets (>400 GB of data). For reproducibility, we provide the scripts for dataset collection and preprocessing in [this repository](https://github.com/vistec-AI/WangchanX/tree/datasets).

- [LLaMa3-8b-WangchanX-sft]() (Release soon)
- [SeaLion-7b-WangchanX-sft](https://huggingface.co/airesearch/WangchanLion7B)
- [PolyLM-WangchanX-sft]() (Release soon)

## ⚡ Evaluation (0-shot)

We evaluate each LLM in terms of (i) Correctness Q1 (higher is better), (ii) Helpfulness Q2 (higher is better), (iii) Irrelevancy Q3 (lower is better), and (iv) Out-of-Context Q4 (lower is better). In addition, we use 100 questions from XQuAD. Please visit [WangchanX-Eval](https://github.com/vistec-AI/WangchanX-Eval) for more details about evaluation and benchmarking Thai LLMs.

| Model                                                                                            | Q1     | Q2     | Q3     | Q4    |
| ------------------------------------------------------------------------------------------------ | ------ | ------ | ------ | ----- |
| [LLaMa3-8b-WangchanX-sft-Demo](https://huggingface.co/airesearch/LLaMa3-8b-WangchanX-sft-Demo)   | **92** | **23** | **14** | 4     |
| [SeaLion-7b-WangchanX-sft](https://huggingface.co/airesearch/WangchanLion7B)                     | 68     | 5      | 19     | 4     |
| [typhoon-7b-WangchanX-sft-Demo](https://huggingface.co/airesearch/typhoon-7b-WangchanX-sft-Demo) | 83     | 17     | **14** | 6     |
| [PolyLM-13b-WangchanX-sft-Demo](https://huggingface.co/airesearch/PolyLM-13b-WangchanX-sft-Demo) | 76     | 16     | 18     | **2** |

## 📦 Installation

1. Please install all dependencies in `requirements.txt` using pip install as

```bash
pip3 install -r requirements.txt
```

2. Please install Flash Attention 2 using pip install as

```bash
pip3 install flash-attn --no-build-isolation
```

3. Go to the `Fine-tuning` section and select the training strategy that is suitable for your constraints.

## 📋 Prepare Dataset (Optional)

1. If you want to use a custom dataset, you need to reformat the file by editing it.

```bash
python3 reformat.py
```

2. If you want to use the demo dataset, you can download it from [this](https://huggingface.co/datasets/airesearch/concat_six_dataset_th_en).

This dataset includes 6 datasets:

- [pythainlp/han-instruct-dataset-v2.0](https://huggingface.co/datasets/pythainlp/han-instruct-dataset-v2.0)
- [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- databricks/databricks-dolly-15k (translated English to Thai by Gemini)
- [math_14k](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/math_14k.json)
- math_14k (translated English to Thai by Gemini)
- [iapp_wiki_qa_squad](https://huggingface.co/datasets/iapp_wiki_qa_squad)

## 🛠 Fine-tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vistec-AI/WangchanX/blob/main/notebooks/Train_WangchanX_pipeline.ipynb)

To start fine-tuning your own LLM, we recommend using QLoRa fine-tuning because it consumes much fewer resources compared to fully fine-tuning the LLM. Please note that the provided examples are all LLaMa3. The main template for the script is structured as

```bash
{RUNNER} scripts/run_{MODE}.py {RECIPE}
```

The main parameters are

| Parameter             | Description                                                                                                                                                                                                            |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RUNNER`              | Can be `python` for single-GPU fine-tuning or `accelerate` with the argument `--config_file {ACCELERATION_CONFIG}` for multi-GPU training.                                                                             |
| `ACCELERATION_CONFIG` | The mode to launch the trainer in multiple setups. Mainly, there are vanilla multi-GPU and ZeRO3 offloading for lower GPU memory usage with IO overhead. Available configurations are in `recipes/accelerate_configs`. |
| `MODE`                | Can be `sft` (supervised fine-tuning) or `dpo` (direct preference optimization).                                                                                                                                       |
| `RECIPE`              | Based on the model types in the `recipes` folder.                                                                                                                                                                      |

<details>
<summary>QLoRa fine-tuning example</summary>
<br>
The simplest way to start fine-tuning your LLM is to use plain Python on a <strong>single GPU</strong>. You can do the supervised fine-tuning (SFT) and direct preference optimization (DPO) as in the following step.
<br>
<br>
<pre lang="bash">
#Step 1 - SFT
python scripts/run_sft.py recipes/llama3-8b/sft/config_qlora.yaml<br>
#Step 2 - DPO (optional)
python scripts/run_dpo.py recipes/llama3-8b/dpo/config_qlora.yaml

</pre>
Alternatively, you can exploit <strong>multi-gpus</strong> training by using the bellowing scripts.
<br>
<br>
<pre lang="bash">
# Step 1 - SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=4 scripts/run_sft.py recipes/llama3-8b/sft/config_qlora.yaml<br>
# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=4 scripts/run_dpo.py recipes/llama3-8b/dpo/config_qlora.yaml
</pre>
Please note that the number of arguments num_processes should be the number of your available GPUs. We use the the default num_processes=4.
</details>
<details>
<summary>Full fine-tuning example</summary>
<br>
You can fine-tune the whole model using the following scripts.
<br>
<br>
<pre lang="bash">
# Step 1 - SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml scripts/run_sft.py recipes/llama3-8b/sft/config_full.yaml<br>
# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml scripts/run_dpo.py recipes/llama3-8b/dpo/config_full.yaml
</pre>
In case you have limited GPU resources but still want to do the full fine-tuing, please consider using DeepSpeed ZeRO3. By adding <code>config_file</code> argument, you are good to go!
<br>
<br>
<pre lang="bash">
# Step 1 - SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/llama3-8b/sft/config_full.yaml<br>
# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/llama3-8b/dpo/config_full.yaml
</pre>
</details>

## 🌟 Inference

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PeUnv89Ao2uHRYYzZVOlUwoBUdYKFbLS?usp=sharing)

### Prepare your model and tokenizer:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model path
path = "airesearch/LLaMa3-8b-WangchanX-sft-Demo"

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto")
```

### Define chat messages:

```python
messages = [
    {"role": "user", "content": "ลิเก กับ งิ้ว ต่างกันอย่างไร"},
]
```

### Tokenize chat messages:

```python
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
print(tokenizer.decode(tokenized_chat[0]))
```

<details close>
  <summary>Output: </summary>
  <br>
    <pre lang="markdown">
<|user|>
ลิเก กับ งิ้ว ต่างกันอย่างไร<|end_of_text|>
<|assistant|></pre>
</details>

### Generate responses:

```python
outputs = model.generate(tokenized_chat, max_length=2048)
print(tokenizer.decode(outputs[0]))
```

<details close>
  <summary>Output: </summary>
  <br>
    <pre lang="markdown">
<|user|>
ลิเก กับ งิ้ว ต่างกันอย่างไร<|end_of_text|>
<|assistant|>
ก่อนอื่นเราต้องรู้ความหมายของคำทั้งสอง คำว่า ลิเก เป็นศิลปะการแสดงแบบดั้งเดิมในประเทศไทย ส่วนคำว่า งิ้วน่าจะเป็นการนำภาษาไทยมาแปลจากคำว่า อินโดปีเลีย (indoplea) ซึ่งเป็นชื่อเรียกดนตรีที่มีต้นกำเนิดจากรัฐอุตตาร์ประเทศ ในอินเดีย และได้แพร่หลายไปยังเอเชียตะวันออกเฉียงใต้ โดยเฉพาะสาธารณรัฐประชาชนจีนและเวียดนาม จึงทำให้เกิดคำว่า งิ้วด้วย แต่ทุกคนไม่รู้ว่ามันก็คืออะไรจริง ๆ แล้ว มันมีความแตกต่างกันมาก เพราะถ้าไปถามชาวบ้านบางแห่งอาจจะบอกว่าเป็นอีกประเภทหนึ่งของเพลงโบราณหรือเพลงพื้นเมือง หรือถ้าพูดตามหลักทางประวัติศาสตร์ก็จะกล่าวว่านั่นคือ การขับร้องเพลงที่ใช้รูปแบบการประสานเสียงแบบฮินดู-ซิกห์วัล ที่ผสมผสานระหว่างภาษาอังกฤษ ภาษาจีนกลาง ภาษาพม่า และภาษาทางเหนือกับภาษาลาว รวมถึงภาษากลุ่มออสเตรโลไนว์ในอดีต ดังนั้นตอนนี้คุณสามารถสรุปได้อย่างแม่นยำว่าสองอย่างเหล่านี้แตกต่างกันอย่างไร: ลิเก คือ ศิลปะการแสดงที่มีมายาวนานกว่า 100 ปีในประเทศไทย เช่น ลิเกล้านนา, ลิเกตลุง, ลิเกล้อ ฯลฯ ขณะที่ งิ้ว หมายถึง เพลงประสานเสียงที่มีรากเหง้าของวงการเพลงคลาสสิคในอินเดีย และแพร่กระจายในเอเชียตะวันตกเฉียงใต้เป็นสิ่งแรกๆ หลังจากการเผยแผ่ศาสนายุคแรกๆ นอกจากนี้ ยังมีการรวมแนวเพลงเพื่อรวมเข้ากับการเต้นร่วมสมัยและบทละครที่มีอิทธิพลจากวรรณกรรมจีน<|end_of_text|></pre>
</details>

## 🚀 Deployment

Ensure you have the Hugging Face pre-trained LLM directory with tokenizer, model, and config files before deployment. Download LLMs using this Python code:

<details>
  <summary>Download</summary>

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model ID
model_id = "airesearch/LLaMa3-8b-WangchanX-sft-Demo"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# Save tokenizer and model
path = "LLaMa3-8b-WangchanX-sft-Demo"
tokenizer.save_pretrained(path)
model.save_pretrained(path)
```

</details>

### Frameworks

<details open>
  <summary>Text Generation Inference</summary>
  <br></br>
  <img src="https://camo.githubusercontent.com/a4473598096b3cbdd8b9d3d189da29962a5d08502b17aa32ea3604bce36bc5ae/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f4e617273696c2f7467695f6173736574732f7265736f6c76652f6d61696e2f7468756d626e61696c2e706e67" alt="TGI" width="500" height="300">
  <br>
  <br>
  Text Generation Inference (TGI) is a toolkit that simplifies the deployment and serving of Large Language Models (LLMs). It offers advanced features such as tensor parallelism, quantization, watermarking, and custom prompt generation, making it easy to deploy and utilize LLMs in various applications. You can find more <a href="https://github.com/huggingface/text-generation-inference?tab=readme-ov-file#get-started">details</a>.
<h4></h4>

- At the current working directory location, prepare the following:

  - The directory containing the pre-trained LLM model from Hugging Face. For example, if you are using the `LLaMa3-8b-WangchanX-sft-Demo` model, the directory should be named `LLaMa3-8b-WangchanX-sft-Demo`.

<br>

- Create a <code>Dockerfile</code> with the following content to build a Docker image:

```Dockerfile
FROM ghcr.io/huggingface/text-generation-inference:2.0
COPY LLaMa3-8b-WangchanX-sft-Demo /data/LLaMa3-8b-WangchanX-sft-Demo
```

- Build the image using the following command:

```bash
docker build -t text-generation-inference -f <Dockerfile> .
```

- Alternatively, you can simply build the image which we already provided in the deployment directory:

```bash
docker build -t text-generation-inference -f deployment/TGI/Dockerfile.TextGenerationInference .
```

- Run the image using this command:

```bash
docker run --gpus all -p 8888:80 text-generation-inference --model-id /data/LLaMa3-8b-WangchanX-sft-Demo #you can add -d flag to run in background
```

- And then you can make requests like this:

```bash
curl 127.0.0.1:8888/generate_stream \
    -X POST \
    -d '{"inputs":"<|user|>ลิเก กับ งิ้ว ต่างกันอย่างไร<|end_of_text|>\n<|assistant|>\n","parameters":{"max_new_tokens":2048}}' \
    -H 'Content-Type: application/json'
```

- Preview:

![TGI](./deployment/img/TGI.gif)

---

**NOTE**

Don't forget to add chat template `<|user|>` message .... `<|end_of_text|>\n<|assistant|>\n` in inputs requests for more nice results.

---

</details>
  <br>
<details open>
  <summary>LocalAI</summary>
<br></br>
<img src="https://github.com/go-skynet/LocalAI/assets/2420543/0966aa2a-166e-4f99-a3e5-6c915fc997dd" alt="TGI" width="500" height="300">
<br>
<br>
LocalAI is a free, open-source OpenAI alternative. It provides a drop-in REST API compatible with OpenAI's specs for local/on-prem inference with LLMs, image/audio generation across model families on consumer hardware sans GPU. You can find more <a href="https://localai.io">details</a>.
<h4></h4>

- At the current working directory location, prepare the following:

  - The directory containing the pre-trained LLM model from Hugging Face. For example, if you are using the `LLaMa3-8b-WangchanX-sft-Demo` model, the directory should be named `LLaMa3-8b-WangchanX-sft-Demo`.

  - The model YAML file. This file can be found in the `deployment/LocalAI` directory. For the `LLaMa3-8b-WangchanX-sft-Demo` model, the YAML file would be named `LLaMa3-8b-WangchanX-sft-Demo.yaml`.
    <br>

- Create a <code>Dockerfile</code> with the following content to build a Docker image:

```Dockerfile
FROM localai/localai:latest-aio-gpu-nvidia-cuda-12
COPY LLaMa3-8b-WangchanX-sft-Demo  /build/models/LLaMa3-8b-WangchanX-sft-Demo
COPY LLaMa3-8b-WangchanX-sft-Demo.yaml /build/models
```

- Build the image using the following command:

```bash
docker build -t localai -f <Dockerfile> .
```

- Alternatively, you can simply build the image which we already provided in the deployment directory:

```bash
docker build -t localai -f deployment/LocalAi/Dockerfile.LocalAi .
```

- Run the image using this command:

```bash
docker run --gpus all -p 8888:8080 localai #you can add -d flag to run in background
```

- And then you can make requests like this:

```bash
curl http://localhost:8888/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{ "model": "LLaMa3-8b-WangchanX-sft-Demo", "messages": [{"role": "user", "content": "ลิเก กับ งิ้ว ต่างกันอย่างไร", "temperature": 0.1}] }'
```

- Preview:

![LocalAi](./deployment/img/LocalAi.gif)

</details>
  <br>
<details open>
  <summary>Ollama</summary>
<br></br>
<img src="https://miro.medium.com/v2/resize:fit:500/0*nfgy5wSyS9IBBl7g.png" alt="Ollama" width="400" height="400">
<br>
<br>
Ollama is an open-source and user-friendly platform that allows you to run large language models (LLMs) locally on your machine. You can find more <a href="https://github.com/ollama/ollama">details</a>.
<h4></h4>

- At the current working directory location, prepare the following:

  - The directory containing the pre-trained LLM model from Hugging Face. For example, if you are using the `LLaMa3-8b-WangchanX-sft-Demo` model, the directory should be named `LLaMa3-8b-WangchanX-sft-Demo`.

    <br>

- Create a <code>Dockerfile</code> with the following content to build a Docker image:

```Dockerfile
FROM ollama/ollama

COPY LLaMa3-8b-WangchanX-sft-Demo /root/LLaMa3-8b-WangchanX-sft-Demo

RUN apt update && apt-get install python3 python3-pip python3-venv git -y

# Clone the ollama repository first
RUN git clone https://github.com/ollama/ollama.git /root/ollama

# Change to the cloned ollama directory
WORKDIR /root/ollama

# Initialize and update git submodules
RUN git submodule update --init --recursive

# Create and activate virtual environment
RUN python3 -m venv .venv
RUN . .venv/bin/activate
RUN python3 -m pip install -r llm/llama.cpp/requirements.txt

# Build the submodule
RUN make -C llm/llama.cpp quantize

# Convert
RUN python3 llm/llama.cpp/convert-hf-to-gguf.py /root/LLaMa3-8b-WangchanX-sft-Demo --outtype f16 --outfile /root/LLaMa3-8b-WangchanX-sft-Demo.gguf
```

- Build the image using the following command:

```bash
docker build -t ollama -f <Dockerfile> .
```

- Alternatively, you can simply build the image which we already provided in the deployment directory:

```bash
docker build -t ollama -f deployment/Ollama/Dockerfile.Ollama .
```

- Run the image using this command:

```bash
docker run -d --gpus all -p 11434:11434  ollama #you can add -d flag to run in background
```

- Create model:

```bash
curl http://localhost:11434/api/create -d '{
  "name": "LLaMa3-8b-WangchanX-sft-Demo",
  "modelfile":"FROM /root/LLaMa3-8b-WangchanX-sft-Demo.gguf\n\n\nTEMPLATE \"\"\"\n{{ if .System }}<|system|>\n{{.System}}<|end_of_text|>\n{{ end }}{{ if .Prompt }}<|user|>\n{{ .Prompt }}<|end_of_text|>\n{{ end }}<|assistant|>\n\"\"\"\n\nPARAMETER stop \"<|end_of_text|>\"\nPARAMETER stop \"<|assistant|>\"\nPARAMETER stop \"<|user|>\"\nPARAMETER stop \"<|system|>\""
}'
```

- And then you can make requests like this:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "LLaMa3-8b-WangchanX-sft-Demo",
  "messages": [
    {
      "role": "user",
      "content": "ลิเก กับ งิ้ว ต่างกันอย่างไร"
    }
  ]
}'
```

- Preview:

![Ollama](./deployment/img/Ollama.gif)

</details>

## ✨ Retrieval Augmented Generation (RAG)

<details open>
<summary>Basic</summary>
<br>
<a href="https://colab.research.google.com/drive/1peyWgBM80OwY3SOy-uN8ZcjR2ME_Qhdq?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>
</details>

<br>

<details open>
<summary>Advanced</summary>
<br></br>
<img src="https://miro.medium.com/v2/resize:fit:1400/0*q9OFsbhJZ44E47bD.png" alt="Flowise" width="800" height="300">
<br>
<br>
Flowise is an open source low-code tool for developers to build customized LLM orchestration flow & AI agents. You can find more <a href="https://docs.flowiseai.com">details</a>.
<h4>Step 1: Establish The Necessary Network</h4>

```bash
docker network create --driver bridge retrieval-augmented-generation
```

<h4>Step 2: Prepare the Embeddings service.</h4>

Prepare the embeddings model YAML file at the current working directory. This file can be found in the `deployment/LocalAI` which is `paraphrase-multilingual-mpnet-base-v2.yaml` and run the following command for create service

```bash
docker build -t localai -f deployment/LocalAi/Dockerfile.LocalAi.Embeddings .
docker run -d -it --network retrieval-augmented-generation --gpus '"device=0"' -p 8888:8080 --name localai-service localai #To select device 0 for GPU, if you have more than one, you can use only the CPU by leaving the 'gpus' flag blank.
```

<h4>Step 3: Prepare LLM service.</h4>

```bash
docker run -d -it --network retrieval-augmented-generation --gpus '"device=1"' -p 11434:11434 --name ollama-service ollama/ollama
```

<h4>Step 4: Create Flowise service.</h4>

```bash
docker run -d -it --network retrieval-augmented-generation --name flowise-service -e PORT=4000 -e FLOWISE_USERNAME=admin -e FLOWISE_PASSWORD=admin -p 4000:4000 elestio/flowiseai
```

- Preview:

![Flowise](./deployment/img/flowise.gif)

</details>

## 🙏 Acknowledgements

We would like to thank all codes and structures from [alignment-handbook](https://github.com/huggingface/alignment-handbook).
This project is sponsored by VISTEC, PTT, SCBX, and SCB.

## 📅 Future Plans

Here are some future plans and what we are doing:

- Adding model and codes for ORPO. Currently, we have codes and preliminary models from the ORPO technique. We are planning to release them soon.
- Thai LLMs benchmark. We are planning to create a machine reading comprehension leaderboard for Thai LLMs. We are happy for any ideas or contributions from everyone.
- Deployment. We are planning to release codes for RAG and ChatBot. This will help Thai NLP engineers and researchers use and deploy LLMs in their works.

## 📜 Citation

If you use WangchanX or WangchanX Eval in your project or publication, please cite the library as follows

```tex
@misc{phatthiyaphaibun2024wangchanlion,
      title={WangchanLion and WangchanX MRC Eval},
      author={Wannaphong Phatthiyaphaibun and Surapon Nonesung and Patomporn Payoungkhamdee and Peerat Limkonchotiwat and Can Udomcharoenchaikit and Jitkapat Sawatphol and Chompakorn Chaksangchaichot and Ekapol Chuangsuwanich and Sarana Nutanong},
      year={2024},
      eprint={2403.16127},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
