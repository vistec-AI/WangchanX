<img src="https://github.com/vistec-AI/vistec-ai.github.io/blob/main/wangchanx_logo_color.png?raw=true" width="250">

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

See [Deployments.md](./deployment/Deployments.md) for details on deploying pre-trained Large Language Models (LLMs) using Text Generation Inference (TGI), LocalAI, and Ollama frameworks.

## ✨ Retrieval Augmented Generation (RAG)

See [RAG.md](./deployment/RAG.md) for details on setting up a Retrieval Augmented Generation system using Flowise, LocalAI, and Ollama frameworks for enhancing language model generation with retrieved knowledge.

## 🙏 Acknowledgements

We would like to thank all codes and structures from [alignment-handbook](https://github.com/huggingface/alignment-handbook).
This project is sponsored by VISTEC, PTT, SCBX, and SCB.

## 📅 Future Plans

Here are some future plans and what we are doing:

- Adding model and codes for ORPO. Currently, we have codes and preliminary models from the ORPO technique. We are planning to release them soon.
- Thai LLMs benchmark. We are planning to create a machine reading comprehension leaderboard for Thai LLMs. We are happy for any ideas or contributions from everyone.

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
