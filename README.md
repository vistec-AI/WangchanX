# WangchanX Fine-tuning Pipeline

This repository contains fine-tuning scripts for both supervised fine-tuning (SFT) and alignment scripts.
Our goal is to create a model-agnostic fine-tuning pipeline and evaluation scripts focusing on the usability of the Thai language.
The repository consists of three training scripts: (i) supervised fine-tuning (SFT), (ii) [direct preference optimization (DPO)](https://arxiv.org/abs/2305.18290), and (iii) [odds ratio preference optimization (ORPO)](https://arxiv.org/abs/2403.07691).
Currently, our project supports 4 base LLMs, such as LLaMa3, SeaLion, SeaLLMs, and PolyLM and we aim to support new LLMs in the future.

## Released Models

We apply our fine-tuning pipeline to various open-source models and publish their weights as follows:

### Demo models

The models that trained on small instruction datasets

- [LLaMa3-8b-WangchanX-sft-vXX-Demo]()
- [SeaLion-7b-WangchanX-sft-Demo]()
- [SeaLLMs-WangchanX-sft-Demo]()
- [PolyLM-WangchanX-sft-Demo]()

### Full models

The models that trained on large instruction datasets (>400 GB of data)

- [LLaMa3-8b-WangchanX-sft-vXX]()
- [SeaLion-7b-WangchanX-sft](https://huggingface.co/airesearch/WangchanLion7B)
- [SeaLLMs-WangchanX-sft]()
- [PolyLM-WangchanX-sft]()

For reproducibility, we provide the scripts for dataset collection and preprocessing in [https://github.com/vistec-AI/WangchanX/tree/datasets](https://github.com/vistec-AI/WangchanX/tree/datasets). 

## Getting Started

1. Please install all dependencies in `requirements.txt` using pip install as

```
pip3 install -r requirements.txt
```

2. Go to the `Fine-tuning` section and select the training strategy that is suitable for your constraints.

## Prepare Dataset (Optional)

1. If you want to use a custom dataset, you need to reformat the file by editing it.

```python
python3 reformat.py
```

2. If you want to use the demo dataset, you can download it from [airesearch/concat_six_dataset_th_en](https://huggingface.co/datasets/airesearch/concat_six_dataset_th_en).

This dataset includes 6 datasets:

- [pythainlp/han-instruct-dataset-v2.0](https://huggingface.co/datasets/pythainlp/han-instruct-dataset-v2.0)
- [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- Dolly (translated English to Thai by Gemini)
- [Math](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/math_14k.json)
- Math (translated by Gemini)
- [iapp_wiki_qa_squad](https://huggingface.co/datasets/iapp_wiki_qa_squad)

## Fine-tuning

To start fine-tuning your own LLM, we recommend using QLoRa fine-tuning because it consumes much fewer resources compared to fully fine-tuning the LLM. The main template for the script is structured as

```
{RUNNER} scripts/run_{MODE}.py {RECIPE}
```

The main parameters are

- `RUNNER`: can simply be the `python` runner for sing-gpu fine-tuning or `accelerate` runner with the following argument `--config_file {ACCELERATION_CONFIG}` when you want to use multi-gpus training
- `ACCELERATION_CONFIG`: is the mode to launch the trainer in multiple setups. Mainly, there're vanilla multi-gpus and ZeRO3 offloading for lower GPU memory usage that comes with the IO overhead. The available configurations are in `recipes/accelerate_configs`
- `MODE`: can be `sft` (supervised fine-tuning) or `dpo` (direct preference optimization)
- `RECIPE`: based on the model types in `recipes` folder

### QLoRa fine-tuning example

The simplest way to start fine-tuning your LLM is to use plain Python on a **single GPU**. You can do the supervised fine-tuning (SFT) and direct preference optimization (DPO) as in the following step.

```
# Step 1 - SFT
python scripts/run_sft.py recipes/llama3-8b/sft/config_qlora.yaml

# Step 2 - DPO (optional)
python scripts/run_dpo.py recipes/llama3-8b/dpo/config_qlora.yaml
```

Alternatively, you can exploit **multi-gpus** training by using the bellowing scripts.

```
# Step 1 - SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=4 scripts/run_sft.py recipes/llama3-8b/sft/config_qlora.yaml

# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=4 scripts/run_dpo.py recipes/llama3-8b/dpo/config_qlora.yaml
```

Please note that the number of arguments `num_processes` should be the number of your available GPUs. We use the the default `num_processes=4`.

### Full fine-tuning example

You can fine-tune the whole model using the following scripts.

```
# Step 1 - SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml scripts/run_sft.py recipes/llama3-8b/sft/config_full.yaml

# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml scripts/run_dpo.py recipes/llama3-8b/dpo/config_full.yaml
```

In case you have limited GPU resources but still want to do the full fine-tuing, please consider using DeepSpeed ZeRO3. By adding `config_file` argument, you are good to go!

```
# Step 1 - SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/llama3-8b/sft/config_full.yaml

# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/llama3-8b/dpo/config_full.yaml
```

### Supported base LLMs

Please note that the provided examples are all LLaMa3. Our pipeline supports more than one LLM. Here is the list of supported base LLMs that we have tested on our scripts.

- LLaMa3
- SeaLion (Please refer to GitHub:[https://github.com/vistec-AI/WangchanLion](https://github.com/vistec-AI/WangchanLion) for the full detail)
- SeaLLMs
- PolyLM

## Inference Example

Inference Example:

```python
print("")
```

## Evaluation

-- table --

Please visit [https://github.com/vistec-AI/WangchanX-Eval](https://github.com/vistec-AI/WangchanX-Eval) for more details about evaluation and benchmarking Thai LLMs.

## Deployment

Coming Soon.

## Acknowledgements

We would like to thank all codes and structures from [alignment-handbook](https://github.com/huggingface/alignment-handbook).
This project is sponsored by VISTEC, PTT, SCBX, and SCB.

## Future Plans

Here are some future plans and what we are doing:

- Adding model and codes for ORPO. Currently, we have codes and preliminary models from the ORPO technique. We are planning to release them soon.
- Thai LLMs benchmark. We are planning to create a machine reading comprehension leaderboard for Thai LLMs. We are happy for any ideas or contributions from everyone.
- Deployment. We are planning to release codes for RAG and ChatBot. This will help Thai NLP engineers and researchers use and deploy LLMs in their works.


## Citation

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
