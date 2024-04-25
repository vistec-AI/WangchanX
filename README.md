# WangchanX Fine-tuning Pipeline
This repository contains fine-tuing scripts for both supervised fine-tuning (SFT) and the alignment scripts. Our goal is to create a model-agnostic fine-tuning pipeline and evaluation scripts focusing on the usability for Thai language.

## Released Models
We apply our fine-tuning pipeline to various open-source models and publish their weights as follows:
- [LLaMa3-8b-WangchanX-sft-vXX](https://huggingface.co/airesearch)
- [SeaLion-7b-WangchanX-sft-vXX](https://huggingface.co/airesearch)
- ...

## Released Dataset
For reproducibility, we provide the scripts for dataset collection and preprocessing in /datasets path ...

## Getting Started

1. Please install all dependencies in `requirements.txt` using pip install as
```
pip3 install -r requirements.txt
```

2. Go to `Fine-tuning` section and select the  training strategy that suitable for your constrains.

## Fine-tuning

To start fine-tuning your own LLM, we recommend using QLoRa fine-tuning because it consume much lesser resources comparing to full fine-tune the LLM. The main template for the script is structurized as 
```
{RUNNER} scripts/run_{MODE}.py {RECIPE}
```
The main parameters are
- `RUNNER`: can simply be the `python` runner for sing-gpu fine-tuning or `accelerate` runner with their following argument `--config_file {ACCELERATION_CONFIG}` when you want to use multi-gpus training
- `ACCELERATION_CONFIG`: is the mode to launch the trainer in multiple setups. Mainly, there're vanilla multi-gpus and ZeRO3 offloading for lower GPU memory usage that comes with the IO overhead. The available configurations are in `recipes/accelerate_configs`
- `MODE`: can be `sft` (supervised fine-tuning) or `dpo` (direct preferene optimization)
- `RECIPE`: based on the model types in `recipes` folder

### QLoRa fine-tuning example

The simplest way to start fine-tune your LLM is to use a plain Python on a **single GPU**. You can do the supervised fine-tuning (SFT) and direct preference optimization (DPO) as in the following step.

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

Please note that the number of argument `num_processes` should be the number of your available GPUs. We use the the default `num_processes=4`.

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
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/llama3-8b/sft/config_full.yaml

# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/llama3-8b/dpo/config_full.yaml
```


### Supported base LLMs
Please note that the provided examples are all LLaMa3. Our pipeline support more than one LLMs. Here are the list of supported base LLMs that we have tested on our scripts.
- LLaMa3
- SeaLion


## Evaluation

-- table --

Please visit https://github.com/vistec-AI/WangchanX-Eval for more detail.

## Deployment
Coming Soon.

## Acknowledgements

We would like to thanks ecosystem (including alignment handbook), sponsor and etc.

## License
..