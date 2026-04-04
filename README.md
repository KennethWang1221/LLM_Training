# LLM Training

Next token generation.

## Setup

### 1) Install dependencies

Install [uv](https://github.com/astral-sh/uv), then run:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2) Download model weights

```bash
git clone https://www.modelscope.cn/gongjy/minimind-3-pytorch.git
git clone git@hf.co:jingyaogong/minimind-3
cd out
ln -s ../minimind-3-pytorch/*.pth .
cd ..
```

### 3) Download dataset

```bash
cd dataset
git clone https://www.modelscope.cn/datasets/gongjy/minimind_dataset.git
ln -s ./minimind_dataset/*.jsonl .
cd ..
```

## Quick Start

Use native PyTorch weights:

```bash
python eval_llm.py --load_from ./model --weight full_sft
```

Use Transformers-format weights:

```bash
python eval_llm.py --load_from ./minimind-3
```

## Training

### Pretrain

```bash
cd trainer && python train_pretrain.py
```

### SFT

```bash
cd trainer && python train_full_sft.py
```

## Inference

Test SFT checkpoint:

```bash
cd ../
python eval_llm.py --weight full_sft
```
