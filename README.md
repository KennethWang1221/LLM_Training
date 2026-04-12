# LLM Training

Reproducible workflow for:
- local model inference
- Dockerized API serving
- MLflow tracking with Postgres + MinIO (production-like local stack)

## 0) Prerequisites

- Docker
- mlflow

## 1) Python Setup

Run from repo root:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 2) Prepare Weights and Data

### 2.1 Model weights

```bash
mkdir -p out
git clone https://www.modelscope.cn/gongjy/minimind-3-pytorch.git
git clone git@hf.co:jingyaogong/minimind-3
cd out
ln -s ../minimind-3-pytorch/*.pth .
cd ..
```

### 2.2 Dataset

```bash
cd dataset
git clone https://www.modelscope.cn/datasets/gongjy/minimind_dataset.git
ln -s ./minimind_dataset/*.jsonl .
cd ..
```

## 3) Quick Local Inference Smoke Check

```bash
python eval_llm.py --load_from ./model --weight full_sft
```

## 4) Training

### 4.1 Pretrain

```bash
cd trainer
python train_pretrain.py
cd ..
```

### 4.2 SFT

```bash
cd trainer
python train_full_sft.py
cd ..
```

## 5) API Serving (Local, non-container)

Terminal A:

```bash
cd scripts
python serve_openai_api.py
```

Terminal B:

```bash
cd scripts
python chat_api.py
```

Optional API test:

```bash
curl --max-time 30 -X POST "http://127.0.0.1:8998/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"minimind","messages":[{"role":"user","content":"hello"}],"stream":false,"max_tokens":64}'
```

## 6) Dockerize Inference Service

Start from repo root:

```bash
docker compose -f deployment/docker/docker-compose.yaml up --build -d
```

Verify service:

```bash
docker compose -f deployment/docker/docker-compose.yaml ps
docker compose -f deployment/docker/docker-compose.yaml logs -f minimind
```

Stop when done:

```bash
docker compose -f deployment/docker/docker-compose.yaml down
```

## 7) MLflow Stack (Postgres + MinIO + MLflow)

Start from repo root:

```bash
docker compose -f deployment/mlflow/docker-compose.yaml up -d
```

Verify services:

```bash
docker compose -f deployment/mlflow/docker-compose.yaml ps
```

Open UI:

```text
http://127.0.0.1:5000
```

Stop MLflow stack when done:

```bash
docker compose -f deployment/mlflow/docker-compose.yaml down
```

## 8) Log and Verify a Run in New MLflow Backend

Log run:

```bash
python3 ./utils/tmp_log_mlflow.py
```

Verify run metadata and artifact URI:

Get `<RUN_ID>` from script output or from the MLflow UI run URL.

```bash
curl -s "http://127.0.0.1:5000/api/2.0/mlflow/runs/get?run_id=<RUN_ID>"
```

Expected: `artifact_uri` starts with `s3://mlflow/...` (stored through MinIO).
