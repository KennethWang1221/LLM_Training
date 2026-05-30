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

### 2.3 OCR pipeline for PDF books

Place raw PDFs under `dataset/pdfs/`. The OCR pipeline keeps every document under its original PDF stem, so `dataset/pdfs/ABCD.pdf` produces:

- `dataset/ocr_stage1/ABCD/`
- `dataset/ocr_stage2/documents/ABCD.document.md`
- `dataset/ocr_stage2/segments/ABCD.segments.jsonl`
- `dataset/final/ABCD/pretrain_ABCD_v1.jsonl`
- `dataset/final/ABCD/sft_ABCD_v1.jsonl`

Run the full pipeline from repo root:

```bash
scripts/run_ocr_pipeline.sh all
```

Run the pipeline for one document only:

```bash
scripts/run_ocr_pipeline.sh all Build_a_Large_Language_Model_From_Scratch
```

By default, OCR extraction runs inside a PaddleOCR container while normalization and JSONL export run locally. To run extraction locally instead:

```bash
OCR_EXECUTION_MODE=local scripts/run_ocr_pipeline.sh extract ABCD
```

Stage breakdown:

- `extract`: PDF -> `dataset/ocr_stage1/<doc_id>/`
- `normalize`: OCR artifacts -> `dataset/ocr_stage2/{documents,segments,reports}/`
- `pretrain`: normalized segments -> `dataset/final/<doc_id>/pretrain_<doc_id>_v1.jsonl`
- `sft`: normalized segments -> `dataset/final/<doc_id>/sft_<doc_id>_v1.jsonl`

Notes:

- The `pretrain` export keeps the cleaned segment text plus provenance metadata.
- The `sft` export is a bootstrap dataset synthesized from cleaned sections; review and improve it before long training runs.
- Generated OCR/data artifacts are ignored by `.gitignore`.

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

## 9) Kubernetes (Minikube)

Use this exact sequence from repo root.

1) Start local Kubernetes cluster.  
```bash
minikube start --driver=docker --cpus=6 --memory=12288 --disk-size=60g --gpus=all
```  
Result: minikube profile status is `OK`, Kubernetes control plane is running, and GPU passthrough is requested.

2) Verify cluster health and GPU availability.  
```bash
minikube addons enable nvidia-device-plugin
kubectl -n kube-system rollout status daemonset/nvidia-device-plugin-daemonset --timeout=180s
kubectl get daemonset -A | rg nvidia
kubectl get node minikube -o jsonpath='{.status.allocatable.nvidia\.com/gpu}{"\n"}'
kubectl get nodes && kubectl get pods -A
```  
Result: node `minikube` is `Ready`, `kube-system` pods are `Running`, and GPU allocatable count is `>=1`.

3) Load your local inference image into minikube runtime.  
```bash
minikube image load minimind-server:v1.0.0 && minikube ssh "sudo crictl images"
```  
Result: image `minimind-server:v1.0.0` is visible in the printed image list.

4) Start repo mount to minikube in a dedicated terminal and keep it alive.  
```bash
minikube mount "$(pwd):/workspace"
```  
Result: mount succeeds and `/workspace` is available inside minikube while this process runs.

5) Ensure checkpoint file in `out/` is a real file (not symlink) for this demo path.  
```bash
rm -f out/full_sft_768.pth && cp minimind-3-pytorch/full_sft_768.pth out/full_sft_768.pth
```  
Result: runtime checkpoint read is stable for mounted minikube path.

6) Create serving namespace.  
```bash
kubectl create namespace llm-serving
```  
Result: namespace `llm-serving` exists and is `Active` (or command returns `already exists`).

7) Deploy base manifests.  
```bash
kubectl apply -k deployment/k8s/base
```  
Result: deployment `minimind-inference` and service `minimind-inference-svc` are applied.

8) Wait for deployment rollout.  
```bash
kubectl -n llm-serving rollout status deployment/minimind-inference --timeout=300s
```  
Result: rollout completes with `successfully rolled out`.

8.1) Verify pod GPU assignment and CUDA runtime.  
```bash
kubectl -n llm-serving describe pod -l app=minimind-inference | rg "nvidia.com/gpu|Limits|Requests"
kubectl -n llm-serving exec deployment/minimind-inference -- python3 -c "import torch; print(torch.cuda.is_available())"
```  
Result: pod resource section includes `nvidia.com/gpu` and runtime check returns `True`.

9) Forward service port to host (run in separate terminal).  
```bash
kubectl -n llm-serving port-forward svc/minimind-inference-svc 19098:8998
```  
Result: local host `127.0.0.1:19098` routes to Kubernetes service port `8998`.

10) Send API request through Kubernetes service path.  
```bash
curl --max-time 60 -X POST "http://127.0.0.1:19098/v1/chat/completions" -H "Content-Type: application/json" -d '{"model":"minimind","messages":[{"role":"user","content":"hello"}],"stream":false,"max_tokens":1}'
```  
Result: API returns JSON response from the pod-backed service.

11) Simulate pod failure to verify self-healing.  
```bash
kubectl -n llm-serving delete pod -l app=minimind-inference
```  
Result: pod is deleted and Deployment controller creates a replacement automatically.

12) Watch replacement pod become healthy.  
```bash
kubectl -n llm-serving get pods -w
```  
Result: new pod transitions to `1/1 Running`; press `Ctrl+C` to stop watch after recovery is confirmed.

13) Simulate bad release (invalid image).  
```bash
kubectl -n llm-serving set image deployment/minimind-inference minimind=nonexistent-image:bad
```  
Result: new revision is created but new pod fails with `ErrImageNeverPull` / `ImagePullBackOff`; on single-GPU local rollout strategy this can temporarily remove the old pod until rollback.

13.1) Confirm bad release behavior (expected to not complete).  
```bash
kubectl -n llm-serving rollout status deployment/minimind-inference --timeout=60s
kubectl -n llm-serving get pods
```  
Result: rollout does not complete successfully within timeout and pod shows image pull/start failure states.

14) Roll back to previous healthy revision (run once).  
```bash
kubectl -n llm-serving rollout undo deployment/minimind-inference
```  
Result: deployment returns to last working image and starts recovery rollout.

14.1) Wait for rollback rollout to finish and confirm pod readiness.  
```bash
kubectl -n llm-serving rollout status deployment/minimind-inference --timeout=300s
kubectl -n llm-serving get pods
```  
Result: rollout completes successfully and pod status is `1/1 Running`.

14.2) Restart port-forward session after pod replacement/rollback (in separate terminal).  
```bash
pkill -f "kubectl -n llm-serving port-forward svc/minimind-inference-svc 19098:8998" || true
kubectl -n llm-serving port-forward svc/minimind-inference-svc 19098:8998
```  
Result: local host `127.0.0.1:19098` is reattached to the current healthy backend pod(s).

15) Verify rollback serving path.  
```bash
curl --max-time 60 -X POST "http://127.0.0.1:19098/v1/chat/completions" -H "Content-Type: application/json" -d '{"model":"minimind","messages":[{"role":"user","content":"rollback test"}],"stream":false,"max_tokens":1}'
```  
Result: API returns JSON response after rollback, confirming release recovery.

Notes:
- If `curl` returns `Empty reply from server`, the existing port-forward tunnel likely attached to an old pod. Stop it and rerun step 14.2.
- Do not run `rollout undo` repeatedly. Each run moves to the previous revision and can flip you away from the desired healthy revision.

16) Clean up when done.  
```bash
kubectl -n llm-serving delete deployment/minimind-inference service/minimind-inference-svc --ignore-not-found
```  
Result: demo workload is removed from cluster.

17) Stop local Kubernetes lab.  
```bash
minikube stop && minikube delete
```  
Result: minikube resources are fully released from your desktop.

## 10) Kubernetes Artifact Runtime (Production-like, no host mount)

Use this sequence to run a production-style artifact pipeline (object storage -> initContainer pull -> k8s serving -> ingress validation -> failure drill -> rollback -> clean teardown).
Validated end-to-end on this repo with single-node minikube + one GPU.

1) Reset to clean state before the run.  
```bash
kubectl delete -k deployment/k8s/artifact-runtime --ignore-not-found || true
kubectl delete namespace llm-serving --ignore-not-found || true
minikube stop || true
minikube delete || true
docker compose -f deployment/mlflow/docker-compose.yaml down || true
```  
Result: no old Section 10 workloads/clusters are left, so this run starts from scratch.

2) Start MLflow/MinIO stack (artifact backend).  
```bash
docker compose -f deployment/mlflow/docker-compose.yaml up -d
```  
Result: MinIO endpoint is available at `http://127.0.0.1:9000`.

3) Ensure checkpoint in `out/` is a real file (not symlink).  
```bash
rm -f out/full_sft_768.pth && cp minimind-3-pytorch/full_sft_768.pth out/full_sft_768.pth
```  
Result: upload reads concrete bytes and avoids symlink path issues.

4) Upload checkpoint and model assets to MinIO bucket `model-artifacts`.  
```bash
docker run --rm --network host --entrypoint /bin/sh -v "$(pwd)/out:/src/out:ro" -v "$(pwd)/model:/src/model:ro" minio/mc:latest -c "mc alias set local http://127.0.0.1:9000 minio minioadmin && mc mb -p local/model-artifacts || true && mc rm --recursive --force local/model-artifacts/model || true && mc cp /src/out/full_sft_768.pth local/model-artifacts/full_sft_768.pth && mc mirror --overwrite /src/model local/model-artifacts/model"
```  
Result: bucket contains `full_sft_768.pth` and full `model/` tree.

5) Verify uploaded artifacts exist in MinIO.  
```bash
docker run --rm --network host --entrypoint /bin/sh minio/mc:latest -c "mc alias set local http://127.0.0.1:9000 minio minioadmin && mc ls local/model-artifacts && mc ls local/model-artifacts/model"
```  
Result: both checkpoint and model files are listed.

6) Start local Kubernetes cluster with GPU passthrough enabled.  
```bash
minikube start --driver=docker --cpus=6 --memory=12288 --disk-size=60g --gpus=all
```  
Result: cluster is active and requests GPU exposure from host to minikube node runtime.

7) Enable NVIDIA device plugin, verify GPU resource, then enable ingress controller.  
```bash
minikube addons enable nvidia-device-plugin
kubectl -n kube-system rollout status daemonset/nvidia-device-plugin-daemonset --timeout=180s
kubectl get daemonset -A | rg nvidia
kubectl get node minikube -o jsonpath='{.status.allocatable.nvidia\.com/gpu}{"\n"}'
minikube addons enable ingress && kubectl -n ingress-nginx rollout status deployment/ingress-nginx-controller --timeout=300s
```  
Result: NVIDIA plugin is running and node reports GPU allocatable count (must be `>=1`), then ingress controller reports rollout success.

8) Start tunnel in a separate terminal and keep it running.  
```bash
minikube tunnel
```  
Result: host routes to minikube ingress/service CIDRs are available (enter sudo password in that terminal when prompted).

9) Load serving image into minikube runtime.  
```bash
minikube image load minimind-server:v1.0.0 && minikube ssh "sudo crictl images"
```  
Result: `minimind-server:v1.0.0` is visible in the printed image list.

10) Create serving namespace.  
```bash
kubectl create namespace llm-serving
```  
Result: namespace `llm-serving` exists.

11) Apply artifact-runtime manifests.  
```bash
kubectl apply -k deployment/k8s/artifact-runtime
```  
Result: secret, deployment, service, and ingress are created.

12) Wait for rollout completion.  
```bash
kubectl -n llm-serving rollout status deployment/minimind-inference-artifact --timeout=300s
```  
Result: initContainer download + app startup complete.

13) Verify pod/service/ingress/endpoints and GPU runtime are healthy.  
```bash
kubectl -n llm-serving get pods,svc,ingress,endpoints
kubectl -n llm-serving describe pod -l app=minimind-inference-artifact | rg "nvidia.com/gpu|Limits|Requests"
kubectl -n llm-serving exec deployment/minimind-inference-artifact -- python3 -c "import torch; print(torch.cuda.is_available())"
```  
Result: pod is `1/1 Running`, service has endpoint, GPU resource appears in pod limits/requests, and `torch.cuda.is_available()` returns `True`.

14) Verify ingress path with non-inference endpoint.  
```bash
MINIKUBE_IP=$(minikube ip) && curl --max-time 20 "http://${MINIKUBE_IP}/docs" -H "Host: minimind.local"
```  
Result: Swagger HTML returns quickly, proving ingress path connectivity.

15) Send inference request through ingress (production-style access path).  
```bash
MINIKUBE_IP=$(minikube ip) && curl --max-time 300 -X POST "http://${MINIKUBE_IP}/v1/chat/completions" -H "Host: minimind.local" -H "Content-Type: application/json" -d '{"model":"minimind","messages":[{"role":"user","content":"artifact runtime test"}],"stream":false,"max_tokens":1}'
```  
Result: API returns JSON completion from artifact-backed runtime.

16) Run self-heal drill (delete pod and verify recovery).  
```bash
kubectl -n llm-serving delete pod -l app=minimind-inference-artifact && kubectl -n llm-serving get pods -w
```  
Result: replacement pod appears and returns to `1/1 Running`; press `Ctrl+C` after recovery is confirmed.

17) Run bad-release drill (invalid image).  
```bash
kubectl -n llm-serving set image deployment/minimind-inference-artifact minimind=nonexistent-image:bad
```  
Result: new revision is created but pod enters image pull failure state.

18) Confirm failed rollout state.  
```bash
kubectl -n llm-serving rollout status deployment/minimind-inference-artifact --timeout=60s || true
kubectl -n llm-serving get pods
```  
Result: rollout does not complete and failing image state is visible.

19) Roll back to previous healthy revision.  
```bash
kubectl -n llm-serving rollout undo deployment/minimind-inference-artifact && kubectl -n llm-serving rollout status deployment/minimind-inference-artifact --timeout=300s
```  
Result: deployment returns to healthy revision.

20) Re-verify ingress + inference after rollback.  
```bash
MINIKUBE_IP=$(minikube ip) && curl --max-time 20 "http://${MINIKUBE_IP}/docs" -H "Host: minimind.local"
MINIKUBE_IP=$(minikube ip) && curl --max-time 300 -X POST "http://${MINIKUBE_IP}/v1/chat/completions" -H "Host: minimind.local" -H "Content-Type: application/json" -d '{"model":"minimind","messages":[{"role":"user","content":"rollback test"}],"stream":false,"max_tokens":1}'
```  
Result: service is reachable and responses succeed after rollback.

21) Simulate CI/CD release flow (build -> tag -> deploy -> verify) with one command chain.  
```bash
GIT_SHA=$(git rev-parse --short HEAD) && docker build -t minimind-server:${GIT_SHA} -f deployment/docker/Dockerfile . && minikube image load minimind-server:${GIT_SHA} && kubectl -n llm-serving set image deployment/minimind-inference-artifact minimind=minimind-server:${GIT_SHA} && kubectl -n llm-serving rollout status deployment/minimind-inference-artifact --timeout=300s
```  
Result: release process is automated and traceable to a git revision tag. This works on single-GPU local clusters because deployment strategy is configured as `maxUnavailable: 1` and `maxSurge: 0`.

22) Enforce immutable release references (no mutable `latest`).  
```bash
kubectl -n llm-serving get deploy minimind-inference-artifact -o jsonpath='{.spec.template.spec.containers[0].image}{"\n"}'
```  
Result: deployment references a revisioned image tag (for example `minimind-server:<git_sha>`), not `latest`.

23) Externalize secrets at deploy time (no hardcoded credentials in runtime path).  
```bash
kubectl -n llm-serving create secret generic model-artifacts-minio --from-literal=endpoint=http://host.minikube.internal:9000 --from-literal=access_key=minio --from-literal=secret_key=minioadmin --dry-run=client -o yaml | kubectl apply -f -
```  
Result: artifact credentials are injected from Kubernetes Secret object at deploy time.

24) Enforce rollout safety gate before traffic validation.  
```bash
kubectl -n llm-serving rollout status deployment/minimind-inference-artifact --timeout=300s && kubectl -n llm-serving get pods -l app=minimind-inference-artifact
```  
Result: release is considered promotable only when rollout succeeds and pod readiness is healthy.

25) Keep rollback command as a required release control.  
```bash
kubectl -n llm-serving rollout undo deployment/minimind-inference-artifact && kubectl -n llm-serving rollout status deployment/minimind-inference-artifact --timeout=300s
```  
Result: bad release can be reverted quickly with deterministic recovery.

26) Record release evidence for audit and SLO review.  
```bash
kubectl -n llm-serving get deploy,svc,ingress,pods -o wide
```  
Result: deployment state snapshot is available for release notes, incident review, and SLO tracking.

27) Remove Section 10 workloads.  
```bash
kubectl delete -k deployment/k8s/artifact-runtime --ignore-not-found
```  
Result: artifact-runtime resources are deleted.

28) Fully release local infra (clean end state).  
```bash
minikube stop && minikube delete
docker compose -f deployment/mlflow/docker-compose.yaml down
```  
Result: cluster and MinIO backend are stopped/removed; desktop resources are released.

Note: ingress host `minimind.local` is passed via `Host` header in curl, so no `/etc/hosts` edit is required for CLI tests. Stop `minikube tunnel` with `Ctrl+C` when done. GPU reduces timeout risk significantly, but long prompts, cold starts, or cluster contention can still cause occasional timeout spikes.

## 11) Start/Stop Quick Commands (All Backends)

Use these from repo root when you want fast lifecycle control.

Docker inference stack (Section 6):
```bash
# start
docker compose -f deployment/docker/docker-compose.yaml up --build -d

# stop
docker compose -f deployment/docker/docker-compose.yaml down
```

MLflow + MinIO stack (Section 7):
```bash
# start
docker compose -f deployment/mlflow/docker-compose.yaml up -d

# stop
docker compose -f deployment/mlflow/docker-compose.yaml down
```

Section 9 Kubernetes base demo:
```bash
# start cluster
minikube start --driver=docker --cpus=6 --memory=12288 --disk-size=60g --gpus=all

# start helper processes (separate terminals)
minikube mount "$(pwd):/workspace"
kubectl -n llm-serving port-forward svc/minimind-inference-svc 19098:8998

# stop helper processes
pkill -f "minikube mount" || true
pkill -f "kubectl -n llm-serving port-forward svc/minimind-inference-svc 19098:8998" || true

# stop cluster
minikube stop && minikube delete
```

Section 10 Kubernetes artifact runtime:
```bash
# start cluster
minikube start --driver=docker --cpus=6 --memory=12288 --disk-size=60g --gpus=all

# start helper process (separate terminal; enter sudo password)
minikube tunnel

# stop helper process
pkill -f "minikube tunnel" || true

# stop cluster
minikube stop && minikube delete
```

Standalone container cleanup (if previously started via `docker run`):
```bash
docker stop minimind-inference-server || true
docker rm minimind-inference-server || true
```

Full stop (release all local backend resources):
```bash
docker compose -f deployment/docker/docker-compose.yaml down || true
docker compose -f deployment/mlflow/docker-compose.yaml down || true
docker stop minimind-inference-server || true
docker rm minimind-inference-server || true
pkill -f "minikube tunnel" || true
pkill -f "minikube mount" || true
pkill -f "kubectl -n llm-serving port-forward" || true
minikube stop || true
minikube delete || true
docker ps
```
