import mlflow
import subprocess
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("llm-inference-industrial")
git_sha = subprocess.check_output(
  ["git", "rev-parse", "--short", "HEAD"], text=True
).strip()
with mlflow.start_run(run_name="minimind_v1_0_0_candidate"):
  mlflow.log_param("model_id", "minimind-local")
  mlflow.log_param("model_version", "v1.0.0")
  mlflow.log_param("weights_path", "out/full_sft_768.pth")   # adjust if needed
  mlflow.log_param("tokenizer_path", "model/")               # adjust if needed
  mlflow.log_param("git_commit", git_sha)
  mlflow.log_param("training_recipe", "full_sft")
  mlflow.log_metric("smoke_pass", 1)
  mlflow.set_tag("status", "candidate")
  mlflow.set_tag("stage", "dev")
  mlflow.set_tag("owner", "kenneth")