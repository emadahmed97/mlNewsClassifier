import os
import subprocess
import requests

from src.config import MODEL_REGISTRY, logger
from src.serve import ModelDeployment

github_username = os.environ.get("GITHUB_USERNAME")

logger.info("hit serve_model")
subprocess.check_output(["aws", "s3", "cp", f"s3://ml-news-classifier/mlflow/", str(MODEL_REGISTRY), "--recursive"])
subprocess.check_output(["aws", "s3", "cp", f"s3://ml-news-classifier/results/", "./", "--recursive"])

run_id = [line.strip() for line in open("run_id.txt")][0]
#logger.info("hit run_id")
#logger.info(run_id)
entrypoint = ModelDeployment.bind(run_id='a8f8972823b743bfa77dba1073e085cb', threshold = 0.9)


# Inference
data = {"query": "What is the default batch size for map_batches?"}
response = requests.post("http://127.0.0.1:8000/query", json=data)
print(response.json())


# Inference
data = {"query": "What is the default batch size for map_batches?"}
response = requests.post("http://127.0.0.1:8000/query", json=data)
print(response.json())