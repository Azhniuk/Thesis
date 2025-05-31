from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
import os

try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception:
    credential = InteractiveBrowserCredential()

subscription_id = "c6da0d49-2d16-43f8-942e-500b5bf9da7a"
resource_group = "Sofi"
workspace_name = "SpermRecog"

# Подключение к workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name
)
yolo_env = Environment(
    name="yolov8-cpu-env",
    description="Environment for YOLOv8 training on CPU",
    conda_file="conda_cpu.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"  
)

compute_name = "spermcluster"  
job = command(
    code="./",  
    command="python train_yolo_original.py",
    environment=yolo_env,
    compute=compute_name,
    display_name="yolov8n-batch32-imgsz1024-cpu",
    experiment_name="yolov8-training"
)

returned_job = ml_client.jobs.create_or_update(job)
print(f"Submitted job: {returned_job.name}")
print(f"Job URL: {returned_job.studio_url}")