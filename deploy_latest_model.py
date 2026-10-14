"""
Deploy wine quality model using boto3 directly.
Bypasses SageMaker Python SDK serving bugs entirely.
"""
import boto3
import os
import time
import tarfile
import tempfile
import shutil

region = "us-east-2"
role = os.environ["SAGEMAKER_ROLE_ARN"]
endpoint_name = "wine-quality-endpoint"
model_name = "wine-quality-model"

boto_session = boto3.Session(region_name=region)
sm = boto_session.client("sagemaker")
s3 = boto_session.client("s3")

IMAGE_URI = f"720646828776.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

print(f"Region: {region}")
print(f"Role: {role}")

# ═══════════════════════════════════════════════════════════════════
# STEP 1: Find latest completed training job
# ═══════════════════════════════════════════════════════════════════
print("\n[1/5] Finding latest completed training job...")

jobs = sm.list_training_jobs(
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=20,
)["TrainingJobSummaries"]

latest_job = None
for job in jobs:
    if job["TrainingJobStatus"] == "Completed":
        latest_job = job["TrainingJobName"]
        break

if latest_job is None:
    raise Exception("No completed training job found")

print(f"  Training job: {latest_job}")

job_details = sm.describe_training_job(TrainingJobName=latest_job)
model_artifact = job_details["ModelArtifacts"]["S3ModelArtifacts"]
print(f"  Model artifact: {model_artifact}")

# ═══════════════════════════════════════════════════════════════════
# STEP 2: Repackage model.tar.gz with inference code inside
# ═══════════════════════════════════════════════════════════════════
print("\n[2/5] Repackaging model with inference code...")

bucket = model_artifact.split("/")[2]
key = "/".join(model_artifact.split("/")[3:])

tmpdir = tempfile.mkdtemp()
original_tar = os.path.join(tmpdir, "original.tar.gz")
extract_dir = os.path.join(tmpdir, "contents")
new_tar = os.path.join(tmpdir, "model.tar.gz")

s3.download_file(bucket, key, original_tar)

os.makedirs(extract_dir, exist_ok=True)
with tarfile.open(original_tar, "r:gz") as tar:
    tar.extractall(extract_dir)
print(f"  Extracted: {os.listdir(extract_dir)}")

# Remove any old code/ directory
code_dir = os.path.join(extract_dir, "code")
if os.path.exists(code_dir):
    shutil.rmtree(code_dir)
os.makedirs(code_dir)

# Write inference.py
with open(os.path.join(code_dir, "inference.py"), "w") as f:
    f.write("""import joblib
import os
import numpy as np
import xgboost as xgb


FEATURE_NAMES = [
    "Wine", "Alcohol", "Malic.acid", "Ash", "Acl", "Mg",
    "Phenols", "Flavanoids", "Nonflavanoid.phenols", "Proanth",
    "Color.int", "Hue", "OD"
]


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model


def input_fn(request_body, content_type):
    if content_type == "text/csv":
        lines = request_body.strip().split("\\n")
        parsed = []
        for line in lines:
            row = [float(x.strip()) for x in line.split(",")]
            parsed.append(row)
        return np.array(parsed)
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    dmatrix = xgb.DMatrix(input_data, feature_names=FEATURE_NAMES)
    prediction = model.get_booster().predict(dmatrix)
    return prediction


def output_fn(prediction, accept):
    return ",".join(str(round(float(p), 4)) for p in prediction)
""")

# Write MINIMAL requirements.txt - ONLY xgboost
with open(os.path.join(code_dir, "requirements.txt"), "w") as f:
    f.write("xgboost==2.0.3\n")

print(f"  code/ contents: {os.listdir(code_dir)}")

# Repackage tar
with tarfile.open(new_tar, "w:gz") as tar:
    for item in os.listdir(extract_dir):
        tar.add(os.path.join(extract_dir, item), arcname=item)

# Upload
new_key = f"models/deploy/model-{int(time.time())}.tar.gz"
s3.upload_file(new_tar, bucket, new_key)
new_model_uri = f"s3://{bucket}/{new_key}"
print(f"  Uploaded: {new_model_uri}")

shutil.rmtree(tmpdir)

# ═══════════════════════════════════════════════════════════════════
# STEP 3: Cleanup ALL existing resources
# ═══════════════════════════════════════════════════════════════════
print("\n[3/5] Cleaning up existing resources...")

# Delete endpoint (wait if in-progress)
try:
    ep = sm.describe_endpoint(EndpointName=endpoint_name)
    status = ep["EndpointStatus"]
    print(f"  Endpoint status: {status}")

    # Wait for any in-progress state to finish
    while status in ("Creating", "Updating", "RollingBack", "Deleting"):
        print(f"  Endpoint is {status}, waiting 30s...")
        time.sleep(30)
        try:
            status = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
        except sm.exceptions.ClientError:
            print("  Endpoint gone")
            status = "Gone"
            break

    if status != "Gone":
        # Now it's InService or Failed — safe to delete
        sm.delete_endpoint(EndpointName=endpoint_name)
        print("  Delete requested, waiting for removal...")
        while True:
            try:
                time.sleep(15)
                sm.describe_endpoint(EndpointName=endpoint_name)
            except sm.exceptions.ClientError:
                print("  Endpoint deleted")
                break

except sm.exceptions.ClientError:
    print("  No existing endpoint")

# Delete endpoint configs
try:
    configs = sm.list_endpoint_configs(NameContains="wine-quality")
    for cfg in configs.get("EndpointConfigs", []):
        sm.delete_endpoint_config(EndpointConfigName=cfg["EndpointConfigName"])
        print(f"  Deleted config: {cfg['EndpointConfigName']}")
except Exception:
    pass

# Delete old models
try:
    sm.delete_model(ModelName=model_name)
    print(f"  Deleted model: {model_name}")
except Exception:
    pass

try:
    models = sm.list_models(SortBy="CreationTime", SortOrder="Descending", MaxResults=10)
    for m in models["Models"]:
        mn = m["ModelName"]
        if "wine" in mn.lower() or "sagemaker-scikit" in mn.lower():
            sm.delete_model(ModelName=mn)
            print(f"  Deleted model: {mn}")
except Exception:
    pass

print("  Waiting 15s...")
time.sleep(15)

# ═══════════════════════════════════════════════════════════════════
# STEP 4: Create model with EXPLICIT env vars (no SDK magic)
# ═══════════════════════════════════════════════════════════════════
print("\n[4/5] Creating SageMaker model...")

sm.create_model(
    ModelName=model_name,
    PrimaryContainer={
        "Image": IMAGE_URI,
        "ModelDataUrl": new_model_uri,
        "Environment": {
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
            "SAGEMAKER_REGION": region,
        },
    },
    ExecutionRoleArn=role,
)
print(f"  Model created: {model_name}")

# ═══════════════════════════════════════════════════════════════════
# STEP 5: Create endpoint config + endpoint
# ═══════════════════════════════════════════════════════════════════
print("\n[5/5] Creating endpoint...")

config_name = f"{endpoint_name}-config-{int(time.time())}"

sm.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[
        {
            "VariantName": "primary",
            "ModelName": model_name,
            "InstanceType": "ml.m5.large",
            "InitialInstanceCount": 1,
        }
    ],
)
print(f"  Endpoint config: {config_name}")

sm.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=config_name,
)
print(f"  Endpoint creation started: {endpoint_name}")

# Wait for InService
print("  Waiting for endpoint to be InService...")
waiter = sm.get_waiter("endpoint_in_service")
waiter.wait(
    EndpointName=endpoint_name,
    WaiterConfig={"Delay": 30, "MaxAttempts": 40},
)

final_status = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
print(f"\n{'='*60}")
print(f"  ENDPOINT STATUS: {final_status}")
print(f"  ENDPOINT NAME:   {endpoint_name}")
print(f"{'='*60}")
