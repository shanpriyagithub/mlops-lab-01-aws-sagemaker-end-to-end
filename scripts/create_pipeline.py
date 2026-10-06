import os
import boto3
import sagemaker

from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep


REGION = os.environ["AWS_REGION"]
ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
BUCKET = os.environ["S3_BUCKET"]
MLFLOW_URI = os.environ["MLFLOW_TRACKING_URI"].rstrip("/")

PIPELINE_NAME = "wine-mlflow-pipeline"


print(f"DEBUG: AWS_REGION={REGION}")
print(f"DEBUG: S3_BUCKET={BUCKET}")
print(f"DEBUG: SAGEMAKER_ROLE_ARN={ROLE_ARN}")
print(f"DEBUG: MLFLOW_TRACKING_URI={MLFLOW_URI}")


try:
    # ----------------------------------------
    # Create SageMaker session
    # ----------------------------------------
    sess = sagemaker.Session(
        boto3.Session(region_name=REGION)
    )

    train_s3_uri = f"s3://{BUCKET}/data/wine.csv"

    # ----------------------------------------
    # Create SageMaker estimator
    # ----------------------------------------
    estimator = SKLearn(
        entry_point="train_with_mlflow.py",
        source_dir="scripts",
        role=ROLE_ARN,
        instance_type="ml.m5.large",
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=sess,
        environment={
            "MLFLOW_TRACKING_URI": MLFLOW_URI
        },
        output_path=f"s3://{BUCKET}/models"
    )

    # ----------------------------------------
    # Create training step
    # ----------------------------------------
    step_train = TrainingStep(
        name="TrainWineModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=train_s3_uri,
                content_type="text/csv"
            )
        }
    )

    # ----------------------------------------
    # Create pipeline
    # ----------------------------------------
    pipeline = Pipeline(
        name=PIPELINE_NAME,
        steps=[step_train],
        sagemaker_session=sess
    )

    # ----------------------------------------
    # Upsert pipeline
    # ----------------------------------------
    print(
        f"DEBUG: Upserting pipeline "
        f"'{PIPELINE_NAME}' in region '{REGION}'..."
    )

    pipeline.upsert(role_arn=ROLE_ARN)

    # ----------------------------------------
    # Start pipeline execution
    # ----------------------------------------
    print(
        f"DEBUG: Starting pipeline execution "
        f"for '{PIPELINE_NAME}'..."
    )

    execution = pipeline.start()

    print("Pipeline started:", execution.arn)

    # ----------------------------------------
    # Wait for pipeline execution
    # ----------------------------------------
    try:
        execution.wait()

    except Exception as e:
        print("\n========================================")
        print("PIPELINE EXECUTION FAILED")
        print("========================================")
        print("Execution ARN:", execution.arn)
        print("Error:", e)

        print("\nPipeline execution details:")
        print(execution.describe())

        print("\nStep details:")

        steps = execution.list_steps()

        for step in steps:
            print("----------------------------------------")
            print("Step:", step.get("StepName"))
            print("Status:", step.get("StepStatus"))
            print("Failure:", step.get("FailureReason"))
            print("Metadata:", step.get("Metadata"))

        raise

    print("\n========================================")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("========================================")

    print("Execution ARN:", execution.arn)

    print("\nStep details:")

    for step in execution.list_steps():
        print("----------------------------------------")
        print("Step:", step.get("StepName"))
        print("Status:", step.get("StepStatus"))
        print("Failure:", step.get("FailureReason"))
        print("Metadata:", step.get("Metadata"))


except Exception as e:
    print("\n========================================")
    print("PIPELINE CREATION/START FAILED")
    print("========================================")
    print(f"ERROR: {str(e)}")

    import traceback
    traceback.print_exc()

    raise
