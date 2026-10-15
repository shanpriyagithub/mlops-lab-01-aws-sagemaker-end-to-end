
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"

REGION = os.getenv("AWS_REGION", "us-east-2")
ENDPOINT_NAME = os.getenv(
    "SAGEMAKER_ENDPOINT_NAME",
    "wine-quality-endpoint",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Wine Quality Prediction API",
    description="Predict wine quality using an AWS SageMaker endpoint.",
    version="1.0.0",
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Uses the EC2 instance IAM role or configured AWS credentials.
sagemaker_runtime = boto3.client(
    "sagemaker-runtime",
    region_name=REGION,
)

# The feature order must match the order used during model training.
FEATURE_NAMES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
]


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Display the wine prediction form."""

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "prediction": None,
            "error": None,
            "features": {},
        },
    )


@app.get("/health")
async def health():
    """Check whether the API application is running."""

    return {
        "status": "healthy",
        "region": REGION,
        "endpoint": ENDPOINT_NAME,
    }


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    fixed_acidity: float = Form(...),
    volatile_acidity: float = Form(...),
    citric_acid: float = Form(...),
    residual_sugar: float = Form(...),
    chlorides: float = Form(...),
    free_sulfur_dioxide: float = Form(...),
    total_sulfur_dioxide: float = Form(...),
    density: float = Form(...),
    pH: float = Form(...),
    sulphates: float = Form(...),
):
    """Validate wine features and request a SageMaker prediction."""

    features = {
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
    }

    try:
        values = [float(features[name]) for name in FEATURE_NAMES]

        if not all(math.isfinite(value) for value in values):
            raise ValueError("All feature values must be finite numbers.")

        # CSV format: one row, with features in training order.
        payload = ",".join(str(value) for value in values)

        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Accept="text/csv",
            Body=payload,
        )

        result = response["Body"].read().decode("utf-8").strip()

        # Supports a simple numeric prediction or a JSON response.
        try:
            parsed: Any = json.loads(result)
            if isinstance(parsed, dict):
                prediction = parsed.get(
                    "prediction",
                    parsed.get("predictions", parsed),
                )
            else:
                prediction = parsed
        except json.JSONDecodeError:
            prediction = result

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "prediction": prediction,
                "error": None,
                "features": features,
            },
        )

    except ValueError as exc:
        logger.warning("Invalid wine features: %s", exc)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "prediction": None,
                "error": str(exc),
                "features": features,
            },
            status_code=400,
        )

    except (BotoCoreError, ClientError) as exc:
        logger.exception("SageMaker prediction request failed")
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "prediction": None,
                "error": (
                    "Unable to get a prediction from SageMaker. "
                    "Check the endpoint status and EC2 IAM permissions."
                ),
                "features": features,
            },
            status_code=502,
        )

    except Exception:
        logger.exception("Unexpected prediction error")
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "prediction": None,
                "error": "An unexpected error occurred. Check the application logs.",
                "features": features,
            },
            status_code=500,
        )


# --------------------------------------------------
# JSON API for testing and integration
# --------------------------------------------------

@app.post("/predict-json")
async def predict_json(request: Request):
    """Accept a JSON object containing all ten wine features."""

    try:
        data = await request.json()

        missing = [name for name in FEATURE_NAMES if name not in data]
        if missing:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing features", "fields": missing},
            )

        values = [float(data[name]) for name in FEATURE_NAMES]

        if not all(math.isfinite(value) for value in values):
            return JSONResponse(
                status_code=400,
                content={"error": "All feature values must be finite numbers."},
            )

        payload = ",".join(str(value) for value in values)

        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Accept="text/csv",
            Body=payload,
        )

        result = response["Body"].read().decode("utf-8").strip()

        try:
            prediction = json.loads(result)
        except json.JSONDecodeError:
            prediction = result

        return {
            "prediction": prediction,
            "endpoint": ENDPOINT_NAME,
            "region": REGION,
        }

    except (BotoCoreError, ClientError):
        logger.exception("SageMaker JSON prediction request failed")
        return JSONResponse(
            status_code=502,
            content={"error": "SageMaker prediction request failed."},
        )

    except (ValueError, TypeError):
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid feature values."},
        )

    except Exception:
        logger.exception("Unexpected JSON prediction error")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error."},
        )
