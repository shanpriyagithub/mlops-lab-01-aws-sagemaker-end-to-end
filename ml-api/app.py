```python
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
    description="Predict wine quality using AWS SageMaker.",
    version="1.0.0",
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

sagemaker_runtime = boto3.client(
    "sagemaker-runtime",
    region_name=REGION,
)

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
# Helpers
# --------------------------------------------------

def render_page(
    request: Request,
    prediction: Any = None,
    error: str = None,
    features: dict = None,
    status_code: int = 200,
):
    """Render index.html with a consistent context."""

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "prediction": prediction,
            "error": error,
            "features": features or {},
        },
        status_code=status_code,
    )


def get_prediction(values: list[float]) -> Any:
    """Call SageMaker and parse its response."""

    payload = ",".join(str(value) for value in values)

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Accept="text/csv",
        Body=payload,
    )

    result = response["Body"].read().decode("utf-8").strip()

    try:
        parsed = json.loads(result)

        if isinstance(parsed, dict):
            return parsed.get(
                "prediction",
                parsed.get("predictions", parsed),
            )

        return parsed

    except json.JSONDecodeError:
        return result


def validate_features(data: dict) -> list[float]:
    """Validate and return values in model training order."""

    missing = [name for name in FEATURE_NAMES if name not in data]

    if missing:
        raise ValueError(
            "Missing required features: " + ", ".join(missing)
        )

    values = [float(data[name]) for name in FEATURE_NAMES]

    if not all(math.isfinite(value) for value in values):
        raise ValueError("All feature values must be finite numbers.")

    return values


# --------------------------------------------------
# Web pages and health check
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request},
    )

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "region": REGION,
        "endpoint": ENDPOINT_NAME,
    }


# --------------------------------------------------
# HTML prediction form
# --------------------------------------------------

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
        values = validate_features(features)
        prediction = get_prediction(values)

        return render_page(
            request=request,
            prediction=prediction,
            features=features,
        )

    except ValueError as exc:
        logger.warning("Invalid wine features: %s", exc)

        return render_page(
            request=request,
            error=str(exc),
            features=features,
            status_code=400,
        )

    except (BotoCoreError, ClientError):
        logger.exception("SageMaker prediction request failed")

        return render_page(
            request=request,
            error=(
                "Unable to get a prediction from SageMaker. "
                "Check that the endpoint is InService and that "
                "the EC2 IAM role has permission to invoke it."
            ),
            features=features,
            status_code=502,
        )

    except Exception:
        logger.exception("Unexpected prediction error")

        return render_page(
            request=request,
            error="An unexpected error occurred. Check application logs.",
            features=features,
            status_code=500,
        )


# --------------------------------------------------
# JSON prediction API
# --------------------------------------------------

@app.post("/predict-json")
async def predict_json(request: Request):
    try:
        data = await request.json()

        if not isinstance(data, dict):
            return JSONResponse(
                status_code=400,
                content={"error": "Expected a JSON object."},
            )

        values = validate_features(data)
        prediction = get_prediction(values)

        return {
            "prediction": prediction,
            "endpoint": ENDPOINT_NAME,
            "region": REGION,
        }

    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": str(exc)},
        )

    except (BotoCoreError, ClientError):
        logger.exception("SageMaker JSON prediction request failed")

        return JSONResponse(
            status_code=502,
            content={"error": "SageMaker prediction request failed."},
        )

    except Exception:
        logger.exception("Unexpected JSON prediction error")

        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error."},
        )
```
