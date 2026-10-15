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

# --------------------------------------------------
# Application initialization
# --------------------------------------------------

app = FastAPI(
    title="Wine Quality Prediction API",
    description="Predict wine quality using an AWS SageMaker endpoint.",
    version="1.0.0",
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

sagemaker_runtime = boto3.client(
    "sagemaker-runtime",
    region_name=REGION,
)

# The standard 11 input features used by the UCI wine
# quality dataset, in training-data column order.
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
    "alcohol",
]


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def parse_prediction(response_body: str) -> Any:
    """Parse a SageMaker prediction response."""
    try:
        return json.loads(response_body)
    except (json.JSONDecodeError, TypeError):
        return response_body.strip()


def predict_wine_quality(features: list[float]) -> Any:
    """
    Invoke a SageMaker Scikit-learn endpoint.

    This request format is for endpoints that accept
    CSV input and return JSON predictions.
    """
    csv_payload = ",".join(str(value) for value in features)

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Accept="application/json",
        Body=csv_payload,
    )

    body = response["Body"].read().decode("utf-8")
    return parse_prediction(body)


def render_home(
    request: Request,
    *,
    prediction: Any = None,
    error: str | None = None,
    input_values: dict[str, Any] | None = None,
    status_code: int = 200,
):
    """Render the home page using the correct TemplateResponse API."""
    context = {
        "request": request,
        "prediction": prediction,
        "error": error,
        "input_values": input_values or {},
        "endpoint_name": ENDPOINT_NAME,
        "region": REGION,
    }

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=context,
        status_code=status_code,
    )


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Display the wine quality prediction form."""
    return render_home(request)


@app.get("/health")
async def health():
    """Check whether the FastAPI application is running."""
    return {
        "status": "healthy",
        "service": "wine-quality-prediction-api",
        "endpoint": ENDPOINT_NAME,
        "region": REGION,
    }


@app.get("/api/health")
async def api_health():
    """Health check with endpoint configuration."""
    return {
        "status": "healthy",
        "endpoint_name": ENDPOINT_NAME,
        "region": REGION,
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
    alcohol: float = Form(...),
):
    """Validate submitted wine features and call SageMaker."""

    values = {
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
        "alcohol": alcohol,
    }

    # Reject non-finite values and obviously invalid input.
    for name, value in values.items():
        if not math.isfinite(value):
            return render_home(
                request,
                error=f"{name} must be a finite number.",
                input_values=values,
                status_code=400,
            )

    if any(value < 0 for value in values.values()):
        return render_home(
            request,
            error="Wine feature values cannot be negative.",
            input_values=values,
            status_code=400,
        )

    features = [values[name] for name in FEATURE_NAMES]

    try:
        result = predict_wine_quality(features)

        logger.info(
            "Prediction completed using endpoint %s",
            ENDPOINT_NAME,
        )

        return render_home(
            request,
            prediction=result,
            input_values=values,
        )

    except (BotoCoreError, ClientError):
        logger.exception("AWS SageMaker invocation failed")

        return render_home(
            request,
            error=(
                "Could not contact the SageMaker prediction endpoint. "
                "Check the endpoint status, AWS permissions, and server logs."
            ),
            input_values=values,
            status_code=502,
        )

    except Exception:
        logger.exception("Unexpected prediction error")

        return render_home(
            request,
            error="An unexpected error occurred while making the prediction.",
            input_values=values,
            status_code=500,
        )


@app.post("/api/predict")
async def api_predict(request: Request):
    """
    JSON API alternative.

    Example JSON:
    {
      "fixed_acidity": 7.4,
      "volatile_acidity": 0.7,
      "citric_acid": 0.0,
      "residual_sugar": 1.9,
      "chlorides": 0.076,
      "free_sulfur_dioxide": 11,
      "total_sulfur_dioxide": 34,
      "density": 0.9978,
      "pH": 3.51,
      "sulphates": 0.56,
      "alcohol": 9.4
    }
    """
    try:
        data = await request.json()

        if not isinstance(data, dict):
            return JSONResponse(
                status_code=400,
                content={"error": "Request body must be a JSON object."},
            )

        missing = [
            name for name in FEATURE_NAMES
            if name not in data
        ]

        if missing:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Missing required features.",
                    "missing_features": missing,
                },
            )

        features = []

        for name in FEATURE_NAMES:
            try:
                value = float(data[name])
            except (TypeError, ValueError):
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"{name} must be a number."
                    },
                )

            if not math.isfinite(value) or value < 0:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": (
                            f"{name} must be a finite, "
                            "non-negative number."
                        )
                    },
                )

            features.append(value)

        result = predict_wine_quality(features)

        return {
            "prediction": result,
            "endpoint_name": ENDPOINT_NAME,
            "region": REGION,
        }

    except (BotoCoreError, ClientError):
        logger.exception("AWS SageMaker invocation failed")
        return JSONResponse(
            status_code=502,
            content={
                "error": "SageMaker prediction failed. Check server logs."
            },
        )

    except Exception:
        logger.exception("Unexpected API prediction error")
        return JSONResponse(
            status_code=500,
            content={"error": "Unexpected prediction error."},
        )
```
