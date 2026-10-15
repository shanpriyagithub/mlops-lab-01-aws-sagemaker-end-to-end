import boto3
import json
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# SageMaker endpoint name
ENDPOINT_NAME = "wine-quality-endpoint"
REGION = "us-east-2"

# SageMaker runtime client
runtime = boto3.client(
    "sagemaker-runtime",
    region_name=REGION
)

# Create the app BEFORE defining routes
app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )
    
# Prediction API
@app.post("/predict")
def predict(
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
    alcohol: float = Form(...)
):

    # convert to CSV format with headers
    headers = "fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol"
    data = f"{fixed_acidity},{volatile_acidity},{citric_acid},{residual_sugar},{chlorides},{free_sulfur_dioxide},{total_sulfur_dioxide},{density},{pH},{sulphates},{alcohol}"
    payload = f"{headers}\n{data}"

    try:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=payload
        )

        result = response["Body"].read().decode()
        
        # Parse the JSON response from SageMaker
        prediction_response = json.loads(result)
        
        # Extract prediction value from the response
        if isinstance(prediction_response, dict) and "predictions" in prediction_response:
            prediction_value = prediction_response["predictions"][0]["score"]
        elif isinstance(prediction_response, list):
            prediction_value = prediction_response[0]["score"]
        else:
            prediction_value = result

        return {
            "prediction": str(round(float(prediction_value), 2))
        }
    except Exception as e:
        return {
            "prediction": f"Error: {str(e)}"
        }
