"""
This is app code which is backend for model- it uses model to give out prediction, it is backbone of the system
"""

# src/Classifier/app/main.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.Classifier.components.model_prediction import PredictionPipeline
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    Acres: float = Form(...),
    Distance_CAISO: float = Form(...),
    Distance_GTET100: float = Form(...),
    Distance_GTET200: float = Form(...),
    Shape__Length: float = Form(...),
    Shape__Area: float = Form(...),
    Percentile_GTET100: str = Form(...),
    Percentile_GTET200: str = Form(...),
    Percentile_CAISO: str = Form(...),
    Install_Type: str = Form(...),
    Urban_Rural: str = Form(...),
    County: str = Form(...)
):
    input_data = {
        "Acres": Acres,
        "Distance to Substation (Miles) CAISO": Distance_CAISO,
        "Distance to Substation (Miles) GTET 100 Max Voltage": Distance_GTET100,
        "Distance to Substation (Miles) GTET 200 Max Voltage": Distance_GTET200,
        "Shape__Length": Shape__Length,
        "Shape__Area": Shape__Area,
        "Percentile (GTET 100 Max Voltage)": Percentile_GTET100,
        "Percentile (GTET 200 Max Voltage)": Percentile_GTET200,
        "Percentile (CAISO)": Percentile_CAISO,
        "Install Type": Install_Type,
        "Urban or Rural": Urban_Rural,
        "County": County
    }
    pipeline = PredictionPipeline()
    prediction = pipeline.predict(input_data)
    if prediction == 1:
        prediction = "Is Fesible to have Solar Panels"
    else:
        prediction = "Is Not Feasible to have Solar Panels"

    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})
