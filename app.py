# src/Classifier/app/main.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.Classifier.components.model_prediction import PredictionPipeline

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(
    Acres: float = Form(...),
    Distance_to_Substation_Miles_CAISO: float = Form(...),
    Distance_to_Substation_Miles_GTET_100_Max_Voltage: float = Form(...),
    Distance_to_Substation_Miles_GTET_200_Max_Voltage: float = Form(...),
    Shape__Length: float = Form(...),
    Shape__Area: float = Form(...),
    Percentile_GTET_100_Max_Voltage: str = Form(...),
    Percentile_GTET_200_Max_Voltage: str = Form(...),
    Percentile_CAISO: str = Form(...),
    Install_Type: str = Form(...),
    Urban_or_Rural: str = Form(...),
    County: str = Form(...)
):
    input_data = {
        "Acres": Acres,
        "Distance_to_Substation_Miles_CAISO": Distance_to_Substation_Miles_CAISO,
        "Distance_to_Substation_Miles_GTET_100_Max_Voltage": Distance_to_Substation_Miles_GTET_100_Max_Voltage,
        "Distance_to_Substation_Miles_GTET_200_Max_Voltage": Distance_to_Substation_Miles_GTET_200_Max_Voltage,
        "Shape__Length": Shape__Length,
        "Shape__Area": Shape__Area,
        "Percentile_GTET_100_Max_Voltage": Percentile_GTET_100_Max_Voltage,
        "Percentile_GTET_200_Max_Voltage": Percentile_GTET_200_Max_Voltage,
        "Percentile_CAISO": Percentile_CAISO,
        "Install_Type": Install_Type,
        "Urban_or_Rural": Urban_or_Rural,
        "County": County
    }
    pipeline = PredictionPipeline()
    prediction = pipeline.predict(input_data)

    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})
