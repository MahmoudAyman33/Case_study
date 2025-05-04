import joblib 
import numpy as np
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
app = FastAPI()

MODEL_PATH = Path(r"C:/Users/4t4/Desktop/ml training apptrainers/task4/xgb_heart_disease.pkl")
SCALER_PATH = Path(r"C:/Users/4t4/Desktop/ml training apptrainers/task4/scaler_heart_disease.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

def predict_calories(input_data):
    scaled_input = scaler.transform(input_data)
    return model.predict(scaled_input)[0]


@app.get("/")
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": "--"})

#Gender	Age	Height	Weight	Duration	Heart_Rate	Body_Temp	Calories

@app.post("/predict/", response_class=HTMLResponse)
async def predict(
    request: Request,
    gender: int = Form(...),
    age: int = Form(...),
    height: float = Form(...),
    weight: float = Form(...),
    duration: float = Form(...),
    heart_rate: float = Form(...),
    body_temp: float = Form(...)
):
    input_data = np.array([[gender, age, height, weight, duration, heart_rate, body_temp]])

    predicted_calories = predict_calories(input_data)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": round(predicted_calories, 2)
    })