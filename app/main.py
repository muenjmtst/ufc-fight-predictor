from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel, Field
import joblib
from pathlib import Path
import pandas as pd
import numpy as np

app = FastAPI()

CURRENT_FILE = Path(__file__).resolve() # получить абсолютный путь к текущему файлу
BASE_DIR = CURRENT_FILE.parent.parent # поднялись до корня
MODEL_PATH = BASE_DIR / "models" / "ufc_logistic_regression.joblib"
model_bundle = joblib.load(MODEL_PATH)

feature_columns = model_bundle['feture_columns']

 

class Fighter(BaseModel):
    name:str
    age:int
    reach: int

class Fighterv1(BaseModel):
    age_diff: int
    reach_diff: float
    height_diff: float
    win_streak_diff: int
    lose_streak_diff: int
    wins_diff: int
    losses_diff: int
    experience_diff: int
    win_rate_diff: float | None
    R_is_debutant: int
    B_is_debutant: int
    R_age: int
    R_Height_cms: float
    R_Reach_cms: float
    R_wins: int
    R_losses: int
    R_current_win_streak: int
    R_current_lose_streak: int
    R_total_rounds_fought: int
    R_total_title_bouts: int
    R_avg_SIG_STR_pct: float
    R_avg_TD_landed: float
    R_avg_TD_pct: float
    R_avg_SUB_ATT: float
    R_win_by_KO_TKO: int = Field(alias="R_win_by_KO/TKO")
    R_win_by_Submission: int
    R_Stance: str
    B_age: int
    B_Height_cms: float
    B_Reach_cms: float
    B_wins: int
    B_losses: int
    B_current_win_streak: int
    B_current_lose_streak: int
    B_total_rounds_fought: int
    B_total_title_bouts: int
    B_avg_SIG_STR_pct: float
    B_avg_TD_landed: float
    B_avg_TD_pct: float
    B_avg_SUB_ATT: float
    B_win_by_KO_TKO: int = Field(alias="B_win_by_KO/TKO")
    B_win_by_Submission: int
    B_Stance: str
    weight_class: str
    title_bout: int
    no_of_rounds: int

@app.get('/')
def return_json():
    return { 
        'model': 'ufc_predictor',
        'version': 1.0,
        'is_active': True,
        "fighters": ["Fighter A", "Fighter B"]
    }

@app.get('/health')
def return_health():
    return {
        'status': 'ok'
    }

@app.post('/compare')
def receive_fighter(fighter_a:Fighter, fighter_b:Fighter):
    age_diff = abs(fighter_a.age - fighter_b.age)

    return {
        'fighter_a_name': fighter_a.name,
        'fighter_b_name': fighter_b.name,
        'age_diff': age_diff
    }


@app.get('/model_info')
def get_model_info():
    model = model_bundle['model']
    threshold = model_bundle['threshold']
    n_features = len(model_bundle['feture_columns'])

    return{
        'model_type': type(model['model']).__name__,
        'threshold': threshold,
        'n_features': n_features,
        'status': 'loaded'
    }

@app.post('/predict')
def predict_for_fighter(fighter_a: Fighterv1):
    translate = fighter_a.model_dump(by_alias=True)#чтобы обратно перевести в правильный формат
    dataframe = pd.DataFrame([translate])[feature_columns]#фильтрация по фичам, чтобы лишние не попали


    dataframe["win_rate_diff"] = pd.to_numeric(dataframe["win_rate_diff"], errors="coerce")


    model = model_bundle['model']
    threshold = model_bundle['threshold']

    proba = float(model.predict_proba(dataframe)[0][1])
    red_win_probability = proba
    blue_win_probability = 1 - proba

    

    predictions = int(red_win_probability >= threshold)

    if predictions == 1:
        winner = 'Red'
    else:
        winner = 'Blue'

    return{
        'predicted_winner': winner,
        'red_win_probability': red_win_probability,
        'blue_win_probability': blue_win_probability,
        'threshold': threshold
    }


    