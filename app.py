from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

from sklearn.preprocessing import LabelEncoder
app = FastAPI(title = 'Random Forest Weather Model')
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


model = load(pathlib.Path('models/random_forest/randomforestweather.joblib'))
le_y = load(pathlib.Path('models/random_forest/le_y.joblib'))

class input(BaseModel):
    country: str="AFGANISTAN"
    year: int=2023
    month: int=11
    day: int=6
    hour: int=10

class OutputData(BaseModel):
    condition_text: str="Clear"

@app.post('/weather')
def weather(data: input):
    le = LabelEncoder()
    data.country = le.fit_transform([data.country])[0]

    model_input = np.array([v for k,v in data.dict().items()]).reshape(1,-1)
    result = model.predict(model_input)
    print(result)
    output = le_y.inverse_transform(result)[0]
    print(output)


    
    return {'condition_text': str(output)}
