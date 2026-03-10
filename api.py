from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load('cancer_model.pkl')

@app.get('/')
def home():
    return{'message':'Cancer prediction API'}
@app.post('/predict')
def predict(data: list):
    prediction = model.predict([data])
    return {'prediction':int(prediction[0])}