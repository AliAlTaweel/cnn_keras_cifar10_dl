from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from src.model import load_model
from src.predict import predict_image

app = FastAPI()

model = load_model()

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    prediction = predict_image(model, image)
    
    return {
        "prediction": prediction
    }