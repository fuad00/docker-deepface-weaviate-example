from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import numpy as np
import cv2
import os

app = FastAPI(title="DeepFace Weaviate API")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Читаем адрес Weaviate из переменной окружения
WEAVIATE_URL = os.getenv("DEEPFACE_CONNECTION_DETAILS",
                         "http://weaviate_db:8080")
MODEL_NAME = "Facenet"
DB_TYPE = "weaviate"


async def read_image(file: UploadFile):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    return img


@app.get("/")
async def root():
    return {"message": "DeepFace API is running"}


@app.post("/register")
async def register(img_name: str = Form(...), file: UploadFile = File(...)):
    img = await read_image(file)
    try:
        # Регистрация лица
        result = DeepFace.register(
            img=img,
            img_name=img_name,
            model_name=MODEL_NAME,
            database_type=DB_TYPE,
            connection_details=WEAVIATE_URL
        )
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search(file: UploadFile = File(...)):
    img = await read_image(file)
    try:
        results = DeepFace.search(
            img=img,
            model_name=MODEL_NAME,
            database_type=DB_TYPE,
            connection_details=WEAVIATE_URL,
            distance_metric="cosine"
        )

        # Парсим результаты DataFrame в JSON
        matches = []
        for df in results:
            if not df.empty:
                matches.extend(
                    df[['img_name', 'distance', 'confidence']].to_dict(orient="records"))

        return {"matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
