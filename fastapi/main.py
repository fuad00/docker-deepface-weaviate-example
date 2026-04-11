from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import numpy as np
import cv2
import os

from models import (
    AnalyzeResponse,
    VerifyResponse,
    RepresentResponse,
    DetectResponse,
    RegisterResponse,
    SearchResponse,
    ModelsInfoResponse,
    ErrorResponse,
)

# ── Constants ───────────────────────────────────────────────────────────────

RECOGNITION_MODELS = [
    "VGG-Face", "Facenet", "Facenet512", "OpenFace",
    "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace",
    "GhostFaceNet",
]

DETECTOR_BACKENDS = [
    "opencv", "ssd", "dlib", "mtcnn", "fastmtcnn",
    "retinaface", "mediapipe", "yolov8", "yunet", "centerface",
]

DISTANCE_METRICS = ["cosine", "euclidean", "euclidean_l2"]

ANALYSIS_ACTIONS = ["age", "gender", "race", "emotion"]

# ── Weaviate config ─────────────────────────────────────────────────────────

WEAVIATE_URL = os.getenv("DEEPFACE_CONNECTION_DETAILS", "http://weaviate_db:8080")
MODEL_NAME = os.getenv("DEEPFACE_MODEL", "Facenet")
DB_TYPE = "weaviate"

# ── App ─────────────────────────────────────────────────────────────────────

DESCRIPTION = """
## Face Recognition & Analysis API

Powered by **DeepFace** (multi-model face recognition library) and **Weaviate** (vector database).

### Capabilities

| Group | What it does |
|-------|-------------|
| **Analysis** | Predict age, gender, emotion, race from a face photo |
| **Verification** | Compare two photos — same person or not? |
| **Embeddings** | Extract numerical face vectors (128-d to 4096-d depending on model) |
| **Detection** | Find and locate all faces in an image |
| **Database** | Register faces into Weaviate and search by similarity |
"""

app = FastAPI(
    title="DeepFace API",
    description=DESCRIPTION,
    version="1.0.0",
    openapi_tags=[
        {"name": "Health", "description": "Service health and status"},
        {
            "name": "Analysis",
            "description": "Facial attribute analysis — age, gender, emotion, race",
        },
        {
            "name": "Verification",
            "description": "Face verification — determine if two images show the same person",
        },
        {
            "name": "Embeddings",
            "description": "Extract facial embedding vectors for downstream use",
        },
        {
            "name": "Detection",
            "description": "Detect and locate faces in an image",
        },
        {
            "name": "Database",
            "description": "Weaviate vector DB — register and search faces by similarity",
        },
        {
            "name": "Info",
            "description": "Available models, detectors, and metrics",
        },
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ─────────────────────────────────────────────────────────────────


async def read_image(file: UploadFile) -> np.ndarray:
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image. Send a valid JPEG/PNG file.")
    return img


# ── Health ──────────────────────────────────────────────────────────────────


@app.get("/", tags=["Health"], summary="Health check")
async def root():
    """Returns service status."""
    return {"status": "ok", "message": "DeepFace API is running"}


# ── Analysis ────────────────────────────────────────────────────────────────


@app.post(
    "/api/v1/analyze",
    tags=["Analysis"],
    summary="Analyze facial attributes",
    response_model=AnalyzeResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def analyze(
    file: UploadFile = File(..., description="Face image (JPEG/PNG)"),
    actions: list[str] = Query(
        default=["age", "gender", "race", "emotion"],
        description="Attributes to analyze: age, gender, race, emotion",
    ),
    detector_backend: str = Query(
        default="opencv",
        description="Face detector backend",
        enum=DETECTOR_BACKENDS,
    ),
    enforce_detection: bool = Query(
        default=True,
        description="Raise error if no face detected",
    ),
    align: bool = Query(default=True, description="Align face before analysis"),
):
    """
    Analyze facial attributes in an uploaded image.

    Returns predictions for each detected face: **age**, **gender** (with probabilities),
    **dominant emotion** (angry, disgust, fear, happy, sad, surprise, neutral),
    and **dominant race** (asian, indian, black, white, middle eastern, latino hispanic).
    """
    img = await read_image(file)
    try:
        results = DeepFace.analyze(
            img_path=img,
            actions=actions,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            silent=True,
        )
        return {"results": results}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Verification ────────────────────────────────────────────────────────────


@app.post(
    "/api/v1/verify",
    tags=["Verification"],
    summary="Verify two faces",
    response_model=VerifyResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def verify(
    file1: UploadFile = File(..., description="First face image"),
    file2: UploadFile = File(..., description="Second face image"),
    model_name: str = Query(
        default="VGG-Face",
        description="Recognition model",
        enum=RECOGNITION_MODELS,
    ),
    distance_metric: str = Query(
        default="cosine",
        description="Distance metric for comparison",
        enum=DISTANCE_METRICS,
    ),
    detector_backend: str = Query(
        default="opencv",
        description="Face detector backend",
        enum=DETECTOR_BACKENDS,
    ),
    enforce_detection: bool = Query(
        default=True,
        description="Raise error if no face detected",
    ),
):
    """
    Compare two face images and determine if they belong to the same person.

    Returns `verified: true/false`, the computed **distance**, and the **threshold**
    used for the decision. Lower distance = more similar.
    """
    img1 = await read_image(file1)
    img2 = await read_image(file2)
    try:
        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name=model_name,
            distance_metric=distance_metric,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            silent=True,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Embeddings ──────────────────────────────────────────────────────────────


@app.post(
    "/api/v1/represent",
    tags=["Embeddings"],
    summary="Extract face embeddings",
    response_model=RepresentResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def represent(
    file: UploadFile = File(..., description="Face image (JPEG/PNG)"),
    model_name: str = Query(
        default="VGG-Face",
        description="Recognition model (determines embedding dimensions)",
        enum=RECOGNITION_MODELS,
    ),
    detector_backend: str = Query(
        default="opencv",
        description="Face detector backend",
        enum=DETECTOR_BACKENDS,
    ),
    enforce_detection: bool = Query(
        default=True,
        description="Raise error if no face detected",
    ),
    align: bool = Query(default=True, description="Align face before embedding"),
):
    """
    Extract facial embedding vectors from an image.

    Embedding dimensions vary by model: VGG-Face=4096, Facenet=128,
    Facenet512=512, ArcFace=512, etc.
    """
    img = await read_image(file)
    try:
        results = DeepFace.represent(
            img_path=img,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            normalization="base",
        )
        return {"results": results}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Detection ───────────────────────────────────────────────────────────────


@app.post(
    "/api/v1/detect",
    tags=["Detection"],
    summary="Detect faces in image",
    response_model=DetectResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def detect(
    file: UploadFile = File(..., description="Image to scan for faces (JPEG/PNG)"),
    detector_backend: str = Query(
        default="opencv",
        description="Face detector backend",
        enum=DETECTOR_BACKENDS,
    ),
    enforce_detection: bool = Query(
        default=True,
        description="Raise error if no face detected",
    ),
    align: bool = Query(default=True, description="Align detected faces"),
):
    """
    Detect and locate all faces in an image.

    Returns bounding box coordinates and detection confidence for each face.
    Does **not** return the face pixel data (use this for counting/locating faces).
    """
    img = await read_image(file)
    try:
        faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
        )
        result = []
        for f in faces:
            result.append({
                "facial_area": f.get("facial_area", {}),
                "confidence": f.get("confidence"),
            })
        return {"faces": result, "count": len(result)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Weaviate Database ───────────────────────────────────────────────────────


@app.post(
    "/api/v1/register",
    tags=["Database"],
    summary="Register a face in Weaviate",
    response_model=RegisterResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def register(
    file: UploadFile = File(..., description="Face image to register"),
    img_name: str = Form(..., description="Unique name/label for this face"),
    model_name: str = Query(
        default=None,
        description=f"Recognition model (default: {MODEL_NAME})",
        enum=RECOGNITION_MODELS,
    ),
):
    """
    Register a face embedding in the Weaviate vector database.

    The face is encoded using the selected model and stored with the given `img_name`.
    Later you can search for similar faces via `/api/v1/search`.
    """
    img = await read_image(file)
    model = model_name or MODEL_NAME
    try:
        result = DeepFace.register(
            img=img,
            img_name=img_name,
            model_name=model,
            database_type=DB_TYPE,
            connection_details=WEAVIATE_URL,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/search",
    tags=["Database"],
    summary="Search for matching faces in Weaviate",
    response_model=SearchResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def search(
    file: UploadFile = File(..., description="Query face image"),
    model_name: str = Query(
        default=None,
        description=f"Recognition model (default: {MODEL_NAME})",
        enum=RECOGNITION_MODELS,
    ),
    distance_metric: str = Query(
        default="cosine",
        description="Distance metric",
        enum=DISTANCE_METRICS,
    ),
):
    """
    Search the Weaviate database for faces similar to the uploaded image.

    Returns a list of matches sorted by similarity (lowest distance = best match).
    """
    img = await read_image(file)
    model = model_name or MODEL_NAME
    try:
        results = DeepFace.search(
            img=img,
            model_name=model,
            database_type=DB_TYPE,
            connection_details=WEAVIATE_URL,
            distance_metric=distance_metric,
        )
        matches = []
        for df in results:
            if not df.empty:
                cols = [c for c in ["img_name", "distance", "confidence"] if c in df.columns]
                matches.extend(df[cols].to_dict(orient="records"))
        return {"matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Info ────────────────────────────────────────────────────────────────────


@app.get(
    "/api/v1/models",
    tags=["Info"],
    summary="List available models and options",
    response_model=ModelsInfoResponse,
)
async def list_models():
    """
    Returns all available recognition models, detector backends,
    distance metrics, and analysis actions supported by this API.
    """
    return {
        "recognition_models": RECOGNITION_MODELS,
        "detector_backends": DETECTOR_BACKENDS,
        "distance_metrics": DISTANCE_METRICS,
        "analysis_actions": ANALYSIS_ACTIONS,
    }
