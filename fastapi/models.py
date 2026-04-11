from pydantic import BaseModel, Field
from typing import Optional


# ── Shared ──────────────────────────────────────────────────────────────────

class FacialArea(BaseModel):
    x: int
    y: int
    w: int
    h: int


class ErrorResponse(BaseModel):
    detail: str


# ── Analysis ────────────────────────────────────────────────────────────────

class AnalyzeResult(BaseModel):
    age: Optional[float] = None
    dominant_gender: Optional[str] = None
    gender: Optional[dict] = Field(None, description="Gender probabilities {'Man': %, 'Woman': %}")
    dominant_emotion: Optional[str] = None
    emotion: Optional[dict] = Field(None, description="Emotion probabilities")
    dominant_race: Optional[str] = None
    race: Optional[dict] = Field(None, description="Race probabilities")
    region: Optional[dict] = Field(None, description="Facial area coordinates")
    face_confidence: Optional[float] = None


class AnalyzeResponse(BaseModel):
    results: list[AnalyzeResult]


# ── Verification ────────────────────────────────────────────────────────────

class VerifyResponse(BaseModel):
    verified: bool
    distance: float
    threshold: float
    model: str
    distance_metric: str
    facial_areas: Optional[dict] = None


# ── Embeddings ──────────────────────────────────────────────────────────────

class EmbeddingResult(BaseModel):
    embedding: list[float]
    facial_area: Optional[dict] = None
    face_confidence: Optional[float] = None


class RepresentResponse(BaseModel):
    results: list[EmbeddingResult]


# ── Detection ───────────────────────────────────────────────────────────────

class DetectedFace(BaseModel):
    facial_area: dict
    confidence: Optional[float] = None


class DetectResponse(BaseModel):
    faces: list[DetectedFace]
    count: int


# ── Weaviate DB ─────────────────────────────────────────────────────────────

class RegisterResponse(BaseModel):
    status: str
    result: Optional[dict] = None


class SearchMatch(BaseModel):
    img_name: Optional[str] = None
    distance: Optional[float] = None
    confidence: Optional[float] = None


class SearchResponse(BaseModel):
    matches: list[SearchMatch]


# ── Models info ─────────────────────────────────────────────────────────────

class ModelsInfoResponse(BaseModel):
    recognition_models: list[str]
    detector_backends: list[str]
    distance_metrics: list[str]
    analysis_actions: list[str]
