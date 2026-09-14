from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
CLASS_MAP_PATH = Path(os.getenv("CLASS_MAP_PATH", BASE_DIR / "class_map.json"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "psl_model.tflite"))


def load_class_map() -> dict[str, str]:
    if not CLASS_MAP_PATH.exists():
        return {}
    raw = json.loads(CLASS_MAP_PATH.read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in raw.items()}


class FeatureRequest(BaseModel):
    features: list[float] = Field(..., min_length=180, max_length=180)
    confidence: float | None = Field(default=None, ge=0, le=1)


class PredictionResponse(BaseModel):
    prediction: str | None
    index: int | None
    confidence: float
    demo: bool
    message: str | None = None


app = FastAPI(title="Dastyar Sign Language Inference API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    labels = load_class_map()
    return {
        "ok": True,
        "model_loaded": MODEL_PATH.exists(),
        "class_map_loaded": bool(labels),
        "labels": len(labels),
        "demo_mode": not MODEL_PATH.exists(),
    }


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(payload: FeatureRequest) -> PredictionResponse:
    labels = load_class_map()
    if not MODEL_PATH.exists():
        return PredictionResponse(
            prediction=None,
            index=None,
            confidence=0,
            demo=True,
            message="psl_model.tflite is not installed; inference is disabled.",
        )

    raise HTTPException(
        status_code=501,
        detail="The model adapter is intentionally unconfigured. Add the exact TFLite input/output contract before enabling inference.",
    )


@app.get("/v1/classes")
def classes() -> dict[str, str]:
    return load_class_map()
