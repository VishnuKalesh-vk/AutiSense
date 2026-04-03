"""
main.py — FastAPI entry point for the Autism Risk Analysis Web App.

DISCLAIMER: This application is for EDUCATIONAL PURPOSES ONLY.
It does NOT provide any form of medical diagnosis.
"""

import os
import shutil
import uuid
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from predictor import predict_risk, _model

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# App initialisation
# ──────────────────────────────────────────────
app = FastAPI(
    title="Autism Risk Analysis API",
    description=(
        "Educational tool only. "
        "This API does NOT provide medical diagnoses."
    ),
    version="1.0.0",
)

# ──────────────────────────────────────────────
# CORS — allow the HTML frontend to call the API
# ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary directory for uploaded images
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# Health-check endpoint
# ──────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    """Simple health-check."""
    return {"status": "ok", "message": "Autism Risk Analysis API is running."}


@app.get("/debug", tags=["Health"])
def debug():
    """Debug endpoint — shows whether the real model is loaded."""
    import os
    return {
        "model_loaded": _model is not None,
        "model_type": type(_model).__name__,
        "model_path_exists": os.path.exists("model/model.h5"),
    }


# ──────────────────────────────────────────────
# Prediction endpoint
# ──────────────────────────────────────────────

DISCLAIMER = (
    "This system is for educational purposes only "
    "and does not provide medical diagnosis."
)


def _confidence_label(score: float) -> str:
    """
    Derive a simple confidence label from the risk score.

    The model's sigmoid output is most decisive at the extremes and
    least decisive near 0.5, so we use that midpoint as the threshold.

    Returns "High" when the model output is far from 0.5 (score ≥ 0.5),
    "Low" when the prediction is near the decision boundary (score < 0.5).
    """
    return "High" if score >= 0.5 else "Low"


@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Accept an image upload and return a risk assessment.

    Returns
    -------
    JSON with keys: risk_score, risk_level, confidence, model_source, message
    """
    # ── 1. Log incoming request ───────────────────────────────
    logger.info(
        "Prediction request received — filename='%s' content_type='%s'",
        file.filename,
        file.content_type,
    )

    # ── 2. Validate file type (HTTP 400 for non-images) ───────
    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning(
            "Rejected upload: unsupported content_type='%s'", file.content_type
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid file type '{file.content_type}'. "
                "Please upload an image (JPEG, PNG, GIF, WEBP, or BMP)."
            ),
        )

    # ── 3. Save to a uniquely named temp file ─────────────────
    temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
    temp_path = os.path.join(TEMP_DIR, temp_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("Upload saved temporarily to '%s'", temp_path)

        # ── 4. Run prediction ─────────────────────────────────
        risk_score, risk_level, model_source = predict_risk(temp_path)

        # ── 5. Derive confidence ──────────────────────────────
        confidence = _confidence_label(risk_score)

        logger.info(
            "Prediction complete — score=%.4f level=%s confidence=%s source=%s",
            risk_score,
            risk_level,
            confidence,
            model_source,
        )

    except HTTPException:
        # Re-raise HTTP exceptions without wrapping them
        raise

    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(exc)}")

    finally:
        # ── 6. Always remove the temp file ────────────────────
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # ── 7. Return structured JSON response ────────────────────
    return JSONResponse(
        content={
            "risk_score":   round(float(risk_score), 4),
            "risk_level":   risk_level,
            "confidence":   confidence,
            "model_source": model_source,
            "message":      DISCLAIMER,
        }
    )
