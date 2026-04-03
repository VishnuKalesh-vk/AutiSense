"""
predictor.py — Image preprocessing and risk prediction logic.

DISCLAIMER: Educational use only. Not a medical diagnosis tool.
"""

import random
import cv2
import numpy as np
import logging

from model_loader import load_model_safe

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
TARGET_SIZE = (224, 224)   # MobileNetV2 expected input
LOW_THRESHOLD = 0.4
HIGH_THRESHOLD = 0.7

# Load the model once at module import time so it is reused across requests.
# load_model_safe() returns a Keras model on success, or None on any failure.
_model = load_model_safe()


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess an image for MobileNetV2 inference using OpenCV.

    Pipeline
    --------
    1. Read image from disk (OpenCV loads as BGR uint8).
    2. Convert BGR → RGB (MobileNetV2 was trained on RGB images).
    3. Resize to 224 × 224 pixels.
    4. Normalise pixel values from [0, 255] to [0.0, 1.0].
    5. Expand dims to add a batch axis: (H, W, C) → (1, H, W, C).

    Parameters
    ----------
    image_path : str
        Path to the image file on disk.

    Returns
    -------
    np.ndarray
        Shape (1, 224, 224, 3), dtype float32, values in [0.0, 1.0].

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If OpenCV cannot decode the file (corrupt or unsupported format),
        or if any processing step produces an unexpected result.
    """
    import os

    # ── Step 0: verify the file exists before handing it to OpenCV ──
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # ── Step 1: read image from disk ────────────────────────────────
    # cv2.imread returns None on decode failure (corrupt / unsupported).
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(
            f"OpenCV could not decode the image at '{image_path}'. "
            "The file may be corrupt or in an unsupported format."
        )

    # ── Step 2: convert BGR → RGB ────────────────────────────────────
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except cv2.error as exc:
        raise ValueError(f"Colour conversion failed for '{image_path}': {exc}") from exc

    # ── Step 3: resize to 224 × 224 ─────────────────────────────────
    # INTER_AREA is preferred for downscaling; it reduces moiré artefacts.
    try:
        img_resized = cv2.resize(img_rgb, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    except cv2.error as exc:
        raise ValueError(f"Resizing failed for '{image_path}': {exc}") from exc

    # ── Step 4: normalise pixel values to [0.0, 1.0] ────────────────
    # Cast to float32 first to avoid integer overflow during division.
    img_normalised = img_resized.astype(np.float32) / 255.0

    # Sanity-check: values must stay within [0, 1]
    if img_normalised.min() < 0.0 or img_normalised.max() > 1.0:
        raise ValueError(
            "Normalisation produced out-of-range values "
            f"(min={img_normalised.min():.4f}, max={img_normalised.max():.4f})."
        )

    # ── Step 5: add batch dimension ──────────────────────────────────
    # Model expects shape (batch, height, width, channels).
    img_batch = np.expand_dims(img_normalised, axis=0)  # → (1, 224, 224, 3)

    logger.debug(
        "Preprocessed '%s' → shape=%s dtype=%s",
        image_path,
        img_batch.shape,
        img_batch.dtype,
    )
    return img_batch


def map_score_to_level(score: float) -> str:
    """
    Convert a continuous risk score to a categorical risk level.

    Risk thresholds
    ---------------
    0.0 – 0.4  → Low
    0.4 – 0.7  → Medium
    0.7 – 1.0  → High

    Parameters
    ----------
    score : float
        Prediction output in range [0, 1].

    Returns
    -------
    str
        One of "Low", "Medium", or "High".
    """
    if score < LOW_THRESHOLD:
        return "Low"
    if score < HIGH_THRESHOLD:
        return "Medium"
    return "High"


def run_inference(img_array: np.ndarray) -> tuple[float, str]:
    """
    Run inference on a preprocessed image batch.

    Chooses between the real trained model and a random fallback depending
    on whether the model was loaded successfully at startup.

    Parameters
    ----------
    img_array : np.ndarray
        Preprocessed image of shape (1, 224, 224, 3), values in [0, 1].

    Returns
    -------
    tuple[float, str]
        (score, source) where:
        - score  : float in [0, 1] — higher means higher risk
        - source : "trained_model" | "fallback_dummy"
    """
    print(f"[predictor] run_inference called. _model={_model}", flush=True)
    if _model is not None:
        # ── Trained model path ────────────────────────────────
        raw_output = _model.predict(img_array, verbose=0)

        # Keras assigns class indices alphabetically:
        #   0 = autistic  (higher risk)
        #   1 = non_autistic (lower risk)
        # The sigmoid output approaches 1 for non_autistic (class 1).
        # Invert so a HIGH score means HIGH autism risk.
        sigmoid_val = float(raw_output[0][0])
        score = 1.0 - sigmoid_val
        source = "trained_model"

        logger.debug("Trained model sigmoid=%.4f → risk score=%.4f", sigmoid_val, score)

    else:
        # ── Fallback dummy path ───────────────────────────────
        # Real model unavailable; generate a random score so the API
        # still returns a well-formed response for testing purposes.
        score = random.uniform(0.0, 1.0)
        source = "fallback_dummy"

        logger.warning(
            "Fallback dummy in use — returning random score %.4f. "
            "Ensure TensorFlow is installed and model.h5 exists.",
            score,
        )

    # Clip to [0, 1] to guard against any floating-point edge cases
    score = max(0.0, min(1.0, score))
    return score, source


def predict_risk(image_path: str) -> tuple[float, str, str]:
    """
    Run end-to-end prediction on a single image file.

    Parameters
    ----------
    image_path : str
        Path to the uploaded image file.

    Returns
    -------
    tuple[float, str, str]
        (risk_score, risk_level, model_source)
        e.g. (0.73, "High", "trained_model")
    """
    # Step 1: preprocess the image into a model-ready array
    img_array = preprocess_image(image_path)

    # Step 2: run inference (handles both trained model and dummy fallback)
    risk_score, source = run_inference(img_array)

    logger.info("Inference source=%s score=%.4f", source, risk_score)

    # Step 3: map the continuous score to a categorical risk level
    risk_level = map_score_to_level(risk_score)

    return risk_score, risk_level, source
