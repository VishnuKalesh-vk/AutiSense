"""
model_loader.py — Safely loads the trained TensorFlow model.

DISCLAIMER: Educational use only. Not a medical diagnosis tool.
"""

import os
import logging
import random
import warnings

logger = logging.getLogger(__name__)

# Absolute path to the saved model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.h5")

print("Loading model from:", MODEL_PATH)
print("File exists:", os.path.exists(MODEL_PATH))


# ──────────────────────────────────────────────────────────────
# Dummy fallback (used when TensorFlow is unavailable or load fails)
# ──────────────────────────────────────────────────────────────
class _DummyModel:
    """
    Minimal model interface that returns a random prediction.
    Used as a fallback so the API keeps returning well-formed responses
    even when the real model cannot be loaded.
    """

    def predict(self, img_array, verbose=0):
        score = random.uniform(0.0, 1.0)
        logger.warning(
            "DummyModel in use — returning random score %.4f. "
            "Install TensorFlow and ensure model.h5 exists for real inference.",
            score,
        )
        return [[score]]


# ──────────────────────────────────────────────────────────────
# Safe model loader
# ──────────────────────────────────────────────────────────────
def load_model_safe():
    """
    Attempt to load the trained Keras model from disk.

    Returns the loaded Keras model on success, or None on any failure.
    No exceptions are raised — all errors are logged as warnings.

    Returns
    -------
    keras.Model | None
    """
    # 1. Check TensorFlow is available
    try:
        import tensorflow as tf
        logger.info("TensorFlow %s imported successfully.", tf.__version__)
    except ImportError as exc:
        logger.warning("TensorFlow is not installed: %s", exc)
        return None

    # 2. Check the model file exists
    if not os.path.exists(MODEL_PATH):
        logger.warning("Model file not found at path: %s", MODEL_PATH)
        return None

    # 3. Load the model
    try:
        print("[model_loader] Attempting tf.keras.models.load_model ...", flush=True)
        # Suppress the legacy HDF5 deprecation warning from Keras
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = tf.keras.models.load_model(MODEL_PATH)

        print(f"[model_loader] Model loaded OK: {model}", flush=True)
        logger.info("Real model loaded successfully from %s", MODEL_PATH)
        return model

    except Exception as exc:
        print(f"[model_loader] EXCEPTION during load: {type(exc).__name__}: {exc}", flush=True)
        logger.warning("Failed to load model from %s: %s", MODEL_PATH, exc)
        return None


# ──────────────────────────────────────────────────────────────
# Public entry point used by predictor.py
# ──────────────────────────────────────────────────────────────
def load_model():
    """
    Return a model ready for inference.

    Calls load_model_safe(); falls back to _DummyModel if it returns None
    so that the rest of the application always receives a usable object.

    Returns
    -------
    keras.Model | _DummyModel
    """
    model = load_model_safe()

    if model is None:
        logger.warning(
            "Real model unavailable — falling back to DummyModel "
            "(random predictions)."
        )
        return _DummyModel()

    return model
