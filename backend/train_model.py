"""
train_model.py — Fine-tune MobileNetV2 on the autism image dataset.

Two-phase training strategy
────────────────────────────
Phase 1 (head training):
  Freeze the entire MobileNetV2 base and train only the new
  classification head for a few epochs. This prevents the randomly
  initialised head from destroying pretrained features.

Phase 2 (fine-tuning):
  Unfreeze the top N layers of the base and continue training with a
  very low learning rate so that high-level features adapt to the
  new domain.

Output
──────
  model/model.h5          ← saved full model (used by predictor.py)
  model/training_plot.png ← accuracy & loss curves

DISCLAIMER: Educational use only. Not a medical diagnosis tool.

Requirements
────────────
  TensorFlow does NOT yet ship wheels for Python 3.13+ (as of 2026-04).
  Run this script with Python 3.11 or 3.12:

      python3.11 -m venv venv_train
      venv_train\\Scripts\\activate        # Windows
      pip install tensorflow matplotlib
      python train_model.py

  The rest of the app (FastAPI server) continues to use the system
  Python 3.14 install; only the training step needs 3.11/3.12.
"""

import os
import sys

# ── Verify TensorFlow is importable before going further ──────
try:
    import tensorflow as tf
    print(f"TensorFlow {tf.__version__} loaded on Python {sys.version.split()[0]}")
except ImportError:
    print(
        "\n[ERROR] TensorFlow is not installed in this environment.\n"
        "TensorFlow does not yet support Python 3.13+.\n\n"
        "Steps to fix:\n"
        "  1. Install Python 3.11 from https://www.python.org/downloads/\n"
        "  2. python3.11 -m venv venv_train\n"
        "  3. venv_train\\Scripts\\activate\n"
        "  4. pip install tensorflow matplotlib\n"
        "  5. python train_model.py\n"
    )
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Reproducibility ───────────────────────────────────────────
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_ROOT   = BASE_DIR / "dataset" / "autism-image-dataset"
MODEL_DIR   = BASE_DIR / "model"
MODEL_PATH  = MODEL_DIR / "model.h5"
PLOT_PATH   = MODEL_DIR / "training_plot.png"

for path in (DATA_ROOT / "train", DATA_ROOT / "test", DATA_ROOT / "valid"):
    if not path.exists():
        sys.exit(
            f"[ERROR] Expected folder not found: {path}\n"
            "Run setup_dataset.py first."
        )

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 32
PHASE1_EPOCHS   = 10    # head-only training
PHASE2_EPOCHS   = 10    # fine-tuning (top layers unfrozen)
PHASE2_UNFREEZE = 30    # number of top base-model layers to unfreeze
LR_PHASE1       = 1e-3
LR_PHASE2       = 1e-5  # must be very low to avoid destroying pretrained weights


# ════════════════════════════════════════════════════════════════
# 1. Data generators
# ════════════════════════════════════════════════════════════════
print("\n── Building data generators ──────────────────────────────")

# Training: apply augmentation to improve generalisation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Validation & test: only rescale — no augmentation
eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    DATA_ROOT / "train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",       # two classes → single sigmoid output
    shuffle=True,
    seed=SEED,
)

valid_gen = eval_datagen.flow_from_directory(
    DATA_ROOT / "valid",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

test_gen = eval_datagen.flow_from_directory(
    DATA_ROOT / "test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

print(f"  Class indices : {train_gen.class_indices}")
print(f"  Train samples : {train_gen.samples}")
print(f"  Valid samples : {valid_gen.samples}")
print(f"  Test  samples : {test_gen.samples}")

# The label for "autistic" depends on alphabetical order. Confirm:
# class_indices == {'autistic': 0, 'non_autistic': 1}
# model output > 0.5  →  non_autistic (label 1)
# model output < 0.5  →  autistic     (label 0)
# predictor.py uses raw score as a "risk" value — the closer to 1.0
# the more the model leans toward non_autistic; invert if needed.


# ════════════════════════════════════════════════════════════════
# 2. Build model
# ════════════════════════════════════════════════════════════════
print("\n── Building model ────────────────────────────────────────")

base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False   # freeze for Phase 1

inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3), name="image_input")
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D(name="gap")(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(128, activation="relu", name="fc1")(x)
x       = layers.Dropout(0.3, name="dropout")(x)
outputs = layers.Dense(1, activation="sigmoid", name="risk_output")(x)

model = Model(inputs, outputs, name="autism_risk_mobilenetv2")
model.summary()


# ════════════════════════════════════════════════════════════════
# 3. Phase 1 — train classification head
# ════════════════════════════════════════════════════════════════
print(f"\n── Phase 1: head training ({PHASE1_EPOCHS} epochs) ──────────────")

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_PHASE1),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

phase1_callbacks = [
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
    callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "best_phase1.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
]

history1 = model.fit(
    train_gen,
    epochs=PHASE1_EPOCHS,
    validation_data=valid_gen,
    callbacks=phase1_callbacks,
    verbose=1,
)


# ════════════════════════════════════════════════════════════════
# 4. Phase 2 — fine-tune top layers of the base model
# ════════════════════════════════════════════════════════════════
print(f"\n── Phase 2: fine-tuning top {PHASE2_UNFREEZE} layers "
      f"({PHASE2_EPOCHS} epochs) ──")

base_model.trainable = True

# Freeze all layers except the last PHASE2_UNFREEZE
for layer in base_model.layers[:-PHASE2_UNFREEZE]:
    layer.trainable = False

trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f"  Trainable base layers: {trainable_count}")

# Re-compile with a much lower LR
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_PHASE2),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

phase2_callbacks = [
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
]

history2 = model.fit(
    train_gen,
    epochs=PHASE2_EPOCHS,
    validation_data=valid_gen,
    callbacks=phase2_callbacks,
    verbose=1,
)


# ════════════════════════════════════════════════════════════════
# 5. Evaluate on held-out test set
# ════════════════════════════════════════════════════════════════
print("\n── Test-set evaluation ───────────────────────────────────")

test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f"  Test loss    : {test_loss:.4f}")
print(f"  Test accuracy: {test_acc * 100:.2f}%")


# ════════════════════════════════════════════════════════════════
# 6. Save final model
# ════════════════════════════════════════════════════════════════
print("\n── Saving model ──────────────────────────────────────────")

# model.h5 is saved by ModelCheckpoint (best val_accuracy checkpoint).
# Verify it was written:
if MODEL_PATH.exists():
    print(f"  ✅ Model saved to {MODEL_PATH}")
else:
    # Fallback: save the current model state
    model.save(str(MODEL_PATH))
    print(f"  ✅ Model saved (fallback) to {MODEL_PATH}")


# ════════════════════════════════════════════════════════════════
# 7. Plot training curves
# ════════════════════════════════════════════════════════════════
print("\n── Plotting training history ──────────────────────────────")

def _concat(key):
    """Merge Phase 1 + Phase 2 history for a given metric."""
    p1 = history1.history.get(key, [])
    p2 = history2.history.get(key, [])
    return p1 + p2

epochs_total = list(range(1, len(_concat("accuracy")) + 1))
phase1_len   = len(history1.history.get("accuracy", []))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("MobileNetV2 Fine-Tuning — Autism Risk Classifier", fontsize=14)

for ax, metric, title in zip(
    axes,
    [("accuracy", "val_accuracy"), ("loss", "val_loss")],
    ["Accuracy", "Loss"],
):
    train_vals = _concat(metric[0])
    val_vals   = _concat(metric[1])
    ax.plot(epochs_total, train_vals, label=f"Train {title}")
    ax.plot(epochs_total, val_vals,   label=f"Val   {title}")
    # Mark Phase 1 / Phase 2 boundary
    if phase1_len > 0 and phase1_len < len(epochs_total):
        ax.axvline(x=phase1_len + 0.5, color="gray", linestyle="--",
                   linewidth=1, label="Phase 2 start")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(str(PLOT_PATH), dpi=150)
print(f"  ✅ Plot saved to {PLOT_PATH}")

print("\n🎉 Training complete.")
print(f"   Model  → {MODEL_PATH}")
print(f"   Plot   → {PLOT_PATH}")
print(
    "\n   NOTE: The app's predictor maps raw sigmoid output to a risk score.\n"
    "   If class_indices == {'autistic': 0, 'non_autistic': 1},\n"
    "   output NEAR 0  → autistic    (higher risk → score stays low)\n"
    "   output NEAR 1  → non_autistic (lower risk → score stays high)\n"
    "   predictor.py uses (1 - output) as the risk score so that\n"
    "   autistic images get a HIGH score, matching the UI expectation.\n"
    "   Verify class_indices above and adjust predictor.py if needed."
)
