# AutiSENSE

> **DISCLAIMER:** This project is for **educational purposes only**.
> It does **not** provide any form of medical diagnosis.
> Always consult a qualified healthcare professional.

---

## Overview

A full-stack web application that accepts an image upload and returns a
**risk score** (Low / Medium / High) produced by a MobileNetV2-based
machine learning model.

Because no real labelled dataset is included, the classification head
is randomly initialised and predictions are not clinically meaningful.
The project is intended to demonstrate a complete ML-in-a-web-app
architecture.

---

## Project Structure

```
autism-risk-detector/
│
├── backend/
│   ├── main.py           ← FastAPI app + /predict endpoint
│   ├── predictor.py      ← Image preprocessing & inference logic
│   ├── model_loader.py   ← MobileNetV2 builder / loader (with fallback)
│   ├── requirements.txt  ← Python dependencies
│   └── model/
│       └── model.h5      ← Auto-generated on first run
│
├── frontend/
│   ├── index.html        ← Single-page UI
│   ├── style.css         ← Responsive card-based design
│   └── script.js         ← Fetch API + dynamic result rendering
│
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI + Uvicorn |
| ML model | TensorFlow / Keras — MobileNetV2 |
| Image processing | OpenCV + NumPy |
| Frontend | HTML5, CSS3, Vanilla JavaScript |

---

## Risk Level Logic

| Score range | Risk Level | Badge colour |
|---|---|---|
| 0.00 – 0.40 | Low | Green |
| 0.40 – 0.70 | Medium | Yellow |
| 0.70 – 1.00 | High | Red |

---

## Getting Started

### 1. Install backend dependencies

```bash
cd autism-risk-detector/backend
py -3.11 -m venv venv_train
.\venv_train\Scripts\Activate.ps1
pip install -r requirements.txt
pip install tensorflow
```

### 2. Start the API server

```bash
uvicorn main:app --reload
```

The API is now available at `http://127.0.0.1:8000`.

> On the **very first run**, `model_loader.py` will download the
> MobileNetV2 ImageNet weights (~14 MB) and save `model/model.h5`.
> Subsequent starts load the saved file directly.

### 3. Open the frontend

Open up a second terminal and execute the following commands

```bash
cd "autism-risk-detector\frontend"
python -m http.server 5500
```

---

## API Reference

### `POST /predict`

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | file | Image (JPEG / PNG / GIF / WEBP / BMP) |

**Response** — `application/json`

```json
{
  "risk_score": 0.6821,
  "risk_level": "Medium",
  "message": "This system is for educational purposes only and does not provide medical diagnosis."
}
```

**Error responses**

| Status | Reason |
|---|---|
| 415 | Unsupported file type |
| 500 | Internal prediction error |

---

## Notes for Developers

- **To fine-tune the model** — replace `model/model.h5` with your own
  trained weights.  The rest of the pipeline stays the same.
- **DummyModel fallback** — if TensorFlow cannot be imported,
  `model_loader.py` returns a `_DummyModel` that generates random
  scores; the API still returns well-formed responses.
- **CORS** — currently set to `allow_origins=["*"]` for development
  convenience.  Restrict this in a production deployment.

---

## License

This project is released for educational use only.
