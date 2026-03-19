# MediScan AI 💊
### OCR-Driven Medicine Detection & Smart Recommendation Platform

> **Project Guide:** Sanjana Bhavsar — Team Lead, Data Vidwan  
> `sanjanaa.bhavsar@gmail.com` · `9727379278`

---

## Overview

MediScan AI is a full-stack AI web application that lets users photograph any medicine strip or tablet packaging, then instantly receive a complete clinical report — drug purpose, dosage, side effects, alternatives, and safety warnings — in English, Hindi, or Gujarati.

### Pipeline

```
Image Upload
    │
    ▼
Image Preprocessing       ← resize, CLAHE, denoise, deskew
    │
    ▼
OCR Engine                ← EasyOCR → Tesseract → Google Vision (fallback chain)
    │
    ▼
NLP Processor             ← clean text, extract candidate drug names
    │
    ▼
Fuzzy Drug Matcher        ← RapidFuzz against 90-drug local database
    │
    ▼
Gemini 2.5 Flash AI       ← structured JSON report (12 fields)
    │
    ▼
Web Interface             ← biopunk-luxury UI, multilingual, audio readout
```

---

## Project Structure

```
MediScanAI/
├── app.py                   ← Flask app, all routes
├── requirements.txt
├── .env.example
│
├── utils/
│   ├── __init__.py
│   ├── image_preprocessor.py  ← OpenCV pipeline (CLAHE, deskew, binarise)
│   ├── ocr_engine.py          ← EasyOCR / Tesseract / Vision fallback
│   ├── nlp_processor.py       ← text cleaning + medicine name extraction
│   └── drug_matcher.py        ← RapidFuzz fuzzy matching
│
├── database/
│   └── drugs.json             ← 90 common medicines with aliases & class
│
└── templates/
    └── index.html             ← Full frontend (single file, no build step)
```

---

## Setup & Run

### 1. Clone / extract the project

```bash
cd MediScanAI
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install system dependencies

```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows: download installer from https://github.com/UB-Mannheim/tesseract/wiki
```

### 5. Download spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 6. Configure environment

```bash
cp .env.example .env
# Edit .env and add your Google Gemini API key
# Get one free at: https://aistudio.google.com/
```

### 7. Run the app

```bash
python app.py
# Open http://localhost:5000
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`      | Serve web app |
| `POST` | `/analyze` | Image + language → full medicine report JSON |
| `POST` | `/chat`    | Follow-up question + context → AI answer |
| `POST` | `/synthesize` | Text + language → MP3 audio (gTTS) |

### `/analyze` Request

```
Content-Type: multipart/form-data
file:     <image file>
language: English | Hindi | Gujarati
```

### `/analyze` Response

```json
{
  "success": true,
  "result": {
    "brand_name": "Crocin 650",
    "generic_name": "Paracetamol",
    "dosage": "650 mg",
    "manufacturer": "GSK",
    "drug_class": "Analgesic / Antipyretic",
    "mechanism_of_action": "Inhibits prostaglandin synthesis in CNS",
    "intended_use": "Fever and mild to moderate pain relief",
    "therapeutic_benefits": "Fast-acting, well-tolerated, OTC availability",
    "side_effects": "Nausea, liver damage in overdose",
    "contraindications": "Severe hepatic impairment, hypersensitivity",
    "safety_warning": "Do not exceed 4g/day. Avoid with alcohol.",
    "alternatives": "Ibuprofen, Aspirin, Nimesulide",
    "ocr_confidence": "91%",
    "_meta": {
      "ocr_raw": "CROCIN 650 PARACETAMOL ...",
      "ocr_candidates": ["Paracetamol", "CROCIN"],
      "fuzzy_match": "Paracetamol",
      "fuzzy_score": 94.0,
      "processing_time": "3.21s",
      "backends": ["easyocr", "tesseract"]
    }
  }
}
```

---

## Features

| Feature | Description |
|---------|-------------|
| 🔍 **OCR Pipeline** | EasyOCR primary + Tesseract fallback + optional Google Vision |
| 🧠 **NLP Cleaning** | Suffix patterns + CamelCase heuristic + spaCy NER |
| 🎯 **Fuzzy Matching** | RapidFuzz token_sort_ratio against 90-drug database |
| 🤖 **AI Analysis** | Gemini 2.5 Flash — 12 structured fields per medicine |
| 🌐 **Multilingual** | English, Hindi, Gujarati (all fields translated) |
| 🔊 **Audio Readout** | gTTS text-to-speech with playback controls |
| 📸 **Camera Capture** | Live camera with scan overlay |
| 📊 **OCR Transparency** | Raw OCR text + confidence bar + fuzzy match score visible to user |
| 🔄 **Alternatives** | Clickable chips that auto-query the AI |
| 📅 **Reminders** | Google Calendar deep-link |
| 💬 **AI Chat** | Follow-up questions grounded in the scan result |
| 📜 **History** | localStorage scan history with reload |

---

## Deliverables Checklist

- [x] OCR-based Medicine Name Detection (EasyOCR + Tesseract + NLP)
- [x] Drug Information Retrieval System (Gemini AI)
- [x] Structured Medicine Database (90 drugs, JSON)
- [x] Interactive Web Application (Flask + single-file frontend)
- [x] Model Performance Evaluation (OCR confidence %, fuzzy match score %)
- [ ] Final Technical Report (to be completed by team)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+, Flask 3.x |
| AI Model | Google Gemini 2.5 Flash |
| OCR | EasyOCR, Tesseract, (Google Vision optional) |
| NLP | spaCy, regex patterns |
| Fuzzy Match | RapidFuzz |
| Image Processing | OpenCV (headless), Pillow |
| TTS | gTTS |
| Frontend | Vanilla HTML/CSS/JS (no build step) |
| Fonts | Playfair Display, Cabinet Grotesk, Fira Code |

---

## License

MIT — for educational and research use.  
Always consult a qualified healthcare professional for medical decisions.
