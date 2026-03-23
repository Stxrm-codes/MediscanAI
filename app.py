"""
MediScan AI — OCR-Driven Medicine Detection & Smart Recommendation Platform
===========================================================================
Project Guide : Sanjana Bhavsar, Team Lead – Data Vidwan
Description   : Accepts a medicine-strip image, runs OCR → NLP cleaning →
                fuzzy drug matching → Gemini AI structured analysis, then
                serves results through a modern web interface.
"""

import os, re, json, time, logging
from io import BytesIO

from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
from PIL import Image
from gtts import gTTS
from google import genai
from google.genai import types

from utils.image_preprocessor import ImagePreprocessor
from utils.ocr_engine          import OCREngine
from utils.nlp_processor       import NLPProcessor
from utils.drug_matcher        import DrugMatcher

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── App & env ─────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not found in .env")

app    = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB

client = genai.Client(api_key=API_KEY)

# ── Module instances ──────────────────────────────────────────────────────────
preprocessor = ImagePreprocessor()
ocr_engine   = OCREngine()
nlp          = NLPProcessor()
matcher      = DrugMatcher()

# ── Gemini safety (allow medical info) ────────────────────────────────────────
SAFETY = [
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE),
]

# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(image_file, language: str) -> dict:
    """
    Full OCR → NLP → Fuzzy-match → Gemini pipeline.
    Returns structured dict with all medicine fields.
    """
    t0 = time.time()

    # 1. Load & preprocess image
    img = Image.open(image_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    enhanced = preprocessor.enhance(img)
    log.info("Step 1 – Preprocessing done (%.2fs)", time.time() - t0)

    # 2. OCR – extract raw text
    raw_text = ocr_engine.extract_text(enhanced)
    log.info("Step 2 – OCR raw: %s", raw_text[:120])

    # 3. NLP – clean & extract candidate names
    clean_text  = nlp.clean(raw_text)
    candidates  = nlp.extract_medicine_names(clean_text)
    log.info("Step 3 – NLP candidates: %s", candidates)

    # 4. Fuzzy match against drug database
    matched, score = matcher.best_match(candidates)
    log.info("Step 4 – Fuzzy match: %s (%.1f%%)", matched, score)

    # 5. Gemini structured extraction
    prompt = f"""
You are a senior clinical pharmacist AI.
Analyse this medicine image.

OCR extracted text (raw): {raw_text[:300] or 'none'}
NLP candidate drug names : {candidates or 'none'}
Best fuzzy DB match      : {matched or 'unknown'} (confidence {score:.0f}%)

Return ONLY a valid JSON object — zero markdown, zero code fences.
Required keys (exact spelling):
  brand_name, generic_name, dosage, manufacturer,
  drug_class, mechanism_of_action,
  intended_use, therapeutic_benefits,
  side_effects, contraindications, safety_warning,
  alternatives, ocr_confidence

Rules:
• "alternatives" → comma-separated string of 2–3 similar drug names
• "ocr_confidence" → your confidence in the OCR result, e.g. "82%"
• Translate ALL field values to {language}
• Use "N/A" for any field you cannot determine from the image
"""

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, enhanced],
        config=types.GenerateContentConfig(safety_settings=SAFETY),
    )

    raw_json = resp.text.strip()
    raw_json = re.sub(r"^```[a-z]*\s*", "", raw_json)
    raw_json = re.sub(r"\s*```$",        "", raw_json).strip()
    m = re.search(r"\{.*\}", raw_json, re.DOTALL)
    result = json.loads(m.group(0) if m else raw_json)

    # Attach pipeline metadata
    result["_meta"] = {
        "ocr_raw"        : raw_text[:400],
        "ocr_candidates" : candidates,
        "fuzzy_match"    : matched,
        "fuzzy_score"    : round(score, 1),
        "processing_time": f"{time.time()-t0:.2f}s",
        "backends"       : ocr_engine.available_backends(),
    }

    log.info("Pipeline complete in %s", result["_meta"]["processing_time"])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """POST /analyze  —  image + language → structured JSON"""
    if "file" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    language = request.form.get("language", "English")
    try:
        result = run_pipeline(request.files["file"], language)
        return jsonify({"result": result, "success": True})
    except Exception as exc:
        log.error("Analyze error: %s", exc, exc_info=True)
        return jsonify({
            "success": False,
            "error"  : str(exc),
            "result" : {
                "brand_name": "Analysis Failed", "generic_name": "N/A",
                "dosage": "N/A", "manufacturer": "N/A", "drug_class": "N/A",
                "mechanism_of_action": "N/A", "intended_use": "N/A",
                "therapeutic_benefits": "N/A", "side_effects": "N/A",
                "contraindications": "N/A", "alternatives": "N/A",
                "safety_warning": "Please upload a clearer image and try again.",
                "ocr_confidence": "0%",
                "_meta": {"ocr_raw": "", "ocr_candidates": [], "fuzzy_match": None,
                          "fuzzy_score": 0, "processing_time": "—", "backends": []},
            },
        }), 200


@app.route("/chat", methods=["POST"])
def chat():
    """POST /chat  —  question + context → AI answer"""
    data     = request.json or {}
    question = data.get("question", "").strip()
    context  = data.get("context",  {})
    language = data.get("language", "English")

    if not question:
        return jsonify({"answer": "Please ask a question."}), 400

    prompt = f"""
You are MediScan AI, an expert medical information assistant.
The user just scanned a medicine. Full details:
{json.dumps(context, indent=2, ensure_ascii=False)}

User question: "{question}"

Answer helpfully in {language}.
Base your answer on the provided data first; use general pharmacology knowledge only when necessary.
Never prescribe specific dosages — always recommend consulting a doctor.
Keep the answer clear and concise.
"""
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(safety_settings=SAFETY),
        )
        return jsonify({"answer": resp.text})
    except Exception as exc:
        log.error("Chat error: %s", exc)
        return jsonify({"answer": f"Error: {exc}"}), 500


@app.route("/synthesize", methods=["POST"])
def synthesize():
    """POST /synthesize  —  text + language → mp3 audio"""
    data     = request.json or {}
    text     = data.get("text", "")
    language = data.get("language", "English")

    if not text:
        return jsonify({"error": "No text"}), 400

    lang_map = {"English": "en", "Hindi": "hi", "Gujarati": "gu"}
    try:
        tts = gTTS(text=text, lang=lang_map.get(language, "en"), slow=False)
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return send_file(buf, mimetype="audio/mpeg")
    except Exception as exc:
        log.error("TTS error: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
if _name_ == '_main_':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
