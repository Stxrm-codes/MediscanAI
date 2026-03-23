import os
from flask import Flask, request, jsonify
from PIL import Image
import io

# ---- OPTIONAL: Only keep required imports ----
# Remove unused heavy libraries like gTTS

# ---- Gemini Client ----
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---- Flask App ----
app = Flask(__name__)

# Limit upload size (IMPORTANT)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # 4MB


# ---- BASIC IMAGE PREPROCESSOR ----
class ImagePreprocessor:
    def enhance(self, img):
        # You can add enhancement if needed
        return img


# ---- OCR ENGINE (Dummy / Replace with your actual OCR) ----
class OCREngine:
    def extract_text(self, img):
        # Replace this with your OCR logic (Tesseract etc.)
        return "sample extracted text"


# ---- NLP MATCHING (Dummy / Replace with your logic) ----
def find_best_match(text):
    # Replace with your NLP + fuzzy matching logic
    return ["Paracetamol", "Ibuprofen"]


# ---- MAIN PIPELINE ----
def run_pipeline(image_file):
    try:
        # ---- Load Image ----
        img = Image.open(image_file)

        # 🔥 Resize image (CRITICAL FIX)
        img.thumbnail((512, 512))

        if img.mode != "RGB":
            img = img.convert("RGB")

        # ---- Preprocess ----
        preprocessor = ImagePreprocessor()
        enhanced = preprocessor.enhance(img)

        # ---- OCR ----
        ocr_engine = OCREngine()
        extracted_text = ocr_engine.extract_text(enhanced)

        # ---- NLP Matching ----
        candidates = find_best_match(extracted_text)

        # ---- Gemini Prompt (TEXT ONLY, NO IMAGE) ----
        prompt = f"""
        Extracted text from medicine image:
        {extracted_text}

        Possible matches:
        {candidates}

        Provide:
        - Correct medicine name
        - Uses
        - Dosage
        - Precautions
        """

        # 🔥 IMPORTANT: DO NOT SEND IMAGE
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        result_text = response.text if response else "No response"

        # ---- Free Memory ----
        del img
        del enhanced

        return {
            "status": "success",
            "extracted_text": extracted_text,
            "candidates": candidates,
            "result": result_text
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# ---- ROUTES ----
@app.route("/")
def home():
    return "MediScanAI is running 🚀"


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    result = run_pipeline(file)
    return jsonify(result)


# ---- RUN ----
if _name_ == '_main_':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)