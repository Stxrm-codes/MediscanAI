"""
utils/ocr_engine.py
────────────────────
Multi-backend OCR with automatic fallback chain.

Priority
--------
1. EasyOCR      — best for angled / stylised medicine labels (no GPU needed)
2. Tesseract    — fast, excellent for clean printed text
3. Google Vision API — cloud fallback (optional, set GOOGLE_VISION_KEY)

Install
-------
pip install easyocr pytesseract
sudo apt install tesseract-ocr          # or brew install tesseract
"""

import io
import logging
import os

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


class OCREngine:
    """Unified OCR interface with graceful backend degradation."""

    def __init__(self, preferred: str = "auto"):
        self.preferred  = preferred
        self._easyocr   = None
        self._tesseract = None
        self._vision    = None
        self._init_all()

    # ── Init ──────────────────────────────────────────────────────────────

    def _init_all(self):
        # EasyOCR
        try:
            import easyocr
            self._easyocr = easyocr.Reader(["en"], gpu=False, verbose=False)
            log.info("EasyOCR ✓")
        except ImportError:
            log.warning("EasyOCR not installed  →  pip install easyocr")

        # Tesseract
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._tesseract = pytesseract
            log.info("Tesseract ✓")
        except Exception:
            log.warning("Tesseract not found   →  install tesseract-ocr binary")

        # Google Vision (optional)
        if os.getenv("GOOGLE_VISION_KEY"):
            try:
                from google.cloud import vision
                self._vision = vision.ImageAnnotatorClient()
                log.info("Google Vision API ✓")
            except ImportError:
                log.warning("google-cloud-vision not installed")

    # ── Public ────────────────────────────────────────────────────────────

    def extract_text(self, pil_img: Image.Image) -> str:
        """Return raw OCR text from PIL image using best available backend."""
        order = (
            [self._run_easyocr, self._run_tesseract, self._run_vision]
            if self.preferred == "auto"
            else {
                "easyocr":  [self._run_easyocr],
                "tesseract":[self._run_tesseract],
                "vision":   [self._run_vision],
            }.get(self.preferred, [self._run_easyocr, self._run_tesseract])
        )

        for fn in order:
            try:
                text = fn(pil_img)
                if text and text.strip():
                    return text
            except Exception as exc:
                log.debug("OCR backend failed: %s", exc)

        log.warning("All OCR backends returned empty string.")
        return ""

    def available_backends(self) -> list:
        out = []
        if self._easyocr:   out.append("easyocr")
        if self._tesseract: out.append("tesseract")
        if self._vision:    out.append("google_vision")
        return out

    # ── Backends ─────────────────────────────────────────────────────────

    def _run_easyocr(self, pil_img: Image.Image) -> str:
        if not self._easyocr:
            raise RuntimeError("EasyOCR not initialised")
        results = self._easyocr.readtext(np.array(pil_img),
                                         detail=0, paragraph=True)
        return "\n".join(results)

    def _run_tesseract(self, pil_img: Image.Image) -> str:
        if not self._tesseract:
            raise RuntimeError("Tesseract not initialised")
        cfg = (
            "--psm 6 --oem 3 "
            "-c tessedit_char_whitelist="
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            "0123456789 .,/-+()"
        )
        return self._tesseract.image_to_string(pil_img, config=cfg)

    def _run_vision(self, pil_img: Image.Image) -> str:
        if not self._vision:
            raise RuntimeError("Google Vision not initialised")
        from google.cloud import vision
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        resp  = self._vision.text_detection(
            image=vision.Image(content=buf.getvalue()))
        texts = resp.text_annotations
        return texts[0].description if texts else ""
