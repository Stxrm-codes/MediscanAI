"""
utils/image_preprocessor.py
────────────────────────────
Image enhancement pipeline that significantly boosts OCR accuracy.

Pipeline
--------
1. Upscale  — ensure ≥ 1200 px wide (OCR struggles below 150 DPI)
2. Grayscale
3. CLAHE    — adaptive contrast enhancement
4. Denoise  — Gaussian blur to reduce label noise
5. Binarise — adaptive threshold for clean black/white text
6. Deskew   — correct label rotation up to ±15°
7. Return   — back to RGB PIL for Gemini vision / Tesseract
"""

import cv2
import numpy as np
from PIL import Image


class ImagePreprocessor:
    MIN_WIDTH = 1200   # minimum pixel width before upscaling

    def enhance(self, pil_img: Image.Image) -> Image.Image:
        """Run the full enhancement pipeline. Returns RGB PIL Image."""
        img = np.array(pil_img)

        # 1. Upscale if needed
        h, w = img.shape[:2]
        if w < self.MIN_WIDTH:
            scale = self.MIN_WIDTH / w
            img   = cv2.resize(img, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

        # 2. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 3. CLAHE contrast
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        # 4. Denoise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # 5. Adaptive threshold
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15, C=4,
        )

        # 6. Deskew
        binary = self._deskew(binary)

        # 7. Back to RGB PIL
        return Image.fromarray(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """Correct small rotations using minAreaRect heuristic."""
        try:
            coords = np.column_stack(np.where(gray > 0))
            angle  = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle += 90
            if abs(angle) < 0.5:
                return gray
            (h, w) = gray.shape
            M      = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            return cv2.warpAffine(gray, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            return gray

    def stats(self, pil_img: Image.Image) -> dict:
        """Useful debug info about image quality."""
        arr = np.array(pil_img.convert("L"))
        return {
            "width":    pil_img.width,
            "height":   pil_img.height,
            "contrast": float(np.std(arr)),
            "mean":     float(np.mean(arr)),
        }
