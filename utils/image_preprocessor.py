import logging
from PIL import Image, ImageEnhance

log = logging.getLogger(__name__)

class ImagePreprocessor:
    def __init__(self, max_size=(1600, 1600)):
        # 1600px is a sweet spot: large enough for clear text, small enough for fast API uploads
        self.max_size = max_size

    def enhance(self, img: Image.Image) -> Image.Image:
        """
        Prepares the image specifically for Gemini's vision model.
        Focuses on resizing for speed and mild enhancements for clarity.
        """
        try:
            # 1. Standardize format
            if img.mode != "RGB":
                img = img.convert("RGB")

            # 2. Resize down if the image is massive (e.g., straight from a 4K phone camera)
            # thumbnail() maintains the aspect ratio automatically
            img.thumbnail(self.max_size, Image.Resampling.LANCZOS)

            # 3. Mild Contrast Boost (helps faded text stand out against the background)
            contrast_enhancer = ImageEnhance.Contrast(img)
            img = contrast_enhancer.enhance(1.2)  # 20% increase in contrast

            # 4. Mild Sharpness Boost (helps define the edges of small, blurry letters)
            sharpness_enhancer = ImageEnhance.Sharpness(img)
            img = sharpness_enhancer.enhance(1.5)  # 50% increase in sharpness

            log.info(f"Preprocessing complete. Final image size: {img.size}")
            return img

        except Exception as e:
            log.error(f"Error during image preprocessing: {e}")
            # Failsafe: If anything goes wrong, just return the original image so the app doesn't crash
            return img