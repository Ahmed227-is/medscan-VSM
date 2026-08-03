import cv2
import numpy as np
import math
from pathlib import Path


class Preprocessor:
    """
    Module de prétraitement des images médicales scannées.
    Améliore la qualité avant OCR.
    """

    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Impossible de charger l'image : {image_path}")
        return image

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Supprime le bruit tout en préservant les détails du texte."""
        return cv2.fastNlMeansDenoising(image, h=10, searchWindowSize=21)

    def correct_skew(self, image: np.ndarray) -> np.ndarray:
        """
        Redresse uniquement si l'inclinaison est significative.
        Détecte l'angle réel avant de corriger.
        """
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )

        if lines is None:
            return image

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 45:
                    angles.append(angle)

        if not angles:
            return image

        median_angle = np.median(angles)

        if abs(median_angle) < 0.5:
            return image

        if abs(median_angle) > 15:
            return image

        h, w = image.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        return cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Améliore le contraste avec CLAHE."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def binarize(self, image: np.ndarray) -> np.ndarray:
        """Binarisation adaptative pour documents très dégradés."""
        return cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

    def compute_quality_score(self, image: np.ndarray) -> float:
        """
        Variance du Laplacien normalisée logarithmiquement.
        Métrique académique standard pour mesurer la netteté.

        Interprétation :
        0.0 - 0.3 : image très floue ou dégradée
        0.3 - 0.6 : qualité moyenne
        0.6 - 1.0 : bonne qualité pour OCR
        """
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        if laplacian_var <= 0:
            return 0.0
        # Normalisation logarithmique — plus robuste que /500
        score = math.log10(laplacian_var + 1) / math.log10(1001)
        return round(min(score, 1.0), 3)

    def process(self, image_path: str) -> dict:
        """
        Pipeline complet de prétraitement.
        Retourne l'image traitée + score de qualité.
        """
        image = self.load_image(image_path)
        gray = self.to_grayscale(image)
        denoised = self.denoise(gray)
        straightened = self.correct_skew(denoised)
        enhanced = self.enhance_contrast(straightened)
        quality_score = self.compute_quality_score(enhanced)

        # Binarisation uniquement si qualité insuffisante
        if quality_score < 0.5:
            final = self.binarize(enhanced)
        else:
            final = enhanced

        output_path = self.output_dir / f"processed_{Path(image_path).name}"
        cv2.imwrite(str(output_path), final)

        return {
            "processed_image": final,
            "processed_path": str(output_path),
            "quality_score": quality_score,
            "needs_llm_fallback": quality_score < 0.3
        }