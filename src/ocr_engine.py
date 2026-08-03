import os
os.environ["FLAGS_use_mkldnn"] = "0"

import logging
import numpy as np
from paddleocr import PaddleOCR

logging.disable(logging.DEBUG)


class OCREngine:
    """
    Module OCR basé sur PaddleOCR 2.7.3.
    Extrait le texte des documents médicaux scannés
    avec score de confiance composite par bloc.
    """

    def __init__(self):
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=False,
            show_log=False
        )
        self.confidence_threshold = 0.6

    def compute_confidence(self, confidences: list) -> dict:
        """
        Score composite plus représentatif que la moyenne simple.

        Formule :
        - 50% moyenne générale
        - 30% ratio de blocs bien lus
        - 20% score minimum (pénalise les très mauvais blocs)

        Interprétation :
        0.0 - 0.4 : extraction très incertaine → fallback LLM
        0.4 - 0.7 : extraction partielle → à vérifier
        0.7 - 1.0 : extraction fiable
        """
        if not confidences:
            return {
                "global_score": 0.0,
                "mean_confidence": 0.0,
                "min_confidence": 0.0,
                "low_confidence_ratio": 1.0
            }

        mean_score = float(np.mean(confidences))
        min_score = float(np.min(confidences))
        low_conf_ratio = sum(
            1 for c in confidences if c < self.confidence_threshold
        ) / len(confidences)

        # Score composite
        composite = (
            mean_score * 0.5 +
            (1 - low_conf_ratio) * 0.3 +
            min_score * 0.2
        )

        return {
            "global_score": round(composite, 3),
            "mean_confidence": round(mean_score, 3),
            "min_confidence": round(min_score, 3),
            "low_confidence_ratio": round(low_conf_ratio, 3)
        }

    def extract_text(self, image_input) -> dict:
        """
        Extrait le texte d'une image ou d'un chemin.
        Retourne texte complet + blocs détaillés + scores.
        """
        result = self.ocr.ocr(image_input, cls=True)

        if not result or not result[0]:
            return {
                "full_text": "",
                "blocks": [],
                "confidence": self.compute_confidence([]),
                "needs_llm_fallback": True
            }

        blocks = []
        confidences = []

        for line in result[0]:
            bbox = line[0]
            text = line[1][0]
            confidence = line[1][1]

            blocks.append({
                "text": text,
                "confidence": round(confidence, 3),
                "bbox": bbox,
                "low_confidence": confidence < self.confidence_threshold
            })
            confidences.append(confidence)

        if not blocks:
            return {
                "full_text": "",
                "blocks": [],
                "confidence": self.compute_confidence([]),
                "needs_llm_fallback": True
            }

        full_text = " ".join([b["text"] for b in blocks])
        confidence_metrics = self.compute_confidence(confidences)

        return {
            "full_text": full_text,
            "blocks": blocks,
            "confidence": confidence_metrics,
            "needs_llm_fallback": confidence_metrics["global_score"] < self.confidence_threshold,
            "total_blocks": len(blocks)
        }

    def extract_from_processed(self, preprocessor_result: dict) -> dict:
        """
        Prend directement la sortie du Preprocessor.
        Décide automatiquement si fallback LLM nécessaire.
        """
        if preprocessor_result["needs_llm_fallback"]:
            return {
                "full_text": "",
                "blocks": [],
                "confidence": self.compute_confidence([]),
                "needs_llm_fallback": True,
                "reason": "Qualité image insuffisante détectée par OpenCV"
            }

        result = self.extract_text(preprocessor_result["processed_path"])
        result["image_quality_score"] = preprocessor_result["quality_score"]
        return result