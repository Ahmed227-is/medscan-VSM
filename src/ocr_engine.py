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
    avec score de confiance composite et qualité linguistique.
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
                "low_confidence_ratio": 1.0,
                "linguistic_quality": {},
                "linguistic_penalty_applied": False
            }

        mean_score = float(np.mean(confidences))
        min_score = float(np.min(confidences))
        low_conf_ratio = sum(
            1 for c in confidences if c < self.confidence_threshold
        ) / len(confidences)

        composite = (
            mean_score * 0.5 +
            (1 - low_conf_ratio) * 0.3 +
            min_score * 0.2
        )

        return {
            "global_score": round(composite, 3),
            "mean_confidence": round(mean_score, 3),
            "min_confidence": round(min_score, 3),
            "low_confidence_ratio": round(low_conf_ratio, 3),
            "linguistic_quality": {},
            "linguistic_penalty_applied": False
        }

    def compute_linguistic_quality(self, text: str) -> dict:
        """
        Mesure la qualité linguistique du texte extrait.
        Détecte si PaddleOCR a produit du charabia sur manuscrit.

        Métriques :
        1. Ratio majuscules anormales (au milieu des mots)
        2. Ratio caractères alphabétiques
        3. Longueur moyenne des mots
        4. Présence de séquences impossibles en français

        Interprétation :
        0.0 - 0.5 : charabia → fallback Qwen
        0.5 - 1.0 : texte correct → PaddleOCR suffisant
        """
        if not text or len(text.strip()) < 10:
            return {
                "linguistic_score": 0.0,
                "is_readable": False,
                "abnormal_caps_ratio": 0.0,
                "alpha_ratio": 0.0,
                "avg_word_length": 0.0,
                "impossible_ratio": 0.0
            }

        words = text.split()
        if not words:
            return {
                "linguistic_score": 0.0,
                "is_readable": False,
                "abnormal_caps_ratio": 0.0,
                "alpha_ratio": 0.0,
                "avg_word_length": 0.0,
                "impossible_ratio": 0.0
            }

        # Métrique 1 — Ratio majuscules anormales
        # "etRospJaluse" → majuscule au milieu = suspect
        # Exclut les acronymes médicaux (UHTCD, HTA, NFS...)
        abnormal_caps = sum(
            1 for w in words
            if len(w) > 2
            and any(c.isupper() for c in w[1:])
            and not w.isupper()
        )
        abnormal_caps_ratio = abnormal_caps / max(len(words), 1)

        # Métrique 2 — Ratio caractères alphabétiques
        alpha_chars = sum(1 for c in text if c.isalpha())
        total_chars = max(len(text), 1)
        alpha_ratio = alpha_chars / total_chars

        # Métrique 3 — Longueur moyenne des mots
        avg_word_length = sum(len(w) for w in words) / max(len(words), 1)
        if 3 <= avg_word_length <= 10:
            length_score = 1.0
        elif avg_word_length < 3:
            length_score = avg_word_length / 3.0
        else:
            length_score = max(0.0, 1.0 - (avg_word_length - 10) / 10.0)

        # Métrique 4 — Séquences impossibles en français
        impossible_sequences = [
            'hr', 'qw', 'bw', 'fw', 'pw', 'gf',
            'zj', 'xr', 'kw', 'vw', 'df', 'cdr',
            'mcc', 'rrt', 'qqr', 'wwe'
        ]
        text_lower = text.lower()
        impossible_count = sum(
            1 for seq in impossible_sequences
            if seq in text_lower
        )
        impossible_ratio = min(impossible_count / 5.0, 1.0)

        # Score linguistique final
        linguistic_score = (
            (1 - abnormal_caps_ratio) * 0.35 +
            alpha_ratio * 0.25 +
            length_score * 0.20 +
            (1 - impossible_ratio) * 0.20
        )
        linguistic_score = round(max(0.0, min(linguistic_score, 1.0)), 3)

        return {
            "linguistic_score": linguistic_score,
            "is_readable": linguistic_score >= 0.5,
            "abnormal_caps_ratio": round(abnormal_caps_ratio, 3),
            "alpha_ratio": round(alpha_ratio, 3),
            "avg_word_length": round(avg_word_length, 3),
            "impossible_ratio": round(impossible_ratio, 3)
        }

    def extract_text(self, image_input) -> dict:
        """
        Extrait le texte d'une image ou d'un chemin.
        Retourne texte complet + blocs détaillés + scores + qualité linguistique.
        """
        result = self.ocr.ocr(image_input, cls=True)

        if not result or not result[0]:
            return {
                "full_text": "",
                "blocks": [],
                "confidence": self.compute_confidence([]),
                "needs_llm_fallback": True,
                "total_blocks": 0
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
                "needs_llm_fallback": True,
                "total_blocks": 0
            }

        full_text = " ".join([b["text"] for b in blocks])
        confidence_metrics = self.compute_confidence(confidences)

        # Calcul qualité linguistique
        linguistic = self.compute_linguistic_quality(full_text)

        # Pénalise le score composite si texte linguistiquement suspect
        if not linguistic['is_readable']:
            penalized = confidence_metrics['global_score'] * linguistic['linguistic_score']
            confidence_metrics['global_score'] = round(penalized, 3)
            confidence_metrics['linguistic_penalty_applied'] = True
        else:
            confidence_metrics['linguistic_penalty_applied'] = False

        confidence_metrics['linguistic_quality'] = linguistic

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
                "reason": "Qualité image insuffisante détectée par OpenCV",
                "total_blocks": 0
            }

        result = self.extract_text(preprocessor_result["processed_path"])
        result["image_quality_score"] = preprocessor_result["quality_score"]
        return result