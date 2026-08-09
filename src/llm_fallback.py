import base64
import json
import requests
from pathlib import Path


class LLMFallback:
    """
    Fallback vers Qwen2.5-VL via Ollama.
    Utilisé quand PaddleOCR échoue ou donne
    des résultats peu fiables.
    100% local — RGPD conforme.
    """

    OLLAMA_URL = "http://localhost:11434/api/generate"
    MODEL = "qwen2.5vl:3b"

    # Seuils de déclenchement
    QUALITY_THRESHOLD = 0.3    # Score qualité OpenCV
    CONFIDENCE_THRESHOLD = 0.6  # Score composite PaddleOCR

    def __init__(self):
        self.available = self._check_ollama()

    def _check_ollama(self) -> bool:
        """Vérifie qu'Ollama tourne et que Qwen est disponible."""
        try:
            response = requests.get(
                "http://localhost:11434/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                available = any('qwen2.5vl' in m for m in models)
                if available:
                    print("    ✓ Ollama + Qwen2.5-VL disponible")
                else:
                    print("    ⚠️ Ollama disponible mais Qwen non trouvé")
                return available
        except Exception:
            print("    ⚠️ Ollama non disponible — démarrez avec: ollama serve")
            return False

    def _image_to_base64(self, image_path: str) -> str:
        """Convertit une image en base64 pour Ollama."""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _build_prompt(self) -> str:
        """Prompt optimisé pour l'extraction médicale française."""
        return """Tu es un expert en documents médicaux français.
Analyse cette image de document médical scanné et extrais TOUT le texte visible.

Instructions :
- Extrais le texte exactement comme il apparaît
- Conserve la structure et l'ordre du texte
- Inclus les dates, médicaments, dosages, noms
- Si le texte est manuscrit, fais de ton mieux
- Réponds UNIQUEMENT avec le texte extrait, rien d'autre

Texte extrait :"""

    def extract_text(self, image_path: str) -> dict:
        """
        Extrait le texte via Qwen2.5-VL.
        Retourne un dictionnaire compatible avec OCREngine.
        """
        if not self.available:
            return {
                "full_text": "",
                "blocks": [],
                "confidence": {
                    "global_score": 0.0,
                    "mean_confidence": 0.0,
                    "min_confidence": 0.0,
                    "low_confidence_ratio": 1.0
                },
                "needs_llm_fallback": False,
                "total_blocks": 0,
                "method": "llm_unavailable"
            }

        try:
            image_b64 = self._image_to_base64(image_path)

            payload = {
                "model": self.MODEL,
                "prompt": self._build_prompt(),
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Faible température = plus factuel
                    "num_predict": 2048
                }
            }

            print(f"    ⟳ Qwen2.5-VL analyse l'image...")
            response = requests.post(
                self.OLLAMA_URL,
                json=payload,
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                text = result.get('response', '').strip()

                if text:
                    print(f"    ✓ Qwen a extrait {len(text.split())} mots")
                    return {
                        "full_text": text,
                        "blocks": [{"text": text, "confidence": 0.85, "bbox": [], "low_confidence": False}],
                        "confidence": {
                            "global_score": 0.85,
                            "mean_confidence": 0.85,
                            "min_confidence": 0.85,
                            "low_confidence_ratio": 0.0
                        },
                        "needs_llm_fallback": False,
                        "total_blocks": 1,
                        "method": "qwen2.5vl"
                    }
                else:
                    return self._empty_result("Qwen a retourné un texte vide")

            else:
                return self._empty_result(f"Erreur Ollama : {response.status_code}")

        except requests.Timeout:
            return self._empty_result("Timeout Qwen — image trop complexe")
        except Exception as e:
            return self._empty_result(f"Erreur : {str(e)}")

    def _empty_result(self, reason: str) -> dict:
        print(f"    ⚠️ {reason}")
        return {
            "full_text": "",
            "blocks": [],
            "confidence": {
                "global_score": 0.0,
                "mean_confidence": 0.0,
                "min_confidence": 0.0,
                "low_confidence_ratio": 1.0
            },
            "needs_llm_fallback": False,
            "total_blocks": 0,
            "method": "llm_failed"
        }

    def should_fallback(
        self,
        quality_score: float,
        ocr_confidence: float,
        linguistic_score: float = 1.0
    ) -> tuple:
        """
        Détermine si le fallback Qwen est nécessaire.
        Retourne (bool, raison)
        """
        if quality_score < self.QUALITY_THRESHOLD:
            return True, f"Qualité image insuffisante ({quality_score} < {self.QUALITY_THRESHOLD})"

        if ocr_confidence < self.CONFIDENCE_THRESHOLD:
            return True, f"Confiance OCR insuffisante ({ocr_confidence} < {self.CONFIDENCE_THRESHOLD})"

        if linguistic_score < 0.5:
            return True, f"Texte linguistiquement suspect ({linguistic_score} < 0.5)"

        return False, None