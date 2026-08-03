import re
from pathlib import Path


# Dictionnaire de mots-clés médicaux par type de document
KEYWORDS = {
    "ordonnance": [
        "ordonnance", "prescrit", "prescription", "comprime", "gelule",
        "mg", "ml", "posologie", "renouveler", "boite", "flacon",
        "matin", "soir", "midi", "jour", "semaine", "mois",
        "cachet", "ampoule", "dose", "traitement", "medicament",
        "prendre", "avaler", "appliquer", "injecter"
    ],
    "compte_rendu": [
        "compte rendu", "conclusion", "diagnostic", "examen",
        "scanner", "irm", "echographie", "biopsie", "endoscopie",
        "chirurgie", "operation", "intervention", "hospitalisation",
        "admission", "sortie", "service", "professeur", "docteur",
        "observation", "antecedents", "histoire", "clinique",
        "bilan", "résultat", "normal", "anormal", "pathologique"
    ],
    "courrier_medecin": [
        "cher confrere", "chere confrere", "madame", "monsieur",
        "adresser", "confier", "suivi", "consultation",
        "je vous adresse", "je vous envoie", "veuillez",
        "cordialement", "salutations", "cabinet", "praticien",
        "medecin traitant", "specialiste", "correspondant"
    ],
    "resultat_biologie": [
        "glycemie", "cholesterol", "triglycerides", "hemoglobine",
        "leucocytes", "plaquettes", "creatinine", "uree", "sodium",
        "potassium", "calcium", "albumine", "ferritine", "tsh",
        "mmol", "g/l", "u/l", "mg/l", "valeur", "norme",
        "reference", "taux", "dosage", "analyse", "laboratoire",
        "prelevement", "echantillon", "serum", "plasma"
    ],
    "resultat_imagerie": [
        "radio", "radiographie", "scanner", "tomodensitometrie",
        "irm", "imagerie", "echographie", "scintigraphie",
        "opacite", "lesion", "masse", "nodule", "kyste",
        "fracture", "calcification", "densite", "signal",
        "centimetre", "millimetre", "cm", "mm", "lobe",
        "poumon", "foie", "rein", "coeur", "cerveau", "os"
    ]
}

DOCUMENT_LABELS_FR = {
    "ordonnance": "ordonnance médicale",
    "compte_rendu": "compte rendu médical hospitalier",
    "courrier_medecin": "courrier entre médecins",
    "resultat_biologie": "résultat d'analyse biologique",
    "resultat_imagerie": "résultat d'imagerie médicale"
}


class DocumentClassifier:
    """
    Classification hybride des documents médicaux.
    Niveau 1 : mots-clés médicaux (rapide)
    Niveau 2 : Zero-Shot xlm-roberta (précis sur cas ambigus)
    """

    KEYWORD_THRESHOLD = 0.35
    ZEROSHOT_THRESHOLD = 0.50

    def __init__(self, use_zeroshot: bool = True):
        self.use_zeroshot = use_zeroshot
        self._zeroshot_model = None
        print("    ✓ Classificateur initialisé (mots-clés actifs)")
        if use_zeroshot:
            self._load_zeroshot()

    def _load_zeroshot(self):
        """Charge le modèle Zero-Shot en local (téléchargé une seule fois)."""
        try:
            from transformers import pipeline
            print("    ⟳ Chargement modèle Zero-Shot (xlm-roberta)...")
            self._zeroshot_model = pipeline(
                "zero-shot-classification",
                model="joeddav/xlm-roberta-large-xnli",
                device=-1  # CPU uniquement
            )
            print("    ✓ Modèle Zero-Shot chargé")
        except Exception as e:
            print(f"    ⚠️ Zero-Shot non disponible : {e}")
            self._zeroshot_model = None

    def _normalize_text(self, text: str) -> str:
        """Normalise le texte pour la comparaison."""
        text = text.lower()
        # Suppression accents basique
        replacements = {
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'à': 'a', 'â': 'a', 'ä': 'a',
            'ù': 'u', 'û': 'u', 'ü': 'u',
            'î': 'i', 'ï': 'i',
            'ô': 'o', 'ö': 'o',
            'ç': 'c', 'œ': 'oe', 'æ': 'ae'
        }
        for accented, plain in replacements.items():
            text = text.replace(accented, plain)
        # Suppression ponctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        return text

    def _classify_by_keywords(self, text: str) -> dict:
        """
        Niveau 1 — Classification par mots-clés médicaux.
        Rapide, zéro modèle nécessaire.
        """
        normalized = self._normalize_text(text)
        words = normalized.split()
        total_words = max(len(words), 1)

        scores = {}
        matched_keywords = {}

        for doc_type, keywords in KEYWORDS.items():
            matches = []
            for kw in keywords:
                kw_normalized = self._normalize_text(kw)
                if kw_normalized in normalized:
                    matches.append(kw)

            # Score = ratio mots-clés trouvés / total mots-clés de la catégorie
            raw_score = len(matches) / len(keywords)

            # Bonus si mots-clés trouvés dans les 100 premiers mots
            first_100 = " ".join(words[:100])
            early_matches = sum(
                1 for kw in matches
                if self._normalize_text(kw) in first_100
            )
            bonus = early_matches * 0.05

            scores[doc_type] = round(min(raw_score + bonus, 1.0), 3)
            matched_keywords[doc_type] = matches

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        return {
            "predicted_type": best_type,
            "score": best_score,
            "all_scores": scores,
            "matched_keywords": matched_keywords[best_type],
            "method": "keywords",
            "confident": best_score >= self.KEYWORD_THRESHOLD
        }

    def _classify_by_zeroshot(self, text: str) -> dict:
        """
        Niveau 2 — Zero-Shot avec xlm-roberta.
        Précis sur cas ambigus, ~2-3 secondes sur CPU.
        """
        if not self._zeroshot_model:
            return None

        # On prend les 512 premiers tokens pour rester dans les limites du modèle
        text_truncated = text[:1500]

        candidate_labels = list(DOCUMENT_LABELS_FR.values())

        result = self._zeroshot_model(
            text_truncated,
            candidate_labels=candidate_labels,
            hypothesis_template="Ce document est {}."
        )

        # Conversion labels FR → clés internes
        labels_to_keys = {v: k for k, v in DOCUMENT_LABELS_FR.items()}
        all_scores = {
            labels_to_keys[label]: round(score, 3)
            for label, score in zip(result['labels'], result['scores'])
        }

        best_label = result['labels'][0]
        best_type = labels_to_keys[best_label]
        best_score = result['scores'][0]

        return {
            "predicted_type": best_type,
            "score": round(best_score, 3),
            "all_scores": all_scores,
            "method": "zero_shot",
            "confident": best_score >= self.ZEROSHOT_THRESHOLD
        }

    def classify(self, text: str) -> dict:
        """
        Point d'entrée principal.
        Hybride : mots-clés d'abord, Zero-Shot si ambiguïté.
        """
        if not text or len(text.strip()) < 10:
            return {
                "predicted_type": "inconnu",
                "score": 0.0,
                "method": "none",
                "confident": False,
                "label_fr": "Document non classifiable",
                "reason": "Texte trop court ou vide"
            }

        # Niveau 1 — mots-clés
        kw_result = self._classify_by_keywords(text)

        if kw_result["confident"]:
            # Score suffisant — on fait confiance aux mots-clés
            kw_result["label_fr"] = DOCUMENT_LABELS_FR.get(
                kw_result["predicted_type"], "inconnu"
            )
            kw_result["zeroshot_used"] = False
            return kw_result

        # Niveau 2 — Zero-Shot si disponible
        if self.use_zeroshot and self._zeroshot_model:
            print("    ⟳ Ambiguïté détectée → Zero-Shot activé...")
            zs_result = self._classify_by_zeroshot(text)
            if zs_result:
                zs_result["label_fr"] = DOCUMENT_LABELS_FR.get(
                    zs_result["predicted_type"], "inconnu"
                )
                zs_result["zeroshot_used"] = True
                zs_result["keyword_scores"] = kw_result["all_scores"]
                return zs_result

        # Fallback — meilleur score mots-clés même si insuffisant
        kw_result["label_fr"] = DOCUMENT_LABELS_FR.get(
            kw_result["predicted_type"], "inconnu"
        )
        kw_result["zeroshot_used"] = False
        kw_result["confident"] = False
        return kw_result