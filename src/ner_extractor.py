"""
ner_extractor.py — MedScan VSM
Module d'extraction d'entités médicales NER, conforme au référentiel VSM HAS / CI-SIS.

HISTORIQUE DE CONCEPTION (pour le dossier technique / soutenance) :
    - v1 : approche hybride regex + DrBERT-4GB-CP-CamemBERT
    - Constat sur documents réels (BANANE_Sophie.pdf, etc.) :
        · Le regex, même exhaustif, ne capture pas la variabilité du langage
          médical réel (formulations libres, abréviations non standard,
          erreurs OCR sur manuscrits) → beaucoup de faux négatifs.
        · DrBERT-4GB-CP-CamemBERT est incompatible avec transformers==5.9.0
          (erreur de tokenizer), et les alternatives testées (DoctoBERT,
          HealthcareNER-Fr) ne sont pas exploitables (non fine-tuné / gated).
    - v2 (ce fichier) : extraction 100% via Qwen2.5-VL en mode TEXTE (pas
      vision) sur le texte déjà extrait par ocr_engine.py / llm_fallback.py.
      Un LLM généraliste bien prompté gère mieux le langage médical libre
      qu'un pipeline regex, et Qwen est déjà intégré/testé dans le projet
      (fallback OCR), donc aucune nouvelle dépendance.

INTERFACE CONSERVÉE (compatibilité avec le reste du pipeline) :
    - NERExtractor().extract(text, document_type="inconnu") -> dict
      (même signature et mêmes clés de sortie que la v1 regex+DrBERT)
    - Nouveau : NERExtractor().extract_document(pages, document_type, tracker)
      pour traiter un document multi-pages avec fusion/dédoublonnage.
"""

import json
import re
import logging
import requests
from difflib import SequenceMatcher
from typing import Optional

logger = logging.getLogger("ner_extractor")

# ============================================================
# CONFIG OLLAMA — cohérent avec llm_fallback.py
# ============================================================
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5vl:3b"
NUM_CTX_MIN = 1024      # plancher : jamais en dessous, même pour un texte très court
NUM_CTX_MAX = 8192      # plafond : jamais au-dessus, pour rester sûr sur 16GB RAM / iGPU
TIMEOUT = 600

# Seuil de similarité pour le dédoublonnage fuzzy (0-1)
FUZZY_DEDUP_THRESHOLD = 0.85

# ============================================================
# SCHÉMA DE SORTIE — conserve exactement les clés de la v1
# (pathologies_actives, antecedents_medicaux, etc.) pour ne
# rien casser en aval (vsm_generator.py, test_pipeline.py)
# ============================================================
OUTPUT_SCHEMA_KEYS = [
    "pathologies_actives",
    "antecedents_medicaux",
    "antecedents_chirurgicaux",
    "allergies_intolerances",
    "traitements_en_cours",
    "constantes_biologiques",
    "vaccinations",
    "facteurs_risque",
    "examens_bilans",
    "dates_importantes",
]

SYSTEM_PROMPT = """Tu es un assistant médical spécialisé dans l'extraction d'informations \
structurées à partir de documents médicaux français (ordonnances, comptes rendus, \
courriers médecin, résultats biologiques, résultats imagerie). Le texte fourni est \
issu d'un OCR et peut contenir des imperfections.

RÈGLES :
1. Extrais UNIQUEMENT les informations explicitement présentes dans le texte. N'invente rien.
2. Si une catégorie n'a rien à extraire, retourne une liste vide — ne force jamais une entrée.
3. Formule chaque entité en une chaîne courte MAIS COMPLÈTE, avec le contexte utile trouvé
   dans le texte (résultat, conclusion, dosage, posologie...). Ne te contente jamais du
   seul nom de l'entité si le texte donne plus de détails autour.
4. Relis le texte entièrement avant de répondre : les dates, en particulier, apparaissent
   souvent à PLUSIEURS endroits différents du document (date de prescription, date
   d'enregistrement, date de l'acte...) — il faut TOUTES les extraire, pas seulement la
   première trouvée.

EXEMPLE :

Texte source :
"Prescrit le 09/10/2014. Enregistré le 10/10/2014. FROTTIS DE DEPISTAGE : Ménopause.
Le fond est propre. CONCLUSION : Frottis satisfaisant, non inflammatoire dans le cadre
d'une ménopause subatrophique. Absence de cellule suspecte sur les frottis examinés."

Réponse attendue :
{
  "pathologies_actives": ["Ménopause subatrophique"],
  "antecedents_medicaux": [],
  "antecedents_chirurgicaux": [],
  "allergies_intolerances": [],
  "traitements_en_cours": [],
  "constantes_biologiques": [],
  "vaccinations": [],
  "facteurs_risque": [],
  "examens_bilans": ["Frottis de dépistage cervico-vaginal : satisfaisant, non inflammatoire, absence de cellule suspecte"],
  "dates_importantes": ["Prescrit le 09/10/2014", "Enregistré le 10/10/2014"]
}

Remarque : le texte source contient DEUX dates distinctes, et les deux sont extraites.
La conclusion médicale ("ménopause subatrophique") est reprise dans les pathologies,
pas seulement mentionnée dans les examens.

Réponds STRICTEMENT en JSON valide, sans texte avant ou après, selon ce schéma exact :

{
  "pathologies_actives": ["string", ...],
  "antecedents_medicaux": ["string", ...],
  "antecedents_chirurgicaux": ["string", ...],
  "allergies_intolerances": ["string", ...],
  "traitements_en_cours": ["string", ...],
  "constantes_biologiques": ["string", ...],
  "vaccinations": ["string", ...],
  "facteurs_risque": ["string", ...],
  "examens_bilans": ["string", ...],
  "dates_importantes": ["string", ...]
}
"""


class NERExtractor:
    """
    Extraction d'entités médicales 100% via Qwen2.5-VL (mode texte, Ollama).
    Conforme au référentiel VSM HAS / CI-SIS.

    Entités extraites :
    - Pathologies actives
    - Antécédents médicaux et chirurgicaux
    - Allergies et intolérances
    - Traitements en cours
    - Constantes biologiques et cliniques
    - Vaccinations
    - Facteurs de risque
    - Examens et bilans
    - Dates importantes
    """

    def __init__(self, ollama_url: str = OLLAMA_URL, model: str = MODEL_NAME,
                 num_ctx_max: int = NUM_CTX_MAX, timeout: int = TIMEOUT):
        self.ollama_url = ollama_url
        self.model = model
        self.num_ctx_max = num_ctx_max
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Normalisation (reprise de la v1, toujours utile en amont de Qwen)
    # ------------------------------------------------------------------
    def _normalize_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('–', '-').replace('—', '-')
        text = text.replace('œ', 'oe').replace('æ', 'ae')
        return text.strip()

    # ------------------------------------------------------------------
    # Point d'entrée principal — UNE page / UN texte
    # (signature conservée : extract(text, document_type))
    # ------------------------------------------------------------------
    def extract(self, text: str, document_type: str = "inconnu") -> dict:
        if not text or len(text.strip()) < 10:
            return self._empty_result("Texte vide ou trop court")

        text = self._normalize_text(text)
        entities = self._call_qwen(text)

        result = {key: entities.get(key, []) for key in OUTPUT_SCHEMA_KEYS}
        result["document_type"] = document_type
        result["extraction_method"] = "qwen_only"
        return result

    def _empty_result(self, reason: str) -> dict:
        result = {key: [] for key in OUTPUT_SCHEMA_KEYS}
        result["document_type"] = "inconnu"
        result["extraction_method"] = "none"
        result["reason"] = reason
        return result

    # ------------------------------------------------------------------
    # num_ctx dynamique : évite d'allouer 8192 tokens de contexte pour
    # une page de quelques lignes. On estime ~1 token ≈ 4 caractères
    # (approximation standard pour du français), on ajoute le system
    # prompt (+few-shot) + une marge pour la réponse JSON générée.
    # ------------------------------------------------------------------
    def _compute_num_ctx(self, text: str) -> int:
        estimated_tokens = len(text) // 4
        prompt_overhead = len(SYSTEM_PROMPT) // 4
        response_margin = 512  # marge pour le JSON de sortie

        needed = estimated_tokens + prompt_overhead + response_margin
        # arrondi au multiple de 512 supérieur (Ollama gère mieux ces tailles)
        needed = ((needed // 512) + 1) * 512

        return max(NUM_CTX_MIN, min(needed, self.num_ctx_max))

    # ------------------------------------------------------------------
    # Appel Qwen (texte pur, pas vision) + parsing robuste
    # ------------------------------------------------------------------
    def _call_qwen(self, text: str) -> dict:
        empty = {k: [] for k in OUTPUT_SCHEMA_KEYS}
        num_ctx = self._compute_num_ctx(text)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Texte médical à analyser :\n\n{text}"},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "num_ctx": num_ctx,
                "temperature": 0.1,
            },
        }

        logger.info(f"Appel Qwen NER — texte={len(text)} caractères, num_ctx={num_ctx}")

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur appel Qwen (NER) : {e}")
            return empty

        try:
            content = response.json()["message"]["content"]
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Réponse Ollama inattendue (NER) : {e}")
            return empty

        return self._parse_json_response(content)

    def _parse_json_response(self, content: str) -> dict:
        empty = {k: [] for k in OUTPUT_SCHEMA_KEYS}

        try:
            parsed = json.loads(content)
            return self._sanitize(parsed)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return self._sanitize(parsed)
            except json.JSONDecodeError:
                pass

        logger.warning("Impossible de parser le JSON Qwen (NER), résultat vide retourné.")
        logger.debug(f"Réponse brute Qwen (NER) : {content[:500]}")
        return empty

    def _sanitize(self, parsed: dict) -> dict:
        clean = {}
        for key in OUTPUT_SCHEMA_KEYS:
            value = parsed.get(key, [])
            if not isinstance(value, list):
                value = []
            clean[key] = [str(v).strip() for v in value if str(v).strip()]
        return clean

    # ------------------------------------------------------------------
    # Extraction sur un DOCUMENT entier (toutes les pages) + fusion
    # ------------------------------------------------------------------
    def extract_document(self, pages: list[str], document_type: str = "inconnu",
                          tracker=None) -> dict:
        """
        pages : liste de textes OCR, un par page (dans l'ordre du document).
        tracker : PipelineTracker optionnel pour logguer la progression.

        Retourne le JSON fusionné et dédoublonné (fuzzy matching) au niveau
        du document entier, avec les mêmes clés que extract().
        """
        per_page_results = []

        for i, page_text in enumerate(pages, start=1):
            if tracker:
                tracker.update(step="ner_extraction", page=i, total=len(pages))

            page_result = self.extract(page_text, document_type=document_type)
            per_page_results.append(page_result)

            nb_entities = sum(len(page_result[k]) for k in OUTPUT_SCHEMA_KEYS)
            logger.info(f"Page {i}/{len(pages)} : NER terminé ({nb_entities} entités).")

        merged = self.merge_entities(per_page_results)
        merged["document_type"] = document_type
        merged["extraction_method"] = "qwen_only"
        return merged

    # ------------------------------------------------------------------
    # Fusion + dédoublonnage fuzzy (sans appel LLM, difflib stdlib)
    # Méthode publique : réutilisable depuis test_pipeline.py pour fusionner
    # les résultats NER de plusieurs pages, avec la même logique que
    # extract_document() (garde toujours la formulation la plus détaillée).
    # ------------------------------------------------------------------
    def merge_entities(self, per_page_results: list[dict]) -> dict:
        merged = {key: [] for key in OUTPUT_SCHEMA_KEYS}

        for page_result in per_page_results:
            for key in OUTPUT_SCHEMA_KEYS:
                for value in page_result.get(key, []):
                    self._add_with_dedup(merged[key], value)

        return merged

    def _add_with_dedup(self, existing_list: list, new_value: str) -> None:
        """
        Ajoute new_value à existing_list, en fusionnant avec un doublon proche.
        Deux critères de doublon (l'un ou l'autre suffit) :
          1. Similarité globale (SequenceMatcher) >= FUZZY_DEDUP_THRESHOLD
             -> attrape les reformulations proches ("diabete type 2" vs
             "diabete de type 2").
          2. Inclusion : la version courte est un sous-ensemble textuel de
             la version longue -> attrape le cas fréquent où Qwen extrait la
             même entité avec plus de détails sur une autre page
             ("Metformine 1000mg" vs "Metformine 1000mg matin et soir").
        Dans les deux cas, on garde la formulation la plus longue/détaillée.
        """
        new_norm = new_value.lower().strip()

        for i, existing_value in enumerate(existing_list):
            existing_norm = existing_value.lower().strip()

            similarity = SequenceMatcher(None, existing_norm, new_norm).ratio()
            # Garde-fou : n'applique la règle d'inclusion qu'à partir de 6
            # caractères, pour éviter qu'une entité courte ("TA", "IMC")
            # ne soit à tort considérée comme un doublon d'une autre entité
            # qui la contient juste par hasard ("HTA" contient "TA").
            is_substring = (
                len(existing_norm) >= 6 and len(new_norm) >= 6
                and (existing_norm in new_norm or new_norm in existing_norm)
            )

            if similarity >= FUZZY_DEDUP_THRESHOLD or is_substring:
                if len(new_value) > len(existing_value):
                    existing_list[i] = new_value
                return

        existing_list.append(new_value)


# ----------------------------------------------------------------------
# Exemple d'utilisation (à intégrer dans test_pipeline.py)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    extractor = NERExtractor()

    sample_text = """
    0780 POLE GYNECOLOGIE OBSTETRIQUE, MEDECINE FCETALE, REPRODUCTION ET GENETIQUE GyNecoLogiE OnsteTRiQuE A > Gynecologie hospitalisation, Consultations externes et echographies, Bloc operatoire, Orthogenie (Centre IVG, planification familiale) Centre d'Accueil des Victimes Presumees d'Abus Sexuels GynecoLogIE OsstetriQue B > Obstetrique hospitalisation, Salle de Naissances, Urgences, Grossesses pathologiques, Medecine Fcetale MEDEcINE ET BIoLOGIE DE LA RepRoDucTION > Consultations CECOS FIV, Hospitalisations de jOur, laboratoire GENETiQue>Consultations, Laboratoires, Fcetopathologie Gynecologie-Obstetrique A Courrier adresse a:Dr Copiea:-Dr le 29 avril 2011 No Dossier: COMPTE-RENDU DE CONSULTATION Chere Je t'adresse comme convenu Madame BANANE Sophie nee le 17/09/1962, pour la prise en charge de dysurie.. Dans ses antecedents on note : Sur le plan familial, RAS Sur le plan chirurgical, Fracture cubitale. Cholecystectomie. Sur le plan medical, Pneumothorax. Sur le plan obstetrical, Une naissance par les voies naturelles. Sur le plan gynecologique,. Ses frottis, mammographies, echographies sont a jour. Cette patiente est menopausee.. L'histoire de la maladie commence en 2008 au traitement chirurgical d'un kyste uretral par les voies naturelles. A la suite de cette intervention, des douleurssupportables ont necessite un traitement par Rivotril et la patiente a vu apparaitre des fuites urinaires apres chaque miction, en quantite selon elle assez importante, necessitant la mise en place d'un protege slip en permanence.
    """

    result = extractor.extract(sample_text, document_type="ordonnance")
    print(json.dumps(result, indent=2, ensure_ascii=False))