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
# FILTRE DE NÉGATION — post-traitement déterministe
# Qwen (3B) a tendance à recopier des mentions de négation
# ("RAS", "sans particularité"...) comme si c'était une vraie
# entité positive, malgré la consigne du prompt. On filtre ça
# après coup plutôt que de compter sur le modèle pour respecter
# la règle "liste vide si rien à signaler" — plus fiable et
# indépendant de la capacité du modèle utilisé.
# ============================================================
NEGATION_TERMS = {
    "ras", "rien a signaler", "rien à signaler",
    "nad", "non applicable",
    "sans particularite", "sans particularité",
    "neant", "néant",
    "aucun", "aucune",
    "sans antecedent", "sans antécédent",
    "sans antecedents", "sans antécédents",
    "sans antecedent notable", "sans antécédent notable",
    "sans antecedents notables", "sans antécédents notables",
    "pas d'antecedent", "pas d'antécédent",
    "pas d'antecedents", "pas d'antécédents",
    "non contributif", "non contributive",
    "sans particularites", "sans particularités",
    "negatif", "négatif", "negative", "négative",
    "non",
}

# ============================================================
# VALIDATION DES DATES — post-traitement, catégorie
# "dates_importantes" uniquement.
# Qwen a tendance à y glisser du texte qui n'est pas une date
# (formules de politesse, instructions de soin, durées...).
# On ne RE-extrait rien : on vérifie juste que l'entité déjà
# sortie par Qwen contient bel et bien une date reconnaissable,
# avec un contrôle de plausibilité (jour 1-31, mois 1-12) pour
# écarter les dates corrompues par l'OCR (ex: "23/C5/2008").
# ============================================================
MONTHS_FR = (
    r'janvier|f[ée]vrier|mars|avril|mai|juin|juillet|'
    r'ao[uû]t|septembre|octobre|novembre|d[ée]cembre'
)

DATE_VALIDATION_PATTERNS = [
    # JJ/MM/AAAA, JJ-MM-AAAA, JJ.MM.AAAA (jour/mois avec vérif plausibilité)
    re.compile(r'\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b'),
    # "12 juin 1996", "1er octobre 2009"
    re.compile(rf'\b\d{{1,2}}(?:er)?\s+(?:{MONTHS_FR})\s+\d{{4}}\b', re.IGNORECASE),
    # "octobre 2009" (mois + année sans jour)
    re.compile(rf'\b(?:{MONTHS_FR})\s+\d{{4}}\b', re.IGNORECASE),
]

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
5. Chaque entité va dans UNE SEULE catégorie, la plus précise possible :
   - un événement déjà opéré va dans antecedents_chirurgicaux, pas dans antecedents_medicaux
   - un événement médical déjà survenu (pathologie passée ou active) ne va PAS dans
     facteurs_risque (les facteurs de risque sont des éléments de mode de vie/hérédité,
     pas des diagnostics déjà posés)
   - une valeur biologique chiffrée (ex: "Hémoglobine 13,4 g/dl") va dans
     constantes_biologiques, pas dans examens_bilans

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
            entries = [str(v).strip() for v in value if str(v).strip()]
            entries = [e for e in entries if not self._is_negation(e)]

            if key == "dates_importantes":
                entries = [e for e in entries if self._contains_valid_date(e)]

            clean[key] = entries

        clean = self._resolve_allergy_conflicts(clean)
        clean = self._resolve_category_priority(clean)
        return clean

    def _is_negation(self, value: str) -> bool:
        """
        Détecte si une entité extraite n'est en fait qu'une mention de
        négation ("RAS", "sans particularité"...) et non une vraie donnée
        clinique. Normalisation : minuscules, accents simplifiés, ponctuation
        de bord retirée. Comparaison stricte sur la chaîne entière (pas de
        sous-chaîne) pour ne jamais filtrer une entité légitime qui
        contiendrait incidemment un de ces mots (ex: "Non fumeuse depuis 2020"
        ne doit pas être filtré, seul un "Non" isolé doit l'être).
        """
        normalized = value.lower().strip().strip(".:-").strip()
        normalized = (normalized
                      .replace('é', 'e').replace('è', 'e').replace('ê', 'e')
                      .replace('à', 'a').replace('ù', 'u'))
        return normalized in NEGATION_TERMS

    def _contains_valid_date(self, value: str) -> bool:
        """
        Vérifie que l'entité contient une date plausible. Pour le format
        numérique JJ/MM/AAAA, on valide en plus que jour et mois sont dans
        des plages réalistes — ça permet d'écarter les dates corrompues par
        l'OCR (ex: "23/C5/2008" ne matche déjà pas le pattern car 'C5' n'est
        pas numérique ; "32/13/2020" matcherait le pattern mais serait rejeté
        ici car jour/mois hors plage).
        """
        for pattern in DATE_VALIDATION_PATTERNS:
            for match in pattern.finditer(value):
                groups = match.groups()
                if len(groups) == 3:  # format numérique JJ/MM/AAAA
                    day, month, _year = groups
                    if 1 <= int(day) <= 31 and 1 <= int(month) <= 12:
                        return True
                    continue  # ce match précis est invalide, on essaie la suite
                else:
                    return True  # format textuel (mois nommé) : déjà fiable
        return False

    def _resolve_allergy_conflicts(self, result: dict) -> dict:
        """
        Contrôle de cohérence critique pour la sécurité du patient : si une
        entité apparaît à la fois dans "allergies_intolerances" et
        "traitements_en_cours", c'est très probablement une confusion du
        modèle (il a classé un médicament pris par la patiente comme une
        allergie), pas une vraie coïncidence. On retire l'entité des
        allergies dans ce cas — un traitement en cours mal classé est un
        moindre mal comparé à une fausse allergie affichée dans un VSM.
        """
        treatments = result.get("traitements_en_cours", [])
        allergies = result.get("allergies_intolerances", [])

        kept_allergies = []
        for allergy in allergies:
            allergy_norm = allergy.lower().strip()
            conflict = False
            for treatment in treatments:
                treatment_norm = treatment.lower().strip()
                similarity = SequenceMatcher(None, allergy_norm, treatment_norm).ratio()
                is_substring = (
                    len(allergy_norm) >= 4
                    and (allergy_norm in treatment_norm or treatment_norm in allergy_norm)
                )
                if similarity >= FUZZY_DEDUP_THRESHOLD or is_substring:
                    conflict = True
                    logger.warning(
                        f"Conflit allergie/traitement détecté et résolu : "
                        f"\"{allergy}\" retiré des allergies (présent aussi dans "
                        f"les traitements en cours : \"{treatment}\")."
                    )
                    break
            if not conflict:
                kept_allergies.append(allergy)

        result["allergies_intolerances"] = kept_allergies
        return result

    def _remove_matches(self, source_list: list, reference_lists: list,
                         reason: str) -> list:
        """
        Fonction générique : retire de source_list toute entité qui
        fuzzy-matche une entité déjà présente dans l'une des reference_lists.
        Utilisée pour appliquer une hiérarchie de priorité entre catégories
        qui se chevauchent sémantiquement.
        """
        kept = []
        for item in source_list:
            item_norm = item.lower().strip()
            matched_ref = None

            for ref_list in reference_lists:
                for ref in ref_list:
                    ref_norm = ref.lower().strip()
                    similarity = SequenceMatcher(None, item_norm, ref_norm).ratio()
                    is_substring = (
                        len(item_norm) >= 6 and len(ref_norm) >= 6
                        and (item_norm in ref_norm or ref_norm in item_norm)
                    )
                    if similarity >= FUZZY_DEDUP_THRESHOLD or is_substring:
                        matched_ref = ref
                        break
                if matched_ref:
                    break

            if matched_ref:
                logger.info(f"{reason} : \"{item}\" retiré (déjà présent sous "
                            f"forme \"{matched_ref}\" dans une catégorie prioritaire).")
            else:
                kept.append(item)

        return kept

    def _resolve_category_priority(self, result: dict) -> dict:
        """
        Résout les chevauchements entre catégories qui se recoupent
        sémantiquement, en appliquant une hiérarchie de priorité (la
        catégorie la plus spécifique/structurée gagne, l'entité est retirée
        de la catégorie la moins spécifique). Constaté sur documents réels :
        une même entité ("Pneumothorax", une valeur d'hémoglobine...) peut
        se retrouver dupliquée dans plusieurs catégories après fusion de
        toutes les pages d'un dossier.

        Règles appliquées :
        1. Antécédent chirurgical > antécédent médical
           (le chirurgical est une information plus précise)
        2. Pathologie active / antécédent (médical ou chirurgical)
           > facteur de risque
           (un événement déjà survenu n'est plus juste un "risque" théorique)
        3. Constante biologique > examen/bilan
           (une valeur numérique structurée est plus utile qu'une mention
           générique dans la liste des examens)
        """
        # Règle 1 : chirurgical prioritaire sur médical générique
        result["antecedents_medicaux"] = self._remove_matches(
            result.get("antecedents_medicaux", []),
            [result.get("antecedents_chirurgicaux", [])],
            reason="Priorité chirurgical > médical",
        )

        # Règle 2 : pathologie/antécédent déjà survenu prioritaire sur facteur de risque
        result["facteurs_risque"] = self._remove_matches(
            result.get("facteurs_risque", []),
            [
                result.get("pathologies_actives", []),
                result.get("antecedents_medicaux", []),
                result.get("antecedents_chirurgicaux", []),
            ],
            reason="Priorité pathologie/antécédent > facteur de risque",
        )

        # Règle 3 : constante biologique structurée prioritaire sur mention d'examen générique
        result["examens_bilans"] = self._remove_matches(
            result.get("examens_bilans", []),
            [result.get("constantes_biologiques", [])],
            reason="Priorité constante biologique > examen générique",
        )

        return result

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

        # Re-applique le contrôle allergie/traitement au niveau document :
        # un conflit peut n'apparaître qu'après fusion (allergie mentionnée
        # page 5, même médicament en traitement page 80, par ex.).
        merged = self._resolve_allergy_conflicts(merged)
        merged = self._resolve_category_priority(merged)

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
    Patient : Jean DUPONT, né le 12/03/1965
    Antécédents : HTA, diabète type 2 depuis 2010
    Allergies : pénicilline
    Traitement actuel : Metformine 1000mg 2x/jour, Amlodipine 5mg 1x/jour
    Consultation du 15/06/2026 pour renouvellement ordonnance.
    """

    result = extractor.extract(sample_text, document_type="ordonnance")
    print(json.dumps(result, indent=2, ensure_ascii=False))