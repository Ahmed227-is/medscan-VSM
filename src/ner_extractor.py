"""
ner_extractor.py — MedScan VSM
Module d'extraction d'entités médicales NER, conforme au référentiel VSM
HAS/CI-SIS IPS-FR_2024.01 (Volet Synthèse médicale, ANS, 13/10/2025).

HISTORIQUE DE CONCEPTION (pour le dossier technique / soutenance) :
    - v1 : approche hybride regex + DrBERT-4GB-CP-CamemBERT — abandonnée
      (regex trop rigide, DrBERT incompatible transformers==5.9.0).
    - v2 : extraction 100% Qwen2.5-VL texte, schéma "listes plates de
      strings" par catégorie, 9 corrections post-traitement (négation,
      dates, conflits allergie/traitement, priorité inter-catégories...).
    - v3 (ce fichier) : recoupement avec le document officiel IPS-FR_2024.01
      a révélé que le VSM HAS attend, pour la plupart des sections, des
      OBJETS structurés {texte, date} et non de simples chaînes — la date
      de début est un champ obligatoire [1..1] pour Problèmes actifs,
      Antécédents, Historique des actes, Allergies. Le "bandeau
      d'horodatage global" ajouté en v2.5 (vsm_generator.py) ne
      répondait pas à cette exigence — il masquait le problème plutôt
      que de le résoudre. v3 corrige ça à la racine : Qwen produit
      directement des objets {texte, date}, catégorie par catégorie.

      Nouvelles sections ajoutées (obligatoires HAS absentes en v2) :
      historique_actes (remplace antecedents_chirurgicaux), dispositifs_medicaux,
      effets_indesirables_medicaments (distinct des allergies).
      constantes_biologiques éclatée en "constantes" (signes vitaux :
      poids, taille, TA, FC...) et "examens_bilans" (résultats de biologie/
      imagerie, qui restent des mentions non structurées par choix — voir
      note de scope ci-dessous).

    SCOPE ASSUMÉ (validé avec le porteur de projet) : on vise la
    conformité au CONTENU métier du VSM (bonnes rubriques, bon contenu,
    date rattachée à l'entité), PAS la conformité technique complète
    CDA R2/XML avec codage CIM-10/SNOMED/LOINC et OID des jeux de valeurs
    officiels — hors de portée réaliste pour un prototype, et non
    nécessaire à l'évaluation du concours (cf. règlement Art. 5 : "VSM
    conforme au référentiel de la HAS", sans exigence d'interopérabilité
    technique CDA explicite).

    RISQUE CONNU ET ASSUMÉ : le schéma de sortie demandé à Qwen est
    nettement plus riche qu'en v2 (14 catégories, dont 8 avec objets
    structurés). On a déjà observé qu'un modèle 3B suit moins bien les
    consignes à mesure que le prompt se complexifie. Les garde-fous
    post-traitement (négation, dates, priorités, conflits) restent tous
    actifs et indépendants de la qualité de suivi de consigne de Qwen —
    mais un audit qualité sur données réelles reste nécessaire après ce
    changement, avant de considérer le module comme stable.

INTERFACE : NERExtractor().extract(text, document_type) -> dict
            NERExtractor().extract_document(pages, document_type, tracker) -> dict
            NERExtractor().merge_entities(per_page_results) -> dict
            NERExtractor().extract_patient_identity(text) -> dict (regex, pas Qwen)
"""

import json
import re
import logging
import requests
from difflib import SequenceMatcher
from collections import Counter
from typing import Optional

logger = logging.getLogger("ner_extractor")

# ============================================================
# CONFIG OLLAMA
# ============================================================
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5vl:3b"
NUM_CTX_MIN = 1024
NUM_CTX_MAX = 8192
TIMEOUT = 600

FUZZY_DEDUP_THRESHOLD = 0.85

# Température : basse (fidélité) sur la 1ère tentative, plus élevée
# (diversité) sur la nouvelle tentative en cas d'échec de parsing —
# voir _request_qwen pour la justification.
BASE_TEMPERATURE = 0.1
RETRY_TEMPERATURE = 0.4

# ============================================================
# SCHÉMA DE SORTIE v3 — conforme IPS-FR_2024.01
#
# OBJECT_CATEGORIES : chaque entrée est un dict avec un champ texte
# + un champ date. STRING_CATEGORIES : chaque entrée reste une simple
# chaîne (sections où le référentiel HAS ne rend pas la date
# structurante, ou trop complexe à fiabiliser pour peu de valeur
# ajoutée à ce stade).
# ============================================================
OBJECT_CATEGORIES = {
    "pathologies_actives": "date_debut",
    "antecedents_medicaux": "date_debut",
    "antecedents_familiaux": "date_debut",
    "historique_actes": "date",
    "allergies_intolerances": "date_debut",
    "effets_indesirables_medicaments": "date_debut",
    "traitements_en_cours": "date_debut",
}

STRING_CATEGORIES = [
    "dispositifs_medicaux",
    "points_attention",
    "vaccinations",
    "facteurs_risque",
    "examens_bilans",
    "dates_importantes",
]

# "constantes" est un cas particulier d'objet (signe_vital + valeur + date)
CONSTANTES_KEY = "constantes"

OUTPUT_SCHEMA_KEYS = list(OBJECT_CATEGORIES.keys()) + [CONSTANTES_KEY] + STRING_CATEGORIES

TRAITEMENT_TYPES_VALIDES = {"long_cours", "aigu"}

# ============================================================
# FILTRE DE NÉGATION — inchangé (v2), s'applique au champ "texte"
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
# VALIDATION DES DATES — réutilisée pour valider tout champ date
# rencontré (date_debut d'une entité, "dates_importantes", etc.)
# ============================================================
MONTHS_FR = (
    r'janvier|f[ée]vrier|mars|avril|mai|juin|juillet|'
    r'ao[uû]t|septembre|octobre|novembre|d[ée]cembre'
)

DATE_VALIDATION_PATTERNS = [
    re.compile(r'\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b'),
    re.compile(rf'\b\d{{1,2}}(?:er)?\s+(?:{MONTHS_FR})\s+\d{{4}}\b', re.IGNORECASE),
    re.compile(rf'\b(?:{MONTHS_FR})\s+\d{{4}}\b', re.IGNORECASE),
]

# ============================================================
# VALIDATION DES ALERTES — inchangée, catégorie "points_attention"
# ============================================================
ALERT_MARKERS = [
    "attention", "contre-indication", "contre indication",
    "formel", "formelle", "formellement",
    "ne jamais", "jamais administrer", "a ne jamais",
    "urgent", "urgence", "danger", "dangereux", "dangereuse",
    "alerte", "a ne pas", "interdit", "interdite", "proscrit", "proscrite",
    "precaution", "vigilance", "risque vital", "mise en garde",
]

# ============================================================
# FILTRE DISPOSITIFS_MEDICAUX — post-traitement déterministe (v3.5)
# Constat sur run réel : malgré la consigne du prompt (avec
# contre-exemple explicite "gastroscope"), Qwen continue d'y ranger du
# matériel d'examen/imagerie utilisé PAR le médecin (échographe,
# scanner, IRM, mammographe, gastroscope, trocarts...) plutôt que des
# dispositifs réellement portés/implantés par le PATIENT. Comme pour
# les négations et les alertes, on ne fait plus confiance au seul
# prompt sur ce point — filtre déterministe en complément.
# ============================================================
EXAM_EQUIPMENT_TERMS = [
    "echograph", "echographie", "gastroscope", "scanner", "irm",
    "mammographi", "radiologi", "radiographi", "osteodensitometri",
    "cystoscope", "coelioscop", "trocart", "fibroscope", "endoscope",
    "colonoscope", "cathetere de biopsie", "doppler",
]

# ============================================================
# FILTRE TRAITEMENTS NON-MÉDICAMENTEUX — post-traitement déterministe
# Constat run réel : "Arrêt du tabac" classé comme "traitement en
# cours" — c'est une recommandation de mode de vie, pas un médicament.
# ============================================================
NON_MEDICATION_TREATMENT_TERMS = [
    "arret du tabac", "arret tabac", "sevrage tabagique",
    "regime alimentaire", "activite physique", "perte de poids",
    "arret de l'alcool", "arret alcool",
]

# ============================================================
# FILTRE ARTEFACTS D'IMPRIMERIE — post-traitement déterministe
# Constat run réel : "T.S.V.P." (Tournez S.V.P., mention de bas de
# page) classé comme "constante" avec une fausse valeur mesurée.
# ============================================================
CONSTANTE_ARTIFACT_TERMS = [
    "t.s.v.p", "tsvp", "tournez svp", "tournez s.v.p",
    "suite page", "voir page suivante",
]

# ============================================================
# SYSTEM PROMPT
# ============================================================
# ============================================================
# SYSTEM PROMPTS — v3.2 : DÉCOUPÉ EN 2 APPELS PAR PAGE
#
# FIX 2 (suite au Fix 1, qui a été réfuté par les tests) : un test sur
# données réelles a montré que le budget de tokens n'était PAS le
# problème (num_ctx/num_predict larges, aucun échec de parsing JSON
# loggé) — mais que Qwen, face à un schéma trop riche en un seul appel
# (14 catégories, dont 8 objets structurés), répond avec un JSON
# valide mais quasi entièrement vide sur la plupart des catégories. Un
# modèle 3B semble "abandonner" silencieusement plutôt que de mal
# répondre quand on lui demande trop de choses à la fois.
#
# Solution : DEUX appels Qwen par page, chacun avec un schéma plus
# restreint (7 catégories au lieu de 14). Coût :2x plus d'appels donc
# plus lent par page, mais chaque appel a une consigne plus simple à
# suivre — objectif : retrouver un taux de remplissage proche de la v2
# (schéma simple, listes plates) tout en gardant les objets {texte,
# date} et les nouvelles sections conformes HAS.
# ============================================================

GROUP1_KEYS = [
    "pathologies_actives", "antecedents_medicaux", "antecedents_familiaux",
    "historique_actes", "allergies_intolerances",
    "effets_indesirables_medicaments", "traitements_en_cours",
]
GROUP2_KEYS = [
    "constantes", "dispositifs_medicaux", "points_attention",
    "vaccinations", "facteurs_risque", "examens_bilans", "dates_importantes",
]

SYSTEM_PROMPT_GROUP1 = """Tu es un assistant médical spécialisé dans l'extraction d'informations \
structurées à partir de documents médicaux français (ordonnances, comptes rendus, \
courriers médecin, résultats biologiques, résultats imagerie). Le texte fourni est \
issu d'un OCR et peut contenir des imperfections.

RÈGLES :
1. Extrais UNIQUEMENT les informations explicitement présentes dans le texte. N'invente rien.
2. Si une catégorie n'a rien à extraire, retourne une liste vide.
3. Formule chaque "texte" de façon courte MAIS COMPLÈTE, avec le contexte utile trouvé
   dans le texte (résultat, dosage, posologie...).
4. Mets une date UNIQUEMENT si elle est explicitement associée à cette entité précise.
   Si aucune date claire ne s'y rattache, mets null — N'INVENTE JAMAIS une date.
5. Chaque entité va dans UNE SEULE catégorie, la plus précise possible.

DÉFINITIONS :
- pathologies_actives : problèmes de santé EN COURS actuellement. {"texte", "date_debut"}
- antecedents_medicaux : problèmes passés, guéris, PERSONNELS au patient (pas la famille).
  {"texte", "date_debut"}
- antecedents_familiaux : antécédents d'un membre de la famille (père, mère, frère/sœur...),
  JAMAIS dans antecedents_medicaux. {"texte", "date_debut", "parent"} où "parent" identifie
  le membre concerné si mentionné, sinon null.
- historique_actes : actes chirurgicaux/diagnostic invasif/thérapeutiques RÉALISÉS
  (ex: appendicectomie, exérèse de kyste, pose de plaque...). {"texte", "date"}
  ATTENTION : un DIAGNOSTIC (ex: "infection urinaire", "dysurie") n'est PAS un acte,
  même s'il est mentionné dans le même contexte — ça va dans pathologies_actives ou
  antecedents_medicaux, jamais dans historique_actes.
- allergies_intolerances : réaction NON PRÉVISIBLE, non liée à la dose. {"texte", "date_debut"}
- effets_indesirables_medicaments : effet secondaire PRÉVISIBLE, dose-dépendant, d'un
  médicament précis (différent d'une allergie). {"texte", "medicament", "date_debut"}
- traitements_en_cours : médicaments pris par le patient. {"texte", "date_debut", "type"}
  où type vaut "long_cours" (chronique/de fond) ou "aigu" (ponctuel). Défaut : "long_cours".

Réponds STRICTEMENT en JSON valide, sans texte avant ou après, selon ce schéma exact :

{
  "pathologies_actives": [{"texte": "string", "date_debut": "string ou null"}, ...],
  "antecedents_medicaux": [{"texte": "string", "date_debut": "string ou null"}, ...],
  "antecedents_familiaux": [{"texte": "string", "date_debut": "string ou null", "parent": "string ou null"}, ...],
  "historique_actes": [{"texte": "string", "date": "string ou null"}, ...],
  "allergies_intolerances": [{"texte": "string", "date_debut": "string ou null"}, ...],
  "effets_indesirables_medicaments": [{"texte": "string", "medicament": "string ou null", "date_debut": "string ou null"}, ...],
  "traitements_en_cours": [{"texte": "string", "date_debut": "string ou null", "type": "long_cours ou aigu"}, ...]
}

EXEMPLE :

Texte source :
"Mme BANANE Sophie. Ménopause subatrophique depuis octobre 2009. Antécédents familiaux :
cancer du sein chez la mère. Traitement en cours : Rivotril 20 gouttes le soir au long
cours. Cure de kyste sous-urétral en juillet 2008 (intervention chirurgicale)."

Réponse attendue :
{
  "pathologies_actives": [{"texte": "Ménopause subatrophique", "date_debut": "octobre 2009"}],
  "antecedents_medicaux": [],
  "antecedents_familiaux": [{"texte": "Cancer du sein", "date_debut": null, "parent": "mère"}],
  "historique_actes": [{"texte": "Cure de kyste sous-urétral", "date": "juillet 2008"}],
  "allergies_intolerances": [],
  "effets_indesirables_medicaments": [],
  "traitements_en_cours": [{"texte": "Rivotril 20 gouttes le soir", "date_debut": null, "type": "long_cours"}]
}
"""

SYSTEM_PROMPT_GROUP2 = """Tu es un assistant médical spécialisé dans l'extraction d'informations \
structurées à partir de documents médicaux français (ordonnances, comptes rendus, \
courriers médecin, résultats biologiques, résultats imagerie). Le texte fourni est \
issu d'un OCR et peut contenir des imperfections.

RÈGLES :
1. Extrais UNIQUEMENT les informations explicitement présentes dans le texte. N'invente rien.
2. Si une catégorie n'a rien à extraire, retourne une liste vide.
3. Formule chaque élément de façon courte MAIS COMPLÈTE, avec le contexte utile trouvé
   dans le texte (résultat, valeur, conclusion...).

DÉFINITION DE "constantes" (signes vitaux uniquement) :
- {"signe_vital", "valeur", "date"}
- Concerne UNIQUEMENT : Poids, Taille, IMC, Fréquence cardiaque, Tension artérielle,
  Température, Saturation O2 (SpO2). PAS les valeurs de biologie sanguine (créatinine,
  cholestérol, hémoglobine...) — celles-ci vont dans "examens_bilans".

CATÉGORIES (listes de chaînes) :
- dispositifs_medicaux : dispositifs implantés ou PORTÉS PAR LE PATIENT (prothèse,
  pacemaker, sonde à demeure, cathéter, fauteuil roulant...). NE PAS inclure le matériel
  d'examen utilisé PAR le médecin pour l'examiner (échographe, gastroscope, appareil
  d'IRM...) — ce n'est pas un dispositif du patient. Liste vide si aucun mentionné.
- points_attention : UNIQUEMENT les alertes explicitement signalées comme telles
  (ex: "ATTENTION", "contre-indication formelle", "à ne jamais administrer"). Liste
  vide dans la grande majorité des documents.
- vaccinations : vaccins mentionnés.
- facteurs_risque : éléments de mode de vie/hérédité (tabac, alcool, sédentarité...),
  PAS un diagnostic déjà posé.
- examens_bilans : résultats de biologie, imagerie, anatomopathologie, avec leur valeur/
  conclusion, SAUF les signes vitaux (voir "constantes" ci-dessus).
- dates_importantes : dates administratives non rattachées à une entité clinique précise
  (date de prescription du courrier, date d'enregistrement...).

Réponds STRICTEMENT en JSON valide, sans texte avant ou après, selon ce schéma exact :

{
  "constantes": [{"signe_vital": "string", "valeur": "string", "date": "string ou null"}, ...],
  "dispositifs_medicaux": ["string", ...],
  "points_attention": ["string", ...],
  "vaccinations": ["string", ...],
  "facteurs_risque": ["string", ...],
  "examens_bilans": ["string", ...],
  "dates_importantes": ["string", ...]
}

EXEMPLE :

Texte source :
"Prescrit le 09/10/2014. FROTTIS DE DEPISTAGE. Poids : 65 kg. TA : 13/7. CONCLUSION :
Frottis satisfaisant, non inflammatoire, absence de cellule suspecte. Gastroscope Olympus
XQ 30 utilisé pour l'examen."

Réponse attendue :
{
  "constantes": [{"signe_vital": "Poids", "valeur": "65 kg", "date": "09/10/2014"}, {"signe_vital": "TA", "valeur": "13/7", "date": "09/10/2014"}],
  "dispositifs_medicaux": [],
  "points_attention": [],
  "vaccinations": [],
  "facteurs_risque": [],
  "examens_bilans": ["Frottis de dépistage cervico-vaginal : satisfaisant, non inflammatoire, absence de cellule suspecte"],
  "dates_importantes": ["Prescrit le 09/10/2014"]
}

Remarque : le gastroscope n'est PAS dans dispositifs_medicaux (matériel du médecin, pas du patient).

DEUXIÈME EXEMPLE — texte OCR de type "tableau de résultats" (fréquent sur les pages de
biologie) : contrairement à l'exemple ci-dessus, ce texte n'a PAS de phrases complètes,
juste des noms de dosages, valeurs, unités et plages de référence collés les uns aux
autres. Il faut quand même extraire CHAQUE valeur individuellement.

Texte source :
"Mme BANANE Sophie dossier du 07/08/00 BIOCHIMIE VITROS Normales Anterieurs
CREATININE 8,4 mg/l 7,0a12,0 74 umol/l 62a106 IONOGRAMME SODIUM 140 mEq/l 135a145
POTASSIUM 4,0 mEq/l 3,5a5,0 CHLORE 105 mEq/l 95a108"

Réponse attendue :
{
  "constantes": [],
  "dispositifs_medicaux": [],
  "points_attention": [],
  "vaccinations": [],
  "facteurs_risque": [],
  "examens_bilans": ["Créatinine : 8,4 mg/l", "Sodium : 140 mEq/l", "Potassium : 4,0 mEq/l", "Chlore : 105 mEq/l"],
  "dates_importantes": ["Dossier du 07/08/00"]
}

Remarque : chaque dosage (nom + valeur + unité) devient une entrée séparée dans
examens_bilans, même sans phrase complète autour. Les plages de référence ("7,0a12,0")
ne sont pas extraites, seulement la valeur mesurée du patient.
"""


class NERExtractor:
    """
    Extraction d'entités médicales via Qwen2.5-VL (mode texte, Ollama),
    schéma conforme HAS/CI-SIS IPS-FR_2024.01 (objets {texte, date}).
    """

    def __init__(self, ollama_url: str = OLLAMA_URL, model: str = MODEL_NAME,
                 num_ctx_max: int = NUM_CTX_MAX, timeout: int = TIMEOUT):
        self.ollama_url = ollama_url
        self.model = model
        self.num_ctx_max = num_ctx_max
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------
    def _normalize_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('–', '-').replace('—', '-')
        text = text.replace('œ', 'oe').replace('æ', 'ae')
        return text.strip()

    # ------------------------------------------------------------------
    # Identité patient — regex déterministe, indépendant de Qwen
    # ------------------------------------------------------------------
    def extract_patient_identity(self, text: str) -> dict:
        identity = {"nom_complet": None, "date_naissance": None, "medecin_traitant": None}

        name_match = re.search(
            r'\b(?:Mme|M\.|Madame|Monsieur)\s+'
            r'([A-ZÀ-Ÿ][A-ZÀ-Ÿ\-]{1,}(?:\s+[A-ZÀ-Ÿ][a-zà-ÿ\-]+){0,1})',
            text
        )
        if name_match:
            identity["nom_complet"] = name_match.group(1).strip()

        dob_match = re.search(
            r"(?:n[ée]\(?e?\)?\s+le|date\s+de\s+naissance\s*[:\-]?)\s*"
            r"(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})",
            text, re.IGNORECASE
        )
        if dob_match:
            day, month, year = re.split(r'[/\-.]', dob_match.group(1))
            if 1 <= int(day) <= 31 and 1 <= int(month) <= 12:
                identity["date_naissance"] = dob_match.group(1)

        gp_match = re.search(
            r'm[ée]decin\s+traitant\s*(?:d[ée]clar[ée])?\s*[:\-]?\s*'
            r'(Dr\.?\s*[A-ZÀ-Ÿ][a-zà-ÿA-ZÀ-Ÿ\-]+)',
            text, re.IGNORECASE
        )
        if gp_match:
            identity["medecin_traitant"] = gp_match.group(1).strip()

        return identity

    def merge_patient_identity(self, per_page_identities: list[dict]) -> dict:
        merged = {}
        for field in ("nom_complet", "date_naissance", "medecin_traitant"):
            values = [p[field] for p in per_page_identities if p.get(field)]
            merged[field] = Counter(values).most_common(1)[0][0] if values else None
        return merged

    # ------------------------------------------------------------------
    # Point d'entrée principal
    # ------------------------------------------------------------------
    def extract(self, text: str, document_type: str = "inconnu",
                page_number: Optional[int] = None) -> dict:
        if not text or len(text.strip()) < 10:
            return self._empty_result("Texte vide ou trop court")

        text = self._normalize_text(text)
        entities = self._call_qwen(text)

        result = {key: entities.get(key, []) for key in OUTPUT_SCHEMA_KEYS}
        result["document_type"] = document_type
        result["extraction_method"] = "qwen_only"
        result["identite_patient"] = self.extract_patient_identity(text)

        self._attach_page_source(result, page_number)
        return result

    def _attach_page_source(self, result: dict, page_number: Optional[int]) -> None:
        """
        Traçabilité (v3.6) : chaque entité garde en mémoire de quelle
        page du PDF original elle provient — indispensable pour qu'un
        médecin qui ne connaît pas le patient puisse vérifier une ligne
        du VSM en 2 secondes plutôt que de relire tout le dossier (voir
        discussion projet). Uniformise au passage toutes les catégories
        en objets — les catégories "string" (examens_bilans, dates...)
        deviennent {"texte": str, "pages_sources": [...]}.

        Qwen n'est jamais informé de la page — c'est une info que SEUL
        le code appelant (test_pipeline.py, qui boucle sur les pages)
        connaît ; on l'attache ici en pur post-traitement.
        """
        sources = [page_number] if page_number is not None else []

        for key in OUTPUT_SCHEMA_KEYS:
            entries = result.get(key, [])
            wrapped = []
            for entry in entries:
                if isinstance(entry, dict):
                    entry["pages_sources"] = list(sources)
                    wrapped.append(entry)
                else:
                    wrapped.append({"texte": str(entry), "pages_sources": list(sources)})
            result[key] = wrapped

    def _empty_result(self, reason: str) -> dict:
        result = {key: [] for key in OUTPUT_SCHEMA_KEYS}
        result["document_type"] = "inconnu"
        result["extraction_method"] = "none"
        result["identite_patient"] = {"nom_complet": None, "date_naissance": None, "medecin_traitant": None}
        result["reason"] = reason
        return result

    # ------------------------------------------------------------------
    # num_ctx dynamique + num_predict explicite
    #
    # v3.1 — FIX 1 : le schéma v3 (objets {texte, date} au lieu de
    # simples chaînes) coûte 2-3x plus de tokens par entité qu'en v2.
    # Une marge de réponse FIXE (768 tokens en v3.0) s'est révélée
    # insuffisante sur les pages à forte densité d'entités (ex: page de
    # résultats de biologie avec 15 valeurs) — Qwen tronque/abandonne
    # silencieusement les catégories les plus coûteuses. On rend cette
    # marge PROPORTIONNELLE à la taille du texte source (plus de texte
    # source ~ plus d'entités probables ~ plus de tokens de sortie
    # nécessaires), et on fixe explicitement num_predict (jamais fait
    # avant — Ollama peut appliquer une limite de génération par défaut
    # indépendante de num_ctx si on ne la précise pas).
    # ------------------------------------------------------------------
    def _compute_num_ctx(self, text: str, system_prompt: str) -> int:
        estimated_tokens = len(text) // 4
        prompt_overhead = len(system_prompt) // 4
        response_margin = self._estimate_response_tokens(estimated_tokens)

        needed = estimated_tokens + prompt_overhead + response_margin
        needed = ((needed // 512) + 1) * 512

        return max(NUM_CTX_MIN, min(needed, self.num_ctx_max))

    def _estimate_response_tokens(self, input_tokens: int) -> int:
        """
        Marge de réponse proportionnelle au texte source, avec un
        plancher confortable. NOTE (v3.2) : un test sur données réelles
        a montré que ce n'était PAS le facteur limitant principal (voir
        Fix 2 / découpage en 2 appels ci-dessous) — mais on garde cette
        marge généreuse par prudence, en complément du vrai fix.
        """
        return max(1536, input_tokens * 2)

    def _compute_num_predict(self, num_ctx: int, prompt_overhead: int) -> int:
        available = num_ctx - prompt_overhead
        return max(1024, min(available, 4096))

    # ------------------------------------------------------------------
    # Appel Qwen + parsing — v3.2 : DEUX appels par page (voir note en
    # tête de fichier sur SYSTEM_PROMPT_GROUP1/GROUP2). Chaque appel
    # retourne un dict PARTIEL non filtré ; on les fusionne (les deux
    # groupes ont des clés disjointes) avant d'appliquer _sanitize()
    # UNE SEULE FOIS sur l'ensemble — tous les filtres/priorités
    # inter-catégories (ex: allergie vs traitement) ont besoin de voir
    # les deux groupes en même temps pour fonctionner correctement.
    # ------------------------------------------------------------------
    def _call_qwen(self, text: str) -> dict:
        raw_group1 = self._call_qwen_group(text, SYSTEM_PROMPT_GROUP1, GROUP1_KEYS)
        raw_group2 = self._call_qwen_group(text, SYSTEM_PROMPT_GROUP2, GROUP2_KEYS)

        merged_raw = {**raw_group1, **raw_group2}
        return self._sanitize(merged_raw)

    def _call_qwen_group(self, text: str, system_prompt: str, expected_keys: list,
                          max_attempts: int = 2) -> dict:
        """
        Un appel Qwen pour UN groupe de catégories, retourne un dict brut
        (non sanitizé). RETRY (v3.3) : si le JSON produit est mal formé,
        on retente une seconde fois avant d'abandonner — un nouvel appel
        avec la même température non nulle (0.1) donne souvent un
        résultat différent, potentiellement valide cette fois. On ne
        retente PAS si Qwen a répondu un JSON valide mais légitimement
        vide (ce n'est pas un échec, juste une page sans contenu pour ce
        groupe) — seul un échec de PARSING déclenche une nouvelle tentative.
        """
        empty = {k: [] for k in expected_keys}
        last_result = empty

        for attempt in range(1, max_attempts + 1):
            content = self._request_qwen(text, system_prompt, expected_keys, attempt)
            if content is None:
                return empty  # erreur réseau/Ollama, inutile de retenter

            parsed, success = self._parse_json_raw(content, expected_keys)
            if success:
                if attempt > 1:
                    logger.info(f"Tentative {attempt}/{max_attempts} : JSON valide obtenu "
                                f"après échec de la première tentative.")
                return parsed

            last_result = parsed
            if attempt < max_attempts:
                logger.warning(f"Tentative {attempt}/{max_attempts} : JSON invalide, "
                                f"nouvelle tentative...")

        logger.warning(f"Échec après {max_attempts} tentatives — résultat vide retourné pour ce groupe.")
        return last_result

    def _request_qwen(self, text: str, system_prompt: str, expected_keys: list,
                       attempt: int) -> Optional[str]:
        """
        Effectue l'appel HTTP à Ollama, retourne le contenu brut (str) ou
        None en cas d'erreur réseau.

        Température variable selon la tentative (v3.4) : à température
        basse (0.1, quasi déterministe), une nouvelle tentative après
        échec de parsing régénère souvent la MÊME erreur — retenter avec
        les mêmes paramètres ne sert à rien. On monte la température sur
        les tentatives suivantes pour donner une vraie chance d'obtenir
        une réponse différente, potentiellement valide.
        """
        num_ctx = self._compute_num_ctx(text, system_prompt)
        prompt_overhead = (len(system_prompt) + len(text)) // 4
        num_predict = self._compute_num_predict(num_ctx, prompt_overhead)
        temperature = BASE_TEMPERATURE if attempt == 1 else RETRY_TEMPERATURE

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Texte médical à analyser :\n\n{text}"},
            ],
            "stream": False,
            "format": "json",
            "options": {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": temperature},
        }

        logger.info(f"Appel Qwen NER (groupe {list(expected_keys)[:2]}..., "
                    f"tentative {attempt}, temperature={temperature}) — "
                    f"texte={len(text)} caractères, num_ctx={num_ctx}, num_predict={num_predict}")

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur appel Qwen (NER, groupe) : {e}")
            return None

        try:
            return response.json()["message"]["content"]
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Réponse Ollama inattendue (NER, groupe) : {e}")
            return None

    def _parse_json_raw(self, content: str, expected_keys: list) -> tuple:
        """
        Parse le JSON brut d'UN groupe, SANS appliquer _sanitize (fait
        une seule fois après fusion des deux groupes, dans _call_qwen).
        Retourne (dict, succès) — succès=False uniquement en cas
        d'échec de PARSING (pour déclencher le retry), pas quand le
        JSON est valide mais légitimement vide.
        """
        empty = {k: [] for k in expected_keys}

        try:
            return json.loads(content), True
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0)), True
            except json.JSONDecodeError:
                pass

        logger.debug(f"Réponse brute Qwen (groupe) non parsable : {content[:500]}")
        return empty, False

    # ------------------------------------------------------------------
    # Sanitize — dispatch par type de catégorie (objet / constante / string)
    # ------------------------------------------------------------------
    def _sanitize(self, parsed: dict) -> dict:
        clean = {}

        for key, date_field in OBJECT_CATEGORIES.items():
            raw_list = parsed.get(key, [])
            clean[key] = self._sanitize_object_list(raw_list, date_field, key)

        clean[CONSTANTES_KEY] = self._sanitize_constantes(parsed.get(CONSTANTES_KEY, []))

        for key in STRING_CATEGORIES:
            raw_list = parsed.get(key, [])
            entries = self._sanitize_string_list(raw_list)
            entries = [e for e in entries if not self._is_negation(e)]

            if key == "dates_importantes":
                entries = [e for e in entries if self._contains_valid_date(e)]
                # Fix v3.7 : une "date importante" est une mention courte
                # (ex: "Enregistré le 09/10/2014"), jamais un paragraphe
                # entier. Constaté sur run réel : Qwen a recopié le contenu
                # intégral d'une page (plusieurs centaines de mots) parce
                # qu'une date y était mentionnée quelque part — le filtre
                # de validation ci-dessus ne vérifie que la PRÉSENCE d'une
                # date, pas que l'entrée EST une date. Même plafond que
                # points_attention (200 caractères), même raison.
                entries = [e for e in entries if len(e) <= 200]
            if key == "points_attention":
                entries = [e for e in entries if self._contains_alert_marker(e)]
                entries = [e for e in entries if len(e) <= 200]
            if key == "dispositifs_medicaux":
                entries = [e for e in entries if not self._is_exam_equipment(e)]

            clean[key] = entries

        clean = self._resolve_allergy_conflicts(clean)
        clean = self._resolve_attention_conflicts(clean)
        clean = self._resolve_category_priority(clean)

        return clean

    def _sanitize_object_list(self, raw_list: list, date_field: str, category: str) -> list:
        """
        Nettoie une liste d'objets {texte, <date_field>, [medicament], [type]}.
        Accepte aussi, en filet de sécurité, une liste de simples chaînes
        (si Qwen ignore la consigne d'objet malgré le prompt) en les
        convertissant en {texte: str, date_field: None}.
        """
        if not isinstance(raw_list, list):
            return []

        clean = []
        for item in raw_list:
            if isinstance(item, str):
                texte = item.strip()
                entry = {"texte": texte, date_field: None}
            elif isinstance(item, dict):
                texte = str(item.get("texte", "")).strip()
                if not texte:
                    continue
                entry = {"texte": texte}
                raw_date = item.get(date_field)
                entry[date_field] = self._validate_date_field(raw_date)

                if category == "effets_indesirables_medicaments":
                    medicament = item.get("medicament")
                    entry["medicament"] = str(medicament).strip() if medicament else None

                if category == "antecedents_familiaux":
                    parent = item.get("parent")
                    entry["parent"] = str(parent).strip() if parent else None

                if category == "traitements_en_cours":
                    t_type = str(item.get("type", "")).strip().lower().replace(" ", "_")
                    entry["type"] = t_type if t_type in TRAITEMENT_TYPES_VALIDES else "long_cours"
            else:
                continue

            if not entry["texte"] or self._is_negation(entry["texte"]):
                continue

            if category == "traitements_en_cours" and self._is_non_medication_treatment(entry["texte"]):
                continue

            clean.append(entry)

        return clean

    def _sanitize_constantes(self, raw_list: list) -> list:
        if not isinstance(raw_list, list):
            return []

        clean = []
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            signe_vital = str(item.get("signe_vital", "")).strip()
            valeur = str(item.get("valeur", "")).strip()
            if not signe_vital or not valeur:
                continue
            if self._is_print_artifact(signe_vital):
                continue
            date = self._validate_date_field(item.get("date"))
            clean.append({"signe_vital": signe_vital, "valeur": valeur, "date": date})

        return clean

    def _sanitize_string_list(self, raw_list: list) -> list:
        """
        Nettoie une liste de chaînes. Filet de sécurité (v3.5) : si Qwen
        renvoie un OBJET au lieu d'une chaîne dans une catégorie "string"
        (constaté en pratique : {'nom': 'Mammographie...', 'valeur': '...',
        'date': '...'} au lieu d'une phrase), on le reformate en texte
        lisible plutôt que d'afficher le dict Python brut tel quel.
        """
        if not isinstance(raw_list, list):
            return []

        result = []
        for v in raw_list:
            text = self._dict_to_readable_string(v) if isinstance(v, dict) else str(v).strip()
            if text:
                result.append(text)
        return result

    def _dict_to_readable_string(self, d: dict) -> str:
        """Reformate un objet inattendu en phrase lisible, en essayant les
        clés courantes dans un ordre logique avant de tout concaténer."""
        parts = []
        for key in ('nom', 'texte', 'signe_vital', 'valeur', 'resultat', 'date'):
            value = d.get(key)
            if value:
                parts.append(str(value).strip())

        if not parts:
            parts = [str(v).strip() for v in d.values() if v]

        if not parts:
            return ""
        if len(parts) <= 2:
            return " : ".join(parts)
        return f"{parts[0]} : {', '.join(parts[1:])}"

    def _validate_date_field(self, raw_date) -> Optional[str]:
        """
        Valide un champ date individuel (pas une catégorie entière comme
        en v2). Retourne la date si elle contient un format reconnaissable
        et plausible, sinon None — jamais on ne rejette l'entité entière
        pour une date invalide, on met juste le champ à None.
        """
        if not raw_date or not isinstance(raw_date, str):
            return None
        raw_date = raw_date.strip()
        if not raw_date or raw_date.lower() in ("null", "none", "n/a"):
            return None
        return raw_date if self._contains_valid_date(raw_date) else None

    # ------------------------------------------------------------------
    # Négation / dates / alertes — logique de détection inchangée (v2)
    # ------------------------------------------------------------------
    def _is_negation(self, value: str) -> bool:
        normalized = value.lower().strip().strip(".:-").strip()
        normalized = (normalized
                      .replace('é', 'e').replace('è', 'e').replace('ê', 'e')
                      .replace('à', 'a').replace('ù', 'u'))
        return normalized in NEGATION_TERMS

    def _contains_valid_date(self, value: str) -> bool:
        for pattern in DATE_VALIDATION_PATTERNS:
            for match in pattern.finditer(value):
                groups = match.groups()
                if len(groups) == 3:
                    day, month, _year = groups
                    if 1 <= int(day) <= 31 and 1 <= int(month) <= 12:
                        return True
                    continue
                else:
                    return True
        return False

    def _contains_alert_marker(self, value: str) -> bool:
        normalized = value.lower()
        normalized = (normalized
                      .replace('é', 'e').replace('è', 'e').replace('ê', 'e')
                      .replace('à', 'a').replace('ù', 'u'))
        return any(marker in normalized for marker in ALERT_MARKERS)

    def _is_exam_equipment(self, value: str) -> bool:
        """
        Détecte si une entrée de dispositifs_medicaux est en fait du
        matériel d'examen/imagerie (échographe, scanner, gastroscope...)
        plutôt qu'un dispositif porté/implanté par le patient. Filtre
        déterministe, indépendant de la consigne du prompt.
        """
        normalized = value.lower()
        normalized = (normalized
                      .replace('é', 'e').replace('è', 'e').replace('ê', 'e')
                      .replace('à', 'a').replace('ù', 'u'))
        return any(term in normalized for term in EXAM_EQUIPMENT_TERMS)

    def _is_non_medication_treatment(self, value: str) -> bool:
        """Détecte une recommandation de mode de vie classée à tort comme
        traitement médicamenteux (ex: "Arrêt du tabac")."""
        normalized = value.lower()
        normalized = (normalized
                      .replace('é', 'e').replace('è', 'e').replace('ê', 'e')
                      .replace('à', 'a').replace('ù', 'u'))
        return any(term in normalized for term in NON_MEDICATION_TREATMENT_TERMS)

    def _is_print_artifact(self, value: str) -> bool:
        """Détecte une mention d'imprimerie/mise en page (ex: "T.S.V.P.")
        classée à tort comme une constante médicale mesurée."""
        normalized = value.lower()
        normalized = (normalized
                      .replace('é', 'e').replace('è', 'e').replace('ê', 'e')
                      .replace('à', 'a').replace('ù', 'u'))
        return any(term in normalized for term in CONSTANTE_ARTIFACT_TERMS)

    # ------------------------------------------------------------------
    # Helper générique : texte affichable d'une entrée, quelle que soit
    # sa forme (objet avec "texte", objet "constante" avec signe_vital/
    # valeur, ou simple chaîne) — utilisé par tous les filtres croisés.
    # ------------------------------------------------------------------
    def _display_text(self, entry) -> str:
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            if "texte" in entry:
                return entry["texte"]
            if "signe_vital" in entry:
                return f"{entry.get('signe_vital', '')} {entry.get('valeur', '')}".strip()
        return str(entry)

    def _leading_word(self, text: str) -> str:
        words = re.findall(r"[A-Za-zÀ-ÿ]+", text)
        return words[0].lower() if words else ""

    # ------------------------------------------------------------------
    # Conflit allergies / traitements (sécurité) — adapté aux objets
    # ------------------------------------------------------------------
    def _resolve_allergy_conflicts(self, result: dict) -> dict:
        treatments = result.get("traitements_en_cours", [])
        allergies = result.get("allergies_intolerances", [])

        kept_allergies = []
        for allergy in allergies:
            allergy_text = self._display_text(allergy).lower().strip()
            conflict = False
            for treatment in treatments:
                treatment_text = self._display_text(treatment).lower().strip()
                similarity = SequenceMatcher(None, allergy_text, treatment_text, autojunk=False).ratio()
                is_substring = (
                    len(allergy_text) >= 4
                    and (allergy_text in treatment_text or treatment_text in allergy_text)
                )
                if similarity >= FUZZY_DEDUP_THRESHOLD or is_substring:
                    conflict = True
                    logger.warning(
                        f"Conflit allergie/traitement détecté et résolu : "
                        f"\"{self._display_text(allergy)}\" retiré des allergies "
                        f"(présent aussi en traitement : \"{self._display_text(treatment)}\")."
                    )
                    break
            if not conflict:
                kept_allergies.append(allergy)

        result["allergies_intolerances"] = kept_allergies
        return result

    # ------------------------------------------------------------------
    # Conflit points_attention / traitements — garde le traitement,
    # retire seulement l'alerte suspecte (cf. décision projet)
    # ------------------------------------------------------------------
    def _resolve_attention_conflicts(self, result: dict) -> dict:
        treatments = result.get("traitements_en_cours", [])
        attentions = result.get("points_attention", [])

        treatment_names = [self._leading_word(self._display_text(t)) for t in treatments]
        kept_attentions = []

        for attention in attentions:
            attention_text = self._display_text(attention)
            attention_name = self._leading_word(attention_text.split(':')[0])
            has_conflict = (
                attention_name and len(attention_name) >= 4
                and attention_name in treatment_names
            )
            if has_conflict:
                logger.warning(
                    f"Contradiction détectée entre points_attention et "
                    f"traitements_en_cours pour \"{attention_name}\" : "
                    f"\"{attention_text}\" retiré des alertes (traitement jugé plus "
                    f"fiable, répété dans le dossier). Révision manuelle recommandée."
                )
            else:
                kept_attentions.append(attention)

        result["points_attention"] = kept_attentions
        return result

    # ------------------------------------------------------------------
    # Priorité inter-catégories — adapté aux objets + historique_actes
    # remplace antecedents_chirurgicaux comme catégorie "chirurgicale"
    # ------------------------------------------------------------------
    def _remove_matches(self, source_list: list, reference_lists: list, reason: str) -> list:
        kept = []
        for item in source_list:
            item_text = self._display_text(item).lower().strip()
            matched_ref = None

            for ref_list in reference_lists:
                for ref in ref_list:
                    ref_text = self._display_text(ref).lower().strip()
                    similarity = SequenceMatcher(None, item_text, ref_text, autojunk=False).ratio()
                    is_substring = (
                        len(item_text) >= 6 and len(ref_text) >= 6
                        and (item_text in ref_text or ref_text in item_text)
                    )
                    if similarity >= FUZZY_DEDUP_THRESHOLD or is_substring:
                        matched_ref = ref_text
                        break
                if matched_ref:
                    break

            if matched_ref:
                logger.info(f"{reason} : \"{item_text}\" retiré (déjà présent sous "
                            f"forme \"{matched_ref}\" dans une catégorie prioritaire).")
            else:
                kept.append(item)

        return kept

    def _resolve_category_priority(self, result: dict) -> dict:
        """
        Règles :
        1. historique_actes (acte réalisé) > antecedents_medicaux générique
        2. pathologie active / antécédent (médical ou acte) > facteur de risque
        3. constantes (signes vitaux structurés) > examens_bilans générique
        """
        result["antecedents_medicaux"] = self._remove_matches(
            result.get("antecedents_medicaux", []),
            [result.get("historique_actes", [])],
            reason="Priorité historique_actes > antécédent médical",
        )

        result["facteurs_risque"] = self._remove_matches(
            result.get("facteurs_risque", []),
            [
                result.get("pathologies_actives", []),
                result.get("antecedents_medicaux", []),
                result.get("historique_actes", []),
            ],
            reason="Priorité pathologie/antécédent > facteur de risque",
        )

        result["examens_bilans"] = self._remove_matches(
            result.get("examens_bilans", []),
            [result.get(CONSTANTES_KEY, [])],
            reason="Priorité constante > examen générique",
        )

        return result

    # ------------------------------------------------------------------
    # Extraction document entier + fusion
    # ------------------------------------------------------------------
    def extract_document(self, pages: list[str], document_type: str = "inconnu",
                          tracker=None) -> dict:
        per_page_results = []

        for i, page_text in enumerate(pages, start=1):
            if tracker:
                tracker.update(step="ner_extraction", page=i, total=len(pages))

            page_result = self.extract(page_text, document_type=document_type, page_number=i)
            per_page_results.append(page_result)

            nb_entities = sum(len(page_result[k]) for k in OUTPUT_SCHEMA_KEYS)
            logger.info(f"Page {i}/{len(pages)} : NER terminé ({nb_entities} entités).")

        merged = self.merge_entities(per_page_results)
        merged["document_type"] = document_type
        merged["extraction_method"] = "qwen_only"
        merged["identite_patient"] = self.merge_patient_identity(
            [p["identite_patient"] for p in per_page_results]
        )
        return merged

    def merge_entities(self, per_page_results: list[dict]) -> dict:
        merged = {key: [] for key in OUTPUT_SCHEMA_KEYS}

        for page_result in per_page_results:
            for key in OUTPUT_SCHEMA_KEYS:
                for value in page_result.get(key, []):
                    self._add_with_dedup(merged[key], value, key)

        merged = self._resolve_allergy_conflicts(merged)
        merged = self._resolve_attention_conflicts(merged)
        merged = self._resolve_category_priority(merged)

        return merged

    def _add_with_dedup(self, existing_list: list, new_value, category: str) -> None:
        """
        Dédoublonnage fuzzy générique (texte + éventuellement objet).
        En cas de doublon : garde l'entrée avec un texte plus détaillé,
        ET si l'une a une date renseignée et l'autre non, garde celle
        AVEC la date (ne perd jamais une date au profit d'un texte
        légèrement plus long sans date). Fusionne aussi les
        "pages_sources" des deux entrées (union) — une entité confirmée
        sur plusieurs pages garde la trace de TOUTES ses pages sources,
        pas juste une seule (v3.6, traçabilité).
        """
        new_text = self._display_text(new_value)
        new_norm = new_text.lower().strip()

        for i, existing_value in enumerate(existing_list):
            existing_text = self._display_text(existing_value)
            existing_norm = existing_text.lower().strip()

            similarity = SequenceMatcher(None, existing_norm, new_norm, autojunk=False).ratio()
            is_substring = (
                len(existing_norm) >= 6 and len(new_norm) >= 6
                and (existing_norm in new_norm or new_norm in existing_norm)
            )

            if similarity >= FUZZY_DEDUP_THRESHOLD or is_substring:
                winner = new_value if self._is_better_entry(new_value, existing_value, category) else existing_value
                merged_sources = self._merge_page_sources(new_value, existing_value)
                if isinstance(winner, dict):
                    winner["pages_sources"] = merged_sources
                existing_list[i] = winner
                return

        existing_list.append(new_value)

    def _merge_page_sources(self, a, b) -> list:
        """Union triée et dédupliquée des pages_sources de deux entrées."""
        sources_a = a.get("pages_sources", []) if isinstance(a, dict) else []
        sources_b = b.get("pages_sources", []) if isinstance(b, dict) else []
        return sorted(set(sources_a) | set(sources_b))

    def _is_better_entry(self, candidate, existing, category: str) -> bool:
        """Détermine si `candidate` doit remplacer `existing` lors d'une fusion."""
        if isinstance(candidate, str) or isinstance(existing, str):
            return len(self._display_text(candidate)) > len(self._display_text(existing))

        date_field = OBJECT_CATEGORIES.get(category) or ("date" if category == CONSTANTES_KEY else None)
        if date_field:
            candidate_has_date = bool(candidate.get(date_field))
            existing_has_date = bool(existing.get(date_field))
            if candidate_has_date and not existing_has_date:
                return True
            if existing_has_date and not candidate_has_date:
                return False

        return len(self._display_text(candidate)) > len(self._display_text(existing))


# ----------------------------------------------------------------------
# Exemple d'utilisation
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    extractor = NERExtractor()

    sample_text = """
    Mme BANANE Sophie, née le 17/09/1962. Poids : 65 kg. TA : 13/7.
    Antécédents : ménopause subatrophique depuis octobre 2009.
    Cure de kyste sous-urétral en juillet 2008.
    Traitement actuel : Rivotril 20 gouttes le soir au long cours.
    """

    result = extractor.extract(sample_text, document_type="courrier")
    print(json.dumps(result, indent=2, ensure_ascii=False))