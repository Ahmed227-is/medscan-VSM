"""
test_ner_sample.py — MedScan VSM
Test rapide du NER v3 (avec identite_patient, antecedents_familiaux,
points_attention) sur un échantillon de pages, SANS refaire l'OCR.

Réutilise le JSON déjà généré par un run complet précédent
(data/processed/{stem}_ocr_result.json), qui contient déjà le
full_text de chaque page. On sélectionne automatiquement quelques
pages intéressantes pour ce test :
    - une page contenant une mention familiale (test antecedents_familiaux)
    - une page contenant l'identité patient (test identite_patient)
    - une page "simple" au hasard (test de non-régression générale)

Objectif : valider en 2-3 minutes que les nouveaux ajouts n'ont pas
dégradé la qualité, avant de relancer un run complet de 114 pages
(qui prend ~2h50 au rythme actuel).
"""

import sys
import json
from pathlib import Path

sys.path.append('/home/ahmed/medscan_vsm')

from src.ner_extractor import NERExtractor

FAMILY_KEYWORDS = ['famili', 'mère', 'pere', 'père', 'frère', 'soeur', 'sœur']
ATTENTION_KEYWORDS = ['attention', 'contre-indication', 'ne jamais', 'formelle']
IDENTITY_KEYWORDS = ['mme', 'madame', 'monsieur', ' m. ']
DENSE_BIO_KEYWORDS = ['mmol/l', 'mg/dl', 'g/dl', 'ui/l', 'meq/l', 'biochimie']

# Pages à retester explicitement (échecs constatés lors des runs précédents),
# indépendamment de la sélection automatique par mots-clés.
FORCED_PAGE_NUMBERS = {
    15: "retest page 15 (échec de parsing JSON constaté précédemment)",
}


def select_sample_pages(pages: list, max_pages: int = 5) -> list:
    """Sélectionne automatiquement des pages représentatives pour le test."""
    selected = []
    used_indices = set()

    # Priorité absolue : pages forcées (échecs connus à revalider)
    for i, page in enumerate(pages):
        page_num = page.get('page_number', i + 1)
        if page_num in FORCED_PAGE_NUMBERS:
            selected.append((i, FORCED_PAGE_NUMBERS[page_num]))
            used_indices.add(i)

    def find_page(keywords):
        for i, page in enumerate(pages):
            if i in used_indices:
                continue
            text_lower = page['full_text'].lower()
            if any(kw in text_lower for kw in keywords):
                return i
        return None

    def find_densest_bio_page():
        """Page avec le plus grand nombre d'occurrences d'unités biologiques —
        cible directement le cas qui a posé problème (Fix 1, budget de réponse)."""
        best_idx, best_score = None, 0
        for i, page in enumerate(pages):
            if i in used_indices:
                continue
            text_lower = page['full_text'].lower()
            score = sum(text_lower.count(kw) for kw in DENSE_BIO_KEYWORDS)
            if score > best_score:
                best_idx, best_score = i, score
        return best_idx

    for keywords, label in [
        (FAMILY_KEYWORDS, "antécédents familiaux"),
        (IDENTITY_KEYWORDS, "identité patient"),
    ]:
        idx = find_page(keywords)
        if idx is not None:
            used_indices.add(idx)
            selected.append((idx, label))

    dense_idx = find_densest_bio_page()
    if dense_idx is not None:
        used_indices.add(dense_idx)
        selected.append((dense_idx, "page dense en biologie (test Fix 1/3)"))

    # Complète avec une page "simple" (première page non déjà sélectionnée)
    for i, page in enumerate(pages):
        if len(selected) >= max_pages:
            break
        if i not in used_indices:
            selected.append((i, "page générique (non-régression)"))
            used_indices.add(i)

    return selected[:max_pages]


def test_ner_sample(json_path: str):
    print("=" * 60)
    print("TEST NER RAPIDE — échantillon de pages (sans refaire l'OCR)")
    print("=" * 60)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pages = data['pages']
    print(f"\nDocument : {data.get('source', json_path)}")
    print(f"Total pages disponibles : {len(pages)}")

    sample = select_sample_pages(pages, max_pages=5)

    if not sample:
        print("⚠️ Aucune page trouvée dans le JSON.")
        return

    ner = NERExtractor()

    for idx, label in sample:
        page = pages[idx]
        page_num = page.get('page_number', idx + 1)
        print(f"\n{'-' * 60}")
        print(f"Page {page_num} — sélectionnée pour : {label}")
        print(f"{'-' * 60}")
        print(f"Extrait du texte : {page['full_text'][:150].strip()}...")
        print(f"Longueur du texte : {len(page['full_text'])} caractères")

        result = ner.extract(
            page['full_text'],
            document_type=page.get('classification', {}).get('predicted_type', 'inconnu')
        )

        print(f"✓ Pathologies         : {result['pathologies_actives']}")
        print(f"✓ Antéc. médicaux     : {result['antecedents_medicaux']}")
        print(f"✓ Antéc. familiaux    : {result['antecedents_familiaux']}")
        print(f"✓ Historique actes    : {result['historique_actes']}")
        print(f"✓ Allergies           : {result['allergies_intolerances']}")
        print(f"✓ Effets indésirables : {result['effets_indesirables_medicaments']}")
        print(f"✓ Traitements         : {result['traitements_en_cours']}")
        print(f"✓ Constantes          : {result['constantes']}")
        print(f"✓ Dispositifs médic.  : {result['dispositifs_medicaux']}")
        print(f"✓ Facteurs risque     : {result['facteurs_risque']}")
        print(f"✓ Vaccinations        : {result['vaccinations']}")
        print(f"✓ Examens/bilans      : {result['examens_bilans']}")
        print(f"✓ Points attention    : {result['points_attention']}")
        print(f"✓ Dates               : {result['dates_importantes']}")
        print(f"✓ Identité détectée   : {result['identite_patient']}")

        nb_entites = sum(len(result[k]) for k in [
            'pathologies_actives', 'antecedents_medicaux', 'antecedents_familiaux',
            'historique_actes', 'allergies_intolerances', 'effets_indesirables_medicaments',
            'traitements_en_cours', 'constantes', 'dispositifs_medicaux',
            'facteurs_risque', 'vaccinations', 'examens_bilans',
            'points_attention', 'dates_importantes',
        ])
        print(f"→ Total entités extraites sur cette page : {nb_entites}")

    print(f"\n{'=' * 60}")
    print("TEST TERMINÉ")
    print("Vérifie en particulier :")
    print("  - la page 'dense en biologie' : les valeurs (Créatinine,")
    print("    Sodium...) doivent apparaître dans Examens/bilans")
    print("  - la page 15 (retest) : regarde les logs ci-dessus pour")
    print("    'Tentative 2/2' (retry déclenché) et si le résultat final")
    print("    contient des entités malgré l'échec initial")
    print("=" * 60)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    json_path = "data/processed/BANANE_Sophie_ocr_result.json"
    test_ner_sample(json_path)