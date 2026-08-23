"""
vsm_generator.py — MedScan VSM
Génère un Volet de Synthèse Médicale (VSM) structuré, conforme au
référentiel HAS/CI-SIS IPS-FR_2024.01, à partir du JSON produit par
ner_extractor.py v3 (schéma objet {texte, date} + identite_patient).

CHANGEMENTS vs v2 (voir discussion projet) :
    - Compatible avec le nouveau schéma NER v3 : la plupart des catégories
      sont maintenant des listes d'OBJETS {texte, date}, plus de simples
      chaînes. La date est rattachée à chaque entité individuellement —
      c'est le vrai fix du "problème de date", pas un bandeau global.
    - Nouvelles sections : Historique des actes, Dispositifs médicaux,
      Effets indésirables liés aux médicaments (toutes issues du
      référentiel HAS, absentes en v2).
    - Placeholders honnêtes pour les 5 sections HAS obligatoires
      ([1..1] dans le référentiel : Problèmes actifs, Historique des
      actes, Allergies, Traitements médicamenteux, Dispositifs médicaux).
      Le référentiel propose "Pas de X connu" (vérifié absent) ou "Pas
      d'information sur X" (non vérifié) — mais notre outil n'interroge
      jamais le patient, il ne peut donc JAMAIS prétendre avoir vérifié
      une absence. On utilise une troisième formulation, assumée :
      "Aucune mention trouvée dans les documents analysés — nécessite
      une vérification directe auprès du patient."
    - Le bandeau d'horodatage global (v2.5) a été RETIRÉ (v3.6). Après
      vérification du référentiel officiel : un VSM standard n'affiche
      jamais de "période couverte" — seulement une "date de la synthèse"
      unique en en-tête (Art. 3.3), car il est généré à la demande à
      partir d'un dossier patient tenu à jour en continu par le médecin,
      pas en compilant rétrospectivement des décennies d'archives. Ce
      bandeau n'était donc pas une exigence HAS mais un artefact propre
      à notre méthode (extraction depuis un PDF scanné historique) — on
      a préféré le retirer plutôt que de risquer de le faire passer pour
      une section standard aux yeux du jury. Les dates par entité restent
      la seule source de repère temporel dans ce document.

SCOPE ASSUMÉ (inchangé) : conformité au contenu métier du VSM (bonnes
rubriques, bon contenu, date rattachée), pas de conformité technique
CDA R2/XML avec codage CIM-10/SNOMED/LOINC — hors scope du concours.
"""

import re
from difflib import SequenceMatcher

FUZZY_DEDUP_THRESHOLD = 0.85

# Placeholder honnête pour les sections HAS obligatoires vides — on ne
# prétend JAMAIS avoir vérifié une absence auprès du patient.
PLACEHOLDER_OBLIGATOIRE = (
    "Aucune mention trouvée dans les documents analysés. "
    "Cette section étant obligatoire au référentiel HAS, une vérification "
    "directe auprès du patient est nécessaire avant validation du VSM."
)
PLACEHOLDER_FACULTATIF = "Aucune information disponible dans les documents analysés."

# ============================================================
# STRUCTURE VSM — mapping des clés NER v3 vers les rubriques HAS
# ============================================================
# type : "objet" (liste de {texte, date}), "string" (liste de chaînes),
#        "traitement" (objet + champ type long_cours/aigu),
#        "effet_indesirable" (objet + champ medicament)
VSM_SECTIONS = [
    {
        "id": "points_attention", "titre": "Points d'attention",
        "cle_ner": "points_attention", "type": "string", "obligatoire": False,
    },
    {
        "id": "allergies", "titre": "Allergies et intolérances",
        "cle_ner": "allergies_intolerances", "type": "objet", "obligatoire": True,
    },
    {
        "id": "effets_indesirables", "titre": "Effets indésirables liés aux médicaments",
        "cle_ner": "effets_indesirables_medicaments", "type": "effet_indesirable", "obligatoire": False,
    },
    {
        "id": "historique_actes", "titre": "Historique des actes",
        "cle_ner": "historique_actes", "type": "objet", "obligatoire": True,
    },
    {
        "id": "traitements", "titre": "Traitements en cours",
        "cle_ner": "traitements_en_cours", "type": "traitement", "obligatoire": True,
    },
    {
        "id": "dispositifs", "titre": "Dispositifs médicaux",
        "cle_ner": "dispositifs_medicaux", "type": "string", "obligatoire": True,
    },
    {
        "id": "pathologies_antecedents", "titre": "Pathologies actives et antécédents",
        "type": "sous_sections",
        "sous_sections": [
            ("Pathologies actives", "pathologies_actives", "objet", True),
            ("Antécédents médicaux", "antecedents_medicaux", "objet", False),
            ("Antécédents familiaux", "antecedents_familiaux", "objet", False),
        ],
    },
    {
        "id": "mode_vie_risques", "titre": "Mode de vie et facteurs de risque",
        "cle_ner": "facteurs_risque", "type": "string", "obligatoire": False,
    },
]

VSM_ANNEXES = [
    {"id": "constantes", "titre": "Constantes (référence)", "cle_ner": "constantes", "type": "constante"},
    {"id": "examens", "titre": "Examens et bilans (référence)", "cle_ner": "examens_bilans", "type": "string"},
    {"id": "vaccinations", "titre": "Vaccinations (référence)", "cle_ner": "vaccinations", "type": "string"},
    {"id": "dates", "titre": "Repères chronologiques divers (référence)", "cle_ner": "dates_importantes", "type": "string"},
]


class VSMGenerator:
    """
    Transforme le JSON produit par NERExtractor v3 (schéma objet) en un
    VSM structuré conforme aux rubriques HAS/CI-SIS IPS-FR_2024.01.
    """

    def generate(self, ner_result: dict) -> dict:
        vsm = {
            "identite": ner_result.get("identite_patient", {}),
            "sections": [],
            "annexes": [],
        }

        for section in VSM_SECTIONS:
            if section["type"] == "sous_sections":
                sous_sections = []
                for titre_sous, cle, type_sous, obligatoire in section["sous_sections"]:
                    entries = self._consolidate(ner_result.get(cle, []), type_sous)
                    sous_sections.append({
                        "titre": titre_sous, "contenu": entries,
                        "type": type_sous, "obligatoire": obligatoire,
                    })
                vsm["sections"].append({"titre": section["titre"], "sous_sections": sous_sections})
            else:
                entries = self._consolidate(ner_result.get(section["cle_ner"], []), section["type"])
                vsm["sections"].append({
                    "titre": section["titre"], "contenu": entries,
                    "type": section["type"], "obligatoire": section["obligatoire"],
                })

        for annexe in VSM_ANNEXES:
            entries = self._consolidate(ner_result.get(annexe["cle_ner"], []), annexe["type"])
            vsm["annexes"].append({"titre": annexe["titre"], "contenu": entries, "type": annexe["type"]})

        return vsm

    # ------------------------------------------------------------------
    # Texte affichable d'une entrée, quel que soit son type — même
    # logique que ner_extractor.py (_display_text) pour cohérence.
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

    def _entry_date(self, entry, type_: str):
        if not isinstance(entry, dict):
            return None
        if type_ == "constante":
            return entry.get("date")
        if "historique_actes" in str(type_):
            return entry.get("date")
        return entry.get("date_debut") or entry.get("date")

    # ------------------------------------------------------------------
    # Consolidation légère : dédoublonnage fuzzy DANS une même catégorie,
    # en préservant toujours la date si l'une des deux entrées en a une
    # (même règle que ner_extractor.py — ne jamais perdre une date lors
    # d'une fusion).
    # ------------------------------------------------------------------
    def _consolidate(self, entries: list, type_: str) -> list:
        consolidated = []
        for entry in entries:
            self._add_with_dedup(consolidated, entry, type_)
        return consolidated

    def _add_with_dedup(self, existing_list: list, new_value, type_: str) -> None:
        new_text = self._display_text(new_value).lower().strip()

        for i, existing_value in enumerate(existing_list):
            existing_text = self._display_text(existing_value).lower().strip()

            similarity = SequenceMatcher(None, existing_text, new_text, autojunk=False).ratio()
            is_substring = (
                len(existing_text) >= 6 and len(new_text) >= 6
                and (existing_text in new_text or new_text in existing_text)
            )

            if similarity >= FUZZY_DEDUP_THRESHOLD or is_substring:
                if self._is_better_entry(new_value, existing_value, type_):
                    existing_list[i] = new_value
                return

        existing_list.append(new_value)

    def _is_better_entry(self, candidate, existing, type_: str) -> bool:
        if isinstance(candidate, str) or isinstance(existing, str):
            return len(self._display_text(candidate)) > len(self._display_text(existing))

        cand_date = self._entry_date(candidate, type_)
        exist_date = self._entry_date(existing, type_)
        if cand_date and not exist_date:
            return True
        if exist_date and not cand_date:
            return False

        return len(self._display_text(candidate)) > len(self._display_text(existing))

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Rendu Markdown lisible
    # ------------------------------------------------------------------
    def render_markdown(self, vsm: dict) -> str:
        lines = []

        identite = vsm.get("identite", {})
        lines.append("# Volet de Synthèse Médicale")
        lines.append("")
        lines.append("## Identité du patient")
        lines.append(f"- **Nom** : {identite.get('nom_complet') or '*Non détecté*'}")
        lines.append(f"- **Date de naissance** : {identite.get('date_naissance') or '*Non détectée*'}")
        lines.append(f"- **Médecin traitant déclaré** : {identite.get('medecin_traitant') or '*Non détecté*'}")
        lines.append("")

        for section in vsm["sections"]:
            lines.append(f"## {section['titre']}")

            if "sous_sections" in section:
                for sous in section["sous_sections"]:
                    lines.append(f"### {sous['titre']}")
                    lines.extend(self._render_entries(sous["contenu"], sous["type"], sous["obligatoire"]))
            else:
                lines.extend(self._render_entries(section["contenu"], section["type"], section["obligatoire"]))

            lines.append("")

        if vsm["annexes"]:
            lines.append("---")
            lines.append("## Annexes (informations de référence)")
            lines.append("")
            for annexe in vsm["annexes"]:
                lines.append(f"### {annexe['titre']}")
                lines.extend(self._render_entries(annexe["contenu"], annexe["type"], obligatoire=False))
                lines.append("")

        return "\n".join(lines)

    def _render_entries(self, entries: list, type_: str, obligatoire: bool) -> list:
        if not entries:
            placeholder = PLACEHOLDER_OBLIGATOIRE if obligatoire else PLACEHOLDER_FACULTATIF
            return [f"*{placeholder}*"]

        lines = []
        for entry in entries:
            lines.append(f"- {self._format_entry(entry, type_)}")
        return lines

    def _format_entry(self, entry, type_: str) -> str:
        if type_ == "string":
            return entry

        if type_ == "constante":
            date_str = f" ({entry['date']})" if entry.get("date") else ""
            return f"{entry['signe_vital']} : {entry['valeur']}{date_str}"

        if type_ == "traitement":
            texte = entry["texte"]
            type_traitement = "long cours" if entry.get("type") == "long_cours" else "aigu"
            date_str = f", débuté {entry['date_debut']}" if entry.get("date_debut") else ""
            return f"{texte} — *{type_traitement}*{date_str}"

        if type_ == "effet_indesirable":
            texte = entry["texte"]
            med = f" (médicament : {entry['medicament']})" if entry.get("medicament") else ""
            date_str = f" — depuis {entry['date_debut']}" if entry.get("date_debut") else ""
            return f"{texte}{med}{date_str}"

        # "objet" générique (pathologies, antécédents, historique_actes, allergies)
        texte = entry.get("texte", str(entry))
        date_val = entry.get("date_debut") or entry.get("date")
        date_str = f" ({date_val})" if date_val else ""

        parent = entry.get("parent")
        parent_str = f" — {parent}" if parent else ""

        return f"{texte}{parent_str}{date_str}"


# ----------------------------------------------------------------------
# Exemple d'utilisation
# ----------------------------------------------------------------------
if __name__ == "__main__":
    example_ner_result = {
        "identite_patient": {
            "nom_complet": "BANANE Sophie", "date_naissance": "17/09/1962", "medecin_traitant": None,
        },
        "pathologies_actives": [
            {"texte": "Ménopause subatrophique", "date_debut": "octobre 2009"},
        ],
        "antecedents_medicaux": [{"texte": "Tabagisme", "date_debut": None}],
        "antecedents_familiaux": [{"texte": "Néoplasie du sein (mère)", "date_debut": None}],
        "historique_actes": [
            {"texte": "Cholécystectomie", "date": "28/05/2002"},
            {"texte": "Exérèse d'un kyste sous-urétral", "date": "26/09/2008"},
        ],
        "allergies_intolerances": [],
        "effets_indesirables_medicaments": [],
        "traitements_en_cours": [
            {"texte": "RIVOTRIL 20 gouttes le soir", "date_debut": None, "type": "long_cours"},
        ],
        "constantes": [{"signe_vital": "Poids", "valeur": "65 kg", "date": "09/10/2014"}],
        "dispositifs_medicaux": [],
        "points_attention": [],
        "vaccinations": [],
        "facteurs_risque": ["Intoxication tabagique non sevrée à 30 paquets/année"],
        "examens_bilans": ["Mammographies : normal"],
        "dates_importantes": ["Prescrit le 09/10/2014"],
    }

    generator = VSMGenerator()
    vsm = generator.generate(example_ner_result)
    print(generator.render_markdown(vsm))