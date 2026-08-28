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
    - Annotation de récence actif/historique (v3.7 → v3.8, corrigée).
      Problème identifié : Qwen classe chaque page indépendamment, sans
      jamais comparer avec les autres pages du dossier — un
      "Pneumothorax" mentionné comme actif sur une page de 2006 reste en
      "pathologies_actives" même après fusion avec 113 autres pages plus
      récentes.
      v3.7 (ABANDONNÉE) : reclassement AUTOMATIQUE en antécédent au-delà
      d'un seuil fixe en années. Erreur reconnue : le seuil "correct"
      dépend du TYPE de pathologie (une infection aiguë et une maladie
      chronique ne "vieillissent" pas pareil), donc aucun chiffre unique
      n'est objectif ni généralisable — le seuil initial avait d'ailleurs
      été calibré pour faire fonctionner un seul exemple, pas validé
      objectivement.
      v3.8 (ACTUELLE) : on ne déplace plus rien automatiquement. Chaque
      pathologie active reçoit juste un fait calculé et neutre —
      "annees_depuis_mention" — visible en mode audit. Le jugement
      clinique (est-ce encore actif ?) reste entièrement au médecin qui
      valide, cohérent avec le principe déjà établi : ce n'est pas à
      l'outil de trancher une décision clinique. Une entité sans date
      n'est jamais annotée (aucune preuve). Le calcul n'agit que s'il y a
      assez de dates dans le dossier pour être fiable (sinon on
      n'annote rien plutôt que de deviner sur un échantillon pauvre).

SCOPE ASSUMÉ (inchangé) : conformité au contenu métier du VSM (bonnes
rubriques, bon contenu, date rattachée), pas de conformité technique
CDA R2/XML avec codage CIM-10/SNOMED/LOINC — hors scope du concours.
"""

import re
from difflib import SequenceMatcher

FUZZY_DEDUP_THRESHOLD = 0.85

MIN_DATES_FOR_RECLASSIFICATION = 5  # échantillon minimum pour faire confiance au calcul

MONTHS_FR = (
    r'janvier|f[ée]vrier|mars|avril|mai|juin|juillet|'
    r'ao[uû]t|septembre|octobre|novembre|d[ée]cembre'
)

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
        ner_result = self._annotate_recency(ner_result)

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
    # Annotation de récence (v3.8) — voir docstring en tête de fichier.
    # NE DÉPLACE PLUS RIEN AUTOMATIQUEMENT (v3.7 → v3.8) : décider si une
    # pathologie est encore active dépend du TYPE de pathologie (aiguë vs
    # chronique), information qu'on n'a pas de façon fiable. Aucun seuil
    # unique en années n'est objectif ni généralisable à tous les cas.
    # On se contente d'annoter chaque entrée avec un fait calculable et
    # neutre — "dernière mention il y a X ans" — et on laisse le médecin
    # qui valide juger si c'est encore pertinent. Cohérent avec le
    # principe déjà établi : ce n'est pas à l'outil de prendre une
    # décision clinique.
    # ------------------------------------------------------------------
    def _annotate_recency(self, ner_result: dict) -> dict:
        reference_year = self._compute_reference_year(ner_result)
        if reference_year is None:
            return ner_result  # pas assez de dates fiables pour un calcul significatif

        pathologies = ner_result.get("pathologies_actives", [])
        for entry in pathologies:
            if not isinstance(entry, dict):
                continue
            entry_year = self._extract_year(entry.get("date_debut"))
            entry["annees_depuis_mention"] = (reference_year - entry_year) if entry_year is not None else None

        return ner_result

    def _compute_reference_year(self, ner_result: dict) -> int | None:
        """
        Année la plus récente trouvée dans le dossier (hors date de
        naissance), utilisée comme référence pour juger si une
        pathologie "active" est en réalité ancienne. Ne retourne une
        valeur que si assez de dates fiables sont disponibles — sinon on
        préfère ne rien reclasser plutôt que de deviner sur un
        échantillon trop pauvre.
        """
        date_naissance = (ner_result.get("identite_patient") or {}).get("date_naissance")
        all_date_strings = []

        for entry in ner_result.get("dates_importantes", []):
            all_date_strings.append(self._display_text(entry))

        for key in ("pathologies_actives", "antecedents_medicaux", "antecedents_familiaux",
                    "historique_actes", "allergies_intolerances",
                    "effets_indesirables_medicaments", "traitements_en_cours"):
            for entry in ner_result.get(key, []):
                if isinstance(entry, dict):
                    date_val = entry.get("date_debut") or entry.get("date")
                    if date_val:
                        all_date_strings.append(date_val)

        for entry in ner_result.get("constantes", []):
            if isinstance(entry, dict) and entry.get("date"):
                all_date_strings.append(entry["date"])

        if date_naissance:
            all_date_strings = [d for d in all_date_strings if date_naissance not in d]

        all_years = [self._extract_year(d) for d in all_date_strings]
        all_years = [y for y in all_years if y is not None]

        if len(all_years) < MIN_DATES_FOR_RECLASSIFICATION:
            return None

        return max(all_years)

    def _extract_year(self, date_str) -> int | None:
        if not date_str or not isinstance(date_str, str):
            return None

        match = re.search(r'\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b', date_str)
        if match:
            return self._normalize_year(match.group(3))

        match = re.search(rf'\b(?:{MONTHS_FR})\s+(\d{{4}})\b', date_str, re.IGNORECASE)
        if match:
            return int(match.group(1))

        match = re.search(r'\b(19\d{2}|20\d{2})\b', date_str)
        if match:
            return int(match.group(1))

        return None

    def _normalize_year(self, year_str: str) -> int:
        if len(year_str) == 4:
            return int(year_str)
        y = int(year_str)
        return 1900 + y if y >= 30 else 2000 + y

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
                winner = new_value if self._is_better_entry(new_value, existing_value, type_) else existing_value
                if isinstance(winner, dict):
                    winner["pages_sources"] = self._merge_page_sources(new_value, existing_value)
                existing_list[i] = winner
                return

        existing_list.append(new_value)

    def _merge_page_sources(self, a, b) -> list:
        """Union triée et dédupliquée des pages_sources de deux entrées."""
        sources_a = a.get("pages_sources", []) if isinstance(a, dict) else []
        sources_b = b.get("pages_sources", []) if isinstance(b, dict) else []
        return sorted(set(sources_a) | set(sources_b))

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
    def render_markdown(self, vsm: dict, show_sources: bool = False) -> str:
        """
        show_sources=False (par défaut) : rendu VSM standard, sans
        référence de page — c'est le document "officiel", à ne pas
        distinguer visuellement d'un VSM produit par un vrai logiciel
        médical (les VSM réels n'ont jamais cette info, puisqu'ils
        partent d'un dossier électronique déjà fiable).

        show_sources=True : mode traçabilité/audit — affiche
        "[source : page X]" en fin de ligne. Réservé à un usage
        interne (debug, tests) ou à l'interface interactive
        (plateforme) qui l'exposera comme fonctionnalité d'innovation,
        pas dans le document final imprimé/exporté.
        """
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
                    lines.extend(self._render_entries(sous["contenu"], sous["type"], sous["obligatoire"], show_sources))
            else:
                lines.extend(self._render_entries(section["contenu"], section["type"], section["obligatoire"], show_sources))

            lines.append("")

        if vsm["annexes"]:
            lines.append("---")
            lines.append("## Annexes (informations de référence)")
            lines.append("")
            for annexe in vsm["annexes"]:
                lines.append(f"### {annexe['titre']}")
                lines.extend(self._render_entries(annexe["contenu"], annexe["type"], obligatoire=False, show_sources=show_sources))
                lines.append("")

        return "\n".join(lines)

    def _render_entries(self, entries: list, type_: str, obligatoire: bool, show_sources: bool = False) -> list:
        if not entries:
            placeholder = PLACEHOLDER_OBLIGATOIRE if obligatoire else PLACEHOLDER_FACULTATIF
            return [f"*{placeholder}*"]

        lines = []
        for entry in entries:
            lines.append(f"- {self._format_entry(entry, type_, show_sources)}")
        return lines

    def _format_entry(self, entry, type_: str, show_sources: bool = False) -> str:
        source_suffix = self._format_source_suffix(entry) if show_sources else ""

        if type_ == "string":
            texte = entry.get("texte", str(entry)) if isinstance(entry, dict) else entry
            return f"{texte}{source_suffix}"

        if type_ == "constante":
            date_str = f" ({entry['date']})" if entry.get("date") else ""
            return f"{entry['signe_vital']} : {entry['valeur']}{date_str}{source_suffix}"

        if type_ == "traitement":
            texte = entry["texte"]
            type_traitement = "long cours" if entry.get("type") == "long_cours" else "aigu"
            date_str = f", débuté {entry['date_debut']}" if entry.get("date_debut") else ""
            return f"{texte} — *{type_traitement}*{date_str}{source_suffix}"

        if type_ == "effet_indesirable":
            texte = entry["texte"]
            med = f" (médicament : {entry['medicament']})" if entry.get("medicament") else ""
            date_str = f" — depuis {entry['date_debut']}" if entry.get("date_debut") else ""
            return f"{texte}{med}{date_str}{source_suffix}"

        # "objet" générique (pathologies, antécédents, historique_actes, allergies)
        texte = entry.get("texte", str(entry))
        date_val = entry.get("date_debut") or entry.get("date")
        date_str = f" ({date_val})" if date_val else ""

        parent = entry.get("parent")
        parent_str = f" — {parent}" if parent else ""

        recency_str = ""
        if show_sources:
            annees = entry.get("annees_depuis_mention")
            if annees is not None and annees > 0:
                recency_str = f" [dernière mention il y a {annees} an{'s' if annees > 1 else ''} — à réévaluer]"

        return f"{texte}{parent_str}{date_str}{recency_str}{source_suffix}"

    def _format_source_suffix(self, entry) -> str:
        """
        Traçabilité (v3.6) : ajoute une référence vers la/les page(s)
        source(s) en fin de ligne. Une entité confirmée sur plusieurs
        pages liste toutes ses sources — un vrai signal de fiabilité
        pour le médecin qui valide (voir discussion projet : la
        validation se fait par sondage ciblé, pas par relecture
        intégrale du dossier).
        """
        if not isinstance(entry, dict):
            return ""
        pages = entry.get("pages_sources") or []
        if not pages:
            return ""
        if len(pages) == 1:
            return f" [source : page {pages[0]}]"
        pages_str = ", ".join(str(p) for p in pages)
        return f" [sources : pages {pages_str}]"


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