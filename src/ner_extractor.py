import re
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification


# ============================================================
# PATTERNS REGEX EXHAUSTIFS — basés sur référentiel HAS/CI-SIS
# ============================================================

PATTERNS = {

    # --------------------------------------------------------
    # MÉDICAMENTS ET TRAITEMENTS
    # --------------------------------------------------------
    "medicaments": [
        # Médicament + dosage + posologie complète
        r'\b([A-ZÀ-ÿ][a-zA-ZÀ-ÿ\-]+(?:\s+[A-ZÀ-ÿ][a-zA-ZÀ-ÿ\-]+)?'
        r'\s+\d+(?:[.,]\d+)?\s*(?:mg|ml|g|µg|mcg|UI|MUI|mmol|ng|mg/ml|mg/kg)'
        r'(?:\s+\d+\s*(?:comprimé|gélule|cp|cps|ampoule|sachet)s?)?'
        r'(?:\s+(?:matin\s+et\s+soir|matin\s+midi\s+et\s+soir|'
        r'par\s+jour|le\s+matin|le\s+soir|matin|midi|soir|'
        r'au\s+coucher|à\s+jeun|au\s+réveil|en\s+continu))?)\b',

        # Médicament + forme galénique
        r'\b([A-ZÀ-ÿ][a-zA-ZÀ-ÿ\-]+\s+'
        r'\d+\s*(?:comprimé|gélule|cp|cps|caps|ampoule|sachet|'
        r'suppositoire|patch|spray|goutte|solution|sirop|'
        r'injectable|perfusion)s?)\b',

        # Médicament avec DCI + nom commercial
        r'\b([A-ZÀ-ÿ][a-zA-ZÀ-ÿ\-]+\s*\([A-ZÀ-ÿ][a-zA-ZÀ-ÿ\-]+\)'
        r'\s*\d+\s*(?:mg|ml|g|µg))\b',
    ],

    # --------------------------------------------------------
    # DATES — tous formats médicaux français
    # --------------------------------------------------------
    "dates": [
        # JJ/MM/AAAA, JJ-MM-AAAA, JJ.MM.AAAA — groupe 0 complet
        r'\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b',

        # Mois/AAAA
        r'\b\d{1,2}[/\-\.]\d{4}\b',

        # Texte long : "le 2 octobre 2007"
        r'\b(?:le\s+)?\d{1,2}\s+'
        r'(?:janvier|février|mars|avril|mai|juin|juillet|août|'
        r'septembre|octobre|novembre|décembre)\s+\d{4}\b',

        # Année avec contexte médical
        r'\b(?:en|depuis|jusqu\'en|avant|après|né\s+en|opéré\s+en|'
        r'diagnostiqué\s+en|hospitalisé\s+en)\s+(\d{4})\b',
    ],

    # --------------------------------------------------------
    # CONSTANTES BIOLOGIQUES ET CLINIQUES — groupe 0 complet
    # --------------------------------------------------------
    "constantes": [
        
        # Tension artérielle — pattern simplifié qui fonctionne
        r'TA\s*[:\-]\s*(\d+/\d+)',
        r'(?:tension\s+art[eé]rielle|pression\s+art[eé]rielle)\s*[:\-]?\s*(\d+/\d+)',
        
        # Poids avec unité
        r'\b(?:[Pp]oids)\s*[:\-]?\s*\d{2,3}(?:[.,]\d+)?\s*(?:kg|kilos?)\b',

        # Taille avec unité
        r'\b(?:[Tt]aille)\s*[:\-]?\s*\d{2,3}\s*(?:cm|m)\b',

        # IMC
        r'\b(?:IMC|indice\s+de\s+masse\s+corporelle)'
        r'\s*[:\-]?\s*\d{1,2}(?:[.,]\d+)?\s*(?:kg/m²)?\b',

        # HbA1c
        r'\b(?:HbA1c|hémoglobine\s+glyquée)'
        r'\s*[:\-]?\s*\d+(?:[.,]\d+)?\s*(?:%|mmol/mol)?\b',

        # Glycémie
        r'\b(?:glycémie|glucose)'
        r'\s*[:\-]?\s*\d+(?:[.,]\d+)?\s*(?:g/l|mmol/l|mg/dl)?\b',

        # Créatinine
        r'\b(?:créatinine|creatinine)'
        r'\s*[:\-]?\s*\d+(?:[.,]\d+)?\s*(?:µmol/l|mg/l|umol/l)?\b',

        # Cholestérol LDL HDL
        r'\b(?:cholestérol|LDL|HDL|triglycérides?)'
        r'\s*[:\-]?\s*\d+(?:[.,]\d+)?\s*(?:g/l|mmol/l|mg/dl)?\b',

        # Fréquence cardiaque
        r'\b(?:FC|fréquence\s+cardiaque|pouls)'
        r'\s*[:\-]?\s*\d{2,3}\s*(?:bpm|/min)?\b',

        # Saturation O2
        r'\b(?:SpO2|saturation|SaO2)\s*[:\-]?\s*\d{2,3}\s*(?:%)?\b',

        # Température
        r'\b(?:température|T°)\s*[:\-]?\s*\d{2}(?:[.,]\d+)?\s*(?:°C)?\b',

        # INR
        r'\bINR\s*[:\-]?\s*\d+(?:[.,]\d+)?\b',

        # PSA
        r'\bPSA\s*[:\-]?\s*\d+(?:[.,]\d+)?\s*(?:ng/ml)?\b',

        # DFG
        r'\b(?:DFG|clairance|MDRD|CKD-EPI)'
        r'\s*[:\-]?\s*\d+(?:[.,]\d+)?\s*(?:ml/min)?\b',

        # SA (semaines d'aménorrhée)
        r'\b\d+\s*(?:SA|semaines?\s+d\'aménorrhée)\b',
    ],

    # --------------------------------------------------------
    # ALLERGIES ET INTOLÉRANCES
    # --------------------------------------------------------
    "allergies": [
    # Capture X et Y : "Allergie à la Pénicilline et à l'Aspirine"
    r'[Aa]llergi\w*\s+(?:à|au|aux)\s+(?:la\s+|l\'|le\s+)?(\w+)'
    r'(?:\s+et\s+(?:à|au|aux)\s+(?:la\s+|l\'|le\s+)?(\w+))?',

    # Format liste "Allergies : X, Y"
    r'[Aa]llergi(?:e|es)\s*[:\-]\s*(\w[\w\s,\-]+?)(?:\n|$|\.\s)',

    # Intolérance
    r'[Ii]ntol[eé]rance\s+(?:à\s+(?:la\s+|l\'|le\s+)?|au\s+|aux\s+)(\w+)',

    # Contre-indication
    r'[Cc]ontre[-\s]indication\s+(?:à\s+(?:la\s+|l\'|le\s+)?|au\s+|aux\s+)(\w+)',

    # Hypersensibilité
    r'[Hh]ypersensibilit[eé]\s+(?:à\s+(?:la\s+|l\'|le\s+)?|au\s+|aux\s+)(\w+)',
    ],

    # --------------------------------------------------------
    # VACCINATIONS — calendrier vaccinal français complet
    # --------------------------------------------------------
    "vaccinations": [
        # Vaccin + nom avec statut
        r'\b(?:vaccin(?:ation|é|ée|és|ées)?\s+(?:contre\s+)?|'
        r'rappel\s+(?:de\s+)?|primo[-\s]vaccination\s+(?:contre\s+)?)'
        r'([A-ZÀ-ÿ][a-zA-ZÀ-ÿ\s\-]+?)'
        r'(?:\s+à\s+jour|\s+(?:fait|effectué|réalisé|incomplet)|\.|,|\n)',

        # Vaccins nommés calendrier vaccinal français
        r'\b(DTP|DTPolio|DTCaPolio|BCG|ROR|MMR|'
        r'Hépatite\s+[AB]|HBV|HAV|'
        r'Pneumocoque|Méningocoque|Méningite\s+[ABCWY]|'
        r'Grippe\s+saisonnière|Influenza|'
        r'HPV|Papillomavirus|Gardasil|Cervarix|'
        r'Covid[-\s]?19|SARS-CoV-2|'
        r'Zona|Varicelle|Rotavirus|'
        r'Typhoïde|Fièvre\s+jaune|Rage|Tétanos|'
        r'Coqueluche|Rougeole|Rubéole|Oreillons)\b',

        # Statut vaccinal
        r'\b(?:vaccin(?:s|ation)?)\s+'
        r'(?:à\s+jour|non\s+à\s+jour|incomplet(?:s)?|à\s+mettre\s+à\s+jour)\b',
    ],

    # --------------------------------------------------------
    # ANTÉCÉDENTS CHIRURGICAUX — CCAM
    # --------------------------------------------------------
    "antecedents_chirurgicaux": [
        # Interventions et procédures chirurgicales
        r'\b(?:intervention(?:s)?\s+chirurgicale?s?|'
        r'chirurgie|opération|'
        r'opéré(?:e)?\s+(?:de|d\'|du)|'
        r'intervention\s+(?:de|du|pour)|'
        r'appendicectomie|cholécystectomie|hystérectomie|'
        r'gastrectomie|colectomie|hépatectomie|'
        r'néphrectomie|prostatectomie|thyroïdectomie|'
        r'mastectomie|tumorectomie|'
        r'appendicite\s+opérée|hernie\s+(?:hiatale\s+)?opérée|'
        r'hernie\s+(?:hiatale|inguinale|discale)|'
        r'prothèse\s+(?:de\s+hanche|de\s+genou|totale|partielle)|'
        r'implant|greffe|transplantation|'
        r'bypass|pontage\s+(?:coronarien|aorto[-\s]coronarien)|'
        r'résection|ablation|exérèse|'
        r'arthroscopie|arthrodèse|'
        r'fracture\s+(?:opérée|chirurgicale)|'
        r'FOGD\s+(?:thérapeutique|opératoire)|'
        r'césarienne|accouchement\s+par\s+césarienne)\b',
    ],

    # --------------------------------------------------------
    # FACTEURS DE RISQUE — HAS
    # --------------------------------------------------------
    "facteurs_risque": [
        # Tabac — toutes formes
        r'\b(?:tabac|tabagisme|fumeur|fumeuse|'
        r'ex[-\s]fumeur|ex[-\s]fumeuse|sevrage\s+tabagique|'
        r'consommation\s+tabagique|'
        r'\d+\s*(?:paquet[s]?[-\s]an(?:née)?[s]?|PA)|'
        r'(?:>|<|≥|≤|\d+)\s*(?:PA|paquet[s]?\s*[-/]\s*an(?:née)?)|'
        r'intoxication\s+tabagique(?:\s+chronique)?|'
        r'non\s+sevré(?:e)?)\b',

        # Alcool
        r'\b(?:alcool|alcoolisme|consommation\s+d\'alcool|'
        r'éthylisme|OH|intoxication\s+alcoolique|'
        r'sevrage\s+alcoolique|'
        r'\d+\s*verres?/(?:semaine|jour|an)|'
        r'abstinent(?:e)?|sobre)\b',

        # Activité physique
        r'\b(?:sédentarité|sédentaire|'
        r'activité\s+physique\s+(?:insuffisante|réduite|nulle|régulière)|'
        r'sportif|sport\s+régulier|'
        r'marche\s+régulière)\b',

        # Surpoids / obésité
        r'\b(?:surpoids|obésité|obèse|'
        r'obésité\s+(?:morbide|sévère|modérée|de\s+grade\s+[123]))\b',

        # Facteurs cardiovasculaires
        r'\b(?:dyslipidémie|hypercholestérolémie|'
        r'hypertriglycéridémie|hyperlipidémie|'
        r'syndrome\s+métabolique|'
        r'risque\s+cardiovasculaire\s+(?:élevé|modéré|faible)|'
        r'SCORE\s+cardiovasculaire)\b',

        # Antécédents familiaux
        r'\b(?:antécédent[s]?\s+familiaux?|ATCD\s+familiaux?)\s+'
        r'(?:de\s+)?(?:diabète|HTA|hypertension|infarctus|'
        r'cancer|AVC|maladie\s+coronarienne|mort\s+subite)\b',

        # Facteurs professionnels
        r'\b(?:exposition\s+(?:professionnelle|au\s+bruit|aux\s+produits\s+chimiques)|'
        r'amiante|silice|risque\s+professionnel|'
        r'travail\s+(?:de\s+nuit|posté|en\s+équipe|pénible))\b',

        # Alimentation
        r'\b(?:alimentation\s+(?:déséquilibrée|riche\s+en\s+graisses)|'
        r'régime\s+(?:déséquilibré|hypercalorique)|'
        r'malnutrition)\b',
    ],

    # --------------------------------------------------------
    # PATHOLOGIES ACTIVES — CIM-10 et CISP
    # --------------------------------------------------------
    "pathologies": [
        # Diabète — toutes formes
        r'\b(?:diabète\s+(?:de\s+)?type\s+[12I]|'
        r'diabète\s+type\s+[12]|DT[12]|'
        r'diabète\s+(?:insulino[-\s]?dépendant|'
        r'non\s+insulino[-\s]?dépendant|'
        r'gestationnel|MODY|secondaire)|'
        r'diabétique)\b',

        # Cardio-vasculaire
        r'\b(?:hypertension\s+artérielle|HTA|'
        r'insuffisance\s+cardiaque\s+(?:gauche|droite|globale)?|IC|'
        r'fibrillation\s+auriculaire|FA|flutter\s+auriculaire|'
        r'infarctus\s+(?:du\s+myocarde)?|IDM|SCA|'
        r'angor|angine\s+de\s+poitrine|'
        r'coronaropathie|maladie\s+coronarienne|'
        r'AVC|accident\s+vasculaire\s+cérébral|'
        r'AIT|accident\s+ischémique\s+transitoire|'
        r'artérite|AOMI|thrombose|'
        r'embolie\s+pulmonaire|EP|'
        r'phlébite|thrombophlébite|TVP|'
        r'dissection\s+aortique|anévrisme)\b',

        # Respiratoire
        r'\b(?:asthme|BPCO|'
        r'bronchite\s+chronique\s+obstructive?|'
        r'emphysème|insuffisance\s+respiratoire\s+(?:chronique|aiguë)?|'
        r'apnée\s+(?:obstructive\s+)?du\s+sommeil|SAOS|SAHOS|'
        r'pneumonie|pleurésie|'
        r'tuberculose|séquelle\s+tuberculeuse|'
        r'cancer\s+(?:du\s+)?poumon|néoplasie\s+pulmonaire|'
        r'fibrose\s+pulmonaire|sarcoïdose)\b',

        # Digestif
        r'\b(?:reflux\s+gastro[-\s]?[oœ]sophagien|RGO|'
        r'ulcère\s+(?:gastrique|duodénal|gastro[-\s]?duodénal)|'
        r'gastrite|Helicobacter\s+pylori|HP|'
        r'maladie\s+de\s+Crohn|rectocolite\s+hémorragique|RCH|MICI|'
        r'colopathie\s+fonctionnelle|SII|'
        r'cirrhose|hépatite\s+[ABC]|stéatose\s+hépatique|NASH|'
        r'pancréatite|lithiase\s+biliaire|'
        r'hernie\s+hiatale|'
        r'cancer\s+(?:colorectal|du\s+côlon|du\s+rectum|'
        r'de\s+l\'[oœ]sophage|gastrique|du\s+pancréas))\b',

        # Rhumatologique
        r'\b(?:arthrose\s+(?:du\s+genou|de\s+hanche|cervicale|lombaire)?|'
        r'polyarthrite\s+rhumatoïde|PR|'
        r'spondylarthrite\s+(?:ankylosante)?|SA|'
        r'goutte|hyperuricémie|'
        r'ostéoporose|ostéopénie|'
        r'lombalgie\s+(?:chronique|aiguë)?|'
        r'cervicalgie|dorsalgie|'
        r'sciatique|sciatalgie|névralgie\s+cervico[-\s]brachiale|'
        r'canal\s+carpien|'
        r'fibromyalgie)\b',

        # Endocrinologique
        r'\b(?:hypothyroïdie|hyperthyroïdie|thyrotoxicose|'
        r'thyroïdite\s+(?:de\s+Hashimoto|auto-immune)?|'
        r'goitre|nodule\s+thyroïdien|'
        r'insuffisance\s+surrénalienne|maladie\s+d\'Addison|'
        r'syndrome\s+de\s+Cushing|'
        r'hyperparathyroïdie|hypoparathyroïdie|'
        r'ostéomalacie|rachitisme)\b',

        # Neurologique / Psychiatrique
        r'\b(?:épilepsie|crise\s+épileptique|'
        r'maladie\s+de\s+Parkinson|syndrome\s+parkinsonien|'
        r'sclérose\s+en\s+plaques|SEP|'
        r'migraine|céphalées\s+(?:chroniques|de\s+tension)?|'
        r'dépression\s+(?:majeure|chronique|saisonnière)?|'
        r'anxiété\s+(?:généralisée)?|troubles?\s+anxieux|'
        r'trouble\s+panique|phobie|'
        r'schizophrénie|trouble\s+bipolaire|'
        r'démence|maladie\s+d\'Alzheimer|troubles?\s+cognitifs|'
        r'AVC\s+séquellaire|hémiplégie|'
        r'neuropathie\s+(?:diabétique|périphérique)?|'
        r'syndrome\s+dépressif|burn.out)\b',

        # Rénal / Urologique
        r'\b(?:insuffisance\s+rénale\s+(?:chronique|aiguë)|IRC|IRA|'
        r'néphropathie\s+(?:diabétique|hypertensive)?|'
        r'glomérulonéphrite|syndrome\s+néphrotique|'
        r'lithiase\s+(?:rénale|urinaire|urétérale|urétéro-rénale)|'
        r'pyélonéphrite|cystite\s+(?:chronique|récidivante)?|'
        r'cancer\s+(?:du\s+)?rein|cancer\s+(?:de\s+la\s+)?prostate|'
        r'hypertrophie\s+bénigne\s+(?:de\s+la\s+)?prostate|HBP|'
        r'incontinence\s+urinaire|dysurie|pollakiurie)\b',

        # Gynécologique / Obstétrical
        r'\b(?:cancer\s+du\s+sein|'
        r'cancer\s+de\s+l\'endomètre|'
        r'cancer\s+du\s+col\s+(?:de\s+l\'utérus)?|'
        r'endométriose|adénomyose|'
        r'fibrome\s+(?:utérin)?|myome|'
        r'kyste\s+(?:ovarien|de\s+l\'ovaire)|'
        r'ménopause|préménopause|périménopause|'
        r'syndrome\s+des\s+ovaires\s+polykystiques|SOPK|'
        r'grossesse\s+(?:extra[-\s]utérine|GEU)?|'
        r'fausse\s+couche|avortement|IVG|'
        r'accouchement\s+(?:prématuré|par\s+voies\s+naturelles)?|'
        r'pré[-\s]éclampsie|éclampsie|'
        r'diabète\s+gestationnel)\b',

        # Oncologie générale
        r'\b(?:cancer|carcinome|adénocarcinome|'
        r'mélanome|lymphome\s+(?:hodgkinien|non\s+hodgkinien)?|'
        r'leucémie\s+(?:aiguë|chronique|lymphoïde|myéloïde)?|'
        r'myélome\s+multiple|'
        r'néoplasie|tumeur\s+(?:maligne|bénigne)?|métastase|'
        r'chimiothérapie|radiothérapie|immunothérapie|hormonothérapie|'
        r'rémission\s+(?:complète|partielle)?|rechute|récidive)\b',

        # Infectieux
        r'\b(?:VIH|SIDA|infection\s+à\s+VIH|'
        r'hépatite\s+[ABC]\s+(?:chronique|aiguë)?|'
        r'infection\s+(?:urinaire|pulmonaire|cutanée|ostéo-articulaire)|'
        r'sepsis|choc\s+septique|bactériémie|'
        r'endocardite|méningite|encéphalite)\b',

        # Dermatologique
        r'\b(?:psoriasis|eczéma|dermatite\s+atopique|'
        r'urticaire\s+chronique|pemphigoïde|'
        r'mélanome|carcinome\s+(?:basocellulaire|spinocellulaire))\b',

        # Ophtalmologique
        r'\b(?:glaucome|DMLA|dégénérescence\s+maculaire|'
        r'cataracte|rétinopathie\s+diabétique|'
        r'décollement\s+de\s+rétine)\b',

        # Hématologique
        r'\b(?:anémie\s+(?:ferriprive|par\s+carence\s+en\s+B12|'
        r'hémolytique|aplasique|falciforme)?|'
        r'thrombopénie|polyglobulie|'
        r'drépanocytose|thalassémie|'
        r'hémophilie|maladie\s+de\s+Willebrand)\b',
    ],

    # --------------------------------------------------------
    # EXAMENS ET BILANS
    # --------------------------------------------------------
    "examens": [
        # Imagerie
        r'\b(?:radio(?:graphie)?|Rx|'
        r'scanner|TDM|tomodensitométrie|'
        r'IRM|imagerie\s+par\s+résonance\s+magnétique|'
        r'échographie|écho(?:graphie)?|doppler|'
        r'scintigraphie|TEP|PET[-\s]scan|'
        r'mammographie|ostéodensitométrie|DEXA|'
        r'endoscopie|coloscopie|fibroscopie|gastroscopie|'
        r'FOGD|bronchoscopie|cystoscopie|'
        r'angiographie|artériographie)\b',

        # Biologie
        r'\b(?:NFS|numération\s+formule\s+sanguine|hémogramme|'
        r'bilan\s+(?:lipidique|hépatique|rénal|thyroïdien|'
        r'inflammatoire|martial|vitaminique|hormonal)|'
        r'ionogramme\s+(?:sanguin|urinaire)?|'
        r'CRP|VS|fibrinogène|'
        r'ECBU|bandelette\s+urinaire|BU|CBEU|'
        r'HbA1c|glycémie\s+à\s+jeun|HGPO|'
        r'TSH|T3|T4|T3L|T4L|'
        r'PSA|CA\s*125|CA\s*19[-\s]9|ACE|AFP|CA\s*15[-\s]3|'
        r'électrophorèse\s+(?:des\s+protéines)?|'
        r'protéinurie|microalbuminurie|'
        r'coagulation|TP|TCA|INR|D-dimères|facteur\s+V|'
        r'ferritine|fer\s+sérique|transferrine|'
        r'vitamine\s+(?:D|B12|B9)|acide\s+folique|'
        r'créatinine|urée|acide\s+urique|'
        r'transaminases|ASAT|ALAT|GGT|PAL|bilirubine|'
        r'albumine|protéines\s+totales)\b',

        # Fonctionnel
        r'\b(?:ECG|électrocardiogramme|'
        r'EFR|épreuve\s+fonctionnelle\s+respiratoire|'
        r'VEMS|CVF|DEP|'
        r'épreuve\s+d\'effort|test\s+d\'effort|'
        r'holter\s+(?:ECG|tensionnel)?|'
        r'MAPA|monitoring\s+tensionnel|'
        r'audiogramme|audiométrie|'
        r'électromyogramme|EMG|'
        r'EEG|électroencéphalogramme|'
        r'potentiels\s+évoqués)\b',
    ],
}

# ============================================================
# MOTS-CLÉS CONTEXTUELS EXHAUSTIFS
# ============================================================

ANTECEDENT_KEYWORDS = [
    'atcd', 'atcds', 'antcd', 'atcd.',
    'antécédent', 'antécédents', 'antecedent', 'antecedents',
    'antécédente', 'antécédentes',
    'antécédent chirurgical', 'antécédents chirurgicaux',
    'antécédent médical', 'antécédents médicaux',
    'antécédent familial', 'antécédents familiaux',
    'antécédent obstétrical', 'antécédents obstétricaux',
    'antécédent gynécologique', 'antécédents gynécologiques',
    'antécédent personnel', 'antécédents personnels',
    'histoire de la maladie', 'hdm', 'hdlm', 'hdlt',
    'histoire médicale', 'passé médical',
    'souffre de', 'souffrait de', 'souffre depuis',
    'diagnostiqué', 'diagnostiquée', 'diagnostic de',
    'connu pour', 'connue pour', 'connu comme',
    'suivi pour', 'suivie pour', 'prise en charge pour',
    'traité pour', 'traitée pour',
    'porteur de', 'porteuse de',
    'présente', 'présentait', 'présente un', 'présente une',
    'a présenté', 'a développé', 'a souffert de',
    'opéré de', 'opérée de', 'opéré en', 'opérée en',
    'hospitalisé pour', 'hospitalisée pour',
    'hospitalisé en', 'hospitalisée en',
    'consulte pour', 'motif de consultation',
    'sur le plan médical', 'sur le plan chirurgical',
    'sur le plan gynécologique', 'sur le plan obstétrical',
    'sur le plan familial', 'sur le plan cardiologique',
    'sur le plan neurologique', 'sur le plan psychiatrique',
    'dans ses antécédents', 'dans ses atcd',
    'notion de', 'antérieurement',
]

TRAITEMENT_KEYWORDS = [
    'traitement', 'traitements', 'thérapeutique', 'thérapeutiques',
    'prescription', 'prescriptions', 'ordonnance',
    'prend', 'prenait', 'prendre',
    'prescrit', 'prescrite', 'prescrits', 'prescrites',
    'administré', 'administrée', 'administrés',
    'sous', 'mis sous', 'mise sous', 'passé sous',
    'initié', 'initiée', 'débuté', 'débutée',
    'arrêté', 'arrêtée', 'suspendu', 'suspendue', 'stoppé',
    'continué', 'maintenu', 'poursuivi', 'reconduit',
    'traitement en cours', 'traitement habituel',
    'traitement au long cours', 'traitement chronique',
    'traitement de fond', 'traitement de crise',
    'traitement symptomatique', 'traitement curatif',
    'posologie', 'dose', 'dosage', 'schéma thérapeutique',
    'prise', 'prises', 'comprimé', 'gélule', 'ampoule',
    'ttt', 'TTT', 'Ttt',
    'examens prescrits', 'traitements prescrits',
    'médicaments prescrits', 'ordonnancé',
]

ALLERGIE_KEYWORDS = [
    'allergie', 'allergies', 'allergique', 'allergiques',
    'intolérance', 'intolérances', 'intolérant', 'intolérante',
    'hypersensibilité', 'hypersensible',
    'contre-indication', 'contre-indiqué', 'contre-indiquée',
    'réaction allergique', 'réaction anaphylactique',
    'choc anaphylactique', 'anaphylaxie',
    'urticaire', 'angioedème', 'oedème de Quincke',
    'allergie médicamenteuse', 'allergie alimentaire',
    'allergie cutanée', 'allergie respiratoire',
    'terrain allergique', 'atopique', 'atopie',
    'eczéma allergique', 'rhinite allergique',
]

VACCINATION_KEYWORDS = [
    'vaccin', 'vaccins', 'vaccination', 'vaccinations',
    'vacciné', 'vaccinée', 'vaccinés', 'vaccinées',
    'rappel', 'rappels', 'primo-vaccination',
    'calendrier vaccinal', 'carnet de vaccination',
    'carnet de santé',
    'à jour', 'non à jour', 'incomplet', 'à mettre à jour',
    'immunisation', 'immunisé', 'immunisée',
    'prochain rappel', 'dernière vaccination',
]

FACTEUR_RISQUE_KEYWORDS = [
    'facteur de risque', 'facteurs de risque',
    'fdr', 'fdrs',
    'mode de vie', 'habitus', 'hygiène de vie',
    'tabac', 'tabagisme', 'fumeur', 'fumeuse',
    'alcool', 'alcoolisme', "consommation d'alcool",
    'obésité', 'surpoids', 'poids excessif',
    'sédentarité', 'activité physique',
    'stress', 'anxiété', 'facteur psychosocial',
    'antécédents familiaux', 'hérédité', 'prédisposition',
    'profession', 'exposition professionnelle',
    'alimentation', 'régime alimentaire',
    'risque cardiovasculaire', 'terrain vasculaire',
]


class NERExtractor:
    """
    Module d'extraction d'entités médicales NER.
    Conforme au référentiel VSM de la HAS et CI-SIS.

    Approche hybride exhaustive :
    1. Règles/Regex → médicaments, dates, constantes,
                      allergies, vaccinations, pathologies,
                      facteurs de risque, examens
    2. DrBERT-4GB   → antécédents, diagnostics complexes,
                      entités contextuelles

    Entités extraites (conformes VSM HAS) :
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

    def __init__(self, use_drbert: bool = True):
        self.use_drbert = use_drbert
        self._ner_pipeline = None
        if use_drbert:
            self._load_drbert()

    def _load_drbert(self):
        """Charge DrBERT-4GB-CP-CamemBERT."""
        try:
            print("    ⟳ Chargement DrBERT-4GB-CP-CamemBERT...")
            tokenizer = AutoTokenizer.from_pretrained(
                "Dr-BERT/DrBERT-4GB-CP-CamemBERT"
            )
            model = AutoModelForTokenClassification.from_pretrained(
                "Dr-BERT/DrBERT-4GB-CP-CamemBERT"
            )
            self._ner_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=-1
            )
            print("    ✓ DrBERT-4GB chargé")
        except Exception as e:
            print(f"    ⚠️ DrBERT non disponible : {e}")
            print("    → Extraction par règles uniquement")
            self._ner_pipeline = None

    def _normalize_text(self, text: str) -> str:
        """Normalise le texte pour l'extraction."""
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('–', '-').replace('—', '-')
        text = text.replace('œ', 'oe').replace('æ', 'ae')
        return text.strip()

    def _extract_by_regex(self, text: str) -> dict:
        """Extraction exhaustive par règles et patterns regex."""
        results = {
            "medicaments": [],
            "dates": [],
            "constantes": [],
            "allergies": [],
            "vaccinations": [],
            "antecedents_chirurgicaux": [],
            "facteurs_risque": [],
            "pathologies": [],
            "examens": []
        }

        for category, patterns in PATTERNS.items():
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        
                        # Cas spécial allergies — capturer groupe 1 ET groupe 2
                        if category == 'allergies':
                            for i in range(1, (match.lastindex or 0) + 1):
                                grp = match.group(i)
                                if grp:
                                    entity = grp.strip()
                                    if entity and len(entity) > 2:
                                        if entity not in results[category]:
                                            results[category].append(entity)
                            continue
                        
                        # Constantes → group(0) complet avec contexte
                        if category == 'constantes':
                            entity = match.group(0).strip()

                        # Dates → group(0) complet
                        elif category == 'dates':
                            entity = match.group(0).strip()

                        # Autres → group(1) si existe
                        elif match.lastindex and match.lastindex >= 1:
                            entity = match.group(1).strip()
                        else:
                            entity = match.group(0).strip()

                        if entity and len(entity) > 2:
                            if entity not in results[category]:
                                results[category].append(entity)
                                
                except Exception:
                    continue

        return results

    def _get_context(self, text: str, word: str, window: int = 100) -> str:
        """Retourne le contexte autour d'un mot."""
        text_lower = text.lower()
        word_lower = word.lower()
        pos = text_lower.find(word_lower)
        if pos == -1:
            return ""
        return text_lower[max(0, pos - window):pos + window]

    def _extract_by_drbert(self, text: str) -> dict:
        """Extraction par DrBERT-4GB avec classification contextuelle."""
        if not self._ner_pipeline:
            return {
                "antecedents": [],
                "diagnostics": [],
                "traitements": [],
                "autres": []
            }

        results = {
            "antecedents": [],
            "diagnostics": [],
            "traitements": [],
            "autres": []
        }

        try:
            # Découpage en chunks de 400 caractères
            max_length = 400
            words = text.split()
            chunks = []
            current_chunk = []
            current_len = 0

            for word in words:
                current_chunk.append(word)
                current_len += len(word) + 1
                if current_len > max_length:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_len = 0

            if current_chunk:
                chunks.append(' '.join(current_chunk))

            all_entities = []
            for chunk in chunks:
                entities = self._ner_pipeline(chunk)
                all_entities.extend(entities)

            for entity in all_entities:
                word = entity.get('word', '').strip()
                score = entity.get('score', 0)

                if not word or score < 0.7 or len(word) < 2:
                    continue

                context = self._get_context(text, word)

                is_antecedent = any(kw in context for kw in ANTECEDENT_KEYWORDS)
                is_traitement = any(kw in context for kw in TRAITEMENT_KEYWORDS)

                if is_antecedent and word not in results["antecedents"]:
                    results["antecedents"].append(word)
                elif is_traitement and word not in results["traitements"]:
                    results["traitements"].append(word)
                elif word not in results["autres"]:
                    results["autres"].append(word)

        except Exception as e:
            print(f"    ⚠️ Erreur DrBERT NER : {e}")

        return results

    def _deduplicate(self, entities: list) -> list:
        """Supprime les doublons en ignorant la casse."""
        seen = set()
        unique = []
        for entity in entities:
            key = entity.lower().strip()
            if key not in seen and len(key) > 2:
                seen.add(key)
                unique.append(entity)
        return unique
    
    def _filter_dates(self, dates: list) -> list:
        """Supprime les dates partielles contenues dans des dates complètes."""
        filtered = []
        for date in dates:
            is_substring = any(
                date != other and date in other
                for other in dates
            )
            if not is_substring:
                filtered.append(date)
        return filtered

    def extract(self, text: str, document_type: str = "inconnu") -> dict:
        """
        Point d'entrée principal.
        Extraction hybride exhaustive conforme VSM HAS.
        """
        if not text or len(text.strip()) < 10:
            return self._empty_result("Texte vide ou trop court")

        text = self._normalize_text(text)

        regex_results = self._extract_by_regex(text)
        drbert_results = self._extract_by_drbert(text)

        result = {
            "pathologies_actives": self._deduplicate(
                regex_results.get("pathologies", [])
            ),
            "antecedents_medicaux": self._deduplicate(
                drbert_results.get("antecedents", []) +
                regex_results.get("pathologies", [])
            ),
            "antecedents_chirurgicaux": self._deduplicate(
                regex_results.get("antecedents_chirurgicaux", [])
            ),
            "allergies_intolerances": self._deduplicate(
                regex_results.get("allergies", [])
            ),
            "traitements_en_cours": self._deduplicate(
                regex_results.get("medicaments", []) +
                drbert_results.get("traitements", [])
            ),
            "constantes_biologiques": self._deduplicate(
                regex_results.get("constantes", [])
            ),
            "vaccinations": self._deduplicate(
                regex_results.get("vaccinations", [])
            ),
            "facteurs_risque": self._deduplicate(
                regex_results.get("facteurs_risque", [])
            ),
            "examens_bilans": self._deduplicate(
                regex_results.get("examens", [])
            ),
            "dates_importantes": self._filter_dates(
                self._deduplicate(regex_results.get("dates", []))
            ),
            "autres_entites": self._deduplicate(
                drbert_results.get("autres", [])
            ),
            "document_type": document_type,
            "extraction_method": (
                "hybrid_regex_drbert"
                if self._ner_pipeline
                else "regex_only"
            )
        }

        return result

    def _empty_result(self, reason: str) -> dict:
        return {
            "pathologies_actives": [],
            "antecedents_medicaux": [],
            "antecedents_chirurgicaux": [],
            "allergies_intolerances": [],
            "traitements_en_cours": [],
            "constantes_biologiques": [],
            "vaccinations": [],
            "facteurs_risque": [],
            "examens_bilans": [],
            "dates_importantes": [],
            "autres_entites": [],
            "document_type": "inconnu",
            "extraction_method": "none",
            "reason": reason
        }