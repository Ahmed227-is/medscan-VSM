"""
recover_from_checkpoint.py — MedScan VSM
Récupération MANUELLE après un crash en fin de pipeline (ex: crash dans
merge_entities(), comme vécu en pratique après 10h39 de traitement sur
143 pages).

⚠️ À UTILISER CONSCIEMMENT, PAS AUTOMATIQUEMENT ⚠️
Ce script réutilise les résultats NER déjà calculés PAGE PAR PAGE, tels
qu'ils étaient au moment du crash. Si tu as modifié ner_extractor.py
DEPUIS ce crash (correctif, nouveau prompt...), ce script va terminer le
traitement avec les ANCIENS résultats (pré-correctif), pas les nouveaux.

Utilise ce script UNIQUEMENT si :
  - le crash vient d'arriver, et
  - tu n'as fait AUCUN changement à ner_extractor.py depuis.

Sinon, relance le pipeline complet depuis zéro avec test_pipeline.py.

Usage :
    python3 recover_from_checkpoint.py data/raw/DATTE_Heloise.pdf
"""

import sys
import json
from pathlib import Path

sys.path.append('/home/ahmed/medscan_vsm')

from src.ner_extractor import NERExtractor
from src.vsm_generator import VSMGenerator


def recover(file_path: str):
    stem = Path(file_path).stem
    checkpoint_path = Path("data/processed") / f"{stem}_checkpoint.jsonl"

    if not checkpoint_path.exists():
        print(f"❌ Aucun checkpoint trouvé pour ce document : {checkpoint_path}")
        return

    print("=" * 60)
    print("RÉCUPÉRATION DEPUIS CHECKPOINT")
    print("=" * 60)
    print(f"\n⚠️  Vérifie que tu n'as PAS modifié ner_extractor.py depuis le crash.")
    reponse = input("Continuer la récupération ? (o/n) : ").strip().lower()
    if reponse != "o":
        print("Annulé.")
        return

    pages_results = []
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pages_results.append(json.loads(line))

    print(f"\n✓ {len(pages_results)} page(s) récupérée(s) depuis le checkpoint")

    ner = NERExtractor()
    vsm_generator = VSMGenerator()

    merged_ner = ner.merge_entities([p['ner'] for p in pages_results])
    merged_identity = ner.merge_patient_identity([p['ner']['identite_patient'] for p in pages_results])

    full_document_text = "\n".join([p['full_text'] for p in pages_results])
    all_types = [p['classification']['predicted_type'] for p in pages_results]
    dominant_type = max(set(all_types), key=all_types.count)

    ocr_methods = [p['ocr_method'] for p in pages_results]
    zeroshot_used = sum(1 for p in pages_results if p['classification'].get('zeroshot_used', False))

    ner_result_complet = {**merged_ner, "identite_patient": merged_identity}
    vsm = vsm_generator.generate(ner_result_complet)
    vsm_markdown = vsm_generator.render_markdown(vsm)

    final_result = {
        "source": file_path,
        "document_type": dominant_type,
        "total_pages": len(pages_results),
        "stats": {
            "pages_paddleocr": ocr_methods.count('paddleocr'),
            "pages_qwen": sum(1 for m in ocr_methods if 'qwen' in str(m)),
            "pages_zeroshot": zeroshot_used,
            "pages_inconnu": all_types.count('inconnu'),
            "pages_manual_review": sum(1 for p in pages_results if p.get('requires_manual_review', False)),
        },
        "ner_global": merged_ner,
        "identite_patient": merged_identity,
        "vsm": vsm,
        "pages": pages_results,
        "full_text": full_document_text,
        "recovered_from_checkpoint": True,
    }

    output_json = Path("data/processed") / f"{stem}_ocr_result.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2, default=str)

    output_vsm = Path("data/processed") / f"{stem}_VSM.md"
    with open(output_vsm, 'w', encoding='utf-8') as f:
        f.write(vsm_markdown)

    print(f"\n{'=' * 60}")
    print("VOLET DE SYNTHÈSE MÉDICALE (VSM)")
    print(f"{'=' * 60}")
    print(vsm_markdown)
    print(f"\n✓ JSON brut sauvegardé : {output_json}")
    print(f"✓ VSM sauvegardé       : {output_vsm}")

    checkpoint_path.unlink()
    print(f"✓ Checkpoint nettoyé (récupération terminée avec succès)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python3 recover_from_checkpoint.py <chemin_du_pdf>")
        sys.exit(1)
    recover(sys.argv[1])