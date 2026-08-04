import sys
import json
from pathlib import Path
from tqdm import tqdm

sys.path.append('/home/ahmed/medscan_vsm')

from src.pdf_handler import PDFHandler
from src.preprocessor import Preprocessor
from src.ocr_engine import OCREngine
from src.pipeline_tracker import PipelineTracker
from src.document_classifier import DocumentClassifier
from src.llm_fallback import LLMFallback


def test_pipeline(file_path: str):
    print("=" * 60)
    print("TEST PIPELINE MEDSCAN VSM")
    print("=" * 60)

    tracker = PipelineTracker(source_file=file_path)

    try:
        # ÉTAPE 0 : Détection format
        print("\n[0] Détection du format...")
        tracker.start_step("pdf_handler")
        pdf_handler = PDFHandler()
        document = pdf_handler.prepare_document(file_path)
        tracker.complete_step("pdf_handler", {
            "type": document['type'],
            "total_pages": document['total_pages']
        })
        print(f"    ✓ Type     : {document['type']}")
        print(f"    ✓ Pages    : {document['total_pages']}")

        preprocessor = Preprocessor()
        ocr_engine = OCREngine()
        llm_fallback = LLMFallback()

        # Zero-Shot activé — utilisé quand keywords insuffisants
        classifier = DocumentClassifier(use_zeroshot=True)

        pages_results = []

        for page_info in tqdm(document['pages'], desc="Traitement pages", unit="page"):
            page_num = page_info['page_number']
            page_path = page_info['path']
            print(f"\n--- Page {page_num}/{document['total_pages']} ---")

            # ÉTAPE 1 : Prétraitement
            print(f"\n[1] Prétraitement OpenCV...")
            tracker.start_step("preprocessor")
            try:
                preprocess_result = preprocessor.process(page_path)
                tracker.complete_step("preprocessor", {
                    "page": page_num,
                    "quality_score": preprocess_result['quality_score'],
                    "needs_llm_fallback": preprocess_result['needs_llm_fallback']
                })
                print(f"    ✓ Score qualité  : {preprocess_result['quality_score']}")
                print(f"    ✓ Fallback LLM   : {preprocess_result['needs_llm_fallback']}")
            except Exception as e:
                tracker.fail_step("preprocessor", str(e))
                raise

            # ÉTAPE 2 : OCR + Fallback Qwen si nécessaire
            print(f"\n[2] Extraction OCR...")
            tracker.start_step("ocr")
            ocr_result = None

            try:
                ocr_result = ocr_engine.extract_from_processed(preprocess_result)
                conf = ocr_result['confidence']

                # Vérification fallback nécessaire
                needs_fallback, fallback_reason = llm_fallback.should_fallback(
                    quality_score=preprocess_result['quality_score'],
                    ocr_confidence=conf['global_score']
                )

                if needs_fallback:
                    # PaddleOCR insuffisant → Qwen prend le relais
                    tracker.complete_step("ocr", {
                        "page": page_num,
                        "skipped": True,
                        "reason": fallback_reason
                    })
                    print(f"    ⚠️ Fallback déclenché : {fallback_reason}")

                    tracker.start_step("llm_fallback")
                    ocr_result = llm_fallback.extract_text(
                        preprocess_result['processed_path']
                    )
                    conf = ocr_result['confidence']
                    tracker.complete_step("llm_fallback", {
                        "page": page_num,
                        "method": ocr_result.get('method'),
                        "global_score": conf['global_score'],
                        "words_extracted": len(ocr_result['full_text'].split())
                    })
                    print(f"    ✓ Méthode         : {ocr_result.get('method')}")
                    print(f"    ✓ Mots extraits   : {len(ocr_result['full_text'].split())}")

                else:
                    # PaddleOCR suffisant
                    tracker.complete_step("ocr", {
                        "page": page_num,
                        "global_score": conf['global_score'],
                        "mean_confidence": conf['mean_confidence'],
                        "min_confidence": conf['min_confidence'],
                        "low_confidence_ratio": conf['low_confidence_ratio'],
                        "total_blocks": ocr_result.get('total_blocks', 0)
                    })
                    tracker.skip_step(
                        "llm_fallback",
                        f"PaddleOCR suffisant (score={conf['global_score']})"
                    )
                    print(f"    ✓ Score composite : {conf['global_score']}")
                    print(f"    ✓ Moyenne conf    : {conf['mean_confidence']}")
                    print(f"    ✓ Minimum conf    : {conf['min_confidence']}")
                    print(f"    ✓ Blocs détectés  : {ocr_result.get('total_blocks', 0)}")
                    ling = conf.get('linguistic_quality', {})
                    print(f"    ✓ Score linguistique   : {ling.get('linguistic_score', 'N/A')}")
                    print(f"    ✓ Texte lisible        : {ling.get('is_readable', 'N/A')}")
                    print(f"    ✓ Pénalité appliquée   : {conf.get('linguistic_penalty_applied', False)}")

            except Exception as e:
                tracker.fail_step("ocr", str(e))
                raise

            # ÉTAPE 3 : Classification
            print(f"\n[3] Classification documentaire...")
            tracker.start_step("classification")
            try:
                classification = classifier.classify(ocr_result['full_text'])
                tracker.complete_step("classification", {
                    "page": page_num,
                    "predicted_type": classification['predicted_type'],
                    "score": classification['score'],
                    "method": classification['method'],
                    "confident": classification['confident'],
                    "zeroshot_used": classification.get('zeroshot_used', False)
                })
                print(f"    ✓ Type détecté   : {classification['label_fr']}")
                print(f"    ✓ Score          : {classification['score']}")
                print(f"    ✓ Méthode        : {classification['method']}")
                print(f"    ✓ Zero-Shot used : {classification.get('zeroshot_used', False)}")
                print(f"    ✓ Confiant       : {classification['confident']}")
            except Exception as e:
                tracker.fail_step("classification", str(e))
                raise

            pages_results.append({
                "page_number": page_num,
                "page_path": page_path,
                "quality_score": preprocess_result['quality_score'],
                "full_text": ocr_result['full_text'],
                "confidence": ocr_result['confidence'],
                "blocks": ocr_result['blocks'],
                "needs_llm_fallback": ocr_result['needs_llm_fallback'],
                "total_blocks": ocr_result.get('total_blocks', 0),
                "ocr_method": ocr_result.get('method', 'paddleocr'),
                "classification": classification
            })

        # Fusion texte toutes pages
        full_document_text = "\n".join([p['full_text'] for p in pages_results])

        # Type dominant du document
        all_types = [p['classification']['predicted_type'] for p in pages_results]
        dominant_type = max(set(all_types), key=all_types.count)

        # Statistiques globales
        ocr_methods = [p['ocr_method'] for p in pages_results]
        zeroshot_used = sum(
            1 for p in pages_results
            if p['classification'].get('zeroshot_used', False)
        )

        # Structure finale
        final_result = {
            "document_id": tracker.document_id,
            "source": file_path,
            "type": document['type'],
            "document_type": dominant_type,
            "total_pages": document['total_pages'],
            "stats": {
                "pages_paddleocr": ocr_methods.count('paddleocr'),
                "pages_qwen": sum(1 for m in ocr_methods if 'qwen' in str(m)),
                "pages_zeroshot": zeroshot_used,
                "pages_inconnu": all_types.count('inconnu')
            },
            "pages": pages_results,
            "full_text": full_document_text,
            "tracking": tracker.get_summary()
        }

        # Sauvegarde JSON
        output_json = Path("data/processed") / f"{Path(file_path).stem}_ocr_result.json"
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2, default=str)

        # Affichage résumé
        print(f"\n{'=' * 60}")
        print(f"RÉSUMÉ FINAL")
        print(f"{'=' * 60}")
        print(f"✓ Type dominant        : {dominant_type}")
        print(f"✓ Pages PaddleOCR      : {final_result['stats']['pages_paddleocr']}")
        print(f"✓ Pages Qwen fallback  : {final_result['stats']['pages_qwen']}")
        print(f"✓ Pages Zero-Shot      : {final_result['stats']['pages_zeroshot']}")
        print(f"✓ Pages non classifiées: {final_result['stats']['pages_inconnu']}")
        print(f"✓ Résultat sauvegardé  : {output_json}")

        tracker.complete_pipeline()
        return final_result

    except Exception as e:
        tracker.fail_pipeline(str(e))
        raise


if __name__ == "__main__":
    result = test_pipeline("data/raw/BANANE_Sophie.pdf")