import os
from pathlib import Path
from pdf2image import convert_from_path
import cv2
import numpy as np


class PDFHandler:
    """
    Convertit les documents PDF scannés en images
    pour le pipeline de traitement.
    Gère : PDF multi-pages, images uniques, lots d'images.
    """

    SUPPORTED_IMAGES = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
    SUPPORTED_DOCS = ['.pdf']

    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def is_pdf(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_DOCS

    def is_image(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_IMAGES

    def pdf_to_images(self, pdf_path: str, dpi: int = 300) -> list:
        """
        Convertit chaque page du PDF en image haute résolution.
        DPI 300 = qualité optimale pour l'OCR médical.
        Retourne la liste des chemins des images générées.
        """
        print(f"    Conversion PDF → images (DPI={dpi})...")

        try:
            pages = convert_from_path(
                pdf_path,
                dpi=dpi,
                fmt='PNG'
            )
        except Exception as e:
            raise RuntimeError(f"Erreur conversion PDF : {e}")

        image_paths = []
        pdf_name = Path(pdf_path).stem

        for i, page in enumerate(pages):
            page_path = self.output_dir / f"{pdf_name}_page_{i + 1}.png"
            page.save(str(page_path), 'PNG')
            image_paths.append({
                "page_number": i + 1,
                "path": str(page_path),
                "source": pdf_path
            })
            print(f"    ✓ Page {i + 1}/{len(pages)} convertie → {page_path.name}")

        return image_paths

    def load_as_numpy(self, image_path: str) -> np.ndarray:
        """Charge une image en tableau numpy pour OpenCV."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Impossible de charger : {image_path}")
        return image

    def prepare_single(self, file_path: str) -> dict:
        """
        Prépare un seul fichier (PDF ou image).
        Retourne toujours une structure normalisée avec liste de pages.
        """
        file_path = str(file_path)

        if not Path(file_path).exists():
            raise FileNotFoundError(f"Fichier introuvable : {file_path}")

        if self.is_pdf(file_path):
            print(f"    Format détecté : PDF")
            pages = self.pdf_to_images(file_path)
            return {
                "type": "pdf",
                "source": file_path,
                "pages": pages,
                "total_pages": len(pages)
            }

        elif self.is_image(file_path):
            print(f"    Format détecté : Image")
            return {
                "type": "image",
                "source": file_path,
                "pages": [{
                    "page_number": 1,
                    "path": file_path,
                    "source": file_path
                }],
                "total_pages": 1
            }

        else:
            raise ValueError(
                f"Format non supporté : {Path(file_path).suffix}\n"
                f"Formats acceptés : {self.SUPPORTED_IMAGES + self.SUPPORTED_DOCS}"
            )

    def prepare_batch(self, folder_path: str) -> list:
        """
        Traite un dossier entier contenant plusieurs documents.
        Utile quand un médecin a numérisé tout son cabinet.
        Retourne une liste de documents préparés.
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Dossier introuvable : {folder_path}")

        supported_extensions = self.SUPPORTED_IMAGES + self.SUPPORTED_DOCS
        files = [
            f for f in folder.iterdir()
            if f.suffix.lower() in supported_extensions
        ]

        if not files:
            raise ValueError(f"Aucun document supporté dans : {folder_path}")

        files = sorted(files)
        print(f"    {len(files)} document(s) trouvé(s) dans {folder_path}")

        documents = []
        for i, file in enumerate(files):
            print(f"\n    Document {i + 1}/{len(files)} : {file.name}")
            try:
                doc = self.prepare_single(str(file))
                documents.append(doc)
            except Exception as e:
                print(f"    ⚠️ Erreur sur {file.name} : {e} — ignoré")

        return documents

    def prepare_document(self, file_path: str) -> dict:
        """
        Point d'entrée principal — compatibilité avec l'ancien code.
        Accepte PDF ou image et retourne structure normalisée.
        """
        result = self.prepare_single(file_path)

        # Compatibilité avec test_pipeline.py existant
        # Convertit la liste de dicts en liste de chemins simples
        result['pages_paths'] = [p['path'] for p in result['pages']]
        return result