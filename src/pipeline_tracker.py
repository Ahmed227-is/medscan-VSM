import json
import uuid
import time
from datetime import datetime
from pathlib import Path
from enum import Enum


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineTracker:
    """
    Suivi en temps réel de chaque étape du pipeline.
    Génère les logs RGPD et les données pour l'interface Streamlit.
    """

    STEPS = [
        "pdf_handler",
        "preprocessor",
        "ocr",
        "llm_fallback",
        "classification",
        "ner",
        "structuration",
        "vsm_generation"
    ]

    def __init__(self, source_file: str, logs_dir: str = "logs"):
        self.document_id = self._generate_id()
        self.source_file = source_file
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "pipeline").mkdir(exist_ok=True)
        (self.logs_dir / "errors").mkdir(exist_ok=True)
        (self.logs_dir / "audit").mkdir(exist_ok=True)

        self.log_path = self.logs_dir / "pipeline" / f"{self.document_id}.json"
        self._step_start_times = {}

        # Initialisation du log
        self.data = {
            "document_id": self.document_id,
            "source": source_file,
            "submitted_at": self._now(),
            "status": "processing",
            "steps": {
                step: {
                    "status": StepStatus.PENDING.value,
                    "started_at": None,
                    "completed_at": None,
                    "duration_seconds": None,
                    "details": {},
                    "error": None
                }
                for step in self.STEPS
            },
            "audit_trail": [],
            "errors": [],
            "completed_at": None,
            "total_duration_seconds": None
        }

        self._add_audit("document_submitted", f"Document soumis : {source_file}")
        self._save()
        print(f"    📋 Tracker initialisé : {self.document_id}")

    def _generate_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique = str(uuid.uuid4())[:8]
        return f"doc_{timestamp}_{unique}"

    def _now(self) -> str:
        return datetime.now().isoformat()

    def _save(self):
        """Sauvegarde le log JSON en temps réel."""
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def _add_audit(self, action: str, details: str):
        """Ajoute une entrée dans l'audit trail RGPD."""
        self.data["audit_trail"].append({
            "timestamp": self._now(),
            "action": action,
            "details": details
        })

    def start_step(self, step: str):
        """Marque le début d'une étape."""
        if step not in self.STEPS:
            return
        self._step_start_times[step] = time.time()
        self.data["steps"][step]["status"] = StepStatus.RUNNING.value
        self.data["steps"][step]["started_at"] = self._now()
        self._add_audit(f"{step}_started", f"Étape {step} démarrée")
        self._save()
        print(f"    ⟳ {step} démarré...")

    def complete_step(self, step: str, details: dict = None):
        """Marque la fin réussie d'une étape."""
        if step not in self.STEPS:
            return
        duration = round(time.time() - self._step_start_times.get(step, time.time()), 3)
        self.data["steps"][step]["status"] = StepStatus.COMPLETED.value
        self.data["steps"][step]["completed_at"] = self._now()
        self.data["steps"][step]["duration_seconds"] = duration
        if details:
            self.data["steps"][step]["details"] = details
        self._add_audit(
            f"{step}_completed",
            f"Étape {step} terminée en {duration}s"
        )
        self._save()
        print(f"    ✓ {step} terminé en {duration}s")

    def fail_step(self, step: str, error: str, correction: str = None):
        """Marque l'échec d'une étape avec détail de l'erreur."""
        if step not in self.STEPS:
            return
        duration = round(time.time() - self._step_start_times.get(step, time.time()), 3)
        error_entry = {
            "step": step,
            "timestamp": self._now(),
            "error": error,
            "correction_applied": correction,
            "duration_before_fail": duration
        }
        self.data["steps"][step]["status"] = StepStatus.FAILED.value
        self.data["steps"][step]["error"] = error
        self.data["steps"][step]["duration_seconds"] = duration
        self.data["errors"].append(error_entry)

        # Log erreur séparé
        error_path = self.logs_dir / "errors" / f"{self.document_id}_{step}.json"
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump(error_entry, f, ensure_ascii=False, indent=2)

        self._add_audit(f"{step}_failed", f"Erreur : {error}")
        if correction:
            self._add_audit(f"{step}_corrected", f"Correction : {correction}")
        self._save()
        print(f"    ❌ {step} échoué : {error}")

    def skip_step(self, step: str, reason: str):
        """Marque une étape comme ignorée."""
        if step not in self.STEPS:
            return
        self.data["steps"][step]["status"] = StepStatus.SKIPPED.value
        self.data["steps"][step]["details"] = {"reason": reason}
        self._add_audit(f"{step}_skipped", reason)
        self._save()
        print(f"    ⏭ {step} ignoré : {reason}")

    def complete_pipeline(self):
        """Marque le pipeline complet comme terminé."""
        start = datetime.fromisoformat(self.data["submitted_at"])
        total = round((datetime.now() - start).total_seconds(), 3)
        self.data["status"] = "completed"
        self.data["completed_at"] = self._now()
        self.data["total_duration_seconds"] = total
        self._add_audit("pipeline_completed", f"Pipeline terminé en {total}s")
        self._save()
        print(f"\n    🏁 Pipeline terminé en {total}s")
        print(f"    📁 Log sauvegardé : {self.log_path}")

    def fail_pipeline(self, reason: str):
        """Marque le pipeline comme échoué."""
        self.data["status"] = "failed"
        self.data["completed_at"] = self._now()
        self._add_audit("pipeline_failed", reason)
        self._save()
        print(f"\n    💥 Pipeline échoué : {reason}")

    def get_summary(self) -> dict:
        """
        Retourne un résumé pour l'interface Streamlit.
        """
        completed = sum(
            1 for s in self.data["steps"].values()
            if s["status"] == StepStatus.COMPLETED.value
        )
        failed = sum(
            1 for s in self.data["steps"].values()
            if s["status"] == StepStatus.FAILED.value
        )
        return {
            "document_id": self.document_id,
            "status": self.data["status"],
            "steps_completed": completed,
            "steps_failed": failed,
            "total_steps": len(self.STEPS),
            "errors": self.data["errors"],
            "total_duration": self.data["total_duration_seconds"]
        }