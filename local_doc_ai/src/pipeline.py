"""
pipeline.py
===========
Orchestrates the full document processing pipeline:
    ingestion → classification → extraction → (index building)

Produces structured results that are written to output.json.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ingestion import Document, ingest_folder
from .classifier import (
    ClassificationResult,
    classify_text,
    load_zero_shot_pipeline,
)
from .extractor import extract_fields

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """A document after classification and extraction."""
    filename: str
    filepath: Path
    doc_class: str
    confidence: float
    classification_method: str
    extracted_fields: Dict[str, Any]
    clean_text: str = ""
    pages: int = 0
    error: Optional[str] = None

    def to_output_dict(self) -> Dict[str, Any]:
        """
        Produce the JSON-serialisable record for output.json.
        Schema:  {"class": "...", field1: val1, ...}
        """
        record: Dict[str, Any] = {"class": self.doc_class}

        if self.error:
            record["_error"] = self.error
        if self.doc_class not in ("Other", "Unclassifiable"):
            record.update(self.extracted_fields)

        return record


class DocumentPipeline:
    """
    Full pipeline: ingest → classify → extract.

    After calling run(), the `.documents` attribute holds a list of
    ProcessedDocument objects (which also carry .clean_text for retrieval).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_zero_shot: bool = False,
        zero_shot_model: str = "typeform/distilbert-base-uncased-mnli",
    ):
        self.model_name = model_name
        self.use_zero_shot = use_zero_shot
        self.zero_shot_model = zero_shot_model
        self.documents: List[ProcessedDocument] = []

        self._zero_shot_pipeline = None
        if use_zero_shot:
            self._zero_shot_pipeline = load_zero_shot_pipeline(zero_shot_model)

    def run(self, folder: Path) -> Dict[str, Any]:
        """
        Process all documents in folder.

        Returns:
            output_json dict  {filename: {class, ...fields}}
        """
        raw_docs: List[Document] = ingest_folder(folder)

        if not raw_docs:
            logger.warning("No documents ingested from %s", folder)
            return {}

        self.documents = []
        output: Dict[str, Any] = {}

        for doc in raw_docs:
            logger.info("Processing: %s", doc.filename)

            if doc.error and not doc.clean_text:
                processed = ProcessedDocument(
                    filename=doc.filename,
                    filepath=doc.filepath,
                    doc_class="Unclassifiable",
                    confidence=0.0,
                    classification_method="error",
                    extracted_fields={},
                    clean_text="",
                    pages=doc.pages,
                    error=doc.error,
                )
            else:
                # ── Classify ──────────────────────────────────────────────
                clf: ClassificationResult = classify_text(
                    text=doc.clean_text,
                    use_zero_shot=self.use_zero_shot,
                    zero_shot_pipeline=self._zero_shot_pipeline,
                )
                logger.info(
                    "  → Class: %-15s  conf=%.3f  method=%s",
                    clf.doc_class, clf.confidence, clf.method
                )

                # ── Extract ───────────────────────────────────────────────
                fields = extract_fields(clf.doc_class, doc.clean_text)

                processed = ProcessedDocument(
                    filename=doc.filename,
                    filepath=doc.filepath,
                    doc_class=clf.doc_class,
                    confidence=clf.confidence,
                    classification_method=clf.method,
                    extracted_fields=fields,
                    clean_text=doc.clean_text,
                    pages=doc.pages,
                    error=doc.error,
                )

            self.documents.append(processed)
            output[processed.filename] = processed.to_output_dict()

        return output
