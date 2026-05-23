"""
ingestion.py
============
Handles reading PDF and plain-text files from a folder, extracting
and cleaning the raw text content.

Supported formats:
    • .pdf  — via pdfminer.six (layout-aware) with PyPDF2 as fallback
    • .txt  — direct read
    • .md   — direct read
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── optional imports (graceful degradation) ───────────────────────────────────

try:
    from pdfminer.high_level import extract_text as pdfminer_extract
    from pdfminer.pdfparser import PDFSyntaxError
    _HAS_PDFMINER = True
except ImportError:
    _HAS_PDFMINER = False
    logger.warning("pdfminer.six not installed — PDF support limited to PyPDF2 fallback.")

try:
    import pypdf
    _HAS_PYPDF = True
except ImportError:
    try:
        import PyPDF2 as pypdf          # legacy name
        _HAS_PYPDF = True
    except ImportError:
        _HAS_PYPDF = False
        logger.warning("pypdf / PyPDF2 not installed — PDF fallback unavailable.")


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class Document:
    """Represents a single ingested document."""
    filename: str           # basename, e.g. "invoice_1.pdf"
    filepath: Path          # absolute path
    raw_text: str           # extracted text (may be empty on failure)
    clean_text: str = ""    # cleaned / normalised text
    pages: int = 0          # number of pages (PDF only)
    error: Optional[str] = None  # non-None if extraction failed

    def __post_init__(self):
        if self.raw_text and not self.clean_text:
            self.clean_text = clean_text(self.raw_text)


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_pdf_text(filepath: Path) -> tuple[str, int]:
    """
    Extract text from a PDF.
    Tries pdfminer.six first (better layout handling), falls back to pypdf.

    Returns:
        (text, page_count)
    """
    text = ""
    pages = 0

    # --- pdfminer.six ---
    if _HAS_PDFMINER:
        try:
            text = pdfminer_extract(str(filepath)) or ""
            # Get page count separately if text was extracted
            if text.strip():
                # Estimate pages from form-feeds inserted by pdfminer
                pages = max(1, text.count("\x0c") + 1)
                return text, pages
        except (PDFSyntaxError, Exception) as exc:
            logger.debug("pdfminer failed on %s: %s — trying pypdf", filepath.name, exc)

    # --- pypdf fallback ---
    if _HAS_PYPDF:
        try:
            reader = pypdf.PdfReader(str(filepath))
            pages = len(reader.pages)
            parts = []
            for page in reader.pages:
                parts.append(page.extract_text() or "")
            text = "\n".join(parts)
            return text, pages
        except Exception as exc:
            logger.error("pypdf failed on %s: %s", filepath.name, exc)

    return text, pages


def extract_text_file(filepath: Path) -> str:
    """Read a plain-text file, trying common encodings."""
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return filepath.read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return filepath.read_bytes().decode("ascii", errors="replace")


# ── Text cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise extracted text:
      • Replace form-feeds and unusual whitespace
      • Collapse multiple blank lines to one
      • Strip leading/trailing whitespace per line
      • Remove null bytes
    """
    if not text:
        return ""

    text = text.replace("\x0c", "\n")   # PDF form-feed → newline
    text = text.replace("\x00", "")     # null bytes
    text = re.sub(r"\r\n?", "\n", text) # normalise line-endings

    # Collapse sequences of whitespace-only lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


# ── Folder ingestion ───────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".text"}


def ingest_folder(folder: Path) -> List[Document]:
    """
    Walk a folder (non-recursive by default) and ingest all supported files.

    Args:
        folder: Path to the directory.

    Returns:
        List of Document objects, one per file.
    """
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    docs: List[Document] = []
    files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        logger.warning("No supported files found in %s", folder)
        return docs

    logger.info("Found %d files in %s", len(files), folder)

    for filepath in files:
        logger.info("  Ingesting: %s", filepath.name)
        doc = _ingest_file(filepath)
        docs.append(doc)
        status = "OK" if not doc.error else f"ERROR: {doc.error}"
        logger.debug("    %s — chars=%d  pages=%d  status=%s",
                     filepath.name, len(doc.clean_text), doc.pages, status)

    return docs


def _ingest_file(filepath: Path) -> Document:
    """Extract and clean text from a single file."""
    ext = filepath.suffix.lower()
    raw_text = ""
    pages = 0
    error = None

    try:
        if ext == ".pdf":
            raw_text, pages = extract_pdf_text(filepath)
            if not raw_text.strip():
                error = "Empty or unreadable PDF (possibly scanned image)"
        elif ext in (".txt", ".md", ".text"):
            raw_text = extract_text_file(filepath)
            pages = 1
        else:
            error = f"Unsupported extension: {ext}"
    except Exception as exc:
        error = str(exc)
        logger.exception("Failed to ingest %s", filepath.name)

    return Document(
        filename=filepath.name,
        filepath=filepath,
        raw_text=raw_text,
        clean_text=clean_text(raw_text),
        pages=pages,
        error=error,
    )
