"""
classifier.py
=============
Classifies documents into one of:
    Invoice | Resume | Utility Bill | Other | Unclassifiable

Strategy (two-tier):
    1. Keyword scoring  — fast, deterministic, works offline with zero extra models.
       Each category has a weighted keyword vocabulary. The category with the
       highest normalised score wins (if it clears a confidence threshold).

    2. Zero-shot fallback (optional, --use-zero-shot)
       Uses a HuggingFace zero-shot classification pipeline
       (e.g. typeform/distilbert-base-uncased-mnli) when keyword confidence is low.
       This still runs 100% locally — no API calls.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Category labels ────────────────────────────────────────────────────────────

CLASSES = ["Invoice", "Resume", "Utility Bill", "Other", "Unclassifiable"]

# ── Keyword vocabulary (term → weight) ────────────────────────────────────────
# Higher weight = stronger signal.  Regex patterns allowed.

KEYWORD_VOCAB: Dict[str, List[Tuple[str, float]]] = {
    "Invoice": [
        (r"\binvoice\b", 3.0),
        (r"\binvoice\s*(?:number|no\.?|#)", 4.0),
        (r"\binv[-_]?\d+", 3.5),
        (r"\bbill\s*to\b", 2.5),
        (r"\bship\s*to\b", 1.5),
        (r"\bpayment\s*due\b", 2.0),
        (r"\bsubtotal\b", 2.0),
        (r"\btax\b", 1.0),
        (r"\btotal\s*amount\b", 2.5),
        (r"\bamount\s*due\b", 2.5),
        (r"\bpurchase\s*order\b", 2.0),
        (r"\bunit\s*price\b", 2.0),
        (r"\bquantity\b", 1.0),
        (r"\bdue\s*date\b", 1.5),
        (r"\bremit\b", 1.5),
        (r"\bvat\b", 1.5),
        (r"\b(?:net|gross)\s+\d", 1.0),
    ],
    "Resume": [
        (r"\bresume\b", 3.0),
        (r"\bcurriculum\s+vitae\b", 3.5),
        (r"\b(?:work\s+)?experience\b", 2.0),
        (r"\bemployment\s+history\b", 2.5),
        (r"\beducation\b", 1.5),
        (r"\bskills?\b", 1.5),
        (r"\bcertification\b", 1.5),
        (r"\bsummary\b", 1.0),
        (r"\bobjective\b", 1.5),
        (r"\blinkedin\b", 2.0),
        (r"\bgithub\b", 1.5),
        (r"\breferences\b", 1.5),
        (r"\b(?:bachelor|master|phd|b\.?s\.?|m\.?s\.?|mba)\b", 2.0),
        (r"\b(?:gpa|cgpa)\b", 2.0),
        (r"\bprojects?\b", 1.0),
        (r"\bproficiency\b", 1.5),
        (r"\byears?\s+of\s+experience\b", 2.5),
    ],
    "Utility Bill": [
        (r"\butility\b", 2.5),
        (r"\baccount\s+(?:number|no\.?|#)", 3.0),
        (r"\bkwh?\b", 3.5),
        (r"\bkilowatt[- ]hour", 3.5),
        (r"\belectricity\b", 2.5),
        (r"\bgas\s+(?:bill|usage|consumption)\b", 2.5),
        (r"\bwater\s+(?:bill|usage|consumption)\b", 2.5),
        (r"\bservice\s+address\b", 2.0),
        (r"\bmeter\s+(?:reading|number)\b", 3.0),
        (r"\bstatement\s+(?:date|period)\b", 1.5),
        (r"\bbilling\s+period\b", 2.0),
        (r"\bcurrent\s+(?:charges?|usage)\b", 2.0),
        (r"\bprevious\s+balance\b", 2.0),
        (r"\bpayment\s+due\s+date\b", 1.5),
        (r"\bcustomer\s+(?:account|number|id)\b", 1.5),
        (r"\busage\b", 1.0),
        (r"\btherms?\b", 2.5),
        (r"\bwatts?\b", 1.5),
        (r"\bpower\s+(?:company|provider)\b", 2.0),
    ],
}

# Minimum confidence score to accept keyword-based classification
KEYWORD_THRESHOLD = 2.0
# If top category score is less than this multiple of second-best, use zero-shot
MARGIN_RATIO = 1.4


# ── Scoring ────────────────────────────────────────────────────────────────────

def _score_text(text: str) -> Dict[str, float]:
    """Compute raw keyword scores for each category."""
    text_lower = text.lower()
    scores: Dict[str, float] = {cat: 0.0 for cat in KEYWORD_VOCAB}

    for category, patterns in KEYWORD_VOCAB.items():
        for pattern, weight in patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                # Logarithmic diminishing returns for repeated matches
                import math
                scores[category] += weight * (1 + 0.5 * math.log(len(matches)))

    return scores


@dataclass
class ClassificationResult:
    doc_class: str
    confidence: float               # 0–1, normalised
    scores: Dict[str, float]        # raw keyword scores
    method: str                     # "keyword" | "zero_shot" | "fallback"


def classify_text(
    text: str,
    use_zero_shot: bool = False,
    zero_shot_pipeline=None,
) -> ClassificationResult:
    """
    Classify a single document's text.

    Args:
        text:               Cleaned document text.
        use_zero_shot:      Whether to use the zero-shot fallback.
        zero_shot_pipeline: A loaded HuggingFace pipeline (or None).

    Returns:
        ClassificationResult
    """
    if not text or not text.strip():
        return ClassificationResult(
            doc_class="Unclassifiable",
            confidence=0.0,
            scores={},
            method="fallback",
        )

    # ── Step 1: keyword scoring ──
    scores = _score_text(text)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_cat, top_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0

    total = sum(scores.values())
    confidence = top_score / total if total > 0 else 0.0

    keyword_ok = (
        top_score >= KEYWORD_THRESHOLD
        and (second_score == 0 or top_score / max(second_score, 0.01) >= MARGIN_RATIO)
    )

    if keyword_ok:
        return ClassificationResult(
            doc_class=top_cat,
            confidence=round(confidence, 3),
            scores=scores,
            method="keyword",
        )

    # ── Step 2: zero-shot fallback ──
    if use_zero_shot and zero_shot_pipeline is not None:
        try:
            # Truncate text to first 512 tokens worth of characters
            snippet = text[:2000]
            candidate_labels = ["Invoice", "Resume", "Utility Bill", "Other"]
            result = zero_shot_pipeline(snippet, candidate_labels=candidate_labels)
            best_label = result["labels"][0]
            best_score = result["scores"][0]
            return ClassificationResult(
                doc_class=best_label,
                confidence=round(best_score, 3),
                scores=scores,
                method="zero_shot",
            )
        except Exception as exc:
            logger.warning("Zero-shot classification failed: %s", exc)

    # ── Step 3: accept weak keyword result or mark Other/Unclassifiable ──
    if top_score > 0:
        # Accept the top guess only if margin over second place is clear enough,
        # otherwise fall back to "Other"
        if top_score >= 1.0 and (second_score == 0 or top_score / max(second_score, 0.01) >= 1.2):
            label = top_cat
        else:
            label = "Other"
        return ClassificationResult(
            doc_class=label,
            confidence=round(confidence, 3),
            scores=scores,
            method="keyword",
        )

    return ClassificationResult(
        doc_class="Unclassifiable",
        confidence=0.0,
        scores=scores,
        method="fallback",
    )


# ── ZeroShot loader (lazy) ─────────────────────────────────────────────────────

_zero_shot_pipeline = None


def load_zero_shot_pipeline(model_name: str = "typeform/distilbert-base-uncased-mnli"):
    """Load and cache a HuggingFace zero-shot classification pipeline."""
    global _zero_shot_pipeline
    if _zero_shot_pipeline is not None:
        return _zero_shot_pipeline

    try:
        from transformers import pipeline
        logger.info("Loading zero-shot model: %s (this may take a moment) …", model_name)
        _zero_shot_pipeline = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=-1,          # CPU only
        )
        logger.info("Zero-shot model loaded.")
    except Exception as exc:
        logger.error("Could not load zero-shot model (%s): %s", model_name, exc)
        _zero_shot_pipeline = None

    return _zero_shot_pipeline
