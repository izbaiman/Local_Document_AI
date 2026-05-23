"""
extractor.py
============
Extracts structured fields from classified documents using regex patterns
and heuristics.  No ML model required — runs fully offline.

Fields per document type:
    Invoice      → invoice_number, date, company, total_amount
    Resume       → name, email, phone, experience_years
    Utility Bill → account_number, date, usage_kwh, amount_due
    Other /
    Unclassifiable → {} (no extraction)
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

# ══════════════════════════════════════════════════════════════════════════════
# Shared helper patterns
# ══════════════════════════════════════════════════════════════════════════════

# ISO date: 2024-01-31  |  US date: 01/31/2024  |  Written: Jan 31, 2024
_DATE_PATTERNS = [
    # ISO
    r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
    # US numeric
    r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",
    # Written month name
    r"\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"[\s.,]+\d{1,2}[\s.,]+\d{2,4})\b",
    # Reversed written month
    r"\b(\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{2,4})\b",
]

# Dollar / currency amount
_AMOUNT_PATTERNS = [
    r"\$\s*([\d,]+\.?\d*)",
    r"([\d,]+\.?\d*)\s*(?:USD|usd)",
    r"(?:total|amount|balance|due)[^\d$]{0,30}\$?\s*([\d,]+\.\d{2})",
]

_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

_PHONE_RE = re.compile(
    r"(?:\+?\d[\d\s\-().]{7,}\d)"
)


def _find_date(text: str) -> Optional[str]:
    """Return the first date-like string found in text."""
    for pattern in _DATE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def _find_amount(text: str) -> Optional[float]:
    """Return the largest currency amount found (likely the total)."""
    amounts = []
    for pattern in _AMOUNT_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            try:
                val = float(m.group(1).replace(",", ""))
                amounts.append(val)
            except ValueError:
                pass

    # Also capture any dollar-sign amounts
    for m in re.finditer(r"\$\s*([\d,]+\.?\d*)", text):
        try:
            amounts.append(float(m.group(1).replace(",", "")))
        except ValueError:
            pass

    if not amounts:
        return None
    # Return the largest plausible amount (assumes total > line items)
    return round(max(amounts), 2)


# ══════════════════════════════════════════════════════════════════════════════
# Invoice extraction
# ══════════════════════════════════════════════════════════════════════════════

def _extract_invoice_number(text: str) -> Optional[str]:
    patterns = [
        r"invoice\s*(?:number|no\.?|#)\s*[:\-]?\s*([A-Za-z0-9\-_/]+)",
        r"inv\s*[-#]?\s*([A-Za-z0-9\-_/]+)",
        r"#\s*([A-Za-z0-9\-_]+)",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if len(val) >= 2:
                return val
    return None


def _extract_company(text: str) -> Optional[str]:
    """
    Heuristic: look for company name near "From:", "Vendor:", "Bill From:", etc.
    Falls back to the first capitalised proper-noun phrase in the document.
    """
    patterns = [
        r"(?:from|vendor|company|billed?\s+(?:by|from))\s*[:\-]?\s*([A-Z][^\n]{2,60})",
        r"(?:issued?\s+by)\s*[:\-]?\s*([A-Z][^\n]{2,60})",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # Fallback: first line that looks like a company name (capitalised, ≤5 words)
    for line in text.split("\n")[:20]:
        line = line.strip()
        if re.match(r"^[A-Z][A-Za-z &.,\-']+$", line) and 2 <= len(line.split()) <= 6:
            return line

    return None


def extract_invoice(text: str) -> Dict[str, Any]:
    return {
        "invoice_number": _extract_invoice_number(text),
        "date": _find_date(text),
        "company": _extract_company(text),
        "total_amount": _find_amount(text),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Resume extraction
# ══════════════════════════════════════════════════════════════════════════════

def _extract_name(text: str) -> Optional[str]:
    """
    Heuristic: the candidate's name is usually in the first 3–5 lines,
    before any section header.  We look for a 2–4 word title-case sequence.
    """
    header = "\n".join(text.split("\n")[:10])

    # Try explicit label first
    m = re.search(r"(?:name)\s*[:\-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", header)
    if m:
        return m.group(1).strip()

    # Look for a standalone Title Case line (2–4 words, no digits)
    for line in header.split("\n"):
        line = line.strip()
        words = line.split()
        if (
            2 <= len(words) <= 4
            and all(re.match(r"^[A-Z][a-z]+\.?$", w) for w in words)
        ):
            return line

    return None


def _extract_email(text: str) -> Optional[str]:
    m = _EMAIL_RE.search(text)
    return m.group(0).lower() if m else None


def _extract_phone(text: str) -> Optional[str]:
    """Return first phone-like sequence with at least 7 digits."""
    for m in _PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 7 <= len(digits) <= 15:
            return m.group(0).strip()
    return None


def _extract_experience_years(text: str) -> Optional[int]:
    """Parse lines like '5 years of experience' or 'Experience: 3+ years'."""
    patterns = [
        r"(\d+)\+?\s*years?\s+(?:of\s+)?(?:work\s+)?experience",
        r"experience[^\d]{0,20}(\d+)\+?\s*years?",
        r"(\d+)\+?\s*years?\s+(?:of\s+)?(?:professional|relevant)\s+experience",
        r"total\s+experience[^\d]{0,10}(\d+)",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return int(m.group(1))

    # Fallback: count distinct employment years from date ranges
    year_ranges = re.findall(
        r"(\d{4})\s*[-–—to]+\s*(present|\d{4})", text, re.IGNORECASE
    )
    if year_ranges:
        import datetime
        current_year = datetime.datetime.now().year
        total = 0
        for start, end in year_ranges:
            end_year = current_year if end.lower() == "present" else int(end)
            total += max(0, end_year - int(start))
        if total > 0:
            return total

    return None


def extract_resume(text: str) -> Dict[str, Any]:
    return {
        "name": _extract_name(text),
        "email": _extract_email(text),
        "phone": _extract_phone(text),
        "experience_years": _extract_experience_years(text),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Utility Bill extraction
# ══════════════════════════════════════════════════════════════════════════════

def _extract_account_number(text: str) -> Optional[str]:
    patterns = [
        r"account\s*(?:number|no\.?|#)\s*[:\-]?\s*([A-Za-z0-9\-_]+)",
        r"acct\.?\s*(?:#|no)?\s*[:\-]?\s*([A-Za-z0-9\-_]{4,20})",
        r"customer\s*(?:id|number|no)\s*[:\-]?\s*([A-Za-z0-9\-_]+)",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def _extract_kwh(text: str) -> Optional[float]:
    """
    Extract electricity / energy usage in kWh.
    Prioritises contextual patterns (usage / this period) over raw kWh mentions
    to avoid picking up cumulative meter readings.
    """
    # High-priority: labelled usage amount
    priority_patterns = [
        r"usage\s*(?:this\s+period)?[^\d]{0,20}([\d,]+\.?\d*)\s*kwh?",
        r"(?:consumption|used\s+this\s+period)[^\d]{0,20}([\d,]+\.?\d*)\s*kwh?",
        r"kwh?\s+equivalent[^\d]{0,10}([\d,]+\.?\d*)",
        r"therms?\s+used[^\d]{0,20}([\d,]+\.?\d*)",    # gas bills
        r"([\d,]+\.?\d*)\s*kwh?\s+(?:equivalent|estimate)",
    ]
    for p in priority_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                if val < 100_000:   # sanity: skip implausibly large meter totals
                    return round(val, 2)
            except ValueError:
                pass

    # Lower-priority: any kWh mention, capped to avoid meter totals
    for p in [r"([\d,]+\.?\d*)\s*kwh?", r"([\d,]+\.?\d*)\s*kilowatt[- ]hour"]:
        for m in re.finditer(p, text, re.IGNORECASE):
            try:
                val = float(m.group(1).replace(",", ""))
                if 0 < val < 100_000:
                    return round(val, 2)
            except ValueError:
                pass
    return None


def _extract_amount_due(text: str) -> Optional[float]:
    """Look for 'amount due', 'total due', 'balance due', or 'please pay'."""
    patterns = [
        r"(?:amount|total|balance|please\s+pay)\s+due[^\d$]{0,20}\$?\s*([\d,]+\.\d{2})",
        r"(?:total\s+)?charges?[^\d$]{0,20}\$?\s*([\d,]+\.\d{2})",
        r"\$\s*([\d,]+\.\d{2})\s*(?:due|total)",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            try:
                return round(float(m.group(1).replace(",", "")), 2)
            except ValueError:
                pass
    # Fallback to largest amount
    return _find_amount(text)


def extract_utility_bill(text: str) -> Dict[str, Any]:
    return {
        "account_number": _extract_account_number(text),
        "date": _find_date(text),
        "usage_kwh": _extract_kwh(text),
        "amount_due": _extract_amount_due(text),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Dispatcher
# ══════════════════════════════════════════════════════════════════════════════

def extract_fields(doc_class: str, text: str) -> Dict[str, Any]:
    """
    Route to the correct extractor based on classified document type.

    Returns:
        Dict of field → value (None if not found).
    """
    dispatch = {
        "Invoice": extract_invoice,
        "Resume": extract_resume,
        "Utility Bill": extract_utility_bill,
    }
    extractor = dispatch.get(doc_class)
    if extractor is None:
        return {}
    return extractor(text)
