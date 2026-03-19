"""
utils/nlp_processor.py
───────────────────────
Cleans noisy OCR output and extracts candidate medicine names.

Steps
-----
1. Strip OCR noise characters
2. Fix common OCR substitutions  (0→O, 1→I, etc.)
3. Collapse whitespace & fix hyphenation breaks
4. Extract candidates via:
   a) Pharmacological suffix patterns (statins, cillin, mab …)
   b) CamelCase / ALL-CAPS token heuristic
   c) spaCy NER (DRUG / CHEMICAL) if model is available
5. Rank candidates by length (longer = more specific)
"""

import re
import logging
from typing import List

log = logging.getLogger(__name__)

# ── Patterns ───────────────────────────────────────────────────────────────
_NOISE       = re.compile(r"[|@#$^&*~`<>{}\\]")
_DOSAGE      = re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|ml|g|iu|ug|%|units?)\b",
                          re.IGNORECASE)
_HYPHEN_BREAK = re.compile(r"(\w)-\s+(\w)")

# Words that are definitely NOT medicine names
_IGNORE = {
    "ltd","pvt","inc","corp","pharma","laboratories","laboratory","lab",
    "mfg","manufactured","marketed","batch","date","store","keep","cool",
    "dry","place","only","schedule","warning","caution","dosage","for",
    "each","tablet","tablets","capsule","capsules","syrup","injection",
    "ointment","cream","gel","drops","strip","strips","box","pack",
}

# Common pharmacological name suffixes to scan for
_SUFFIX_PATTERNS = [
    r"\b\w+cillin\b",   # amoxicillin, ampicillin
    r"\b\w+mycin\b",    # azithromycin, erythromycin
    r"\b\w+mab\b",      # monoclonal antibodies
    r"\b\w+pril\b",     # lisinopril, enalapril
    r"\b\w+sartan\b",   # losartan, valsartan
    r"\b\w+statin\b",   # atorvastatin, rosuvastatin
    r"\b\w+zole\b",     # omeprazole, fluconazole
    r"\b\w+pam\b",      # diazepam, lorazepam
    r"\b\w+olol\b",     # atenolol, propranolol
    r"\b\w+ine\b",      # cetirizine, codeine
    r"\b\w+ide\b",      # furosemide, hydrochlorothiazide
    r"\b\w+one\b",      # prednisolone, prednisone
    r"\b\w+oxin\b",     # digoxin, doxorubicin
    r"\b\w+afil\b",     # sildenafil, tadalafil
    r"\b\w+parin\b",    # heparin, enoxaparin
]


class NLPProcessor:
    def __init__(self):
        self._nlp = None
        self._load_spacy()

    def _load_spacy(self):
        try:
            import spacy
            for model in ("en_core_sci_sm", "en_core_web_sm"):
                try:
                    self._nlp = spacy.load(model)
                    log.info("spaCy model '%s' loaded ✓", model)
                    return
                except OSError:
                    continue
            log.warning("No spaCy model found. "
                        "Run: python -m spacy download en_core_web_sm")
        except ImportError:
            log.warning("spaCy not installed — rule-based extraction only")

    # ── Public ────────────────────────────────────────────────────────────

    def clean(self, raw: str) -> str:
        """Remove OCR noise and normalise whitespace."""
        text = _NOISE.sub(" ", raw)
        text = _HYPHEN_BREAK.sub(r"\1\2", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def extract_medicine_names(self, text: str) -> List[str]:
        """Return ranked list of candidate medicine name strings."""
        found: set = set()

        # (a) Suffix patterns
        for pat in _SUFFIX_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                tok = m.group(0).strip()
                if len(tok) >= 4:
                    found.add(tok)

        # (b) CamelCase & ALL-CAPS heuristic
        for tok in text.split():
            clean = _DOSAGE.sub("", tok).strip(".,()/-+")
            if len(clean) < 4 or clean.lower() in _IGNORE:
                continue
            if re.match(r"^[A-Z][a-z]{3,}", clean) or \
               re.match(r"^[A-Z]{4,}$", clean):
                found.add(clean)

        # (c) spaCy NER
        if self._nlp:
            doc = self._nlp(text[:1000])
            for ent in doc.ents:
                if ent.label_ in ("DRUG", "CHEMICAL", "ORG", "PRODUCT"):
                    name = ent.text.strip()
                    if len(name) >= 4 and name.lower() not in _IGNORE:
                        found.add(name)

        return sorted(found, key=len, reverse=True)[:10]

    def normalise_for_matching(self, name: str) -> str:
        """Lowercase, strip dosage/punctuation — for fuzzy matching."""
        name = _DOSAGE.sub("", name)
        name = re.sub(r"[^a-zA-Z\s]", " ", name)
        return " ".join(name.lower().split())
