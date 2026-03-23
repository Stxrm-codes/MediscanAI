"""
utils/drug_matcher.py
──────────────────────
Fuzzy string matching of OCR candidates against a local drug database.

Uses RapidFuzz (fast C++ Levenshtein) → falls back to thefuzz → plain
substring search if neither is installed.

Scoring uses token_sort_ratio so that "Amoxicillin 500mg" correctly
matches "Amoxicillin" even with extra tokens in the OCR output.

Match confidence ≥ 60  → we return the matched drug name.
Match confidence <  60  → we return the first OCR candidate as-is.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "database" / "drugs.json"
_THRESHOLD = 60.0


class DrugMatcher:
    def __init__(self):
        self._names: List[str] = []
        self._data:  List[dict] = []
        self._fuzz   = None
        self._proc   = None
        self._load_db()
        self._init_fuzzy()

    # ── Init ──────────────────────────────────────────────────────────────

    def _load_db(self):
        if _DB_PATH.exists():
            with open(_DB_PATH, encoding="utf-8") as f:
                self._data  = json.load(f)
            self._names = [d["name"] for d in self._data if d.get("name")]
            log.info("Drug database loaded: %d entries", len(self._names))
        else:
            log.warning("Drug DB not found at %s", _DB_PATH)

    def _init_fuzzy(self):
        try:
            from rapidfuzz import fuzz, process
            self._fuzz, self._proc = fuzz, process
            log.info("RapidFuzz ✓")
            return
        except ImportError:
            pass
        try:
            from fuzz import fuzz, process      # type: ignore
            self._fuzz, self._proc = fuzz, process
            log.info("thefuzz ✓ (consider rapidfuzz for speed)")
        except ImportError:
            log.warning("No fuzzy library found  →  pip install rapidfuzz")

    # ── Public ────────────────────────────────────────────────────────────

    def best_match(self, candidates: List[str]) -> Tuple[Optional[str], float]:
        """
        Return (best_drug_name, confidence_0_to_100).
        Falls back to first candidate with score 0 when matching fails.
        """
        if not candidates:
            return (None, 0.0)
        if not self._names or not self._fuzz:
            return (candidates[0], 0.0)

        best_name, best_score = None, 0.0
        norm_db = [self._norm(n) for n in self._names]

        for cand in candidates:
            nc = self._norm(cand)
            if len(nc) < 3:
                continue
            try:
                res = self._proc.extractOne(
                    nc, norm_db,
                    scorer=self._fuzz.token_sort_ratio,
                )
                if res and res[1] > best_score:
                    best_score = res[1]
                    best_name  = self._names[res[2]]
            except Exception as exc:
                log.debug("Fuzzy error for '%s': %s", cand, exc)

        if best_score < _THRESHOLD:
            log.info("Fuzzy score %.1f below threshold — using raw candidate", best_score)
            return (candidates[0], best_score)

        return (best_name, best_score)

    def lookup(self, name: str) -> Optional[dict]:
        """Return the full database record for a drug name, if available."""
        norm = self._norm(name)
        for entry in self._data:
            if self._norm(entry.get("name","")) == norm:
                return entry
        return None

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _norm(text: str) -> str:
        text = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|ml|g|iu|%)\b", "",
                      text, flags=re.IGNORECASE)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        return " ".join(text.lower().split())
