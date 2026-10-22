import re
import logging
from typing import Dict, Tuple

logger = logging.getLogger("QueryMasker")


class QueryMasker:
    """
    Redacts sensitive identifiers from query text before it is embedded and
    sent toward the cloud-side search path. Runs as a discrete pipeline step
    between receiving the request and generating the query embedding.

    NOTE: pattern set covers common PII shapes (email, SSN, phone, credit card).
    Confirm with the team whether "mask" in the original design also meant
    something else (e.g. embedding-level noise injection) — this module only
    covers text-level redaction, which is the safer default and easiest to
    extend if that's confirmed to be insufficient.
    """
    _patterns: Dict[str, re.Pattern] = {
        "EMAIL": re.compile(r"[\w\.-]+@[\w\.-]+\.\w+"),
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "PHONE": re.compile(r"\b(?:\+?\d{1,2}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    }

    @classmethod
    def mask(cls, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Returns (masked_text, redaction_counts) where redaction_counts maps
        each PII label to how many instances were redacted, for audit logging.
        """
        masked = text
        counts: Dict[str, int] = {}
        for label, pattern in cls._patterns.items():
            matches = pattern.findall(masked)
            if matches:
                counts[label] = len(matches)
                masked = pattern.sub(f"[{label}_REDACTED]", masked)

        if counts:
            logger.info(f"Masked query text: {counts}")

        return masked, counts