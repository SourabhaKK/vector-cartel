"""
security.py -- SecureOps Assistant input scanner (prompt injection gate).

Implements the InputScannerProtocol contract defined in src/contracts.py:
    scan(query: str) -> InputScanResult
    InputScanResult = Tuple[bool, Optional[str]]
        [0] bool          -- True = clean query, False = blocked
        [1] Optional[str] -- reason for blocking, None if clean

Sits in front of the agent graph (wired in src/agent.py as the entry-point
input_gate node) -- it runs BEFORE classify_query, so a blocked query never
reaches the LLM at all.

DESIGN: deterministic keyword/pattern matching, not an LLM call. This is a
deliberate choice -- a regex deny-list is fully testable, has zero latency
and zero API cost, and (unlike an LLM-based classifier) cannot itself be
manipulated by adversarial phrasing of the very input it's screening.
It will not catch every injection technique (no static deny-list does),
but it reliably blocks the common, named attack categories: instruction
override, system-prompt exfiltration, and role/jailbreak attempts.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

InputScanResult = Tuple[bool, Optional[str]]

CLEAN_SCAN_RESULT: InputScanResult = (True, None)

# Each pattern maps to a specific, named reason so a blocked response can
# explain *what category* of attack was detected, not just that something
# was blocked.
_INJECTION_PATTERNS = [
    (r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?\b", "instruction override attempt"),
    (r"\bdisregard\s+(all\s+)?(the\s+)?(above|prior|your)\s+(rules?|instructions?)\b", "instruction override attempt"),
    (r"\bforget\s+(your|all)\s+(instructions?|rules?)\b", "instruction override attempt"),
    (r"\b(reveal|show|print|repeat)\s+(your\s+)?system\s+prompt\b", "system prompt exfiltration attempt"),
    (r"\byou\s+are\s+now\s+(dan|jailbroken)\b", "jailbreak / role override attempt"),
    (r"\bact\s+as\s+(an?\s+)?(unrestricted|jailbroken|uncensored)\b", "jailbreak / role override attempt"),
    (r"\bpretend\s+you\s+have\s+no\s+(safety\s+)?(guidelines?|restrictions?|rules?)\b", "jailbreak / role override attempt"),
    (r"\bbypass\s+your\s+(safety|security)\s+(rules?|guidelines?)\b", "instruction override attempt"),
    (r"^\s*system\s*:\s*new\s+instructions?\b", "instruction override attempt"),
]

_COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), reason) for p, reason in _INJECTION_PATTERNS]


class InputScanner:
    """Deterministic keyword/pattern-based prompt injection scanner."""

    def scan(self, query: str) -> InputScanResult:
        try:
            if not isinstance(query, str):
                return (False, "query is not a string")
            if not query.strip():
                return CLEAN_SCAN_RESULT
            for pattern, reason in _COMPILED_PATTERNS:
                if pattern.search(query):
                    return (False, reason)
            return CLEAN_SCAN_RESULT
        except Exception:  # noqa: BLE001 - contract: never raise to the caller
            return (False, "scan failed, blocking defensively")
