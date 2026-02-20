from __future__ import annotations

import re
from typing import Dict, List

FORBIDDEN_PATTERNS = [
    r"yield\s+farming",
    r"liquidity\s+pool",
    r"staking\s+rewards?",
    r"defi\s+apr",
    r"impermanent\s+loss",
]

REQUIRED_CONCEPTS = {
    "public funds": [r"public\s+funds", r"treasury\s+funds?", r"taxpayer\s+funds?"],
    "sovereign": [r"sovereign", r"national\s+control"],
    "legal": [r"legal", r"lawful", r"regulatory"],
    "audit": [r"audit", r"oversight", r"traceability"],
}



def _find_matches(patterns: List[str], text: str) -> List[str]:
    return [pattern for pattern in patterns if re.search(pattern, text)]



def check_policy(text: str) -> Dict[str, List[str]]:
    normalized = text.lower()

    violations = _find_matches(FORBIDDEN_PATTERNS, normalized)

    missing = []
    for concept, patterns in REQUIRED_CONCEPTS.items():
        if not any(re.search(pattern, normalized) for pattern in patterns):
            missing.append(concept)

    return {"violations": violations, "missing": missing}
