
FORBIDDEN_KEYWORDS = ["yield farming", "liquidity pool", "staking rewards"]
REQUIRED_TERMS = ["public funds", "sovereign", "legal", "audit"]

def check_policy(text):
    violations = [k for k in FORBIDDEN_KEYWORDS if k in text.lower()]
    missing = [k for k in REQUIRED_TERMS if k not in text.lower()]
    return {"violations": violations, "missing": missing}
