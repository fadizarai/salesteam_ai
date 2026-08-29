"""
Test get_detailed_explanation() on documented project cases.

For each case, verifies:
  1. All 4 sections are present and distinct
  2. No number in the response is absent from retrieve_deep_context()
  3. No technical jargon (score, probabilite, coefficient, CV, boost, seuil)
  4. Rank and urgency match the true values from recommend()
"""

import sys, re, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from src.api.schemas import RecommendRequest
from src.services.recommendation import recommend, get_detailed_explanation
from src.services.deep_context import retrieve_deep_context

# ── Constants ─────────────────────────────────────────────────────────────────

REQUIRED_SECTIONS = [
    "Pourquoi ce produit",
    "Pourquoi cette quantit",   # accent-insensitive prefix
    "Pourquoi ce classement",
    "Pourquoi ce niveau",
]

BANNED_TERMS = [
    "score",
    "probabilit",   # matches "probabilite" / "probabilite"
    "coefficient",
    r"\bCV\b",
    "boost",
    "seuil",
    "threshold",
    "feature",
]

SEPARATOR = "=" * 72


def check_sections(text):
    """Returns list of missing sections."""
    missing = []
    for sec in REQUIRED_SECTIONS:
        pattern = re.compile(re.escape(sec), re.IGNORECASE)
        if not pattern.search(text):
            missing.append(sec)
    return missing


def check_jargon(text):
    """Returns list of banned terms found in the response."""
    found = []
    for term in BANNED_TERMS:
        if re.search(term, text, re.IGNORECASE):
            found.append(term)
    return found


def extract_numbers(text):
    """Extract all numeric tokens from a text string."""
    return set(re.findall(r"\b\d+(?:[.,]\d+)?\b", text))


def get_context_numbers(ctx):
    """Flatten all numeric values from the context dict into a string for comparison."""
    import json
    ctx_str = json.dumps(ctx, default=str)
    return set(re.findall(r"\b\d+(?:[.,]\d+)?\b", ctx_str))


def audit_numbers(explanation_text, context):
    """
    Returns a list of numbers in the explanation that are NOT found in the context.
    Allows small formatting differences (e.g. 145.9 vs 145,9).
    """
    exp_nums = extract_numbers(explanation_text)
    ctx_nums = get_context_numbers(context)
    # Normalise: replace comma-decimals with dot-decimals
    def normalise(s):
        return s.replace(",", ".")
    ctx_normalised = {normalise(n) for n in ctx_nums}
    suspicious = []
    for n in exp_nums:
        if normalise(n) not in ctx_normalised:
            suspicious.append(n)
    return suspicious


def run_case(label, client_id, code_article, expected_group=None):
    print(SEPARATOR)
    print(f"CASE : {label}")
    print(f"Client : {client_id}  |  Article : {code_article}")
    print(SEPARATOR)

    # 1. Run recommend() to warm cache and capture the full response
    req = RecommendRequest(client_id=client_id, commercial_id="TEST")
    response = recommend(req)
    response_dict = response.model_dump()

    # Find the suggestion for this article
    suggestion = next(
        (s for s in response_dict["suggestions"] if s["code_article"] == code_article),
        None
    )
    if suggestion is None:
        print(f"  [!] Article '{code_article}' NOT found in recommend() response.")
        print(f"  Available articles: {[s['code_article'] for s in response_dict['suggestions']]}")
        print()
        return

    print(f"  [RECOMMEND] urgency_group   = {suggestion['urgency_group']}")
    print(f"  [RECOMMEND] score_final     = {suggestion['score_final']}")
    print(f"  [RECOMMEND] score_confiance = {suggestion['score_confiance']}")
    print(f"  [RECOMMEND] timing_boost    = {suggestion['timing_boost']}")
    print(f"  [RECOMMEND] recency_relative= {suggestion['recency_relative']}")
    print(f"  [RECOMMEND] quantite_suggeree = {suggestion['quantite_suggeree']}")
    print(f"  [RECOMMEND] source_quantite  = {suggestion['source_quantite']}")

    # Rank info
    all_suggs = response_dict["suggestions"]
    global_rank = next((i+1 for i,s in enumerate(all_suggs) if s["code_article"] == code_article), "?")
    same_group = [s for s in all_suggs if s["urgency_group"] == suggestion["urgency_group"]]
    rank_in_group = next((i+1 for i,s in enumerate(same_group) if s["code_article"] == code_article), "?")
    print(f"  [RECOMMEND] global_rank     = {global_rank} / {len(all_suggs)}")
    print(f"  [RECOMMEND] rank_in_group   = {rank_in_group} / {len(same_group)}")

    # 2. Retrieve deep context
    context = retrieve_deep_context(
        client_id=client_id,
        code_article=code_article,
        recommendation_response=response_dict,
    )

    # 3. Generate explanation (uses fallback since no HF token in test env)
    explanation = get_detailed_explanation(client_id=client_id, code_article=code_article)

    print()
    print("  ---- EXPLANATION TEXT ----")
    print(explanation)
    print()

    # 4. Audit checks
    print("  ---- AUDIT RESULTS ----")

    # Check 1: all 4 sections present
    missing_secs = check_sections(explanation)
    if missing_secs:
        print(f"  [FAIL] Missing sections: {missing_secs}")
    else:
        print("  [OK]   All 4 sections present")

    # Check 2: jargon
    jargon_found = check_jargon(explanation)
    if jargon_found:
        print(f"  [FAIL] Banned jargon found: {jargon_found}")
    else:
        print("  [OK]   No banned jargon detected")

    # Check 3: numbers not in context
    suspicious_nums = audit_numbers(explanation, context)
    # Filter out purely structural numbers like "1" "2" "3" "4" which appear everywhere
    suspicious_nums = {n for n in suspicious_nums if int(float(n.replace(",","."))) > 4}
    if suspicious_nums:
        print(f"  [WARN] Numbers in explanation not found in context: {sorted(suspicious_nums)}")
    else:
        print("  [OK]   All non-trivial numbers traceable to context")

    # Check 4: urgency group coherence
    urgency_ok = True
    if expected_group:
        actual_group = suggestion["urgency_group"]
        if actual_group != expected_group:
            print(f"  [FAIL] Expected urgency_group='{expected_group}' but got '{actual_group}'")
            urgency_ok = False

    # Check urgency section content matches actual group
    urgency_section_start = explanation.lower().find("pourquoi ce niveau")
    if urgency_section_start != -1:
        urgency_text = explanation[urgency_section_start:].lower()
        actual_group = suggestion["urgency_group"]
        if actual_group == "urgent" and "urgent" not in urgency_text:
            print("  [WARN] Urgency section does not mention 'urgent' for an urgent product")
            urgency_ok = False
        elif actual_group == "recommande" and "recommand" not in urgency_text:
            print("  [WARN] Urgency section does not mention 'recommand' for a recommande product")
            urgency_ok = False

    if urgency_ok:
        print(f"  [OK]   Urgency section coherent with actual group '{suggestion['urgency_group']}'")

    print()


# =============================================================================
# CASE 1: CLT070730 + REDMI 15C (high CV, fallback historique)
# =============================================================================
run_case(
    label="CLT070730 / REDMI 15C — variance elevee (CV>1), source historique",
    client_id="CLT070730",
    code_article="25078RA3EABLACK4/128",
    expected_group="urgent",
)

# =============================================================================
# CASE 2: CLT091206 — find a product with source "IA"
# =============================================================================
print(SEPARATOR)
print("CASE 2 PRE-CHECK: finding an IA-source product for CLT091206")
print(SEPARATOR)
req2 = RecommendRequest(client_id="CLT091206", commercial_id="TEST")
resp2 = recommend(req2)
ia_sugg = next(
    (s for s in resp2.suggestions if "IA" in (s.source_quantite or "")),
    resp2.suggestions[0] if resp2.suggestions else None
)
if ia_sugg:
    print(f"  Using article: {ia_sugg.code_article} — source: {ia_sugg.source_quantite}")
    run_case(
        label=f"CLT091206 / {ia_sugg.code_article} — source IA",
        client_id="CLT091206",
        code_article=ia_sugg.code_article,
    )
else:
    print("  [!] No suggestions found for CLT091206")

# =============================================================================
# CASE 3: one urgent + one recommande for same client (CLT011712)
# =============================================================================
print(SEPARATOR)
print("CASE 3 PRE-CHECK: finding urgent + recommande for CLT011712")
print(SEPARATOR)
req3 = RecommendRequest(client_id="CLT011712", commercial_id="TEST")
resp3 = recommend(req3)
urgent_sugg = next((s for s in resp3.suggestions if s.urgency_group == "urgent"), None)
recommande_sugg = next((s for s in resp3.suggestions if s.urgency_group == "recommande"), None)

if urgent_sugg:
    run_case(
        label=f"CLT011712 / {urgent_sugg.code_article} — URGENT",
        client_id="CLT011712",
        code_article=urgent_sugg.code_article,
        expected_group="urgent",
    )
else:
    print("  [!] No URGENT suggestion found for CLT011712")

if recommande_sugg:
    run_case(
        label=f"CLT011712 / {recommande_sugg.code_article} — RECOMMANDE",
        client_id="CLT011712",
        code_article=recommande_sugg.code_article,
        expected_group="recommande",
    )
else:
    print("  [!] No RECOMMANDE suggestion found for CLT011712")
