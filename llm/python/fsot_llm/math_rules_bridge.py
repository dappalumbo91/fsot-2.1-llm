"""
Bridge Math generator (Desktop) rulebooks → FSOT math pathway curriculum.

GSM8K failed partly because emission collapsed to stubs and pure CoT
memorization without binding operations to explicit arithmetic *rules*.
The Math generator encodes math as atomic rule records (id, preconditions,
operation, examples, common_errors) — the right structure for D_eff≈10
chain_hash_number: each step cites a rule, then emits #### n.

Default source:
  C:\\Users\\damia\\Desktop\\Math generator
Override with env FSOT_MATH_GENERATOR_ROOT.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Optional

from fsot_llm.curriculum import FSOT_SYSTEM, _chat
from fsot_llm.domain_routing import allocation_for_pack, inject_route_context
from fsot_llm.external_data import extract_gsm8k_gold, load_pack_rows
from fsot_llm.paths import workspace_root


# GSM8K-relevant rule documents (grade-school quantity coupling)
GSM8K_RULE_DOCS = (
    "ARITHMETIC_RULES.json",
    "ALGEBRA_RULES.json",
    "GENERAL_MATHEMATICS_RULES.json",
)

# Categories that most often appear in word problems
PRIORITY_CATEGORIES = {
    "addition",
    "subtraction",
    "multiplication",
    "division",
    "fraction",
    "fraction addition",
    "percentage",
    "decimal",
    "order of operations",
    "equality",
    "sign",
    "ratio",
    "numeral semantics",
    "domain management",
    "distributive",
}


def math_generator_root() -> Path:
    env = os.environ.get("FSOT_MATH_GENERATOR_ROOT")
    if env:
        return Path(env)
    return Path(r"C:\Users\damia\Desktop\Math generator")


def load_rule_document(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_gsm8k_rules(
    *,
    root: Optional[Path] = None,
    max_rules: int = 120,
) -> list[dict[str, Any]]:
    """Load and prioritize rules for grade-school / GSM8K pathway."""
    root = root or math_generator_root()
    if not root.is_dir():
        raise FileNotFoundError(f"Math generator not found: {root}")

    rules: list[dict[str, Any]] = []
    for name in GSM8K_RULE_DOCS:
        p = root / name
        if not p.is_file():
            continue
        doc = load_rule_document(p)
        doc_id = name.replace("_RULES.json", "").lower()
        for r in doc.get("rules") or []:
            if not isinstance(r, dict) or not r.get("id"):
                continue
            r = dict(r)
            r["_document"] = doc_id
            r["_source_file"] = name
            rules.append(r)

    # Prefer high-priority categories; keep some others for coverage
    pri, rest = [], []
    for r in rules:
        cat = str(r.get("category") or "").lower()
        name = str(r.get("name") or "").lower()
        if any(c in cat or c in name for c in PRIORITY_CATEGORIES):
            pri.append(r)
        else:
            rest.append(r)
    ordered = pri + rest
    return ordered[:max_rules]


def rule_to_card_line(rule: dict[str, Any]) -> str:
    rid = rule.get("id", "?")
    name = rule.get("name", "")
    op = (rule.get("operation") or "").strip()
    if len(op) > 120:
        op = op[:117] + "..."
    return f"[{rid}] {name}: {op}"


def build_compact_rule_card(rules: list[dict[str, Any]], *, max_lines: int = 24) -> str:
    """Short system/context block — rules as constraints on CoT, not free text."""
    lines = [
        "FSOT MATH RULE CARD (apply only valid rules; never invent shortcuts):",
        "- Identify quantities and units first.",
        "- Each calc step should match a rule (add/mul/div/percent/fraction/order).",
        "- Mark calc as <<expr=value>> when helpful.",
        "- End with exactly one line: #### <number>",
        "- Never answer with a one-line stub like 'Read quantities carefully'.",
        "Key rules:",
    ]
    for r in rules[:max_lines]:
        lines.append("- " + rule_to_card_line(r))
    return "\n".join(lines)


def _example_to_cot(rule: dict[str, Any], example: str) -> Optional[str]:
    """Turn a short rule example into a tiny rule-cited chain ending in #### if numeric."""
    rid = rule.get("id", "RULE")
    name = rule.get("name", "")
    op = rule.get("operation") or ""
    # Prefer examples that look like equations
    ex = example.strip()
    if not ex:
        return None
    body = (
        f"Rule {rid} ({name}).\n"
        f"Operation: {op}\n"
        f"Apply: {ex}\n"
    )
    # Try extract a trailing number after = for #### emission practice
    import re

    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", ex.replace(",", ""))
    if nums and "=" in ex:
        body += f"#### {nums[-1]}"
    else:
        body += f"Result form: {rule.get('output_form', 'transformed expression')}"
    return body


def build_rule_curriculum_rows(
    *,
    max_rules: int = 100,
    max_rows: int = 180,
    seed: int = 7,
) -> list[dict[str, Any]]:
    """Chat rows teaching rule application (math pathway only)."""
    rules = load_gsm8k_rules(max_rules=max_rules)
    alloc = allocation_for_pack("gsm8k_test")
    card = build_compact_rule_card(rules, max_lines=16)
    rnd = random.Random(seed)
    rows: list[dict[str, Any]] = []

    # 1) Explicit rule application drills from examples
    for rule in rules:
        examples = list(rule.get("examples") or [])
        if not examples:
            continue
        ex = rnd.choice(examples)
        cot = _example_to_cot(rule, ex)
        if not cot:
            continue
        user = inject_route_context(
            "Apply the named arithmetic/algebra rule step-by-step. "
            "Cite the rule id. If the result is a number, end with #### <number>.\n\n"
            f"Rule: {rule.get('id')} — {rule.get('name')}\n"
            f"Example to execute: {ex}\n\n"
            f"{card}",
            alloc,
        )
        rows.append(_chat(user, cot, FSOT_SYSTEM))
        if len(rows) >= max_rows // 2:
            break

    # 2) Common-error avoidance (negative knowledge as positive instruction)
    for rule in rules:
        errs = list(rule.get("common_errors") or [])
        counters = list(rule.get("counterexamples") or [])
        if not errs and not counters:
            continue
        err = (errs or counters)[0]
        user = inject_route_context(
            "State the correct rule and reject the common error. "
            "Brief reasoning; if a corrected numeric example is clear, end with #### n.\n\n"
            f"Rule {rule.get('id')} ({rule.get('name')})\n"
            f"Common error / counterexample: {err}\n"
            f"Correct operation: {rule.get('operation')}",
            alloc,
        )
        ans = (
            f"Rule {rule.get('id')}: {rule.get('name')}.\n"
            f"Correct: {rule.get('operation')}\n"
            f"Reject: {err}\n"
            f"Always check preconditions: {', '.join(rule.get('preconditions') or ['none'])}."
        )
        rows.append(_chat(user, ans, FSOT_SYSTEM))
        if len(rows) >= max_rows:
            break

    rnd.shuffle(rows)
    return rows[:max_rows]


def tag_rules_for_problem(question: str, rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lightweight keyword → rule hints for word problems."""
    q = question.lower()
    keywords = [
        (["percent", "%", "percentage"], ["percentage", "percent"]),
        (["fraction", "half", "third", "quarter", "/"], ["fraction"]),
        (["times", "product", "each", "per "], ["multiplication"]),
        (["total", "altogether", "sum", "more than", "plus"], ["addition"]),
        (["left", "remain", "less", "difference", "minus"], ["subtraction"]),
        (["divide", "split", "share", "average", "per "], ["division"]),
        (["ratio", "times as many"], ["ratio", "multiplication"]),
        (["order", "first", "then"], ["order of operations"]),
    ]
    hit_cats: set[str] = set()
    for keys, cats in keywords:
        if any(k in q for k in keys):
            hit_cats.update(cats)
    if not hit_cats:
        hit_cats = {"addition", "multiplication", "order of operations"}

    tagged = []
    for r in rules:
        cat = str(r.get("category") or "").lower()
        name = str(r.get("name") or "").lower()
        if any(c in cat or c in name for c in hit_cats):
            tagged.append(r)
    # always include a few core ops
    core_ids = {"AR-100", "AR-140", "AR-160", "AR-201", "AR-202", "AR-147", "AR-262"}
    for r in rules:
        if r.get("id") in core_ids and r not in tagged:
            tagged.append(r)
    return tagged[:8]


def _certify_gsm8k_answer(full: str, gold: str) -> bool:
    """FSOT measurement: assistant #### must equal gold quantity."""
    from fsot_llm.external_data import extract_final_number

    if "####" not in full:
        return False
    pred = extract_final_number(full)
    if pred is None or not gold:
        return False
    g = str(gold).replace(",", "").strip()
    p = str(pred).replace(",", "").strip()
    if g == p:
        return True
    try:
        return abs(float(g) - float(p)) < 1e-6
    except ValueError:
        return False


def build_gsm8k_truth_rows(
    *,
    limit: int = 200,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    TRUTH-ONLY GSM8K: official CoT as assistant target — no rule theater.

    - Assistant = raw train answer (measurement-true information wave)
    - User = thin FSOT_ROUTE + emission instruction only
    - Reject if #### does not certificate to gold
    - Never prefix 'Using rules AR-…' (that was decorative suction / theater)
    """
    alloc = allocation_for_pack("gsm8k_test")
    pool_cap = min(max(limit * 3, 800), 8000)
    pool = load_pack_rows("gsm8k_train", limit=pool_cap)
    rnd = random.Random(seed)
    rnd.shuffle(pool)
    banned = (
        "read quantities carefully",
        "brief answer only",
        "using rules ar-",
        "```",
        "sympy",
    )
    rows: list[dict[str, Any]] = []

    for row in pool:
        if len(rows) >= limit:
            break
        q = (row.get("question") or "").strip()
        gold = extract_gsm8k_gold(row.get("answer") or "")
        full = (row.get("answer") or "").strip()
        if not q or not gold or not full:
            continue
        low = full.lower()
        if any(b in low for b in banned):
            continue
        body = full.split("####")[0].strip()
        if len(body) < 40:
            continue
        if not _certify_gsm8k_answer(full, gold):
            continue
        # Prefer chains with explicit calc annotations when present
        user = inject_route_context(
            "Solve the grade-school math problem with pure arithmetic "
            "(no Python, no code). Show step-by-step reasoning with "
            "<<calc=result>> when helpful. "
            "Never use one-line stubs. "
            "End with exactly one line: #### <number>\n\n"
            f"Problem: {q}",
            alloc,
        )
        # ASSISTANT = official GSM8K answer only (truth)
        rows.append(_chat(user, full, FSOT_SYSTEM))

    return rows


# Back-compat alias — must NOT reintroduce theater
def build_gsm8k_rule_guided_rows(
    *,
    limit: int = 200,
    seed: int = 42,
) -> list[dict[str, Any]]:
    return build_gsm8k_truth_rows(limit=limit, seed=seed)


def build_math_rules_curriculum(
    *,
    gsm8k_n: int = 200,
    rule_n: int = 0,
    stamp: str = "truth",
    truth_only: bool = True,
) -> Path:
    """
    Math fold curriculum.

    Default truth_only=True: official GSM8K CoT only (no rule-prefix theater).
    rule_n>0 only adds SEPARATE rule-drill chats (not mixed into GSM8K targets).
    """
    gsm_rows = build_gsm8k_truth_rows(limit=gsm8k_n)
    # Hard reject theater / uncertified
    def _ok_gsm(row: dict[str, Any]) -> bool:
        msgs = row.get("messages") or []
        if not msgs:
            return False
        content = str(msgs[-1].get("content") or "")
        if content.lower().startswith("using rules"):
            return False
        if "rules in force" in content.lower():
            return False
        if "####" not in content:
            return False
        if "```" in content or "sympy" in content.lower():
            return False
        if "read quantities carefully" in content.lower():
            return False
        return True

    gsm_rows = [r for r in gsm_rows if _ok_gsm(r)]
    rule_rows: list[dict[str, Any]] = []
    if not truth_only and rule_n > 0:
        rule_cap = min(rule_n, max(20, gsm8k_n // 8))
        rule_rows = build_rule_curriculum_rows(max_rows=rule_cap)

    rows = list(gsm_rows) + list(rule_rows)
    random.Random(11).shuffle(rows)

    out = (
        workspace_root()
        / "llm"
        / "data"
        / "curriculum"
        / "pathways"
        / f"math_truth_{stamp}.jsonl"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "mode": "truth_only_gsm8k" if truth_only or rule_n == 0 else "truth_plus_rule_drills",
        "theater_forbidden": True,
        "n_rows": len(rows),
        "n_rule_rows": len(rule_rows),
        "n_gsm8k_rows": len(gsm_rows),
        "certify_hash_eq_gold": True,
        "path": str(out),
    }
    out.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return out


def math_system_with_rules() -> str:
    """Optional richer system string for math training only."""
    try:
        rules = load_gsm8k_rules(max_rules=20)
        card = build_compact_rule_card(rules, max_lines=12)
    except Exception:
        card = "Use valid arithmetic rules; end with #### <number>."
    return (
        FSOT_SYSTEM
        + "\n\nYou solve quantitative problems by applying explicit math rules "
        "(from the FSOT-aligned Math generator rulebooks). "
        "Math is rule-governed: check domain/preconditions, apply operation, "
        "avoid common errors. "
        + card
    )
