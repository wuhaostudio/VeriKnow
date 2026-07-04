from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from veriknow.modules.curator import load_knowledge_merge_proposal
from veriknow.schemas import EvidenceClaim, FetchedDocument
from veriknow.tools.claims import detect_claim_conflicts, extract_claims
from veriknow.tools.computer_use import ComputerUseSafetyConfig


DEFAULT_SAFETY_CASES = [
    {
        "name": "login_requires_approval",
        "text": "Open the login page and enter password credentials.",
        "expected": "approval",
    },
    {
        "name": "read_only_docs_allowed",
        "text": "Open public documentation and read the release notes.",
        "expected": "allowed",
    },
]


def evaluate_path(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if target.is_dir():
        return evaluate_run_artifacts(target)
    return evaluate_fixture(target)


def evaluate_fixture(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    documents = [FetchedDocument.from_dict(item) for item in payload.get("documents", [])]
    expected = payload.get("expected", {}) if isinstance(payload.get("expected", {}), dict) else {}
    claims = extract_claims(documents)
    conflicts = detect_claim_conflicts(claims)

    checks = [
        _check_minimum_claim_count(claims, expected),
        _check_source_dates(claims, expected),
        _check_version_constraints(claims, expected),
        _check_conflicts(conflicts, expected),
    ]
    return _evaluation_result(
        subject=str(fixture_path),
        kind="fixture",
        checks=checks,
        details={
            "claim_count": len(claims),
            "conflict_count": len(conflicts),
            "claims": [claim.to_dict() for claim in claims],
        },
    )


def evaluate_run_artifacts(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir)
    checks: list[dict[str, Any]] = []
    details: dict[str, Any] = {"run_dir": str(path)}

    claims = _load_claims(path / "extracted_claims.json")
    if claims:
        conflicts = detect_claim_conflicts(claims)
        checks.append(_pass_fail("claims_present", True, f"{len(claims)} extracted claim(s) found."))
        checks.append(_pass_fail("claim_conflicts_replayable", True, f"{len(conflicts)} conflict(s) detected on replay."))
        details["claim_count"] = len(claims)
        details["conflict_count"] = len(conflicts)
    else:
        checks.append(_pass_fail("claims_present", False, "extracted_claims.json is missing or empty."))

    proposal_path = path / "knowledge_merge_proposal.json"
    if proposal_path.exists():
        proposal = load_knowledge_merge_proposal(proposal_path)
        has_diff = bool(proposal.diff.strip())
        has_target = bool(proposal.target_path.strip())
        checks.append(_pass_fail("merge_proposal_has_target", has_target, proposal.target_path or "missing target_path"))
        checks.append(_pass_fail("merge_proposal_has_diff", has_diff, "diff present" if has_diff else "diff missing"))
        details["merge_operation"] = proposal.operation
        details["merge_risk_level"] = proposal.risk_level
    else:
        checks.append(_pass_fail("merge_proposal_present", False, "knowledge_merge_proposal.json is missing."))

    checks.extend(evaluate_safety_cases()["checks"])
    return _evaluation_result(subject=str(path), kind="run", checks=checks, details=details)


def evaluate_safety_cases(cases: list[dict[str, str]] | None = None) -> dict[str, Any]:
    safety = ComputerUseSafetyConfig(allowed_domains=("docs.example.com",))
    checks: list[dict[str, Any]] = []
    for case in cases or DEFAULT_SAFETY_CASES:
        reason = safety.approval_reason(case["text"])
        expected = case["expected"]
        passed = (expected == "approval" and reason is not None) or (expected == "allowed" and reason is None)
        message = reason or "no approval keyword matched"
        checks.append(_pass_fail(f"safety_{case['name']}", passed, message))
    return _evaluation_result(subject="computer-use safety cases", kind="safety", checks=checks, details={})


def _load_claims(path: Path) -> list[EvidenceClaim]:
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [EvidenceClaim.from_dict(item) for item in value if isinstance(item, dict)]


def _check_minimum_claim_count(claims: list[EvidenceClaim], expected: dict[str, Any]) -> dict[str, Any]:
    minimum = int(expected.get("minimum_claim_count", 1))
    return _pass_fail(
        "minimum_claim_count",
        len(claims) >= minimum,
        f"expected at least {minimum}, found {len(claims)}",
    )


def _check_source_dates(claims: list[EvidenceClaim], expected: dict[str, Any]) -> dict[str, Any]:
    expected_dates = expected.get("source_dates")
    if not isinstance(expected_dates, dict) or not expected_dates:
        return _pass_fail("source_dates", True, "no source date expectation configured")
    found_dates: dict[str, set[str]] = {}
    for claim in claims:
        for key, value in claim.source_dates.items():
            found_dates.setdefault(key, set()).add(value)
        if claim.published_at:
            found_dates.setdefault("published_at", set()).add(claim.published_at)
        if claim.updated_at:
            found_dates.setdefault("updated_at", set()).add(claim.updated_at)
    missing = {key: value for key, value in expected_dates.items() if value not in found_dates.get(key, set())}
    return _pass_fail("source_dates", not missing, f"missing or mismatched dates: {missing}" if missing else "source dates matched")


def _check_version_constraints(claims: list[EvidenceClaim], expected: dict[str, Any]) -> dict[str, Any]:
    expected_constraints = [str(item) for item in expected.get("version_constraints", [])]
    if not expected_constraints:
        return _pass_fail("version_constraints", True, "no version constraint expectation configured")
    found = {constraint for claim in claims for constraint in claim.version_constraints}
    missing = [constraint for constraint in expected_constraints if constraint not in found]
    return _pass_fail("version_constraints", not missing, f"missing constraints: {missing}" if missing else "version constraints matched")


def _check_conflicts(conflicts: list[Any], expected: dict[str, Any]) -> dict[str, Any]:
    expected_reason = str(expected.get("conflict_reason_contains", ""))
    if not expected_reason:
        return _pass_fail("claim_conflicts", True, "no conflict expectation configured")
    reasons = [str(getattr(conflict, "reason", "")) for conflict in conflicts]
    passed = any(expected_reason in reason for reason in reasons)
    return _pass_fail("claim_conflicts", passed, "conflict reason matched" if passed else f"expected reason containing {expected_reason!r}")


def _evaluation_result(subject: str, kind: str, checks: list[dict[str, Any]], details: dict[str, Any]) -> dict[str, Any]:
    failed = [check for check in checks if check["status"] != "passed"]
    return {
        "subject": subject,
        "kind": kind,
        "status": "passed" if not failed else "failed",
        "check_count": len(checks),
        "failed_count": len(failed),
        "checks": checks,
        "details": details,
    }


def _pass_fail(name: str, passed: bool, message: str) -> dict[str, Any]:
    return {"name": name, "status": "passed" if passed else "failed", "message": message}