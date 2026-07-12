from __future__ import annotations

import json
import math
import re
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
        try:
            proposal = load_knowledge_merge_proposal(proposal_path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            checks.append(
                _pass_fail(
                    "merge_proposal_readable",
                    False,
                    f"{exc.__class__.__name__}: {exc}",
                )
            )
        else:
            has_diff = bool(proposal.diff.strip())
            has_target = bool(proposal.target_path.strip())
            has_content = bool(proposal.proposed_content.strip())
            valid_base_hash = bool(
                re.fullmatch(r"[0-9a-f]{64}", proposal.base_content_hash)
            )
            checks.append(
                _pass_fail(
                    "merge_proposal_has_target",
                    has_target,
                    proposal.target_path or "missing target_path",
                )
            )
            checks.append(
                _pass_fail(
                    "merge_proposal_has_diff",
                    has_diff,
                    "diff present" if has_diff else "diff missing",
                )
            )
            checks.append(
                _pass_fail(
                    "merge_proposal_has_proposed_content",
                    has_content,
                    "proposed content present" if has_content else "proposed content missing",
                )
            )
            checks.append(
                _pass_fail(
                    "merge_proposal_has_base_hash",
                    valid_base_hash,
                    "base content hash present"
                    if valid_base_hash
                    else "base content hash missing or invalid",
                )
            )
            details["merge_operation"] = proposal.operation
            details["merge_risk_level"] = proposal.risk_level
    else:
        checks.append(_pass_fail("merge_proposal_present", False, "knowledge_merge_proposal.json is missing."))

    llm_artifacts = _load_llm_artifacts(path / "llm")
    if llm_artifacts:
        llm_checks, llm_details = _evaluate_llm_artifacts(llm_artifacts)
        checks.extend(llm_checks)
        details["llm"] = llm_details

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


def _load_llm_artifacts(llm_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    if not llm_dir.exists():
        return []
    artifacts: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(llm_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            payload = {"artifact_error": f"{exc.__class__.__name__}: {exc}"}
        artifacts.append((path, payload if isinstance(payload, dict) else {}))
    return artifacts


def _evaluate_llm_artifacts(
    artifacts: list[tuple[Path, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required_metadata = {
        "provider",
        "model",
        "status",
        "error_code",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "attempts",
    }
    checks: list[dict[str, Any]] = []
    statuses: list[str] = []
    total_tokens = 0
    total_latency_ms = 0.0
    for path, payload in artifacts:
        metadata = payload.get("call_metadata")
        missing = (
            sorted(required_metadata - set(metadata))
            if isinstance(metadata, dict)
            else sorted(required_metadata)
        )
        metadata_valid = (
            not missing
            and isinstance(metadata, dict)
            and bool(str(metadata.get("provider") or "").strip())
            and str(metadata.get("status"))
            in {"completed", "failed", "blocked", "unknown"}
            and _valid_nonnegative_int(metadata.get("attempts"))
            and _valid_optional_nonnegative_number(metadata.get("latency_ms"))
            and _valid_optional_nonnegative_int(metadata.get("input_tokens"))
            and _valid_optional_nonnegative_int(metadata.get("output_tokens"))
            and _valid_optional_nonnegative_int(metadata.get("total_tokens"))
            and _valid_optional_nonnegative_number(metadata.get("estimated_cost_usd"))
        )
        checks.append(
            _pass_fail(
                f"llm_metadata_{path.stem}",
                metadata_valid,
                "complete metadata"
                if metadata_valid
                else f"missing or invalid metadata fields: {missing}",
            )
        )

        prompt_hash = str(payload.get("prompt_hash") or "")
        prompt_stored = payload.get("prompt_stored")
        privacy_ok = (
            bool(re.fullmatch(r"[0-9a-f]{64}", prompt_hash))
            and isinstance(prompt_stored, bool)
            and (
                (prompt_stored and isinstance(payload.get("prompt"), str))
                or (not prompt_stored and payload.get("prompt") is None)
            )
        )
        checks.append(
            _pass_fail(
                f"llm_prompt_policy_{path.stem}",
                privacy_ok,
                "prompt retention policy recorded"
                if privacy_ok
                else "prompt retention metadata is incomplete",
            )
        )

        if isinstance(metadata, dict):
            statuses.append(str(metadata.get("status") or "unknown"))
            total_tokens += _safe_int(metadata.get("total_tokens"))
            total_latency_ms += _safe_float(metadata.get("latency_ms"))

    return checks, {
        "artifact_count": len(artifacts),
        "completed_count": statuses.count("completed"),
        "failed_count": statuses.count("failed"),
        "blocked_count": statuses.count("blocked"),
        "total_tokens": total_tokens,
        "total_latency_ms": round(total_latency_ms, 3),
    }


def _safe_int(value: Any) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError, OverflowError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        number = float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _valid_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _valid_optional_nonnegative_int(value: Any) -> bool:
    return value is None or _valid_nonnegative_int(value)


def _valid_optional_nonnegative_number(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number) and number >= 0


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
