from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from veriknow.schemas import EvidenceBundle, RunRecord, VerificationPlan, VerificationRun


DEFAULT_REVERIFY_INTERVAL_DAYS = 30


def render_report(
    record: RunRecord,
    run_dir: Path | None = None,
    *,
    reverify_interval_days: int = DEFAULT_REVERIFY_INTERVAL_DAYS,
) -> str:
    task = record.task
    run_dir = run_dir or _run_dir_for(record)
    constraints = "\n".join(f"- {item}" for item in task.constraints) or "- None"
    evidence = _load_evidence_bundle(record)
    plan = _load_verification_plan(record)
    verification = _load_verification_run(record)
    status = _report_status(evidence, verification)
    confidence = _overall_confidence(evidence, verification)
    verified_at = verification.completed_at if verification and verification.completed_at else ""
    next_verify_at = _next_verify_at(verified_at, interval_days=reverify_interval_days)
    source_front_matter = _source_front_matter(evidence)
    research_summary = _research_summary(evidence)
    verification_summary = _verification_summary(plan, verification)
    guide = _operation_guide(plan, verification, run_dir)
    screenshots = _screenshots_section(verification, run_dir)
    outdated = _outdated_information_section(evidence, verification)
    manual_checkpoints = _manual_checkpoints_section(plan, verification)
    logs = _logs_section(verification, run_dir)
    sources = _sources_section(evidence)
    return f"""---
title: {_yaml_quote(task.target)}
status: {_yaml_quote(status)}
verified_at: {_yaml_quote(verified_at)}
next_verify_at: {_yaml_quote(next_verify_at)}
confidence: {_yaml_quote(confidence)}
sources:
{source_front_matter}
---

# {task.target}

## Status

- Run ID: `{record.run_id}`
- Report status: {status}
- Verified date: {verified_at or "Not verified yet"}
- Next verification date: {next_verify_at}
- Confidence: {confidence}

## Summary

{research_summary}

## Task

- Target: {task.target}
- Objective: {task.objective}
- Scope: {task.scope}
- Verification required: {task.verification_required}
- Verification method: {task.verification_method}
- Output format: {task.output_format}
- Publish target: {task.publish_target}

## Constraints

{constraints}

## Step-by-Step Guide

{guide}

## Verification Summary

{verification_summary}

## Screenshots

{screenshots}

## Outdated or Unsupported Information

{outdated}

## Manual Checkpoints

{manual_checkpoints}

## Logs

{logs}

## Sources

{sources}
"""


def render_placeholder_report(record: RunRecord) -> str:
    return render_report(record)


def write_report(
    record: RunRecord,
    run_dir: Path,
    *,
    reverify_interval_days: int = DEFAULT_REVERIFY_INTERVAL_DAYS,
) -> Path:
    report_path = run_dir / "report.md"
    report_path.write_text(
        render_report(record, run_dir, reverify_interval_days=reverify_interval_days),
        encoding="utf-8",
    )
    return report_path


def write_placeholder_report(record: RunRecord, run_dir: Path) -> Path:
    return write_report(record, run_dir)


def _load_evidence_bundle(record: RunRecord) -> EvidenceBundle | None:
    evidence_path = record.artifacts.get("evidence")
    if not evidence_path:
        return None
    path = Path(evidence_path)
    if not path.exists():
        return None
    return EvidenceBundle.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _load_verification_plan(record: RunRecord) -> VerificationPlan | None:
    plan_path = record.artifacts.get("verification_plan")
    if not plan_path:
        return None
    path = Path(plan_path)
    if not path.exists():
        return None
    return VerificationPlan.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _load_verification_run(record: RunRecord) -> VerificationRun | None:
    verification_path = record.artifacts.get("verification")
    if not verification_path:
        return None
    path = Path(verification_path)
    if not path.exists():
        return None
    return VerificationRun.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _run_dir_for(record: RunRecord) -> Path | None:
    for path in record.artifacts.values():
        artifact_path = Path(path)
        if artifact_path.name:
            return artifact_path.parent
    return None


def _report_status(evidence: EvidenceBundle | None, verification: VerificationRun | None) -> str:
    if verification is not None:
        return verification.status
    if evidence is not None and evidence.items:
        return "partial"
    return "draft"


def _overall_confidence(
    evidence: EvidenceBundle | None,
    verification: VerificationRun | None = None,
) -> str:
    if evidence is None or not evidence.items:
        return "low"
    if verification is not None:
        statuses = {result.status for result in verification.results}
        if "failed" in statuses or "blocked" in statuses:
            return "low"
        if statuses <= {"passed", "manual"} and any(item.confidence == "high" for item in evidence.items):
            return "high"
        if statuses <= {"passed", "partial", "manual", "skipped"}:
            return "medium"
    if any(item.confidence == "high" for item in evidence.items):
        return "medium"
    return evidence.items[0].confidence


def _source_front_matter(evidence: EvidenceBundle | None) -> str:
    if evidence is None or not evidence.items:
        return "  []"
    return "\n".join(
        [
            f"  - url: {_yaml_quote(item.url)}\n    type: {_yaml_quote(item.source_type)}"
            for item in evidence.items
        ]
    )


def _research_summary(evidence: EvidenceBundle | None) -> str:
    if evidence is None or not evidence.items:
        return (
            "This draft was generated from a normalized VeriKnow task. "
            "Public research and verification have not run yet."
        )
    return evidence.summary


def _operation_guide(
    plan: VerificationPlan | None,
    verification: VerificationRun | None,
    run_dir: Path | None,
) -> str:
    if plan is None or not plan.steps:
        return (
            "1. Review the task summary and constraints.\n"
            "2. Run `veriknow research <query>` to collect evidence.\n"
            "3. Run `veriknow plan <run_id>` and `veriknow verify <run_id>` before publishing."
        )

    result_by_description = {
        result.step_description: result for result in verification.results
    } if verification else {}
    lines: list[str] = []
    for index, step in enumerate(plan.steps, start=1):
        result = result_by_description.get(step.description)
        status = result.status if result else "not run"
        tools = ", ".join(step.tools) if step.tools else "none"
        lines.extend(
            [
                f"{index}. {step.description}",
                f"   - Expected result: {step.expected_result}",
                f"   - Method: {step.method}",
                f"   - Tools: {tools}",
                f"   - Verification status: {status}",
            ]
        )
        if result and result.actual_result:
            lines.append(f"   - Observation: {result.actual_result}")
        if result and result.actions:
            lines.append(f"   - Actions: {'; '.join(result.actions)}")
        if result and result.screenshot_path:
            lines.append(f"   - Screenshot: {_markdown_path(result.screenshot_path, run_dir)}")
        if step.requires_approval:
            lines.append("   - Approval required before automated execution.")
    return "\n".join(lines)


def _verification_summary(
    plan: VerificationPlan | None,
    verification: VerificationRun | None,
) -> str:
    if verification is None:
        planned = len(plan.steps) if plan else 0
        if planned:
            return f"Verification is planned with {planned} step(s), but no verification run has been recorded."
        return "No verification plan or verification run has been recorded."

    counts: dict[str, int] = {}
    for result in verification.results:
        counts[result.status] = counts.get(result.status, 0) + 1
    count_text = ", ".join(f"{status}: {count}" for status, count in sorted(counts.items()))
    completed = verification.completed_at or "not completed"
    return (
        f"Verification run status: {verification.status}.\n\n"
        f"- Completed at: {completed}\n"
        f"- Result counts: {count_text or 'none'}"
    )


def _screenshots_section(verification: VerificationRun | None, run_dir: Path | None) -> str:
    if verification is None:
        return "No screenshots have been captured yet."
    lines: list[str] = []
    for index, result in enumerate(verification.results, start=1):
        if not result.screenshot_path:
            continue
        path = _markdown_path(result.screenshot_path, run_dir)
        lines.extend(
            [
                f"### Step {index}",
                "",
                f"![Step {index} screenshot]({path})",
                "",
                f"- Status: {result.status}",
                f"- Observation: {result.actual_result or 'No observation recorded.'}",
                "",
            ]
        )
    return "\n".join(lines).strip() or "No screenshots have been captured yet."


def _outdated_information_section(
    evidence: EvidenceBundle | None,
    verification: VerificationRun | None,
) -> str:
    lines: list[str] = []
    if evidence is not None:
        for item in evidence.items:
            if item.updated_at or item.published_at:
                continue
            lines.append(
                f"- Source date unknown: [{item.title}]({item.url}). Re-check before relying on time-sensitive details."
            )

    if verification is not None:
        for result in verification.results:
            if result.status in {"failed", "skipped", "partial", "blocked"}:
                lines.append(
                    f"- {result.status.title()} verification: {result.step_description}. {result.actual_result}"
                )

    return "\n".join(lines) if lines else "No outdated or unsupported information was identified."


def _manual_checkpoints_section(
    plan: VerificationPlan | None,
    verification: VerificationRun | None,
) -> str:
    lines: list[str] = []
    if plan is not None:
        for step in plan.steps:
            if step.method == "manual" or step.requires_approval:
                approval = " Requires approval." if step.requires_approval else ""
                lines.append(f"- {step.description} Expected: {step.expected_result}.{approval}")

    if verification is not None:
        for result in verification.results:
            if result.status in {"manual", "skipped"} and result.step_description not in "\n".join(lines):
                lines.append(f"- {result.step_description} Result: {result.actual_result}")

    return "\n".join(lines) if lines else "No manual checkpoints are pending."


def _logs_section(verification: VerificationRun | None, run_dir: Path | None) -> str:
    if verification is None:
        return "No verification logs have been recorded yet."
    lines = [
        f"- {_markdown_path(result.log_path, run_dir)}"
        for result in verification.results
        if result.log_path
    ]
    return "\n".join(lines) if lines else "No verification logs have been recorded yet."


def _sources_section(evidence: EvidenceBundle | None) -> str:
    if evidence is None or not evidence.items:
        return "No sources collected yet."
    lines: list[str] = []
    for index, item in enumerate(evidence.items, start=1):
        snippet = f" - {item.snippet}" if item.snippet else ""
        lines.append(
            f"{index}. [{item.title}]({item.url}) "
            f"({item.source_type}, confidence: {item.confidence}){snippet}"
        )
    return "\n".join(lines)


def _markdown_path(path: str | None, run_dir: Path | None) -> str:
    if not path:
        return ""
    artifact_path = Path(path)
    if run_dir is not None:
        try:
            artifact_path = artifact_path.relative_to(run_dir)
        except ValueError:
            pass
    return artifact_path.as_posix()


def _yaml_quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _next_verify_at(verified_at: str, *, interval_days: int = DEFAULT_REVERIFY_INTERVAL_DAYS) -> str:
    base = datetime.now(timezone.utc)
    if verified_at:
        normalized = verified_at.replace("Z", "+00:00")
        try:
            base = datetime.fromisoformat(normalized)
        except ValueError:
            pass
    return (base.date() + timedelta(days=interval_days)).isoformat()
