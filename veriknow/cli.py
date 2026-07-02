from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from veriknow.config import create_default_config, ensure_data_dirs, load_config
from veriknow.llm import create_llm_client
from veriknow.memory.store import MemoryStore
from veriknow.modules.adaptive_profile import AdaptiveProfile
from veriknow.modules.curator import KnowledgeCurator, load_knowledge_patch
from veriknow.modules.knowledge import MarkdownKnowledgeIndex, title_from_markdown
from veriknow.modules.normalizer import AIRequirementNormalizer, RequirementNormalizer, SUPPORTED_NORMALIZER_STRATEGIES
from veriknow.modules.planner import AIVerificationPlanner, SUPPORTED_PLANNING_STRATEGIES, VerificationPlanner, render_verification_checklist
from veriknow.modules.publisher import publish_document
from veriknow.modules.researcher import AIResearcher, Researcher, SUPPORTED_RESEARCH_STRATEGIES, add_claim_summary
from veriknow.modules.verifier import Verifier
from veriknow.schemas import EvidenceBundle, EvidenceClaim, VerificationPlan
from veriknow.tools.claims import AIClaimExtractor, detect_claim_conflicts, extract_claims
from veriknow.tools.computer_use import ComputerUseSafetyConfig, ComputerUseVerifier
from veriknow.tools.web_fetch import fetch_documents
from veriknow.tools.markdown import write_report
from veriknow.tools.web_search import SearchProviderError, create_search_provider


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "cli":
        args = args[1:]

    parser = build_parser()
    namespace = parser.parse_args(args)
    try:
        namespace.handler(namespace)
    except (KeyError, SearchProviderError, ValueError) as exc:
        parser.exit(2, f"veriknow: error: {exc}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="veriknow",
        description="Local-first knowledge verification workflow.",
    )
    parser.set_defaults(handler=lambda _: parser.print_help())
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Create local config and data directories.")
    init_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    init_parser.set_defaults(handler=handle_init)

    run_parser = subparsers.add_parser("run", help="Normalize a task and create a local run.")
    run_parser.add_argument("request", help="Raw research or verification request.")
    run_parser.add_argument("--dry-run", action="store_true", help="Stop after normalization.")
    run_parser.add_argument(
        "--normalizer",
        choices=sorted(SUPPORTED_NORMALIZER_STRATEGIES),
        default="deterministic",
        help="Task normalization strategy.",
    )
    run_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    run_parser.set_defaults(handler=handle_run)

    research_parser = subparsers.add_parser("research", help="Collect public evidence for a task.")
    research_parser.add_argument("query", nargs="?", help="Research query. Required unless --run-id is used.")
    research_parser.add_argument("--run-id", help="Research an existing run instead of creating a new one.")
    research_parser.add_argument("--limit", type=int, default=None, help="Maximum number of sources to keep.")
    research_parser.add_argument("--search-provider", help="Search provider override, such as static or brave.")
    research_parser.add_argument(
        "--strategy",
        choices=sorted(SUPPORTED_RESEARCH_STRATEGIES),
        default="deterministic",
        help="Research strategy.",
    )
    research_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    research_parser.set_defaults(handler=handle_research)

    plan_parser = subparsers.add_parser("plan", help="Generate a verification plan for a run.")
    plan_parser.add_argument("run_id")
    plan_parser.add_argument(
        "--strategy",
        choices=sorted(SUPPORTED_PLANNING_STRATEGIES),
        default="deterministic",
        help="Verification planning strategy.",
    )
    plan_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    plan_parser.set_defaults(handler=handle_plan)

    verify_parser = subparsers.add_parser("verify", help="Run verification steps for a planned run.")
    verify_parser.add_argument("run_id")
    verify_parser.add_argument(
        "--include-approval-required",
        action="store_true",
        help="Execute steps marked as requiring approval.",
    )
    verify_parser.add_argument(
        "--mode",
        choices=["browser", "computer-use"],
        default="browser",
        help="Verification execution mode.",
    )
    verify_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    verify_parser.set_defaults(handler=handle_verify)

    llm_parser = subparsers.add_parser("llm", help="Inspect configured model provider.")
    llm_parser.set_defaults(handler=lambda _: llm_parser.print_help())
    llm_subparsers = llm_parser.add_subparsers(dest="llm_command")
    llm_check_parser = llm_subparsers.add_parser("check", help="Check whether the configured model provider is available.")
    llm_check_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    llm_check_parser.set_defaults(handler=handle_llm_check)

    memory_parser = subparsers.add_parser("memory", help="Inspect local memory records.")
    memory_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    memory_subparsers = memory_parser.add_subparsers(dest="memory_command")
    runs_parser = memory_subparsers.add_parser("runs", help="List recent runs.")
    runs_parser.add_argument("--limit", type=int, default=20)
    runs_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    runs_parser.set_defaults(handler=handle_memory_runs)
    show_parser = memory_subparsers.add_parser("show", help="Show one run.")
    show_parser.add_argument("run_id")
    show_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    show_parser.set_defaults(handler=handle_memory_show)
    prefs_parser = memory_subparsers.add_parser("preferences", help="List passive preference signals.")
    prefs_parser.add_argument("--limit", type=int, default=50)
    prefs_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    prefs_parser.set_defaults(handler=handle_memory_preferences)
    publications_parser = memory_subparsers.add_parser("publications", help="List publication jobs.")
    publications_parser.add_argument("--limit", type=int, default=20)
    publications_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    publications_parser.set_defaults(handler=handle_memory_publications)

    pref_parser = subparsers.add_parser("preference", help="Append a task-relevant preference signal.")
    pref_parser.add_argument("key")
    pref_parser.add_argument("value")
    pref_parser.add_argument("--task-id")
    pref_parser.add_argument("--source", default="explicit")
    pref_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    pref_parser.set_defaults(handler=handle_preference)

    write_parser = subparsers.add_parser("write", help="Generate a Markdown operation report.")
    write_parser.add_argument("run_id")
    write_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    write_parser.set_defaults(handler=handle_write)

    curate_parser = subparsers.add_parser("curate", help="Generate a knowledge update patch for a run.")
    curate_parser.add_argument("run_id")
    curate_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    curate_parser.set_defaults(handler=handle_curate)

    apply_parser = subparsers.add_parser("apply", help="Apply an approved knowledge patch for a run.")
    apply_parser.add_argument("run_id")
    apply_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    apply_parser.set_defaults(handler=handle_apply)

    kb_parser = subparsers.add_parser("kb", help="Search local Markdown knowledge.")
    kb_parser.set_defaults(handler=lambda _: kb_parser.print_help())
    kb_subparsers = kb_parser.add_subparsers(dest="kb_command")
    kb_search_parser = kb_subparsers.add_parser("search", help="Search local knowledge by keyword.")
    kb_search_parser.add_argument("query")
    kb_search_parser.add_argument("--limit", type=int, default=10)
    kb_search_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    kb_search_parser.set_defaults(handler=handle_kb_search)

    publish_parser = subparsers.add_parser("publish", help="Publish an approved knowledge document.")
    publish_parser.add_argument("document_path")
    publish_parser.add_argument("--target", default="feishu", help="Publish target, such as feishu.")
    publish_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    publish_parser.set_defaults(handler=handle_publish)

    stale_parser = subparsers.add_parser("stale", help="List knowledge documents due for re-verification.")
    stale_parser.add_argument(
        "--exclude-missing",
        action="store_true",
        help="Do not treat documents without next_verify_at as stale.",
    )
    stale_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    stale_parser.set_defaults(handler=handle_stale)

    reverify_parser = subparsers.add_parser("reverify", help="Re-run research and verification for a knowledge document.")
    reverify_parser.add_argument("document_path")
    reverify_parser.add_argument("--mode", choices=["browser", "computer-use"], default="browser")
    reverify_parser.add_argument(
        "--include-approval-required",
        action="store_true",
        help="Execute steps marked as requiring approval.",
    )
    reverify_parser.add_argument("--limit", type=int, default=5, help="Maximum number of sources to keep.")
    reverify_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    reverify_parser.set_defaults(handler=handle_reverify)

    return parser


def handle_init(args: argparse.Namespace) -> None:
    created = create_default_config(args.config)
    config = load_config(args.config)
    ensure_data_dirs(config)
    store = MemoryStore(config)
    store.initialize()
    config_status = "created" if created else "exists"
    print(f"config: {config_status} ({args.config})")
    print(f"data_dir: {config.data_dir}")
    print(f"database: {config.database_path}")


def handle_run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_data_dirs(config)
    artifact = None
    if args.normalizer == "ai":
        result = AIRequirementNormalizer(config, create_llm_client(config)).normalize(args.request)
        task = result.task
        artifact = result.artifact
    else:
        task = RequirementNormalizer(config).normalize(args.request)

    store = MemoryStore(config)
    record = store.create_run(args.request, task)
    artifacts = {}
    if artifact is not None:
        artifact_path = _write_llm_artifact(store.run_dir(record.run_id), "normalizer", artifact.to_dict())
        artifacts["llm_normalizer"] = str(artifact_path)
    status = "dry_run" if args.dry_run else "created"
    record = store.update_run(record.run_id, status=status, artifacts=artifacts)
    print(json.dumps(record.to_dict(), ensure_ascii=False, indent=2))


def handle_research(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_data_dirs(config)
    store = MemoryStore(config)

    if args.run_id:
        record = store.get_run(args.run_id)
        if record is None:
            raise KeyError(f"run not found: {args.run_id}")
    else:
        if not args.query:
            raise ValueError("research query or --run-id is required")
        task = RequirementNormalizer(config).normalize(args.query)
        record = store.create_run(args.query, task)

    limit = args.limit or config.search_result_limit
    search_provider = create_search_provider(config, provider=args.search_provider)
    researcher = Researcher(search_provider)
    research_artifact = None
    if args.strategy == "ai":
        result = AIResearcher(create_llm_client(config), base=researcher).research(
            record.task,
            run_id=record.run_id,
            limit=limit,
        )
        bundle = result.bundle
        research_artifact = result.artifact
    else:
        bundle = researcher.research(record.task, run_id=record.run_id, limit=limit)

    run_dir = store.run_dir(record.run_id)
    evidence_path = run_dir / "evidence.json"
    artifacts = {}
    if researcher.last_raw_search_payloads:
        raw_payloads_path = run_dir / "raw_search_payloads.json"
        raw_payloads_path.write_text(
            json.dumps(researcher.last_raw_search_payloads, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        artifacts["raw_search_payloads"] = str(raw_payloads_path)

    if config.search_fetch_pages:
        raw_dir = run_dir / "raw_pages" if config.search_store_raw_pages else None
        fetched = fetch_documents(bundle.items, limit=limit, raw_dir=raw_dir)
        fetched_path = run_dir / "fetched_documents.json"
        fetched_path.write_text(
            json.dumps([document.to_dict() for document in fetched], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        artifacts["fetched_documents"] = str(fetched_path)
        claim_artifact = None
        if args.strategy == "ai":
            claim_result = AIClaimExtractor(create_llm_client(config)).extract(fetched)
            claims = claim_result.claims
            claim_artifact = claim_result.artifact
        else:
            claims = extract_claims(fetched)
        claim_conflicts = detect_claim_conflicts(claims)
        if claim_artifact is not None:
            artifact_path = _write_llm_artifact(run_dir, "claim_extractor", claim_artifact.to_dict())
            artifacts["llm_claim_extractor"] = str(artifact_path)
        bundle.summary = add_claim_summary(
            bundle.summary,
            claims,
            conflict_count=len(claim_conflicts),
        )
        claims_path = run_dir / "extracted_claims.json"
        claims_path.write_text(
            json.dumps([claim.to_dict() for claim in claims], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        artifacts["extracted_claims"] = str(claims_path)
        if claim_conflicts:
            conflicts_path = run_dir / "claim_conflicts.json"
            conflicts_path.write_text(
                json.dumps([conflict.to_dict() for conflict in claim_conflicts], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            artifacts["claim_conflicts"] = str(conflicts_path)

    evidence_path.write_text(
        json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    related_path = _write_related_knowledge(
        record,
        config,
        run_dir,
        extra_text=bundle.summary,
    )
    artifacts.update(
        {
            "evidence": str(evidence_path),
            "related_knowledge": str(related_path),
        }
    )
    if research_artifact is not None:
        artifact_path = _write_llm_artifact(store.run_dir(record.run_id), "research", research_artifact.to_dict())
        artifacts["llm_research"] = str(artifact_path)
    store.update_run(record.run_id, status="researched", artifacts=artifacts)
    print(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2))


def handle_plan(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_data_dirs(config)
    store = MemoryStore(config)
    record = store.get_run(args.run_id)
    if record is None:
        raise KeyError(f"run not found: {args.run_id}")

    evidence = _load_evidence(record.artifacts.get("evidence"))
    claims = _load_claims(record.artifacts.get("extracted_claims"))
    claim_conflicts = _load_json_list(record.artifacts.get("claim_conflicts"))
    plan_artifact = None
    if args.strategy == "ai":
        result = AIVerificationPlanner(create_llm_client(config)).plan(
            record.task,
            evidence,
            run_id=record.run_id,
            claims=claims,
            claim_conflicts=claim_conflicts,
        )
        plan = result.plan
        plan_artifact = result.artifact
    else:
        plan = VerificationPlanner().plan(record.task, evidence, run_id=record.run_id)
    run_dir = store.run_dir(record.run_id)
    plan_path = run_dir / "verification_plan.json"
    checklist_path = run_dir / "verification_checklist.md"
    plan_path.write_text(
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    checklist_path.write_text(render_verification_checklist(plan), encoding="utf-8")
    artifacts = {
        "verification_plan": str(plan_path),
        "verification_checklist": str(checklist_path),
    }
    if plan_artifact is not None:
        artifact_path = _write_llm_artifact(run_dir, "planner", plan_artifact.to_dict())
        artifacts["llm_planner"] = str(artifact_path)
    store.update_run(
        record.run_id,
        status="planned",
        artifacts=artifacts,
    )
    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))


def handle_verify(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_data_dirs(config)
    store = MemoryStore(config)
    record = store.get_run(args.run_id)
    if record is None:
        raise KeyError(f"run not found: {args.run_id}")

    plan = _load_plan(record.artifacts.get("verification_plan"))
    if plan is None:
        raise ValueError("verification_plan artifact is required; run `veriknow plan <run_id>` first")

    safety = ComputerUseSafetyConfig(
        allowed_domains=config.computer_use_domain_allowlist,
        approval_keywords=config.computer_use_approval_keywords,
    )
    run = Verifier(computer_use=ComputerUseVerifier(safety)).verify(
        plan,
        run_dir=store.run_dir(record.run_id),
        include_approval_required=args.include_approval_required,
        mode=args.mode,
    )
    verification_path = store.run_dir(record.run_id) / "verification.json"
    verification_path.write_text(
        json.dumps(run.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    store.update_run(
        record.run_id,
        status="verified" if run.status == "completed" else f"verification_{run.status}",
        artifacts={"verification": str(verification_path)},
    )
    print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))


def handle_llm_check(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    result = create_llm_client(config).check()
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def handle_memory_runs(args: argparse.Namespace) -> None:
    store = MemoryStore(load_config(args.config))
    runs = store.list_runs(limit=args.limit)
    if not runs:
        print("No runs found.")
        return
    for record in runs:
        print(f"{record.run_id}\t{record.status}\t{record.created_at}\t{record.task.target}")


def handle_memory_show(args: argparse.Namespace) -> None:
    store = MemoryStore(load_config(args.config))
    record = store.get_run(args.run_id)
    if record is None:
        raise KeyError(f"run not found: {args.run_id}")
    print(json.dumps(record.to_dict(), ensure_ascii=False, indent=2))


def handle_memory_preferences(args: argparse.Namespace) -> None:
    store = MemoryStore(load_config(args.config))
    preferences = store.list_preferences(limit=args.limit)
    if not preferences:
        print("No preferences found.")
        return
    for preference in preferences:
        task_id = preference.task_id or "-"
        print(f"{preference.created_at}\t{preference.key}\t{preference.value}\t{task_id}")


def handle_memory_publications(args: argparse.Namespace) -> None:
    store = MemoryStore(load_config(args.config))
    jobs = store.list_publication_jobs(limit=args.limit)
    if not jobs:
        print("No publication jobs found.")
        return
    for job in jobs:
        target_url = job.target_url or "-"
        error_code = job.error_code or "-"
        print(
            f"{job.created_at}\t{job.target}\t{job.status}\t"
            f"{job.document_path}\t{target_url}\t{error_code}"
        )


def handle_preference(args: argparse.Namespace) -> None:
    store = MemoryStore(load_config(args.config))
    profile = AdaptiveProfile(store)
    preference = profile.append_signal(
        args.key,
        args.value,
        source=args.source,
        task_id=args.task_id,
    )
    print(json.dumps(preference.to_dict(), ensure_ascii=False, indent=2))


def handle_write(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    store = MemoryStore(config)
    record = store.get_run(args.run_id)
    if record is None:
        raise KeyError(f"run not found: {args.run_id}")
    report_path = write_report(
        record,
        store.run_dir(record.run_id),
        reverify_interval_days=config.default_reverify_interval_days,
    )
    record = store.update_run(record.run_id, artifacts={"report": str(report_path)})
    print(json.dumps(record.to_dict(), ensure_ascii=False, indent=2))


def handle_curate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_data_dirs(config)
    store = MemoryStore(config)
    record = store.get_run(args.run_id)
    if record is None:
        raise KeyError(f"run not found: {args.run_id}")

    report_path = _required_artifact(record.artifacts.get("report"), "report")
    related_path = _write_related_knowledge(
        record,
        config,
        store.run_dir(record.run_id),
        extra_text=report_path.read_text(encoding="utf-8"),
    )
    curator = KnowledgeCurator()
    patch = curator.create_patch(record, report_path, config.knowledge_dir)
    proposal = curator.create_merge_proposal(record, patch, report_path)
    diff_path, patch_path = curator.write_patch_files(patch, store.run_dir(record.run_id), proposal=proposal)
    proposal_path = store.run_dir(record.run_id) / "knowledge_merge_proposal.json"
    store.update_run(
        record.run_id,
        status="curated",
        artifacts={
            "related_knowledge": str(related_path),
            "patch_diff": str(diff_path),
            "knowledge_patch": str(patch_path),
            "knowledge_merge_proposal": str(proposal_path),
        },
    )
    print(json.dumps(patch.to_dict(), ensure_ascii=False, indent=2))


def handle_apply(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_data_dirs(config)
    store = MemoryStore(config)
    record = store.get_run(args.run_id)
    if record is None:
        raise KeyError(f"run not found: {args.run_id}")

    report_path = _required_artifact(record.artifacts.get("report"), "report")
    patch_path = _required_artifact(record.artifacts.get("knowledge_patch"), "knowledge_patch")
    curator = KnowledgeCurator()
    patch = load_knowledge_patch(patch_path)
    approved = curator.apply_patch(patch, report_path, config.knowledge_dir, patch_path)
    store.complete_run(
        record.run_id,
        artifacts={
            "knowledge_document": approved.target_path,
            "knowledge_patch": str(patch_path),
        },
    )
    print(json.dumps(approved.to_dict(), ensure_ascii=False, indent=2))


def handle_kb_search(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_data_dirs(config)
    results = MarkdownKnowledgeIndex().search(
        args.query,
        config.knowledge_dir,
        limit=args.limit,
    )
    print(json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2))


def handle_publish(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_data_dirs(config)
    store = MemoryStore(config)
    approved = store.is_approved_knowledge_document(args.document_path)
    job = publish_document(args.document_path, target=args.target, config=config, approved=approved)
    store.append_publication_job(job)
    print(json.dumps(job.to_dict(), ensure_ascii=False, indent=2))


def handle_stale(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_data_dirs(config)
    stale = MarkdownKnowledgeIndex().stale_documents(
        config.knowledge_dir,
        include_missing=not args.exclude_missing,
    )
    print(json.dumps([item.to_dict() for item in stale], ensure_ascii=False, indent=2))


def handle_reverify(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_data_dirs(config)
    store = MemoryStore(config)
    document_path = _knowledge_document_path(args.document_path, config.knowledge_dir)
    document_content = document_path.read_text(encoding="utf-8")
    title = title_from_markdown(document_content, document_path)
    request = f"Re-verify the latest information for {title}"
    task = RequirementNormalizer(config).normalize(request)
    record = store.create_run(request, task)
    run_id = record.run_id
    run_dir = store.run_dir(record.run_id)

    bundle = Researcher(create_search_provider(config)).research(record.task, run_id=record.run_id, limit=args.limit)
    evidence_path = run_dir / "evidence.json"
    evidence_path.write_text(
        json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    related_path = _write_related_knowledge(
        record,
        config,
        run_dir,
        extra_text=f"{document_content[:2000]}\n{bundle.summary}",
    )
    store.update_run(
        record.run_id,
        status="reverify_researched",
        artifacts={
            "source_document": str(document_path),
            "evidence": str(evidence_path),
            "related_knowledge": str(related_path),
        },
    )
    record = store.get_run(run_id)
    if record is None:
        raise KeyError(f"run not found: {run_id}")

    plan = VerificationPlanner().plan(record.task, bundle, run_id=record.run_id)
    plan_path = run_dir / "verification_plan.json"
    checklist_path = run_dir / "verification_checklist.md"
    plan_path.write_text(
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    checklist_path.write_text(render_verification_checklist(plan), encoding="utf-8")
    store.update_run(
        record.run_id,
        status="reverify_planned",
        artifacts={
            "verification_plan": str(plan_path),
            "verification_checklist": str(checklist_path),
        },
    )
    record = store.get_run(run_id)
    if record is None:
        raise KeyError(f"run not found: {run_id}")

    safety = ComputerUseSafetyConfig(
        allowed_domains=config.computer_use_domain_allowlist,
        approval_keywords=config.computer_use_approval_keywords,
    )
    verification = Verifier(computer_use=ComputerUseVerifier(safety)).verify(
        plan,
        run_dir=run_dir,
        include_approval_required=args.include_approval_required,
        mode=args.mode,
    )
    verification_path = run_dir / "verification.json"
    verification_path.write_text(
        json.dumps(verification.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    store.update_run(
        record.run_id,
        status="reverify_verified" if verification.status == "completed" else f"reverify_{verification.status}",
        artifacts={"verification": str(verification_path)},
    )
    record = store.get_run(run_id)
    if record is None:
        raise KeyError(f"run not found: {run_id}")

    report_path = write_report(
        record,
        run_dir,
        reverify_interval_days=config.default_reverify_interval_days,
    )
    record = store.update_run(record.run_id, artifacts={"report": str(report_path)})
    curator = KnowledgeCurator()
    patch = curator.create_patch_for_target(
        record,
        report_path,
        document_path,
        config.knowledge_dir,
    )
    proposal = curator.create_merge_proposal(record, patch, report_path)
    diff_path, patch_path = curator.write_patch_files(patch, run_dir, proposal=proposal)
    proposal_path = run_dir / "knowledge_merge_proposal.json"
    record = store.update_run(
        record.run_id,
        status="reverify_curated",
        artifacts={
            "patch_diff": str(diff_path),
            "knowledge_patch": str(patch_path),
            "knowledge_merge_proposal": str(proposal_path),
        },
    )
    print(json.dumps(record.to_dict(), ensure_ascii=False, indent=2))


def _write_llm_artifact(run_dir: Path, name: str, payload: dict) -> Path:
    llm_dir = run_dir / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    path = llm_dir / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

def _load_evidence(path: str | None) -> EvidenceBundle | None:
    if not path:
        return None
    evidence_path = Path(path)
    if not evidence_path.exists():
        return None
    with evidence_path.open(encoding="utf-8") as file:
        return EvidenceBundle.from_dict(json.load(file))


def _load_claims(path: str | None) -> list[EvidenceClaim]:
    if not path:
        return []
    claims_path = Path(path)
    if not claims_path.exists():
        return []
    with claims_path.open(encoding="utf-8") as file:
        raw_claims = json.load(file)
    if not isinstance(raw_claims, list):
        return []
    return [EvidenceClaim.from_dict(item) for item in raw_claims if isinstance(item, dict)]


def _load_json_list(path: str | None) -> list[dict]:
    if not path:
        return []
    json_path = Path(path)
    if not json_path.exists():
        return []
    with json_path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]

def _load_plan(path: str | None) -> VerificationPlan | None:
    if not path:
        return None
    plan_path = Path(path)
    if not plan_path.exists():
        return None
    with plan_path.open(encoding="utf-8") as file:
        return VerificationPlan.from_dict(json.load(file))


def _required_artifact(path: str | None, name: str) -> Path:
    if not path:
        raise ValueError(f"{name} artifact is required")
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise ValueError(f"{name} artifact does not exist: {artifact_path}")
    return artifact_path


def _write_related_knowledge(record, config, run_dir: Path, *, extra_text: str = "") -> Path:
    indexer = MarkdownKnowledgeIndex()
    results = indexer.related_for_run(
        record,
        config.knowledge_dir,
        extra_text=extra_text,
        limit=5,
    )
    return indexer.write_related(results, run_dir)


def _knowledge_document_path(document_path: str, knowledge_dir: Path) -> Path:
    path = Path(document_path)
    if not path.exists():
        raise ValueError(f"knowledge document does not exist: {path}")
    resolved = path.resolve()
    knowledge_resolved = knowledge_dir.resolve()
    if not resolved.is_relative_to(knowledge_resolved):
        raise ValueError(f"document is outside knowledge directory: {path}")
    if path.suffix.lower() != ".md":
        raise ValueError(f"knowledge document must be Markdown: {path}")
    return path
