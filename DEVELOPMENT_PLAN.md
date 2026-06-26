# VeriKnow Development Plan

This plan turns VeriKnow from design into an executable local-first product.

The first usable version should be a local CLI application that can accept a research target, collect evidence, verify key claims, and generate a traceable Markdown guide.

## Development Principles

- Local first: Markdown, SQLite, screenshots, and logs are stored locally.
- Minimal entities: add a module only when it owns a distinct responsibility.
- Evidence before conclusion: every important claim must point to a source or verification result.
- Verification before publishing: Feishu receives only approved Markdown documents.
- Replace safely: generate diffs before changing existing knowledge.

## Target MVP

Input:

```text
veriknow run "Research the latest LangChain multi-agent workflow and create an operation guide"
```

Output:

```text
data/
  runs/<run_id>/
    task.json
    evidence.json
    verification.json
    report.md
    screenshots/
    logs/
  knowledge/
    <category>/<topic>.md
  memory.sqlite
```

The MVP is considered usable when it can:

1. Normalize a vague request into a structured `TaskSpec`.
2. Search public web sources and store evidence with citations.
3. Generate a verification plan.
4. Run at least browser-based verification through Playwright.
5. Capture step-level screenshots.
6. Generate a Markdown operation guide.
7. Store the whole run in local memory.

## Recommended Stack

```text
Language: Python
Workflow: LangGraph
Agent/tool layer: LangChain
Browser verification: Playwright
Storage: SQLite + local files
Knowledge format: Markdown
Config: YAML or TOML
CLI: Typer
Validation: Pydantic
Testing: pytest
```

## Project Structure

```text
VeriKnow/
  README.md
  DEVELOPMENT_PLAN.md
  pyproject.toml
  veriknow/
    __init__.py
    cli.py
    config.py
    schemas.py
    orchestrator/
      graph.py
      state.py
    modules/
      adaptive_profile.py
      normalizer.py
      researcher.py
      verifier.py
      writer.py
      curator.py
      publisher.py
    memory/
      store.py
      migrations/
    tools/
      web_search.py
      browser.py
      markdown.py
      feishu.py
    templates/
      report.md.j2
  tests/
  data/
    .gitkeep
```

## Core Schemas

Implement these first in `veriknow/schemas.py`.

```text
TaskSpec
EvidenceItem
EvidenceBundle
VerificationStep
VerificationPlan
VerificationResult
VerificationRun
KnowledgePatch
MarkdownDocument
UserPreference
RunRecord
```

These schemas are the contract between modules. Keep agents replaceable by making them read and write these objects.

## Phase 0: Repository Bootstrap

Status: Completed.

Goal: make the project runnable as a Python package.

Tasks:

1. Create `pyproject.toml`.
2. Add package directory `veriknow/`.
3. Add `veriknow cli --help`.
4. Add config loading from `config.yaml`.
5. Add local data directory creation.
6. Add basic pytest setup.

Deliverables:

- `veriknow --help` works.
- `pytest` runs.
- `data/` structure is created automatically.

Acceptance:

```text
veriknow init
veriknow run "test task" --dry-run
pytest
```

## Phase 1: Task Normalization and Memory

Status: Completed.

Goal: convert user input into a traceable task and store each run locally.

Tasks:

1. Implement `RequirementNormalizer`.
2. Implement `MemoryStore` with SQLite.
3. Add run creation, update, and completion records.
4. Store raw user request, normalized task, timestamps, status, and artifacts.
5. Add `AdaptiveProfile` as a passive component.

Important constraint:

`AdaptiveProfile` should only store task-relevant preferences, not sensitive or personality labels.

Deliverables:

- `TaskSpec` generated from user input.
- `RunRecord` persisted in SQLite.
- User preference signals can be appended.

Acceptance:

```text
veriknow run "帮我研究某个工具的最新用法" --dry-run
veriknow memory runs
veriknow memory show <run_id>
```

## Phase 2: Public Research

Status: Completed.

Current scope:

- Deterministic local search provider for repeatable tests.
- `veriknow research "<query>"` creates a run, writes `evidence.json`, and records the artifact in SQLite.
- Evidence is ranked by source authority and includes source URLs, source type, snippets, and confidence levels.

Future enhancements:

- Replace or augment the static provider with a live web-search backend.
- Capture publish/update dates when the backend exposes them.

Goal: collect public evidence and generate a source-grounded summary.

Tasks:

1. Implement `Researcher`.
2. Add web search tool abstraction.
3. Store source title, URL, publish/update date if available, source type, and snippet.
4. Generate `EvidenceBundle`.
5. Rank evidence by authority and recency.

Preferred source priority:

1. Official documentation
2. Official GitHub repositories
3. Standards or RFC pages
4. Vendor blogs
5. Community posts

Deliverables:

- `evidence.json`
- source-grounded research summary

Acceptance:

```text
veriknow research "LangChain multi-agent supervisor workflow"
```

The output must include source URLs and confidence levels.

## Phase 3: Verification Planning

Status: Completed.

Current scope:

- `veriknow plan <run_id>` reads the run's `evidence.json` artifact when available.
- The planner generates source-level verification steps, expected results, required tools, screenshot flags, and approval flags.
- The command writes both `verification_plan.json` and `verification_checklist.md`, then records both artifacts in SQLite.

Goal: turn evidence claims into executable verification steps.

Tasks:

1. Extract testable claims from `EvidenceBundle`.
2. Generate `VerificationPlan`.
3. Classify verification method:
   - browser
   - CLI
   - API
   - manual checkpoint
4. Mark risky steps that require approval.

Deliverables:

- `verification_plan.json`
- human-readable verification checklist

Acceptance:

```text
veriknow plan <run_id>
```

The plan should include expected result, required tools, and screenshot points.

## Phase 4: Browser Verification

Status: Completed.

Current scope:

- `veriknow verify <run_id>` reads `verification_plan.json`.
- Browser steps that do not require approval are executed through the browser verifier.
- Playwright is used when installed and available.
- When Playwright is unavailable, the verifier records a traceable partial result with step screenshots and logs so the local workflow remains testable.
- The command writes `verification.json` and records it as a run artifact.

Goal: verify information through controlled browser actions and screenshots.

Tasks:

1. Implement Playwright browser tool.
2. Capture screenshots per important step.
3. Store browser logs and page URLs.
4. Redact sensitive fields before storing screenshots when possible.
5. Produce `VerificationRun`.

Scope for first version:

- public pages
- documentation pages
- console-free workflows
- no login automation unless explicitly enabled

Deliverables:

- `verification.json`
- `screenshots/step-*.png`
- `logs/browser.log`

Acceptance:

```text
veriknow verify <run_id>
```

At least three verification steps should produce screenshots when the task requires a visual guide.

## Phase 5: Markdown Writer

Status: Completed.

Goal: generate operation-manual style Markdown.

Tasks:

1. Implement report template.
2. Include title, status, verified date, confidence, sources, and screenshots.
3. Include step-by-step guide.
4. Include outdated information section.
5. Include verification summary.

Deliverables:

- `report.md`
- local image references

Acceptance:

```text
veriknow write <run_id>
```

The generated Markdown must be readable without opening the memory database.

## Phase 6: Knowledge Curator

Status: Completed.

Current scope:

- `veriknow curate <run_id>` reads the generated `report.md`, indexes local Markdown knowledge, selects the most related existing file or proposes a new `data/knowledge/general/*.md` target.
- The curator writes `patch.diff` and `knowledge_patch.json` into the run directory without modifying the knowledge base.
- `veriknow apply <run_id>` is the explicit approval step. It applies the generated report to the selected knowledge target, marks the patch as approved, and completes the run.
- Apply is constrained to the configured knowledge directory so a modified patch cannot write outside `data/knowledge`.

Goal: compare generated report with existing knowledge and propose updates.

Tasks:

1. Add local knowledge directory indexing.
2. Find related existing Markdown files.
3. Generate `KnowledgePatch`.
4. Produce a diff before replacing content.
5. Add approval flag before applying changes.

Deliverables:

- `patch.diff`
- updated `data/knowledge/**/*.md` after approval

Acceptance:

```text
veriknow curate <run_id>
veriknow apply <run_id>
```

The system must not overwrite existing knowledge without an explicit apply command.

## Phase 7: Internal Knowledge Retrieval

Status: Completed.

Current scope:

- `veriknow kb search "<query>"` searches local Markdown knowledge by keyword.
- Public research and curation write `related_knowledge.json` artifacts for each run.
- Search is deterministic and file-based; vector search is still intentionally deferred.

Goal: search existing local knowledge before creating new content.

Tasks:

1. Implement Markdown indexer.
2. Add keyword search first.
3. Add vector search later only if keyword search is insufficient.
4. Feed related documents into the research and curator stages.

Deliverables:

- local knowledge search
- related document list per run

Acceptance:

```text
veriknow kb search "multi-agent"
```

## Phase 8: Feishu Publisher

Status: Completed.

Current scope:

- `PublicationJob` is implemented as the publish lifecycle record.
- `Publisher` and `FeishuPublisher` are isolated from research, verification, and curation.
- `veriknow publish <document_path> --target feishu` exists.
- Publishing is constrained to Markdown documents inside `data/knowledge` that were approved through `veriknow apply`.
- Publication attempts are recorded in SQLite.
- `veriknow memory publications` lists recorded publication jobs.
- Feishu destination config keys exist for `publisher_allow_stub`, `feishu_base_url`, `feishu_folder_token`, `feishu_document_url_template`, and `feishu_title_strategy`.
- The Feishu adapter retrieves tenant access tokens, creates documents, appends converted Markdown blocks, and records document IDs and URLs when available.
- Missing credentials and API failures are recorded as failed or blocked `PublicationJob` results.

Goal: publish approved Markdown documents to Feishu.

Tasks:

1. Define `Publisher` interface.
2. Implement Feishu API or CLI adapter.
3. Keep credentials outside the repository.
4. Upload only approved final Markdown.
5. Record publication result in memory.

Deliverables:

- `PublicationJob`
- Feishu document URL stored in memory

Acceptance:

```text
veriknow publish <document_path> --target feishu
```

## Phase 9: Computer Use Integration

Status: Completed for the local safety boundary.

Current scope:

- `veriknow verify <run_id> --mode computer-use` routes browser verification steps through a computer-use verifier boundary.
- The computer-use adapter enforces a domain allowlist before any step can run.
- Login, password, payment, billing, destructive, and account-change terms are treated as approval checkpoints.
- Allowed steps record planned actions, observations, logs, and screenshots in `verification.json`.
- The current local build records a traceable partial result when no live computer-use runtime is configured.

Future enhancements:

- Connect the boundary to a real isolated browser or VM runtime.
- Add richer action telemetry once the runtime is selected.
- Expand approval policy from keyword checks to structured risk classification.

Goal: support complex UI verification that Playwright cannot reliably handle.

Tasks:

1. Add computer-use tool adapter.
2. Run it only in an isolated browser or VM.
3. Add domain allowlist.
4. Add approval checkpoints for login, payments, account changes, or destructive actions.
5. Store screenshots, actions, and observations.

Deliverables:

- computer-use verification mode
- safety policy config

Acceptance:

```text
veriknow verify <run_id> --mode computer-use
```

## Phase 10: Scheduled Re-Verification

Status: Completed for the local CLI workflow.

Current scope:

- Generated Markdown reports include `next_verify_at` front matter.
- `veriknow stale` lists local Markdown knowledge documents that are due, missing `next_verify_at`, or have invalid dates.
- `veriknow reverify <document_path>` creates a new run for an existing knowledge document.
- Re-verification runs research, planning, verification, report generation, and patch generation.
- Re-verification does not overwrite the existing knowledge document; the generated patch still requires explicit `veriknow apply <run_id>`.

Future enhancements:

- Add a real scheduler or cron integration around `veriknow stale`.
- Add per-document re-verification intervals.
- Add richer stale policies based on source type, confidence, and verification result.

Goal: keep knowledge fresh.

Tasks:

1. Add `next_verify_at` metadata to Markdown front matter.
2. Add scheduler command.
3. Re-run research and verification for stale documents.
4. Generate update patches.

Deliverables:

- stale document list
- scheduled verification run records

Acceptance:

```text
veriknow stale
veriknow reverify <document_path>
```

## Next Major Release: AI-Driven Expansion

Status: Planned.

Goal: extend the completed local workflow with model-assisted research, source analysis, computer-use execution, and knowledge merge while preserving the current safety model.

The next release should keep this invariant:

```text
AI may propose evidence, actions, plans, and patches.
Only reviewed patches may change the local knowledge base.
Only approved local knowledge documents may be published to Feishu.
```

Architectural direction:

```text
User request
  -> Task normalization
  -> AI-assisted research
  -> Evidence extraction
  -> Verification planning
  -> Browser or computer-use verification
  -> Markdown report
  -> AI-assisted knowledge merge proposal
  -> Human review
  -> Local apply
  -> Feishu publish or update
```

Non-negotiable constraints:

1. Keep SQLite and local Markdown as the source of truth.
2. Keep generated patches reviewable before applying them.
3. Store prompts, model outputs, tool calls, screenshots, logs, and source URLs as run artifacts.
4. Treat model output as untrusted until validated against schemas and safety policies.
5. Block login, payment, destructive, account-change, and credential-handling actions unless explicitly approved.
6. Do not allow model-generated paths to write outside the configured knowledge directory.

## Phase 11: Model Provider Layer

Status: Planned.

Goal: introduce a replaceable model layer without coupling business modules to one vendor.

Proposed structure:

```text
veriknow/
  llm/
    __init__.py
    client.py
    config.py
    prompts.py
    schemas.py
    providers/
      openai.py
      stub.py
```

Initial interface:

```text
LLMClient
  generate_text(prompt, context) -> str
  generate_json(prompt, schema, context) -> dict
  classify(prompt, labels, context) -> str
```

Configuration keys:

```yaml
model_provider: "stub"
model_name: ""
model_api_key_env: "OPENAI_API_KEY"
model_base_url: ""
model_temperature: 0
model_timeout_seconds: 60
model_max_output_tokens: 4000
model_store_prompts: true
```

Tasks:

1. Add model config parsing with safe defaults.
2. Add a deterministic stub provider for tests.
3. Add an OpenAI-compatible provider behind optional dependencies.
4. Validate all JSON model outputs against typed schemas.
5. Persist prompt and response artifacts under `data/runs/<run_id>/llm/`.
6. Add retry and structured error records without hiding failures.

Deliverables:

- `LLMClient` interface
- stub provider
- OpenAI-compatible provider
- prompt and response artifacts
- unit tests for JSON validation and failure handling

Acceptance:

```text
veriknow llm check
pytest
```

The command should report whether the configured model provider is available without making any knowledge-base changes.

## Phase 12: AI-Assisted Requirement Normalization

Status: Planned.

Goal: improve `TaskSpec` generation for vague, multi-part, or domain-specific requests.

Tasks:

1. Add an AI normalization strategy behind the existing `RequirementNormalizer`.
2. Preserve the deterministic normalizer as a fallback.
3. Extract objective, target, scope, constraints, output format, and verification method.
4. Flag ambiguous requests and risky objectives.
5. Store the raw model output and the validated `TaskSpec`.

CLI option:

```text
veriknow run "<request>" --normalizer ai
```

Deliverables:

- AI normalization strategy
- schema validation errors surfaced to the user
- fallback path to deterministic normalization

Acceptance:

```text
veriknow run "调研 OpenAI 最新 Responses API 用法并生成可发布到飞书的操作指南" --normalizer ai --dry-run
```

The resulting `TaskSpec` should be structured, localized, and traceable to the stored model artifact.

## Phase 13: Live Search and AI Evidence Extraction

Status: In progress; live search provider boundary completed.

Current scope:

- `search_provider` config and `veriknow research --search-provider ...` select the search backend.
- Static seed search remains the default deterministic backend for local and test runs.
- Brave Search is available as an optional live backend using `BRAVE_SEARCH_API_KEY` or `search_api_key_env`.
- Missing live-search credentials fail explicitly instead of silently falling back to static results.
- `search_fetch_pages: true` writes normalized page snapshots to `fetched_documents.json`.
- `search_store_raw_pages: true` also stores raw HTML under `data/runs/<run_id>/raw_pages/` and records each raw path.
- Deterministic claim extraction writes `extracted_claims.json` from fetched documents.
- Deterministic claim extraction writes `extracted_claims.json` from fetched documents.

Goal: replace static seed search with real search backends and model-assisted evidence extraction.

Proposed provider structure:

```text
veriknow/tools/web_search.py
  StaticSeedSearchProvider
  BraveSearchProvider
  BingSearchProvider
  SerpApiSearchProvider
  HybridSearchProvider
```

Suggested new schemas:

```text
EvidenceClaim
  text
  source_url
  source_title
  quote
  source_type
  published_at
  updated_at
  confidence
  freshness
  conflicts

FetchedDocument
  url
  title
  text
  fetched_at
  status_code
  content_hash
```

Tasks:

Completed:

1. Add search provider configuration and CLI override.
2. Keep static seed search as the default deterministic backend.
3. Add Brave Search as the first optional live backend.
4. Fetch and normalize source pages into `fetched_documents.json`.
5. Optionally store raw HTML under `raw_pages/` only when explicitly enabled.

Remaining before Phase 15:

1. Promote deterministic `EvidenceClaim` extraction from initial implementation to stable artifact contract.
2. Extract claims, dates, version constraints, quotes, source caveats, and freshness metadata.
3. Add conflict detection across extracted claims.
4. Preserve raw search results, fetched documents, extracted claims, conflicts, and final `EvidenceBundle`.
5. Let AI research use fetched page text when available, with deterministic fallback and stored model artifacts.
6. Add evaluation fixtures for claim extraction and conflict detection before expanding providers.

Deferred:

1. Add Bing, SerpApi, or hybrid search only after the claim extraction contract is stable.
2. Generate multiple search queries from a task only after single-query live search is reliable.

Configuration keys:

```yaml
search_provider: "static"
search_api_key_env: ""
search_result_limit: 8
search_fetch_pages: true
search_store_raw_pages: false
research_strategy: "deterministic"
```

CLI options:

```text
veriknow research "<query>" --search-provider brave --strategy ai
veriknow research --run-id <run_id> --search-provider brave --strategy ai
```

Deliverables:

Completed:

- live search provider boundary
- Brave Search provider
- page fetcher
- normalized fetched document artifact
- optional raw HTML storage

Next deliverables:

- `EvidenceClaim` schema and deterministic artifact
- AI claim extractor using fetched page text
- conflict detection
- source freshness and caveat metadata
- richer `evidence.json`

Acceptance:

```text
veriknow research "latest OpenAI Responses API tool calling" --search-provider brave --strategy ai
```

The evidence output must include concrete source URLs, extracted claims, source dates when available, and confidence reasoning.

## Phase 14: AI Verification Planning

Status: Completed for the explicit AI planning strategy.

Current scope:

- `veriknow plan <run_id> --strategy ai` uses the configured model provider to propose a verification plan.
- The deterministic planner remains the fallback and default strategy.
- AI planning stores the prompt, seed plan, model output, fallback status, and validated plan under `data/runs/<run_id>/llm/planner.json`.
- Model-generated browser and computer-use steps are rejected unless they include a concrete source URL.

Goal: generate higher-quality verification plans from extracted claims and source conflicts.

Tasks:

1. Convert `EvidenceClaim` records into testable verification steps.
2. Classify each step as browser, API, CLI, manual, or computer-use.
3. Identify steps requiring screenshots.
4. Identify steps requiring approval.
5. Add expected evidence for pass/fail decisions.
6. Keep deterministic planning as fallback.

CLI option:

```text
veriknow plan <run_id> --strategy ai
```

Deliverables:

- AI verification planner
- stricter `VerificationStep` validation
- conflict-aware manual checkpoints

Acceptance:

```text
veriknow plan <run_id> --strategy ai
```

The plan should directly reference extracted claims and should not create browser steps without a URL or verifiable expected result.

## Phase 15: AI-Assisted Knowledge Merge

Status: Planned.

Goal: merge verified reports into local Markdown knowledge more precisely than full-file replacement.

Suggested schema:

```text
KnowledgeMergeProposal
  operation: create | update | append | replace_section | mark_stale
  target_path
  target_title
  rationale
  evidence_urls
  conflicts
  diff
  risk_level
```

Tasks:

1. Retrieve related local knowledge documents.
2. Ask the model to choose create, update, append, replace section, or mark stale.
3. Generate section-level diffs when an existing document is selected.
4. Require evidence URLs for every substantial new claim.
5. Record conflicts instead of silently resolving them.
6. Reject proposals that remove source metadata without replacement.
7. Preserve `veriknow apply <run_id>` as the only write path.

CLI option:

```text
veriknow curate <run_id> --strategy ai
```

Deliverables:

- `knowledge_merge_proposal.json`
- section-level `patch.diff`
- explicit conflict list
- safety validation before apply

Acceptance:

```text
veriknow curate <run_id> --strategy ai
veriknow apply <run_id>
```

The system must show what changed, why it changed, which sources support the change, and whether unresolved conflicts remain.

## Phase 16: AI Safety, Evaluation, and Observability

Status: Planned.

Goal: make model-assisted behavior inspectable, testable, and safe enough for routine use.

Tasks:

1. Add golden tests for prompt inputs and expected structured outputs.
2. Add replay tests from stored run artifacts.
3. Add safety tests for path traversal, risky computer-use actions, and unapproved publishing.
4. Add evaluation fixtures for source ranking, claim extraction, and merge quality.
5. Add cost, latency, token usage, and model error metrics to run artifacts.
6. Add a `veriknow inspect <run_id>` command for reviewing all artifacts in one place.
7. Add redaction helpers for secrets, tokens, cookies, and credentials before storing logs.

Deliverables:

- AI evaluation fixtures
- replayable run artifacts
- model usage metadata
- inspection command
- redaction utilities

Acceptance:

```text
veriknow inspect <run_id>
pytest
```

Reviewers should be able to understand which model calls were made, what evidence was used, which actions were executed, and why each knowledge change was proposed.

## Phase 17: Feishu Update and Publication Sync

Status: Planned.

Goal: treat Feishu as a publication target that can create or update documents while local Markdown remains authoritative.

New publication metadata:

```text
local_path
local_content_hash
target
target_document_id
target_url
last_published_at
last_published_hash
remote_revision
status
```

Tasks:

1. Store stable Feishu document IDs per local knowledge document.
2. Add update-existing-document support.
3. Skip publishing when local content hash has not changed.
4. Detect remote update conflicts when Feishu exposes revision metadata.
5. Convert Markdown blocks more faithfully, including headings, lists, links, code blocks, and images.
6. Keep publication jobs idempotent where possible.

CLI options:

```text
veriknow publish <document_path> --target feishu
veriknow publish <document_path> --target feishu --update
veriknow memory publications
```

Deliverables:

- Feishu document mapping
- create/update publish modes
- content hash tracking
- richer Markdown-to-Feishu conversion

Acceptance:

```text
veriknow publish data/knowledge/general/example.md --target feishu --update
```

If the document has already been published and the local content changed, the existing Feishu document should be updated or a conflict should be recorded.

## Phase 18: AI-Driven Computer Use Runtime

Status: Planned.

Goal: connect the existing computer-use safety boundary to a real, isolated browser or VM runtime.

Proposed structure:

```text
veriknow/tools/computer_use/
  safety.py
  runtime.py
  runtime_playwright.py
  agent.py
  verifier.py
```

Runtime interface:

```text
ComputerRuntime
  open(url)
  screenshot()
  click(target)
  type_text(target, text)
  scroll(direction)
  wait(seconds)
  close()
```

Agent action schema:

```json
{
  "action": "click | type | scroll | wait | finish | fail",
  "target": "",
  "text": "",
  "reason": "",
  "requires_approval": false
}
```

Execution loop:

```text
open URL
  -> screenshot
  -> model observes page
  -> model proposes next action
  -> safety policy validates action
  -> runtime executes action
  -> screenshot and log
  -> repeat until finish, fail, blocked, or max steps
```

Tasks:

1. Split the current computer-use module into safety, runtime, agent, and verifier components.
2. Implement a Playwright-backed runtime for public pages.
3. Add a model-driven page observation and action proposal loop.
4. Restrict first release to read-only browsing and documentation verification.
5. Add max-step, max-time, domain, and action allowlist controls.
6. Store every screenshot, action proposal, safety decision, and runtime result.
7. Block or pause on login, payment, deletion, account mutation, file upload, and credential prompts.

Configuration keys:

```yaml
computer_use_runtime: "playwright"
computer_use_max_steps: 12
computer_use_max_seconds: 180
computer_use_read_only: true
computer_use_store_screenshots: true
computer_use_require_approval_for_forms: true
```

CLI options:

```text
veriknow verify <run_id> --mode computer-use --computer-use-runtime playwright
veriknow verify <run_id> --mode computer-use --include-approval-required
```

Deliverables:

- isolated browser runtime
- model action loop
- safety policy decisions
- step-level screenshot sequence
- structured action logs

Acceptance:

```text
veriknow verify <run_id> --mode computer-use --computer-use-runtime playwright
```

For allowed public documentation domains, the verifier should navigate, inspect pages, record screenshots, and finish with a clear pass, partial, failed, or blocked result.

## Execution Order

Build in this order:

1. Phase 0: Bootstrap
2. Phase 1: Normalizer and memory
3. Phase 2: Public research
4. Phase 5: Markdown writer
5. Phase 3: Verification planning
6. Phase 4: Browser verification
7. Phase 6: Knowledge curator
8. Phase 7: Internal knowledge retrieval
9. Phase 8: Feishu publisher
10. Phase 9: Computer use
11. Phase 10: Scheduled re-verification

This order gives an early usable product before adding the riskiest integrations.

For the AI-driven expansion, build in this order:

Completed or underway:

1. Phase 11: Model provider layer.
2. Phase 12: AI-assisted requirement normalization.
3. Phase 13A: Live search provider boundary and fetched page artifacts.
4. Phase 14A: Explicit AI verification planning strategy.

Near-term order:

1. Phase 13B: Structured `EvidenceClaim` extraction from fetched documents.
2. Phase 13C: Source freshness, caveat metadata, and conflict detection.
3. Phase 15A: `KnowledgeMergeProposal` schema and deterministic section-level merge proposal.
4. Phase 15B: AI-assisted section-level merge with evidence URL requirements.
5. Phase 16A: `veriknow inspect <run_id>` and redaction utilities for reviewing model/tool artifacts.
6. Phase 17: Feishu update and publication sync.
7. Phase 16B: replay/evaluation fixtures for claim extraction, merge quality, and safety checks.
8. Phase 18: AI-driven computer-use runtime.

This order improves evidence quality and reviewability before adding autonomous UI execution, which has the highest operational risk.

## First Sprint

Duration: 3 to 5 days.

Scope:

1. Bootstrap Python package.
2. Implement schemas.
3. Implement CLI skeleton.
4. Implement SQLite memory store.
5. Implement dry-run task normalization.
6. Generate a placeholder Markdown report from a `TaskSpec`.

First sprint commands:

```text
veriknow init
veriknow run "研究 LangChain 多智能体协作的最新做法" --dry-run
veriknow memory runs
veriknow write <run_id>
pytest
```

First sprint exit criteria:

- The project can be installed locally.
- A run can be created and inspected.
- A normalized task is stored.
- A Markdown file is generated.
- Tests cover schemas and memory store basics.

## Testing Strategy

Unit tests:

- schema validation
- config loading
- memory store CRUD
- Markdown rendering
- knowledge diff generation

Integration tests:

- CLI dry run
- research to evidence bundle
- verification plan generation
- report generation from stored artifacts

Manual tests:

- browser screenshot capture
- Feishu publishing
- computer-use verification

## Non-Goals for MVP

- Full autonomous enterprise publishing
- Login automation
- Multi-user permission system
- Distributed task queue
- Full vector database infrastructure
- Automatic overwrite of official knowledge

These can be added after the local workflow is reliable.

## Immediate Next Step

Finish Phase 13 before starting AI merge work:

1. Add an `EvidenceClaim` schema with source URL, source title, quote, freshness, caveats, confidence, and conflict fields.
2. Write `extracted_claims.json` from fetched documents, with a deterministic extractor first and an AI extractor behind `--strategy ai`.
3. Feed extracted claims into `evidence.json` summaries without breaking the existing `EvidenceBundle.items` contract.
4. Add conflict detection across claims from different sources and persist `claim_conflicts.json` when conflicts exist.
5. Update `veriknow plan --strategy ai` context to include extracted claims when available.
6. Start Phase 15 only after claim artifacts are stable and covered by tests.
7. Keep all knowledge updates behind patch review and explicit apply.
