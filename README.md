# VeriKnow

VeriKnow is a local-first AI knowledge verification workflow. It turns public web information and local knowledge into verified, traceable Markdown documents.

It is designed for tasks where an answer should be backed by sources, screenshots, logs, and a reviewable knowledge-base patch instead of being pasted directly into long-term documentation.

## What It Does

Given a broad or unclear request, VeriKnow can:

1. Normalize the request into a structured task.
2. Collect public evidence and related local knowledge.
3. Generate a verification plan.
4. Run browser or computer-use-boundary verification.
5. Write a Markdown operation guide with evidence, screenshots, logs, and sources.
6. Generate a knowledge-base patch for review.
7. Apply approved knowledge updates.
8. Publish approved knowledge documents to Feishu.
9. List stale knowledge documents and re-verify them.

The project follows four rules:

- Evidence before conclusion.
- Verification before publishing.
- Diff before replacing knowledge.
- Local traceability for runs, artifacts, screenshots, and logs.

## Requirements

- Python 3.10 or newer
- Optional: Playwright Chromium for real browser verification
- Optional: Feishu credentials for publishing

## Quick Start

Install the package in editable mode with test dependencies:

```bash
python -m pip install -e .[dev]
```

Optional browser verification setup:

```bash
python -m pip install -e .[browser]
playwright install chromium
```

Initialize local config, SQLite memory, and data directories:

```bash
veriknow init
```

Run the full local workflow:

```bash
veriknow research "LangChain multi-agent supervisor workflow"
veriknow plan <run_id>
veriknow verify <run_id>
veriknow write <run_id>
veriknow curate <run_id>
```

Review the generated patch in:

```text
data/runs/<run_id>/patch.diff
```

Apply the reviewed patch to the local knowledge base:

```bash
veriknow apply <run_id>
```

Run tests:

```bash
pytest
```

## Common Workflows

Create a dry-run task and persist it to SQLite:

```bash
veriknow run "帮我研究某个工具的最新用法" --dry-run
```

Collect public evidence:

```bash
veriknow research "LangChain multi-agent supervisor workflow"
veriknow research "latest OpenAI Responses API tool calling" --search-provider brave
```

Brave search requires an API key in `BRAVE_SEARCH_API_KEY` by default, or in the environment variable named by `search_api_key_env`. When `search_fetch_pages` is enabled, research also writes normalized page text to `data/runs/<run_id>/fetched_documents.json` and deterministic claims to `data/runs/<run_id>/extracted_claims.json`; with `search_store_raw_pages`, each fetched document records its raw HTML path.

Generate a verification plan:

```bash
veriknow plan <run_id>
veriknow plan <run_id> --strategy ai
```

AI planning stores its prompt, seed plan, model output, fallback status, and validated plan under `data/runs/<run_id>/llm/planner.json`.

Run browser verification:

```bash
veriknow verify <run_id>
```

Run computer-use verification through the safety boundary:

```bash
veriknow verify <run_id> --mode computer-use
veriknow verify <run_id> --mode computer-use --computer-use-runtime playwright
```

Computer-use mode requires allowed domains in config. It records actions, observations, logs, and screenshots when the step is allowed. Login, payment, destructive, and account-change actions remain behind explicit approval.

Generate a Markdown report:

```bash
veriknow write <run_id>
```

Generate a knowledge update patch without changing the knowledge base:

```bash
veriknow curate <run_id>
```

Apply an approved patch:

```bash
veriknow apply <run_id>
```

Search local Markdown knowledge:

```bash
veriknow kb search "multi-agent"
```

List stale knowledge documents:

```bash
veriknow stale
```

Re-run research and verification for a knowledge document:

```bash
veriknow reverify data/knowledge/general/example.md
```

Re-verification creates a new run, fresh evidence, verification artifacts, a report, and a proposed patch. It does not overwrite the knowledge document; use `veriknow apply <run_id>` after reviewing the patch.

Replay a local evaluation fixture or run artifact directory:

```bash
veriknow eval tests/fixtures/phase13_metadata_eval.json
veriknow eval data/runs/<run_id>
```

Publish an approved local knowledge document to Feishu:

```bash
veriknow publish data/knowledge/general/example.md --target feishu
veriknow publish data/knowledge/general/example.md --target feishu --update
```

The document must come from an approved `veriknow apply <run_id>` result. Draft reports in `data/runs` and unapproved Markdown files are rejected. Update mode reuses the stored publication mapping, skips unchanged local content by hash, and records blocked or failed remote updates without modifying local Markdown.

## Inspect Local Memory

```bash
veriknow memory runs
veriknow memory show <run_id>
veriknow memory publications
veriknow memory publication-mappings
```

Append a task-relevant preference signal:

```bash
veriknow preference output_structure "prefer concise checklists"
```

## Configuration

Default config is written to `config.yaml` by `veriknow init`.

Core keys:

```yaml
data_dir: data
database_path: data/memory.sqlite
default_scope: public_web
default_output_format: markdown
default_publish_target: local
```

Computer-use safety keys:

```yaml
computer_use_domain_allowlist: "docs.langchain.com,github.com"
computer_use_approval_keywords: "login,sign in,password,billing,payment,purchase,delete,destructive,account change,account settings"
computer_use_action_allowlist: "open,screenshot,observe,scroll,wait,finish,fail"
```

Scheduled re-verification:

```yaml
default_reverify_interval_days: 30
```

Search provider keys:

```yaml
search_provider: "static"
search_api_key_env: ""
search_result_limit: 5
search_fetch_pages: false
search_store_raw_pages: false
```

Set `search_provider` to `brave` for live search. Set `search_fetch_pages` to `true` to store normalized fetched page text, and set `search_store_raw_pages` to `true` to also retain raw HTML under `data/runs/<run_id>/raw_pages/`.

Model provider keys:

```yaml
model_provider: "zhipu"
model_name: "glm-5.2"
model_api_key_env: "ZHIPUAI_API_KEY"
model_base_url: "https://open.bigmodel.cn/api/paas/v4"
model_temperature: 0
model_timeout_seconds: 60
model_max_output_tokens: 4000
model_store_prompts: true
```

Optional Feishu publisher keys:

```yaml
publisher_allow_stub: true
feishu_base_url: "https://open.feishu.cn"
feishu_folder_token: ""
feishu_document_url_template: ""
feishu_title_strategy: "filename"
```

Required Feishu environment variables:

```bash
FEISHU_APP_ID=...
FEISHU_APP_SECRET=...
```

## Local Data Layout

```text
data/
  runs/
    <run_id>/
      task.json
      evidence.json
      fetched_documents.json
      extracted_claims.json
      verification_plan.json
      verification_checklist.md
      verification.json
      report.md
      patch.diff
      knowledge_patch.json
      screenshots/
      logs/
  knowledge/
  memory.sqlite
```

## Markdown Output

Generated reports include front matter for status, confidence, source URLs, verification time, and the next scheduled verification date:

```markdown
---
title: "Tool / Method / Protocol Name"
status: "completed | partial | blocked | failed | draft"
verified_at: "YYYY-MM-DDTHH:MM:SS+00:00"
next_verify_at: "YYYY-MM-DD"
confidence: "high | medium | low"
sources:
  - url: "https://example.com"
    type: "official_doc"
---
```

## Architecture

```text
User
  ↓
Adaptive Profile
  ↓
Requirement Normalizer
  ↓
Memory Store
  ├─ Researcher
  ├─ Verifier
  ├─ Writer
  ├─ Curator
  └─ Publisher
```

Key objects:

```text
TaskSpec            structured user request
EvidenceBundle      collected source evidence
VerificationPlan    executable verification steps
VerificationRun     screenshots, logs, actual results
KnowledgePatch      proposed document updates
MarkdownDocument    final verified document
PublicationJob      Feishu upload task
UserProfile         task-relevant user preferences
```

## Project Status

The local-first MVP workflow is complete and covered by tests. The current build uses a deterministic static search provider by default, an optional Brave live-search provider, and optional AI-assisted normalization, research extraction, and verification planning behind explicit strategy flags.

`DEVELOPMENT_PLAN.md` is retained as implementation history and a maintenance roadmap. It is not required for installing or running the project.

Remaining enhancement areas:

- Live web-search backend.
- Stronger isolated browser or VM runtime and model-driven action proposals for computer-use verification.
- Cron or scheduler integration around `veriknow stale`.
- Richer source freshness metadata and confidence policies.
