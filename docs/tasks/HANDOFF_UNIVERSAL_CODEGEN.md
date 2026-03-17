# Agent Handoff: Universal Cross-Domain Code Generation

> Created: 2026-03-17
> Status: Ready to start
> Branch: `main` (create `feat/universal-codegen` before writing code)

---

## 0. Read This First

You are picking up a major feature for AutoResearchClaw: **making the code generation pipeline work across all research domains**, not just ML/AI. Before writing ANY code, read and understand these documents:

1. **Design Document** (your primary blueprint):
   `docs/tasks/universal_codegen_design.md`
   — Contains the full architecture, module designs, ML hardcoding inventory, competitive analysis, and phased implementation plan.

2. **CodeAgent v2 Enhancement Plan** (what was already built):
   `docs/tasks/code_agent_enhancement_plan.md`
   — The current CodeAgent v2 features you'll be extending: Blueprint Distillation, Sequential File Generation, Hard Validation Gates, Targeted Error Repair.

3. **Project Memory** (accumulated knowledge from 25+ runs):
   `.claude/projects/-home-jqliu-projects-AutoResearchClaw/memory/MEMORY.md`
   — Key architectural facts, known issues, user preferences, run history.

---

## 1. Your Goal

**Build a universal code generation framework that can produce runnable experiment code for ANY computational research domain** — physics, chemistry, biology, economics, mathematics, security, robotics, and beyond.

The current system only works for ML/AI. There are 400+ lines of ML-specific hardcoding across `prompts.py`, `executor.py`, `sandbox.py`, and data files. Your job is to **generalize without breaking existing ML functionality**.

### Success criteria (MVP):
- [ ] All 1284 existing tests still pass (zero regression)
- [ ] At least 1 non-ML domain works end-to-end (recommend: computational physics or math)
- [ ] Code Searcher can find relevant GitHub repos and extract API patterns
- [ ] Domain detection accuracy > 90% on a test set of 50 topics

### Stretch goals:
- [ ] 5+ domains working (ML, Physics, Chemistry, Economics, Biology)
- [ ] Flexible metric system (JSON/CSV output, not just stdout regex)
- [ ] Multiple Docker images by domain

---

## 2. What You MUST NOT Do

These are hard rules. Violating them causes real damage:

1. **NEVER break existing ML functionality.** All 1284 tests must pass after every change. Run `python -m pytest tests/ -x -q` frequently.
2. **NEVER rewrite prompts.py or executor.py from scratch.** These are 2395-line and 8318-line battle-tested files. Use the adapter pattern — wrap existing behavior, don't replace it.
3. **NEVER add `Co-Authored-By` lines in commits.** User preference.
4. **NEVER push to `upstream`.** Only push to `origin` (private repo).
5. **NEVER commit API keys, tokens, or credentials.** Config files with keys are gitignored.
6. **NEVER commit test output directories** (`test_outputs*/`, `records/`, `*.log`).
7. **NEVER commit without running tests first.**

---

## 3. What You Should Know

### 3.1 Project structure (key files)

```
researchclaw/
├── pipeline/
│   ├── executor.py          # 8318 lines — THE core pipeline (23 stages)
│   ├── code_agent.py        # 1324 lines — CodeAgent v2 (you'll extend this)
│   └── contracts.py         # Stage contracts and data classes
├── prompts.py               # 2395 lines — ALL LLM prompts (heavy ML hardcoding)
├── experiment/
│   └── sandbox.py           # Docker sandbox execution + metric parsing
├── agents/
│   ├── benchmark_agent/     # 4-agent benchmark selection (ML domains only)
│   └── figure_agent/        # 5-agent figure generation
├── domains/                 # DOES NOT EXIST YET — you create this
├── data/
│   ├── benchmark_knowledge.yaml  # 845 lines, 13 ML domains
│   └── dataset_registry.yaml     # 162 lines, torchvision/HF only
├── templates/
│   ├── conference.py        # LaTeX conference template
│   └── converter.py         # Markdown → LaTeX converter
├── literature/              # OpenAlex, Semantic Scholar, arXiv clients
├── config.py                # All configuration classes
└── docker/                  # Docker image definitions
```

### 3.2 How the pipeline currently generates code

```
Stage 9:  Experiment Design → exp_plan.yaml (baselines, ablations, metrics)
Stage 10: Code Generation   → main.py + supporting files (via CodeAgent v2)
Stage 11: Code Validation    → AST checks + sandbox dry-run
Stage 12: Experiment Run     → Docker sandbox execution
Stage 13: Result Collection  → Parse stdout metrics
Stage 14: Result Analysis    → LLM analysis of metrics
```

CodeAgent v2 flow (Stage 10):
```
1. Blueprint Generation (E-01) → per-file specs with pseudocode
2. Sequential File Generation (E-02) → config → data → model → train → main
3. Hard Validation (E-03) → AST checks for critical issues
4. Error Repair (E-05) → Parse traceback → fix specific file:line
```

### 3.3 Critical codebase facts

- **CodeAgentConfig duplication**: Config exists in BOTH `config.py` AND `code_agent.py`. If you add config fields, update BOTH.
- **executor.py is monolithic**: All 23 stages live in one file. Stage methods are `_stage_XX_name()`. Don't try to split it — too risky.
- **prompts.py uses f-strings with `{` escaping**: Be careful with LaTeX content in prompts (double `{{` for literal braces).
- **Docker sandbox has 3 phases**: Phase 0 (pip install) → Phase 1 (setup.py) → Phase 2 (experiment). Network is disabled before Phase 2.
- **Azure OpenAI endpoint**: `huaxi-mlg4x1rk-eastus2.services.ai.azure.com`, models: gpt-5.2, gpt-5.1, gpt-4.1, gpt-4o.

---

## 4. Your Implementation Plan

Follow the phased plan in the design doc. Here's the recommended order with practical guidance:

### Phase 1: Infrastructure Skeleton (Start Here)

**Create the adapter framework without changing any existing behavior.**

```
Tasks:
1. Create `researchclaw/domains/` package
   - detector.py      — DomainProfile dataclass + detect_domain()
   - prompt_adapter.py — PromptAdapter base class
   - adapters/
     ├── __init__.py
     ├── ml.py         — Wrap ALL current ML behavior (zero change)
     └── generic.py    — Fallback for unknown domains

2. Create domain profile YAMLs
   - researchclaw/domains/profiles/ml_vision.yaml (+ ml_nlp, ml_rl, etc.)
   - Start with 3-4 ML sub-domains to prove the pattern

3. Create UniversalMetricParser
   - researchclaw/experiment/metrics.py
   - Must support: JSON (new), CSV (new), stdout regex (existing behavior)
   - Existing stdout parsing must be default fallback

4. Wire domain detection into executor.py
   - detect_domain() call at Stage 9 entry
   - Pass DomainProfile through to code_agent
   - BUT: if domain == ML, use EXACTLY the current code path (adapter wraps it)

5. Tests
   - tests/test_domain_detector.py
   - tests/test_prompt_adapter.py
   - tests/test_metric_parser.py
   - Confirm all 1284 existing tests still pass
```

### Phase 2: Code Searcher

**The most impactful new feature — lets the system learn from existing code before generating.**

```
Tasks:
1. Create `researchclaw/agents/code_searcher/` package
   - github_client.py  — GitHub REST API (repo search + code search)
   - query_gen.py      — LLM generates search queries from topic + domain
   - pattern_extractor.py — LLM extracts API patterns from retrieved code
   - cache.py          — Disk cache with 30-day TTL

2. Integrate into Blueprint generation
   - code_agent.py: before generating blueprint, optionally search GitHub
   - Inject found patterns into blueprint prompt as reference material

3. Handle API limits gracefully
   - GitHub: 10 req/min for code, 30 req/min for repos
   - If rate-limited: skip search, continue with LLM knowledge only
   - Cache results aggressively

4. Tests
   - Mock GitHub API responses for unit tests
   - Integration test with real API (optional, needs GITHUB_TOKEN)
```

### Phase 3: First Non-ML Domain

**Prove the architecture works end-to-end on a new domain.**

```
Recommended: Computational Physics or Numerical Mathematics
Why: Simple dependencies (numpy/scipy), clear evaluation (convergence order),
     LLM has strong knowledge, easy to verify correctness.

Tasks:
1. Create physics/math domain profile YAML
2. Create physics/math PromptAdapter
3. Create Docker image config (or just use generic + pip install)
4. Adapt experiment schema for convergence studies
5. End-to-end test: give a PDE solver topic, generate code, run in sandbox
```

---

## 5. Things to Watch Out For

### 5.1 Architecture traps

- **Don't over-abstract early.** Build concrete adapters for 2-3 domains first, THEN extract common patterns. Premature abstraction here will be wrong.
- **Don't create a "domain registry" service.** Keep it simple — YAML files loaded at init. No database, no API, no runtime registration.
- **Don't modify sandbox.py's metric parsing for Phase 1.** Add the new JSON/CSV parser alongside it, not replacing it.

### 5.2 Known technical debt

- `executor.py` has 6 bugs (BUG-12 to BUG-29) from Run 18v3 that are tracked but not yet fixed. See `docs/tasks/run18_bug_tracker.md`. Don't fix them now — focus on the universal codegen task.
- There are uncommitted changes in `executor.py`, `contracts.py`, and `conference.py` on the current working tree. These may be work-in-progress from another task. **Don't discard them** — stash or work around them.

### 5.3 Testing strategy

```bash
# Run full test suite (must pass after every change)
python -m pytest tests/ -x -q

# Run specific test files
python -m pytest tests/test_benchmark_agent.py -q   # 43 tests
python -m pytest tests/test_figure_agent.py -q      # 45 tests

# Quick smoke test for code generation (no full pipeline)
python scripts/test_codegen_v2.py --topic "..." --config config_test.yaml
```

### 5.4 Git workflow

```bash
# Before starting work
git fetch origin && git merge origin/main

# Create feature branch
git checkout -b feat/universal-codegen

# Commit often, descriptive messages, NO Co-Authored-By
git commit -m "feat: add domain detector with ML adapter"

# Push to origin only
git push -u origin feat/universal-codegen
```

---

## 6. You Are Encouraged To

This is not a paint-by-numbers task. You should exercise judgment:

1. **Do your own research.** The design doc is a starting point, not gospel. If you find a better approach (e.g., a better code search API, a smarter adapter pattern), use it.

2. **Challenge the design.** If something in `universal_codegen_design.md` seems wrong or overengineered, document your reasoning and take a different path. Write your analysis in `docs/tasks/universal_codegen_notes.md`.

3. **Add domains not in the plan.** If you see an easy win (e.g., tabular data science, signal processing), go for it.

4. **Improve the Code Searcher.** The design doc mentions GitHub REST API, but you could also explore:
   - `gh` CLI for authenticated searches
   - Sourcegraph MCP server
   - Papers with Code API (find code repos linked to papers)
   - arXiv + GitHub cross-referencing

5. **Borrow ideas from competitors.** AI-Scientist-v2's BFTS tree search, AI-Researcher's Resource Analysts, OpenHands' event-sourced architecture — if any of these patterns would help, adapt them.

6. **Write tests for everything.** The project has 1284 tests and they've caught many regressions. Every new module should have comprehensive tests.

---

## 7. Task Tracking

**Use Markdown documents to track your progress.** Create and maintain:

```
docs/tasks/universal_codegen_progress.md    — Your progress log
docs/tasks/universal_codegen_notes.md       — Your research notes and design decisions
docs/tasks/universal_codegen_design.md      — The design doc (update as needed)
```

### Progress log format:

```markdown
# Universal CodeGen Progress

## Phase 1: Infrastructure

### Task 1.1: Domain Profile dataclass
- Status: IN_PROGRESS / DONE / BLOCKED
- Files: researchclaw/domains/detector.py
- Notes: ...
- Blockers: ...

### Task 1.2: ...
```

### When you complete a phase:
1. Update the progress log
2. Run the full test suite and record the result
3. Commit with a clear message
4. Push to `origin`

---

## 8. Quick Reference

| Item | Value |
|------|-------|
| Primary design doc | `docs/tasks/universal_codegen_design.md` |
| Feature branch | `feat/universal-codegen` |
| Test command | `python -m pytest tests/ -x -q` |
| Test count (baseline) | 1284 tests |
| Main files to modify | `code_agent.py`, `prompts.py`, `executor.py`, `config.py` |
| New package to create | `researchclaw/domains/` |
| New agent to create | `researchclaw/agents/code_searcher/` |
| Push target | `origin` only (NEVER `upstream`) |
| Commit style | No Co-Authored-By, descriptive messages |
| Python version | 3.11 |
| Hardware | NVIDIA RTX 6000 Ada (49GB VRAM) |
| LLM endpoint | Azure OpenAI (gpt-5.2, gpt-5.1, gpt-4.1, gpt-4o) |

---

Good luck. Build something great.
