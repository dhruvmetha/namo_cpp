---
name: model-selection
description: Use whenever spawning a subagent (Agent tool) and choosing its `model` (opus/sonnet/haiku/fable) and `effort` (low/medium/high/xhigh/max) — before EVERY delegation, not just expensive-looking ones. Fires IN ADDITION to task-level skills like superpowers:dispatching-parallel-agents, subagent-driven-development, or deep-research — those decide what/how to delegate, this one sets each agent's model+effort, so invoke both. Trigger on "spawn an agent to research/do X", "delegate this to a subagent", "fork agents to run in parallel", "which model should this agent use", "is this worth Opus or something cheaper", "pick effort level", or any orchestration step that sets model/effort.
---

# Model + effort selection for subagents

Two independent dials: **model** (which brain) and **effort** (how hard it thinks, all tokens: text + tool calls + thinking). Cost multiplies across both — a mechanical task at Opus+xhigh can cost 15-20x the same task at Haiku+default for zero quality gain. Default to the cheapest tier that can plausibly succeed; escalate only after a demonstrated failure, not preemptively.

**⛔ Hard constraints [USER]:** never run Fable at `xhigh`/`max` effort or in Claude Code's "ultracode" mode. Never default to the biggest model — match tier to task.

## TL;DR — pick from this table first

| Task archetype | Model | Effort | Why |
|---|---|---|---|
| Read-only recon / file search | Haiku | default (no effort knob) | Pure lookup/pattern-match, no reasoning depth to spend tokens on |
| Mechanical bulk transform (apply a known fix pattern everywhere) | Haiku (Sonnet `low` if slight judgment needed) | default / `low` | Mechanical = no reasoning required; bulk = many tokens, so per-token price dominates |
| Data aggregation / tabulation of existing outputs | Haiku or Sonnet | default / `low` | Read + summarize, minimal reasoning depth |
| Code review (routine correctness/style) | Sonnet | `high` (default) | Sonnet 5 is "near-Opus quality on coding" — save Opus for the hardest reviews |
| Adversarial verification (is this *actually* done, red-team a claim) | Opus | `high`–`xhigh` | Needs the strongest reasoning that isn't cost-prohibitive; this is exactly where a weak model gets fooled by confident-sounding output |
| Hard multi-step reasoning / architecture design | Opus | `xhigh` | `xhigh` is the documented recommended starting point for coding/agentic work on Opus 4.8 |
| Research + synthesis + long-form writing | Sonnet (`high`) for most; Opus (`high`–`xhigh`) for deep multi-source synthesis | `high` | Writing/synthesis quality tracks reasoning depth more than raw model tier for most cases |
| Quick factual lookup | Haiku | default | Latency + cost matter more than depth |
| Drafting docs (boilerplate/templated) | Haiku | default | No judgment required |
| Drafting docs (needs structure/judgment calls) | Sonnet | `medium`–`high` | Balance of quality and cost |
| Genuinely frontier-hard, long-horizon (>30 min) problem where Opus itself is the ceiling | Fable | `high` (never `xhigh`/`max`) | Reserve for cases nothing else can solve — see cost note below |

This project's own convention already encodes the pattern: `scout` (Bash/Read/Grep/Glob only) does recon and mechanical fan-out; `experiment-runner` (Opus/xhigh) does reasoning-heavy experiment design and judgment calls. Default new subagents the same way.

## Model tiers — capability, speed, cost

Current generation, per [Models overview](https://platform.claude.com/docs/en/about-claude/models/overview) and independently confirmed via web search (July 2026):

| Model | API ID | Input $/MTok | Output $/MTok | Relative cost vs Haiku | Positioning |
|---|---|---|---|---|---|
| Claude Fable 5 | `claude-fable-5` | $10 | $50 | 10x | "Next-generation intelligence for long-running agents" — Anthropic's most capable **widely released** model |
| Claude Opus 4.8 | `claude-opus-4-8` | $5 | $25 | 5x | "For complex agentic coding and enterprise work" — the frontier tier for day-to-day hard reasoning |
| Claude Sonnet 5 | `claude-sonnet-5` | $3 ($2 intro, through 2026-08-31) | $15 ($10 intro) | 3x (2x during intro) | "Best combination of speed and intelligence" — near-Opus quality on coding/agentic work |
| Claude Haiku 4.5 | `claude-haiku-4-5` | $1 | $5 | 1x (baseline) | "Fastest model with near-frontier intelligence" — the bulk/cheap tier |

Frontier/hardest-reasoning tier = **Fable 5**. Cheap/fast/bulk tier = **Haiku 4.5**. Opus 4.8 and Sonnet 5 sit between them — Opus for genuinely hard reasoning, Sonnet as the default workhorse (near-Opus quality on coding/agentic tasks at ~60% of the cost, or ~40% during Sonnet 5's intro pricing).

Context/output: Fable 5, Opus 4.8, Sonnet 5 all have a 1M-token context window and 128K max output. Haiku 4.5 has 200K context and 64K max output — plenty for recon/mechanical work, rarely a real constraint at that tier.

Sources: [Models overview](https://platform.claude.com/docs/en/about-claude/models/overview.md), cross-checked via web search July 2026 (independent pricing writeups all agree on $10/$50, $5/$25, $3/$15 with $2/$10 Sonnet 5 intro through 2026-08-31, $1/$5).

## Effort levels — what they actually change

Per [Effort](https://platform.claude.com/docs/en/build-with-claude/effort.md): effort affects **all tokens in the response** — text, tool calls, and thinking (when enabled) — not just a thinking budget. This is why effort and model tier compound: a chattier, more-tool-call-heavy high-effort run costs more on every axis at once.

| Level | What changes | Typical use |
|---|---|---|
| `low` | Fewest, most-consolidated tool calls; proceeds straight to action, no preamble; terse completion messages | Subagents, simple/scoped tasks, latency-sensitive bulk work |
| `medium` | Balanced — moderate token savings | Agentic tasks needing a speed/cost/quality balance |
| `high` (default) | Equivalent to omitting the param entirely | Complex reasoning, difficult coding, most agentic tasks |
| `xhigh` | Extended capability for long-horizon work; documented for tasks running >30 min with token budgets in the millions | Long-running agentic/coding tasks — the *recommended starting point* for coding/agentic work on Opus 4.7/4.8 |
| `max` | Absolute ceiling, no token constraint | Genuinely frontier problems only — usually adds cost for little quality gain; can cause overthinking on structured/less-intelligence-sensitive tasks |

Effort is "a behavioral signal, not a strict token budget" — even at `low`, the model will still think on a sufficiently hard problem, just less than at `high` for the same problem.

**Model support gaps (verify before setting):**
- `xhigh` is supported on Fable 5, Opus 4.8, Opus 4.7, and Sonnet 5 — **not** on Opus 4.6, Sonnet 4.6, or Haiku 4.5.
- **Haiku 4.5 has no `effort` parameter at all** (absent from the supported-model list in the docs). Haiku uses classic extended thinking (`budget_tokens`) instead of adaptive thinking + effort. Default (thinking off) is normally right for the bulk/recon work Haiku is used for — don't try to set `effort` on a Haiku subagent.
- Every model in the current lineup (Fable 5, Opus 4.8, Sonnet 5) defaults to `high` if you don't set it explicitly.

Source: [Effort](https://platform.claude.com/docs/en/build-with-claude/effort.md).

## Why "never Fable at xhigh/max or ultracode"

Fable 5 is already 2x Opus 4.8's per-token price. `xhigh` on Fable is documented for long-horizon tasks with **token budgets in the millions**. Claude Code's "ultracode" mode isn't a separate API effort level — it's `xhigh` effort *plus* standing permission for the agent to launch multiagent (fan-out) workflows on top of that. Stacking Fable's 2x price × millions-of-tokens `xhigh` runs × unbounded multiagent fan-out is the single fastest way to blow a budget in this environment — hence the hard ban. Reserve Fable for a problem Opus 4.8 at `xhigh` has already failed on, and even then run it at `high`, one agent at a time, not in ultracode. (Source: [Effort](https://platform.claude.com/docs/en/build-with-claude/effort.md) note on ultracode.)

## Cost-saving heuristics

- **Cheapest-tier-first, escalate on evidence.** Try Haiku/Sonnet-low; escalate to Opus only after it demonstrably fails or the task archetype table says otherwise up front. Don't pre-escalate because the task "sounds hard."
- **Fan out cheap, fan in expensive.** Many parallel Haiku/Sonnet-low recon or mechanical subagents feeding one Opus/xhigh synthesis step beats running every subagent at Opus tier.
- **Don't run high effort on mechanical work.** Effort scales tool-call chattiness too — high effort on a bulk rename/mechanical task doesn't just think more, it also makes more, longer tool calls, compounding cost for a task with no reasoning depth to exploit.
- **Match effort to horizon.** `xhigh`/`max` are for genuinely long (>30 min) or frontier-hard problems, not "I want the best possible answer" on a routine task — `high` (the default) already covers most complex reasoning and coding.
- **Reuse baselines, don't retrain/re-run.** [USER convention] — if a registered result or checkpoint already answers the question, cite it instead of spawning a fresh expensive run.
- **Sonnet 5's intro pricing** ($2/$10 through 2026-08-31) makes it even cheaper relative to Opus right now than the sticker price suggests — lean on it as the default "real work" tier while the window lasts.
- **scout vs experiment-runner split** (this repo): route recon/mechanical/state-checking to `scout` (cheap tier, narrow tools); route methodology/judgment/experiment design to `experiment-runner` (Opus/xhigh). Don't blur the line by giving scout-shaped work to the expensive agent.
