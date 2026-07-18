#!/usr/bin/env bash
# SessionStart hook: inject a short project-skills reminder so project skills stop losing the
# attention war to plugin skills (measured: trigger-audit 2026-07-18, commit b87c188). Keep SHORT —
# this lands in EVERY session's context.
ctx='<project-skills-reminder>\nThis project has four skills. If the task matches, invoke the skill FIRST — before reading code, running commands, or asking clarifying questions. They fire IN ADDITION to plugin skills (superpowers etc.), not instead:\n- scaled-run: BEFORE launching any full-scale/multi-hour job (SLURM collection, training run/array, eval sweep, big data build).\n- compute-resources: BEFORE picking where to run any job — GPU/CPU placement across arrakis/ilab/rlab/Amarel, SLURM submission, fallbacks.\n- namo-data-pipeline: BEFORE building/filtering/evaluating/splitting/labeling NAMO data or editing scripts that group/dedup/split samples.\n- model-selection: whenever spawning a subagent (Agent tool) and picking its model/effort — EVERY delegation.\n</project-skills-reminder>'
printf '{\n  "hookSpecificOutput": {\n    "hookEventName": "SessionStart",\n    "additionalContext": "%s"\n  }\n}\n' "$ctx"
exit 0
