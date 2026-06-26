#!/bin/bash
# Machine-agnostic NAMO environment activation.
#
#   source scripts/env/activate.sh
#
# All machine-specific facts live in <parent>/.env (one dir above this repo),
# copied from namo/.env.example. This script sources that file and DERIVES the
# composite vars (PYTHONPATH, LD_LIBRARY_PATH). Works on ilab and amarel because
# every difference is isolated to .env. Idempotent; safe to re-source.
#
# Do NOT add `set -e` here — this file is sourced, an exit would kill the shell.

_self="${BASH_SOURCE[0]:-$0}"
_env_dir="$(cd "$(dirname "$_self")" && pwd)"          # <repo>/scripts/env
_repo_here="$(cd "$_env_dir/../.." && pwd)"            # <repo>
_env_file="$(cd "$_repo_here/.." && pwd)/.env"         # <parent>/.env

if [[ ! -f "$_env_file" ]]; then
  echo "✗ No .env at $_env_file"
  echo "  Copy the template:  cp $_repo_here/.env.example $_env_file   (then edit for this machine)"
  return 1 2>/dev/null || exit 1
fi

set -a; source "$_env_file"; set +a
: "${NAMO_REPO:=$_repo_here}"

# Toolchain — only do what this machine asked for in .env.
[[ -n "${NAMO_MODULES:-}" ]] && module load ${NAMO_MODULES}
if [[ -n "${NAMO_CONDA_SH:-}" ]]; then
  source "$NAMO_CONDA_SH"
  conda activate "${NAMO_CONDA_ENV:?set NAMO_CONDA_ENV in .env when NAMO_CONDA_SH is set}"
elif [[ -n "${NAMO_PYTHON:-}" ]]; then
  export PATH="$(dirname "$NAMO_PYTHON"):$PATH"
fi

# MuJoCo libs (belt-and-suspenders on ilab, where the .so bakes a RUNPATH).
export LD_LIBRARY_PATH="${MJ_PATH}/lib:${MJ_PATH}/build/lib${NAMO_CONDA_ENV:+:${NAMO_CONDA_ENV}/lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Canonical imports: build_python (compiled namo_rl) + python + sage_learning.
export PYTHONPATH="${NAMO_REPO}/build_python:${NAMO_REPO}/python${NAMO_SAGE:+:${NAMO_SAGE}}${PYTHONPATH:+:${PYTHONPATH}}"

# BLAS hygiene — multiprocessing.Pool already parallelizes our work.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

cd "$NAMO_REPO" || return 1 2>/dev/null
echo "✓ NAMO env active on $(hostname)"
echo "  NAMO_REPO=$NAMO_REPO"
echo "  MJ_PATH=$MJ_PATH"
echo "  python: $(command -v python)  ($(python --version 2>&1))"
