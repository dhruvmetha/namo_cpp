#!/usr/bin/env bash
# Source only after the machine-local, untracked container.env.
: "${CONTAINER_ROOT:?set CONTAINER_ROOT in container.env}"
: "${FROZEN_ROOT:?set FROZEN_ROOT in container.env}"
: "${PYTHON_ENV:?set PYTHON_ENV in container.env}"

export CONTAINER_ROOT FROZEN_ROOT PYTHON_ENV
export NAMO_REPO="$CONTAINER_ROOT/code"
export NAMO_PYTHON="$PYTHON_ENV/bin/python"
export MJ_PATH="$CONTAINER_ROOT/mujoco"
export NAMO_SCRATCH="$CONTAINER_ROOT/scratch"
export SAGE_REPO="$FROZEN_ROOT/sage_learning"
export PYTHONPATH="$NAMO_REPO/build_python:$NAMO_REPO/python:$NAMO_REPO/scripts:$NAMO_REPO/scripts/sandbox:$NAMO_REPO/scripts/pipeline:$SAGE_REPO"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export NAMO_GLOBAL_SEED=42 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1
unset NAMO_DISABLE_MOVABLE_BLOB_EDGES CMAKE_PREFIX_PATH CPATH CPLUS_INCLUDE_PATH LIBRARY_PATH

activate_native_runtime() {
    # Inside the image only: don't affect host Apptainer or its dependencies.
    export PATH="/usr/bin:/bin:$PYTHON_ENV/bin"
    # Match dhruv's actual search path. Python's own RPATH resolves its Conda
    # dependencies; forcing either Conda or system libraries first changes it.
    export LD_LIBRARY_PATH="$MJ_PATH/build/lib:$MJ_PATH/lib"
}

refuse_existing() {
    local artifact
    for artifact in "$@"; do
        if [[ -e "$artifact" || -L "$artifact" ]]; then
            printf 'refusing existing artifact: %s\n' "$artifact" >&2
            return 1
        fi
    done
}
