#!/bin/bash
# chain_o — own_u J-share (2 decompositions + fresh-v gate). ~5 min.
set -u
VENVPY=/workspace/venvs/jlens/bin/python
cd /workspace/experiments/routing-core

nvidia-smi -L || { echo CHAIN_O_GPU_DEAD; exit 1; }
$VENVPY own_u_jshare.py && echo CHAIN_O_DONE || echo CHAIN_O_FAILED
