#!/bin/bash
# Chain I — audit fix queue (F1, F6, R6, F7, F2, F3, R7), strictly serialized,
# one GPU job at a time, memory-gated, resumable at every step.
#
# GPU-revocation aware: this box runs docker with the systemd cgroup driver
# on cgroup v2, where the NVIDIA stack can lose device access in a RUNNING
# container. Every GPU step is preceded by a canary so the chain exits
# immediately (marker GPU_LOST) instead of burning through remaining steps
# with a dead device; the host supervisor then restarts the container and
# re-runs this chain, which resumes from checkpoints.
set -u
VENVPY=/workspace/venvs/jlens/bin/python
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true   # experiments die first

gpu() {  # canary before every GPU step
  nvidia-smi -L > /dev/null 2>&1 || { echo GPU_LOST; exit 1; }
  python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    echo GPU_LOST; exit 1; }
}

gate() {  # never repeat the OOM outage: wait for >=20GB MemAvailable
  for i in $(seq 1 240); do
    avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    [ "$avail" -ge 20 ] && return 0
    echo "[chain-i] gate: only ${avail}GB free, waiting"; sleep 30
  done
  echo CHAIN_I_GATE_TIMEOUT; exit 1
}

# 0. preempt the J5 penult fit if alive (checkpointed; supervisor resumes it)
if pgrep -f "load_wikitext_prompts|fit_lens" > /dev/null 2>&1; then
  echo "[chain-i] preempting checkpointed penult fit"
  pkill -TERM -f "load_wikitext_prompts|fit_lens"; sleep 20
  echo PENULT_FIT_PREEMPTED
fi

# 1. F1 — random-jcomp control arm (gates J4's causal claim)
if [ ! -f /workspace/experiments/j4-behavioral/results/f1_gen.done ]; then
  gpu; gate
  cd /workspace/experiments/j4-behavioral && \
    python3 j4_dissociation.py --rand-jcomp && \
    touch results/f1_gen.done && echo F1_GEN_DONE || echo F1_FAILED
fi

# 2. F6 — regenerate the missing compare_u.json (report cites it)
if [ ! -f /workspace/experiments/welfare-axis/results/compare_u.json ]; then
  gpu; gate
  cd /workspace/experiments/welfare-axis && \
    python3 compare_u.py && echo F6_DONE || echo F6_FAILED
fi

# 3. R6 — norm/cosine table (CPU, seconds)
if [ ! -f /workspace/experiments/welfare-axis/results/R6_direction_table.json ]; then
  cd /workspace/experiments/welfare-axis && \
    python3 r6_direction_table.py || echo R6_FAILED
fi

# 4. F7 — k-sweep at both treatment stream positions + 12-random null
if [ ! -f /workspace/experiments/jlens-atlas/results/ksweep_ext.json ]; then
  gpu; gate
  cd /workspace/experiments/jlens-atlas && \
    $VENVPY f7_ksweep_ext.py || echo F7_FAILED
fi

# 5. F2 — J-share cohort n=100/polarity (thesis-bearing p-floor fix)
if [ ! -f /workspace/experiments/routing-core/results/jshare_cohort_n100.json ]; then
  gpu; gate
  cd /workspace/experiments/routing-core && \
    $VENVPY f2_jshare_cohort.py || echo F2_FAILED
fi

# 6. F3 — atlas cohort n=100/polarity on the F4 estimand (+ R1 raw rows)
if [ ! -f /workspace/experiments/jlens-atlas/results/atlas_cohort_n100.json ]; then
  gpu; gate
  cd /workspace/experiments/jlens-atlas && \
    $VENVPY f3_atlas_cohort.py || echo F3_FAILED
fi

# 7. R7 — whole-generation readout diagnostic for C6
if [ ! -f /workspace/experiments/routing-core/results/R7_wholegen.json ]; then
  gpu; gate
  cd /workspace/experiments/routing-core && \
    python3 r7_wholegen.py || echo R7_FAILED
fi

echo CHAIN_I_DONE
