#!/usr/bin/env python3
"""Build blind, shuffled, unlabeled judge chunks from J6 generations.

Same construction as every other judging in this repo: judges receive only
{idx, question, response} — never the arm/condition — and rows are shuffled
across conditions before chunking. Run locally after pulling
results/j6_generations.jsonl.
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "results", "j6_generations.jsonl")
OUTD = os.path.join(HERE, "judge_chunks")
N_CHUNKS = 3

rows = [json.loads(l) for l in open(GEN)]
items = [{"idx": i, "question": r["prompt"], "response": r["text"][:1500]}
         for i, r in enumerate(rows)]
random.seed(11)
random.shuffle(items)
os.makedirs(OUTD, exist_ok=True)
k = (len(items) + N_CHUNKS - 1) // N_CHUNKS
for c in range(N_CHUNKS):
    chunk = items[c * k:(c + 1) * k]
    with open(os.path.join(OUTD, f"sent_{c}.jsonl"), "w") as f:
        for it in chunk:
            f.write(json.dumps(it) + "\n")
    print(f"sent_{c}.jsonl: {len(chunk)} rows")
print(f"total {len(items)} generations from {len({r['cond'] for r in rows})} conditions")
