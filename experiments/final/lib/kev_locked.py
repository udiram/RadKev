"""Run a Kev entry point with checkpoint loading serialised across processes (host RAM, 2026-10-06).

    python kev_locked.py LOCKFILE kev_eval.py --run ... --data ... --out ...      # a script
    python kev_locked.py LOCKFILE kev.benchmark --run ... --suite ... --out ...    # a module

Kev's LocalPredictor builds the backbone on the CPU (fp32 unless KEV_DTYPE says otherwise) and then moves it to the GPU, so
several processes loading at once can exceed the job's memory (four 9B shards were OOM-killed). Loading takes the lock;
scoring runs in parallel. Nothing else changes: same class, same arguments, same outputs.
"""
import fcntl
import gc
import runpy
import sys

import kev.predictors as P

lock_path, target = sys.argv[1], sys.argv[2]
sys.argv = [target, *sys.argv[3:]]
_Local = P.LocalPredictor


class LockedLocalPredictor(_Local):
    def __init__(self, *a, **k):
        with open(lock_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try: super().__init__(*a, **k)
            finally:
                gc.collect(); fcntl.flock(f, fcntl.LOCK_UN)


P.LocalPredictor = LockedLocalPredictor
if target.endswith(".py"): runpy.run_path(target, run_name="__main__")
else: runpy.run_module(target, run_name="__main__", alter_sys=True)
