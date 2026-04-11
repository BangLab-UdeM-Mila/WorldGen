"""
Batch Runner
=============
Executes SceneSpec simulations in parallel using subprocesses.

Each scene runs in its own process so Genesis GPU state is fully
isolated.  The number of parallel workers should match the number
of available GPUs (or be 1 on a single-GPU machine to avoid OOM).

Usage
-----
    from pipeline.batch_runner import BatchRunner
    runner = BatchRunner(output_dir="dataset/", workers=1)
    results = runner.run(specs)
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .scene_spec import SceneSpec

_RUNNER_SCRIPT = Path(__file__).parent.parent / "scene_runner.py"


# ─────────────────────────────────────────────────────────────────────────────

def _run_one(
    spec_json: str,
    spec_path: str,
    out_dir: str,
    gpu_id: int,
    timeout: int,
) -> dict:
    """
    Worker function: write spec → invoke scene_runner.py → return result.
    Runs in a subprocess to isolate Genesis GPU state.
    """
    spec_p = Path(spec_path)
    spec_p.parent.mkdir(parents=True, exist_ok=True)
    spec_p.write_text(spec_json)

    env = {
        **os.environ,
        "PYOPENGL_PLATFORM": "egl",
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
    }

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            ["python", str(_RUNNER_SCRIPT),
             "--spec",   str(spec_p),
             "--output", out_dir],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - t0
        return {
            "success":  result.returncode == 0,
            "elapsed":  round(elapsed, 1),
            "stdout":   result.stdout[-2000:],   # tail to avoid huge logs
            "stderr":   result.stderr[-500:] if result.returncode != 0 else "",
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "elapsed": timeout, "stdout": "", "stderr": "TIMEOUT"}
    except Exception as e:
        return {"success": False, "elapsed": 0, "stdout": "", "stderr": str(e)}


# ─────────────────────────────────────────────────────────────────────────────

class BatchRunner:
    """
    Runs a list of SceneSpecs in parallel, one subprocess per scene.

    Parameters
    ----------
    output_dir : str | Path
        Root directory for all scene outputs.
        Each scene gets its own sub-directory: <output_dir>/<scene_id>/
    workers : int
        Number of parallel processes.  Set to 1 for a single GPU.
    gpu_id : int
        CUDA device index to use for all workers.
    timeout : int
        Max seconds per scene before the subprocess is killed.
    """

    def __init__(
        self,
        output_dir: str | Path,
        workers: int = 1,
        gpu_id: int = 0,
        timeout: int = 180,
    ):
        self.output_dir = Path(output_dir)
        self.workers    = workers
        self.gpu_id     = gpu_id
        self.timeout    = timeout

    def run(self, specs: list[SceneSpec]) -> list[dict]:
        """
        Execute all specs, return a list of result dicts.
        Prints a live progress bar to stdout.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        total   = len(specs)
        results = []

        print(f"\n[batch_runner] {total} scenes  workers={self.workers}  "
              f"gpu={self.gpu_id}  output={self.output_dir}\n")

        # Build task args (avoid pickling SceneSpec objects across processes)
        tasks = []
        for spec in specs:
            scene_dir = self.output_dir / spec.scene_id
            spec_path = scene_dir / "spec.json"
            tasks.append((
                spec.to_json(),
                str(spec_path),
                str(scene_dir),
                self.gpu_id,
                self.timeout,
            ))

        done = 0
        t_batch = time.perf_counter()

        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(_run_one, *args): specs[i]
                for i, args in enumerate(tasks)
            }
            for future in as_completed(futures):
                spec   = futures[future]
                result = future.result()
                done  += 1

                status = "OK  " if result["success"] else "FAIL"
                print(
                    f"  [{status}] {spec.scene_id}  "
                    f"obj={spec.object.name:<18}  "
                    f"room={spec.room.type:<8}  "
                    f"{result['elapsed']:.1f}s  "
                    f"[{done}/{total}]"
                )
                if not result["success"] and result["stderr"]:
                    print(f"         stderr: {result['stderr'][:120]}")

                results.append({
                    "scene_id":  spec.scene_id,
                    "seed":      spec.seed,
                    "object":    spec.object.name,
                    "room":      spec.room.type,
                    **result,
                })

        elapsed = time.perf_counter() - t_batch
        n_ok    = sum(1 for r in results if r["success"])
        print(f"\n[batch_runner] {n_ok}/{total} succeeded in {elapsed:.1f}s "
              f"({elapsed/total:.1f}s/scene avg)\n")

        # Write batch summary
        summary_path = self.output_dir / "batch_summary.json"
        summary_path.write_text(json.dumps(results, indent=2))
        print(f"[batch_runner] Summary → {summary_path}")

        return results
