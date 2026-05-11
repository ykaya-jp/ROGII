"""Re-run each of the 4 self-base kernels (exp007 / exp008 v2 / exp009 v2
/ exp010) under each of the 5 CV strategies (baseline / C1 / C2 / C3 / C4)
to generate the OOF table required for AC-5..AC-9 of the CV-strategies
task.

Mechanism:
    For each (exp, cv) pair we fork the kernel's directory into a temp
    working copy, patch the kernel script's CV_STRATEGY constant + kernel
    metadata title/id, attach ``ky7240/rogii-cv-fold-overrides`` as a
    dataset source, push it to Kaggle, wait for COMPLETE, then download
    the produced ``submission.csv`` + OOF artifact back into
    ``outputs/oof/cv-lb-correlation/``.

Prerequisites (= user one-time setup):
    1. ``kaggle datasets create -p outputs/folds/ -u`` from repo root
       to upload the 5 fold parquets as a Kaggle utility dataset.
       Update DATASET_SLUG below to the resulting slug.
    2. Verify ``kaggle competitions submissions -c rogii-wellbore-geology-prediction``
       responds without 403 (= auth ok).

Usage:
    # Single (exp, cv) for smoke
    .venv/bin/python scripts/regenerate_oof.py --exp exp008_case_d_kalman --cv baseline --dry-run

    # Full 4 exp × 5 cv = 20 kernels grid run (sequential)
    .venv/bin/python scripts/regenerate_oof.py --all

    # Full grid, parallel push (Kaggle's queue does the scheduling)
    .venv/bin/python scripts/regenerate_oof.py --all --parallel-push
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNELS_DIR = REPO_ROOT / "kaggle_kernels"
OUT_OOF_DIR = REPO_ROOT / "outputs" / "oof" / "cv-lb-correlation"
FOLD_DIR = REPO_ROOT / "outputs" / "folds"

# Dataset slug (= ky7240/<slug>) that contains baseline.parquet / C1.parquet / ...
# Default assumes the user ran ``kaggle datasets create -p outputs/folds -u`` and
# named it ``rogii-cv-fold-overrides``. Override via --fold-dataset CLI arg.
DEFAULT_FOLD_DATASET = "ky7240/rogii-cv-fold-overrides"

# Self-base kernels = those that train their own LGB/CB/XGB and accept
# FOLD_OVERRIDE_PARQUET. Karnakbaev-only kernels (exp005 series, exp006)
# are deferred to a separate task because they re-use karnakbaev's published
# OOF and would require a fold-aware karnakbaev re-fit pipeline.
SELF_BASE_EXPS = [
    "exp007_edge_q_m",
    "exp008_case_d_kalman",
    "exp009_case_e_edge_o",
    "exp010_fold_reform",
]

CV_STRATEGIES = ["baseline", "C1", "C2", "C3", "C4"]

# Public LB lookup for downstream correlation calculation
PUBLIC_LB = {
    "exp002_lgb":            14.695,
    "exp003_lgb":            17.510,
    "exp005_cache_blend":    10.317,   # v1
    "exp006_tabicl_pflite":  10.503,
    "exp007_edge_q_m":       10.677,
    "exp008_case_d_kalman":   9.957,   # v2 (= current best)
    # v3 = same kernel id, different submission; LB tracked separately when SCORED
    "exp009_case_e_edge_o":  None,     # PENDING as of 2026-05-11 11:14 UTC
    "exp010_fold_reform":    None,     # kernel still RUNNING
}


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def kernel_patch_cv_strategy(kernel_path: Path, cv_strategy: str, fold_dataset: str) -> str:
    """Return patched kernel source with CV_STRATEGY constant injected at the top.

    The injected snippet sets an env var so the existing FOLD_OVERRIDE_PARQUET
    handler in build_edge_q_folds + build_stratified_edge_q_folds picks it up.
    """
    src = kernel_path.read_text()
    dataset_short = fold_dataset.split("/")[-1]
    fold_path_in_kernel = f"/kaggle/input/{dataset_short}/{cv_strategy}.parquet"
    header = (
        f'# ============================================================\n'
        f'# CV STRATEGY OVERRIDE (= kaggle-rogii-cv-strategies-2026-05-11)\n'
        f'# Auto-injected by scripts/regenerate_oof.py — do not edit.\n'
        f'# Strategy: {cv_strategy}\n'
        f'# ============================================================\n'
        f'import os as _cv_strat_os\n'
        f'_cv_strat_os.environ["FOLD_OVERRIDE_PARQUET"] = "{fold_path_in_kernel}"\n'
        f'print("[CV-override] strategy={cv_strategy} parquet=" '
        f'+ repr(_cv_strat_os.environ["FOLD_OVERRIDE_PARQUET"]))\n'
        f'\n'
    )
    return header + src


def kernel_patch_metadata(
    metadata_path: Path,
    exp: str,
    cv_strategy: str,
    fold_dataset: str,
    retry_suffix: str = "",
) -> dict:
    """Return patched kernel-metadata.json with title/id updated + fold dataset attached.

    retry_suffix: appended to id and title to bypass Kaggle "Notebook not found"
        ghost-slug errors (= when a prior push partially-registered the kernel,
        same slug re-push fails). Use e.g. "-r2" or "-r3" to force a new slug.
    """
    meta = json.loads(metadata_path.read_text())
    base_title = meta.get("title", exp)
    base_id = meta.get("id", f"ky7240/{exp.replace('_', '-')}")
    user_prefix, kernel_slug = base_id.split("/", 1)
    suffix = f"-cv-{cv_strategy.lower()}{retry_suffix.lower()}"
    new_id = f"{user_prefix}/{kernel_slug}{suffix}"
    new_title = f"{base_title} [CV={cv_strategy}{retry_suffix.upper()}]"
    meta["id"] = new_id
    meta["title"] = new_title
    sources = list(meta.get("dataset_sources", []))
    if fold_dataset not in sources:
        sources.append(fold_dataset)
    meta["dataset_sources"] = sources
    # Track provenance so cleanup is automatable
    meta["__cv_override__"] = {"exp": exp, "cv": cv_strategy, "fold_dataset": fold_dataset}
    return meta


def push_kernel(work_dir: Path, dry_run: bool = False) -> int:
    """Run ``kaggle kernels push -p <work_dir>``; return CLI exit code."""
    cmd = ["kaggle", "kernels", "push", "-p", str(work_dir)]
    if dry_run:
        print(f"  [DRY-RUN] would run: {' '.join(cmd)}")
        return 0
    print(f"  [push] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(f"  [push] stdout: {proc.stdout.strip()}")
    if proc.returncode != 0:
        print(f"  [push] stderr: {proc.stderr.strip()}", file=sys.stderr)
    return proc.returncode


def wait_kernel_complete(kernel_id: str, poll_seconds: int = 60, timeout_seconds: int = 7200) -> str:
    """Poll ``kaggle kernels status <kernel_id>`` until COMPLETE / ERROR / timeout."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        proc = subprocess.run(
            ["kaggle", "kernels", "status", kernel_id],
            capture_output=True, text=True,
        )
        out = (proc.stdout or "").strip()
        print(f"  [poll {kernel_id}] {out}")
        if "COMPLETE" in out:
            return "COMPLETE"
        if "ERROR" in out or "CancelRequested" in out:
            return "ERROR"
        time.sleep(poll_seconds)
    return "TIMEOUT"


def download_kernel_output(kernel_id: str, dest_dir: Path) -> int:
    """``kaggle kernels output <kernel_id> -p <dest>``; returns exit code."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["kaggle", "kernels", "output", kernel_id, "-p", str(dest_dir)],
        capture_output=True, text=True,
    )
    print(f"  [output] {proc.stdout.strip()}")
    if proc.returncode != 0:
        print(f"  [output] stderr: {proc.stderr.strip()}", file=sys.stderr)
    return proc.returncode


# ────────────────────────────────────────────────────────────────────────────
# Main per-(exp, cv) runner
# ────────────────────────────────────────────────────────────────────────────


def run_one(exp: str, cv: str, fold_dataset: str, dry_run: bool, parallel_push: bool, retry_suffix: str = "") -> dict:
    print(f"\n== exp={exp}  cv={cv}  retry={retry_suffix!r} ==")
    kernel_dir = KERNELS_DIR / exp
    kernel_py = kernel_dir / f"{exp}.py"
    kernel_meta = kernel_dir / "kernel-metadata.json"
    if not kernel_py.exists():
        return {"exp": exp, "cv": cv, "status": "MISSING_KERNEL", "kernel": str(kernel_py)}
    if not kernel_meta.exists():
        return {"exp": exp, "cv": cv, "status": "MISSING_METADATA", "kernel": str(kernel_meta)}

    # Build working copy
    work_dir = REPO_ROOT / ".work" / "regenerate_oof" / f"{exp}__cv-{cv}{retry_suffix}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    patched_py = kernel_patch_cv_strategy(kernel_py, cv, fold_dataset)
    patched_meta = kernel_patch_metadata(kernel_meta, exp, cv, fold_dataset, retry_suffix=retry_suffix)
    (work_dir / f"{exp}.py").write_text(patched_py)
    (work_dir / "kernel-metadata.json").write_text(json.dumps(patched_meta, indent=2))
    new_kernel_id = patched_meta["id"]

    rc_push = push_kernel(work_dir, dry_run=dry_run)
    if rc_push != 0:
        return {"exp": exp, "cv": cv, "kernel_id": new_kernel_id, "status": "PUSH_FAILED"}
    if dry_run:
        return {"exp": exp, "cv": cv, "kernel_id": new_kernel_id, "status": "DRY_RUN_OK"}
    if parallel_push:
        return {"exp": exp, "cv": cv, "kernel_id": new_kernel_id, "status": "PUSHED_ASYNC"}

    final_status = wait_kernel_complete(new_kernel_id)
    if final_status != "COMPLETE":
        return {"exp": exp, "cv": cv, "kernel_id": new_kernel_id, "status": final_status}

    out_dir = OUT_OOF_DIR / f"{exp}__cv-{cv}"
    rc_dl = download_kernel_output(new_kernel_id, out_dir)
    return {
        "exp": exp,
        "cv": cv,
        "kernel_id": new_kernel_id,
        "status": "COMPLETE" if rc_dl == 0 else "OUTPUT_DOWNLOAD_FAILED",
        "out_dir": str(out_dir.relative_to(REPO_ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp", choices=SELF_BASE_EXPS)
    parser.add_argument("--cv", choices=CV_STRATEGIES)
    parser.add_argument("--all", action="store_true",
                        help="Run the full SELF_BASE_EXPS × CV_STRATEGIES grid")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build patched kernel + metadata but don't push to Kaggle")
    parser.add_argument("--parallel-push", action="store_true",
                        help="Push all kernels without waiting for COMPLETE between pushes")
    parser.add_argument("--fold-dataset", default=DEFAULT_FOLD_DATASET,
                        help=f"Kaggle dataset slug containing fold parquets (default: {DEFAULT_FOLD_DATASET})")
    parser.add_argument("--retry-suffix", default="",
                        help="Suffix appended to kernel id/title to bypass Kaggle "
                             "ghost-slug errors (e.g. '-r2'). Empty by default.")
    args = parser.parse_args()

    if args.all:
        pairs = [(e, c) for e in SELF_BASE_EXPS for c in CV_STRATEGIES]
    elif args.exp and args.cv:
        pairs = [(args.exp, args.cv)]
    else:
        parser.error("specify --all or both --exp and --cv")

    OUT_OOF_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    t0 = time.perf_counter()
    for exp, cv in pairs:
        r = run_one(exp, cv, args.fold_dataset, args.dry_run, args.parallel_push, args.retry_suffix)
        results.append(r)

    elapsed = time.perf_counter() - t0
    summary_path = OUT_OOF_DIR / f"regenerate_oof_summary_{int(time.time())}.json"
    summary_path.write_text(json.dumps({
        "n_pairs": len(pairs),
        "fold_dataset": args.fold_dataset,
        "elapsed_sec": elapsed,
        "results": results,
    }, indent=2, ensure_ascii=False))
    print(f"\n== summary == ({elapsed:.1f}s)")
    for r in results:
        print(f"  {r.get('exp')}  cv={r.get('cv')}  status={r.get('status')}  kernel_id={r.get('kernel_id', '-')}")
    print(f"  written to {summary_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
