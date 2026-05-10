"""Live monitor for LightGBM training logs (parses tee-style log files in real time).

Usage (run in a separate terminal while training is in progress):
    uv run python -m rogii.watch
    # or with explicit path:
    uv run python -m rogii.watch outputs/logs/exp002_lgb.log

Renders a continuously-updated terminal dashboard with:
- current fold / round / val RMSE
- per-fold best_iter history
- ASCII sparkline of the validation curve (live)
- elapsed time + ETA estimate
- system CPU/MEM utilisation

Implementation: polls the log file every refresh interval and re-parses tail.
Zero impact on the training process (read-only).
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

DEFAULT_LOG = Path("outputs/logs/exp002_lgb.log")
REFRESH_HZ = 2  # 2 frames per second
TARGET_FOLDS = 5

# regex parsers
RX_FOLD_HEADER = re.compile(r"\[fold (\d+)/(\d+)\]\s+train rows=(\S+)\s+val rows=(\S+)")
RX_LGB_LINE = re.compile(r"^\[(\d+)\]\s+val's rmse:\s*([\d.]+)")
RX_BEST_ITER = re.compile(r"best_iter=\s*(\d+)\s+.*?TVT RMSE all=([\d.]+)\s+hidden=([\d.]+)\s+\(([\d.]+)s\)")
RX_FINAL_RESID = re.compile(r"CV residual RMSE\s*=\s*([\d.]+)")
RX_FINAL_ALL = re.compile(r"CV TVT RMSE all rows\s*=\s*([\d.]+)")
RX_FINAL_HIDDEN = re.compile(r"CV TVT RMSE hidden\s*=\s*([\d.]+)")
RX_SUBMISSION = re.compile(r"wrote submission (\S+).*?\((\S+) rows.*?CV TVT hidden = ([\d.]+)")


@dataclass
class FoldStats:
    fold: int
    n_train: str = ""
    n_val: str = ""
    rounds: list[int] = field(default_factory=list)
    rmses: list[float] = field(default_factory=list)
    best_iter: int | None = None
    rmse_all: float | None = None
    rmse_hidden: float | None = None
    elapsed_sec: float | None = None


@dataclass
class TrainState:
    folds: list[FoldStats] = field(default_factory=list)
    cv_residual: float | None = None
    cv_tvt_all: float | None = None
    cv_tvt_hidden: float | None = None
    submission_path: str | None = None
    completed: bool = False
    started_at: float | None = None
    last_log_lines: list[str] = field(default_factory=list)

    @property
    def current_fold(self) -> FoldStats | None:
        if not self.folds:
            return None
        return self.folds[-1]


def parse_log(path: Path) -> TrainState:
    state = TrainState()
    if not path.exists():
        return state
    state.started_at = path.stat().st_mtime
    raw = path.read_text(errors="ignore")
    lines = raw.splitlines()
    state.last_log_lines = lines[-12:]
    cur: FoldStats | None = None
    for ln in lines:
        m = RX_FOLD_HEADER.search(ln)
        if m:
            cur = FoldStats(fold=int(m.group(1)), n_train=m.group(3), n_val=m.group(4))
            state.folds.append(cur)
            continue
        if cur is not None:
            m = RX_LGB_LINE.search(ln)
            if m:
                cur.rounds.append(int(m.group(1)))
                cur.rmses.append(float(m.group(2)))
                continue
            m = RX_BEST_ITER.search(ln)
            if m:
                cur.best_iter = int(m.group(1))
                cur.rmse_all = float(m.group(2))
                cur.rmse_hidden = float(m.group(3))
                cur.elapsed_sec = float(m.group(4))
                continue
        m = RX_FINAL_RESID.search(ln)
        if m:
            state.cv_residual = float(m.group(1))
        m = RX_FINAL_ALL.search(ln)
        if m:
            state.cv_tvt_all = float(m.group(1))
        m = RX_FINAL_HIDDEN.search(ln)
        if m:
            state.cv_tvt_hidden = float(m.group(1))
        m = RX_SUBMISSION.search(ln)
        if m:
            state.submission_path = m.group(1)
            state.completed = True
    return state


SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 60) -> str:
    if not values:
        return "(no data)"
    if len(values) > width:
        # downsample
        step = len(values) / width
        idx = [int(i * step) for i in range(width)]
        values = [values[i] for i in idx]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return SPARK[0] * len(values)
    bins = len(SPARK) - 1
    out = "".join(SPARK[int((v - lo) / (hi - lo) * bins)] for v in values)
    return out


def _fmt_eta(secs: float) -> str:
    if secs < 60:
        return f"{secs:.0f}s"
    m, s = divmod(int(secs), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def render(state: TrainState, log_path: Path) -> Group:
    title = Text("ROGII LightGBM Live Monitor", style="bold cyan", justify="center")

    overall_progress = Progress(
        TextColumn("[bold]Folds"),
        BarColumn(bar_width=None),
        TextColumn("{task.completed}/{task.total}"),
        expand=True,
    )
    completed_folds = sum(1 for f in state.folds if f.best_iter is not None)
    overall_progress.add_task("folds", total=TARGET_FOLDS, completed=completed_folds)

    cur = state.current_fold
    fold_panel: Panel
    if cur is None:
        fold_panel = Panel("waiting for log…", title="Current fold", border_style="dim")
    else:
        cur_round = cur.rounds[-1] if cur.rounds else 0
        cur_rmse = cur.rmses[-1] if cur.rmses else float("nan")
        spark = sparkline(cur.rmses, width=70)
        body = Group(
            Text(
                f"fold {cur.fold}/{TARGET_FOLDS}  "
                f"train rows={cur.n_train}  val rows={cur.n_val}",
                style="bold",
            ),
            Text(
                f"latest log: round {cur_round}  val_rmse={cur_rmse:.4f}",
                style="green" if cur.best_iter is None else "dim",
            ),
            Text(spark, style="cyan"),
        )
        if cur.best_iter is not None:
            body = Group(
                body,
                Text(
                    f"completed: best_iter={cur.best_iter}  "
                    f"TVT all={cur.rmse_all:.4f}  hidden={cur.rmse_hidden:.4f}  "
                    f"({cur.elapsed_sec:.0f}s)",
                    style="bold green",
                ),
            )
        fold_panel = Panel(body, title="Current fold", border_style="cyan")

    folds_table = Table(
        title="Per-fold summary",
        title_style="bold yellow",
        expand=True,
        show_lines=False,
    )
    folds_table.add_column("Fold", justify="center")
    folds_table.add_column("Best iter", justify="right")
    folds_table.add_column("Last round", justify="right")
    folds_table.add_column("Last RMSE", justify="right")
    folds_table.add_column("TVT all", justify="right")
    folds_table.add_column("TVT hidden", justify="right")
    folds_table.add_column("Time", justify="right")
    for f in state.folds:
        folds_table.add_row(
            str(f.fold),
            str(f.best_iter) if f.best_iter is not None else "running",
            str(f.rounds[-1]) if f.rounds else "—",
            f"{f.rmses[-1]:.4f}" if f.rmses else "—",
            f"{f.rmse_all:.4f}" if f.rmse_all is not None else "—",
            f"{f.rmse_hidden:.4f}" if f.rmse_hidden is not None else "—",
            f"{f.elapsed_sec:.0f}s" if f.elapsed_sec is not None else "—",
        )

    if state.completed:
        overall_box = Panel(
            Text(
                f"DONE  CV TVT hidden = {state.cv_tvt_hidden:.4f}  "
                f"submission: {state.submission_path}",
                style="bold green",
                justify="center",
            ),
            border_style="green",
        )
    else:
        elapsed = time.time() - (state.started_at or time.time())
        # ETA: completed folds + extrapolate from current fold
        eta_secs = float("nan")
        if state.folds:
            done = [f for f in state.folds if f.best_iter is not None]
            if done:
                avg = sum(f.elapsed_sec or 0 for f in done) / max(len(done), 1)
                remain = TARGET_FOLDS - len(done)
                # current fold partial
                if cur and cur.best_iter is None and cur.rounds:
                    # estimate cur fold completion: linear from rounds vs typical best_iter ~ 1500
                    cur_round = cur.rounds[-1]
                    estimate_cur_total = max(cur_round * 1.6, 2000)
                    cur_remaining_frac = max(1 - cur_round / estimate_cur_total, 0.05)
                    eta_secs = avg * cur_remaining_frac + avg * remain
                else:
                    eta_secs = avg * remain
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        overall_box = Panel(
            Text(
                f"elapsed={_fmt_eta(elapsed)}  ETA={_fmt_eta(eta_secs) if eta_secs == eta_secs else '—'}  "
                f"CPU={cpu:>4.0f}%  RAM={mem.percent:>4.0f}% "
                f"({mem.used / 1e9:.1f}/{mem.total / 1e9:.1f} GB)",
                style="bold",
            ),
            title="Status",
            border_style="magenta",
        )

    log_tail = Text("\n".join(state.last_log_lines[-8:]), style="dim")
    log_panel = Panel(log_tail, title=f"log tail ({log_path.name})", border_style="dim")

    return Group(
        Align.center(title),
        overall_box,
        overall_progress,
        fold_panel,
        folds_table,
        log_panel,
    )


def main(argv: list[str]) -> int:
    log_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_LOG
    if not log_path.exists():
        print(f"[watch] waiting for log file to appear: {log_path}", file=sys.stderr)

    console = Console()
    refresh_period = 1.0 / REFRESH_HZ
    with Live(console=console, refresh_per_second=REFRESH_HZ, transient=False) as live:
        while True:
            try:
                state = parse_log(log_path)
                live.update(render(state, log_path))
                if state.completed:
                    # keep showing for a bit then exit cleanly
                    time.sleep(2)
                    break
            except Exception as e:  # noqa: BLE001
                live.update(
                    Panel(
                        Text(f"watcher error: {e}", style="red"),
                        title="error",
                        border_style="red",
                    )
                )
            time.sleep(refresh_period)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
