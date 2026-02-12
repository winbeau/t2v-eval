"""
Progress reporting: live progress bars, multi-GPU progress board, stdout monitoring.
"""

import contextlib
import datetime
import json
import re
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    from .env import logger
except ImportError:
    from vbench_runner.env import logger


# =============================================================================
# Text helpers
# =============================================================================
def _shorten_text(text: str, max_len: int) -> str:
    """Shorten text for compact terminal table display."""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


# =============================================================================
# Per-rank progress reporter
# =============================================================================
class RankProgressReporter:
    """Per-rank progress reporter writing JSON status for controller display."""

    def __init__(
        self,
        progress_dir: Path,
        rank: int,
        local_rank: int,
        assigned_subtasks: list[str],
        visible_devices: list[str] | None = None,
    ):
        self.progress_dir = progress_dir
        self.rank = rank
        self.local_rank = local_rank
        self.assigned_subtasks = list(assigned_subtasks)
        self.visible_devices = visible_devices or []
        self._lock = threading.Lock()
        self._completed_subtasks = 0
        self._current_subtask: str | None = None
        self._current_percent: int | None = None
        self._status_text = "idle"
        self._elapsed_sec = 0
        self._done = False
        self._last_error: str | None = None
        self._write_status()

    @property
    def _status_file(self) -> Path:
        return self.progress_dir / f"rank_{self.rank}.json"

    def _gpu_label(self) -> str:
        if self.visible_devices and 0 <= self.local_rank < len(self.visible_devices):
            return self.visible_devices[self.local_rank]
        return str(self.local_rank)

    def _next_subtask(self) -> str:
        total = len(self.assigned_subtasks)
        if total == 0:
            return "-"

        if self._current_subtask:
            try:
                current_idx = self.assigned_subtasks.index(self._current_subtask)
            except ValueError:
                current_idx = self._completed_subtasks
            next_idx = current_idx + 1
            if 0 <= next_idx < total:
                return self.assigned_subtasks[next_idx]
            return "-"

        if self._completed_subtasks < total:
            return self.assigned_subtasks[self._completed_subtasks]
        return "-"

    def _write_status(self) -> None:
        payload = {
            "rank": self.rank,
            "gpu": self._gpu_label(),
            "assigned_total": len(self.assigned_subtasks),
            "assigned_subtasks": self.assigned_subtasks,
            "completed_subtasks": self._completed_subtasks,
            "current_subtask": self._current_subtask,
            "next_subtask": self._next_subtask(),
            "percent": self._current_percent,
            "status_text": self._status_text,
            "elapsed_sec": int(self._elapsed_sec),
            "done": self._done,
            "error": self._last_error,
            "updated_at": time.time(),
        }
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self._status_file.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        tmp_path.replace(self._status_file)

    def start_task(self, task_name: str, status_text: str = "running") -> None:
        with self._lock:
            self._current_subtask = task_name
            self._current_percent = None
            self._status_text = status_text
            self._elapsed_sec = 0
            self._write_status()

    def update_live(self, percent: int | None, status_text: str, elapsed_sec: int) -> None:
        with self._lock:
            if percent is not None:
                self._current_percent = max(0, min(99, int(percent)))
            else:
                self._current_percent = None
            self._status_text = status_text
            self._elapsed_sec = int(elapsed_sec)
            self._write_status()

    def finish_task(
        self,
        success: bool = True,
        error: str | None = None,
        count_completion: bool = True,
    ) -> None:
        with self._lock:
            if count_completion and self._current_subtask is not None:
                self._completed_subtasks += 1
            self._current_percent = 100
            self._status_text = "done" if success else "failed"
            self._last_error = _shorten_text(str(error), 120) if error else None
            self._current_subtask = None
            self._elapsed_sec = 0
            self._write_status()

    def mark_done(self) -> None:
        with self._lock:
            self._done = True
            self._current_subtask = None
            self._current_percent = 100 if self.assigned_subtasks else 0
            self._status_text = "done"
            self._elapsed_sec = 0
            self._write_status()

    def log_event(self, message: str, level: str = "INFO") -> None:
        """Append a cross-GPU summary event to the shared events buffer file."""
        events_path = self.progress_dir / "events.log"
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] [{level}] [GPU{self._gpu_label()}] {message}\n"
        try:
            with open(events_path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError:
            pass


# =============================================================================
# Multi-GPU progress board (rank-0 only)
# =============================================================================
class MultiGpuProgressBoard:
    """Rank-0 terminal board: bordered table with ANSI overwrite + log tail."""

    LOG_TAIL_LINES = 5

    def __init__(
        self,
        progress_dir: Path,
        assignment_map: dict[int, list[str]],
        gpu_map: dict[int, str] | None = None,
        refresh_sec: float = 1.0,
        log_path: Path | None = None,
        non_tty_snapshot_sec: float = 30.0,
    ):
        self.progress_dir = progress_dir
        self.assignment_map = assignment_map
        self.gpu_map = gpu_map or {}
        self.world_size = len(assignment_map)
        self.refresh_sec = refresh_sec
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._stdout = sys.__stdout__
        self._overwrite = bool(getattr(self._stdout, "isatty", lambda: False)())
        self._start_time = time.time()
        self._block_height = 0
        # Log tailing — prefer events.log (cross-GPU summary), fall back to log_path
        self._events_path = progress_dir / "events.log"
        self._log_path = log_path
        self._log_tail: deque[str] = deque(maxlen=self.LOG_TAIL_LINES)
        self._events_offset = 0
        self._log_offset = 0
        # Non-TTY fallback
        self._last_row_strs: list[str] = []
        self._last_snapshot = 0.0
        self._snapshot_interval = max(1.0, float(non_tty_snapshot_sec))

    def _status_file(self, rank: int) -> Path:
        return self.progress_dir / f"rank_{rank}.json"

    def _read_rank_status(self, rank: int) -> dict:
        path = self._status_file(rank)
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        assigned = self.assignment_map.get(rank, [])
        return {
            "rank": rank,
            "gpu": self.gpu_map.get(rank, str(rank)),
            "assigned_total": len(assigned),
            "assigned_subtasks": assigned,
            "completed_subtasks": 0,
            "current_subtask": None,
            "next_subtask": assigned[0] if assigned else "-",
            "percent": None,
            "status_text": "waiting",
            "elapsed_sec": 0,
            "done": False,
            "error": None,
        }

    def _print_assignment_table(self) -> None:
        rows = []
        for rank in range(self.world_size):
            tasks = self.assignment_map.get(rank, [])
            rows.append(
                {
                    "gpu": f"gpu{self.gpu_map.get(rank, str(rank))}",
                    "tasks": _shorten_text(", ".join(tasks) if tasks else "-", 52),
                    "done": f"0/{len(tasks)}",
                    "next": _shorten_text(tasks[0] if tasks else "-", 20),
                }
            )
        headers = {"gpu": "GPU", "tasks": "Assigned Tasks", "done": "Done", "next": "Next Task"}
        widths = {"gpu": 4, "tasks": 52, "done": 4, "next": 20}
        sep = (
            "+"
            + "+".join("-" * (widths[key] + 2) for key in ["gpu", "tasks", "done", "next"])
            + "+"
        )
        print(sep, file=self._stdout)
        print(
            f"| {headers['gpu']:<{widths['gpu']}} "
            f"| {headers['tasks']:<{widths['tasks']}} "
            f"| {headers['done']:<{widths['done']}} "
            f"| {headers['next']:<{widths['next']}} |",
            file=self._stdout,
        )
        print(sep, file=self._stdout)
        for row in rows:
            print(
                f"| {row['gpu']:<{widths['gpu']}} "
                f"| {row['tasks']:<{widths['tasks']}} "
                f"| {row['done']:<{widths['done']}} "
                f"| {row['next']:<{widths['next']}} |",
                file=self._stdout,
            )
        print(sep, file=self._stdout)
        self._stdout.flush()

    def _derive_next_task(self, status: dict) -> str:
        assigned = list(status.get("assigned_subtasks", []))
        if not assigned:
            return "-"
        current = status.get("current_subtask")
        completed = int(status.get("completed_subtasks", 0))
        if current and current in assigned:
            current_idx = assigned.index(current)
            next_idx = current_idx + 1
            if next_idx < len(assigned):
                return assigned[next_idx]
            return "-"
        if 0 <= completed < len(assigned):
            return assigned[completed]
        return "-"

    def _format_row(self, status: dict) -> dict[str, str]:
        completed = int(status.get("completed_subtasks", 0))
        total = int(status.get("assigned_total", 0))
        percent = status.get("percent")
        pct_text = "--%" if percent is None else f"{int(percent):>3d}%"
        current = status.get("current_subtask") or "-"
        next_task = self._derive_next_task(status)
        state_text = status.get("status_text", "running")
        elapsed = int(status.get("elapsed_sec", 0))
        gpu_label = status.get("gpu", "?")
        done_flag = " [DONE]" if status.get("done", False) else ""
        return {
            "gpu": f"GPU{gpu_label}",
            "done": f"{completed}/{total}",
            "pct": pct_text,
            "cur": _shorten_text(str(current), 24),
            "next": _shorten_text(str(next_task), 24),
            "status": _shorten_text(str(state_text), 20),
            "time": f"{elapsed:>4}s{done_flag}",
        }

    def _update_log_tail(self) -> None:
        """Read new lines from events buffer; fall back to worker log if empty."""
        found = self._read_tail_from(self._events_path, "_events_offset")
        if not found:
            self._read_tail_from(self._log_path, "_log_offset")

    def _read_tail_from(self, path: Path | None, offset_attr: str) -> bool:
        """Read new complete lines from a file, return True if any lines were added."""
        if path is None or not path.exists():
            return False
        try:
            offset = getattr(self, offset_attr, 0)
            with open(path, errors="replace") as f:
                f.seek(offset)
                chunk = f.read(65536)
                if not chunk:
                    return False
                last_nl = chunk.rfind("\n")
                if last_nl < 0:
                    return False
                setattr(self, offset_attr, offset + last_nl + 1)
                added = False
                for line in chunk[:last_nl].splitlines():
                    stripped = line.strip()
                    if not stripped or "█" in stripped:
                        continue
                    if len(stripped) > 120:
                        stripped = stripped[:117] + "..."
                    self._log_tail.append(stripped)
                    added = True
                return added
        except Exception:
            return False

    def _build_table_lines(self, rows: list[dict[str, str]]) -> list[str]:
        cols = ["gpu", "done", "pct", "cur", "next", "status", "time"]
        headers = {
            "gpu": "GPU",
            "done": "Done",
            "pct": "%",
            "cur": "Current",
            "next": "Next",
            "status": "Status",
            "time": "Time",
        }
        widths = {}
        for c in cols:
            widths[c] = max(len(headers[c]), *(len(r[c]) for r in rows))

        def _sep(left: str, mid: str, right: str) -> str:
            return left + mid.join("─" * (widths[c] + 2) for c in cols) + right

        def _row(vals: dict[str, str]) -> str:
            return "│" + "│".join(f" {vals[c]:<{widths[c]}} " for c in cols) + "│"

        elapsed = int(time.time() - self._start_time)
        return [
            _sep("┌", "┬", "┐"),
            _row(headers),
            _sep("├", "┼", "┤"),
            *(_row(r) for r in rows),
            _sep("└", "┴", "┘") + f"  T+{elapsed}s",
        ]

    def _render(self, force: bool = False) -> None:
        statuses = [self._read_rank_status(r) for r in range(self.world_size)]
        rows = [self._format_row(s) for s in statuses]

        self._update_log_tail()

        table = self._build_table_lines(rows)

        # Fixed-height log section below table
        log_display = list(self._log_tail)
        while len(log_display) < self.LOG_TAIL_LINES:
            log_display.append("")
        log_display = log_display[-self.LOG_TAIL_LINES :]

        all_lines = table + [f"  {line}" for line in log_display]

        if self._overwrite:
            if self._block_height > 0:
                self._stdout.write(f"\x1b[{self._block_height}A")
            for line in all_lines:
                self._stdout.write("\r\x1b[2K" + line + "\n")
            self._stdout.flush()
            self._block_height = len(all_lines)
        else:
            row_strs = [str(r) for r in rows]
            if not force and row_strs == self._last_row_strs:
                return
            now = time.time()
            if not force and (now - self._last_snapshot) < self._snapshot_interval:
                return
            self._last_snapshot = now
            print("\n".join(table), file=self._stdout)
            self._stdout.flush()
            self._last_row_strs = row_strs

    def _run(self) -> None:
        while not self._stop.is_set():
            self._render(force=False)
            if self._stop.wait(self.refresh_sec):
                break
        self._render(force=True)

    def start(self) -> None:
        self._print_assignment_table()
        self._start_time = time.time()
        self._render(force=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._render(force=True)


# =============================================================================
# Stdout monitoring
# =============================================================================
def summarize_vbench_stdout(stdout_text: str, subtask: str) -> None:
    """Summarize verbose stdout from VBench/VBench-Long evaluate()."""
    if not stdout_text:
        return

    normalized = stdout_text.replace("\r", "\n")
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if not lines:
        return

    saved_lines = [line for line in lines if line.startswith("Saved ")]
    other_lines = [line for line in lines if not line.startswith("Saved ")]
    other_lines = [
        line
        for line in other_lines
        if "████" not in line
        and "clips=" not in line
        and not line.startswith("[")
        and not ("%" in line and "|" in line and "/" in line)
    ]

    if saved_lines:
        logger.info(f"[{subtask}] Saved split clips: {len(saved_lines)}")

    max_preview = 4
    for line in other_lines[:max_preview]:
        logger.info(f"[{subtask}] {line}")

    hidden = len(other_lines) - max_preview
    if hidden > 0:
        logger.info(f"[{subtask}] ... {hidden} more log lines suppressed")


class StdoutMonitor:
    """Thread-safe stdout monitor for VBench evaluation output."""

    def __init__(self):
        self._lock = threading.Lock()
        self._parts: list[str] = []
        self._line_buffer = ""
        self._saved_count = 0
        self._saved_video_ids: set[str] = set()
        self._line_count = 0
        self._chars = 0
        self._latest_percent: int | None = None
        self._latest_done: int | None = None
        self._latest_total: int | None = None

    def write(self, text: str) -> int:
        if not text:
            return 0
        with self._lock:
            self._parts.append(text)
            self._chars += len(text)
            for match in re.finditer(r"(\d{1,3})%\|", text):
                percent = int(match.group(1))
                if 0 <= percent <= 100:
                    self._latest_percent = percent
            if "%|" in text:
                for match in re.finditer(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)", text):
                    done = int(match.group(1))
                    total = int(match.group(2))
                    if total > 0 and 0 <= done <= total:
                        self._latest_done = done
                        self._latest_total = total
                        self._latest_percent = int(done * 100 / total)
            self._line_buffer += text
            lines = self._line_buffer.splitlines(keepends=True)
            remainder = ""
            if lines and not lines[-1].endswith(("\n", "\r")):
                remainder = lines.pop()
            self._line_buffer = remainder

            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                self._line_count += 1
                if line_stripped.startswith("Saved "):
                    self._saved_count += 1
                    saved_video_id = _extract_saved_video_id(line_stripped)
                    if saved_video_id:
                        self._saved_video_ids.add(saved_video_id)
                match = re.search(r"(\d{1,3})%\|", line_stripped)
                if match:
                    percent = int(match.group(1))
                    if 0 <= percent <= 100:
                        self._latest_percent = percent
                if "%|" in line_stripped:
                    for ratio_match in re.finditer(
                        r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)", line_stripped
                    ):
                        done = int(ratio_match.group(1))
                        total = int(ratio_match.group(2))
                        if total > 0 and 0 <= done <= total:
                            self._latest_done = done
                            self._latest_total = total
                            self._latest_percent = int(done * 100 / total)
        return len(text)

    def flush(self) -> None:
        return

    def snapshot(self) -> tuple[int, int, int, int | None, int, int | None, int | None]:
        with self._lock:
            return (
                self._saved_count,
                self._line_count,
                self._chars,
                self._latest_percent,
                len(self._saved_video_ids),
                self._latest_done,
                self._latest_total,
            )

    def getvalue(self) -> str:
        with self._lock:
            tail = self._line_buffer
            return "".join(self._parts) + tail


# =============================================================================
# Progress bar rendering
# =============================================================================
def _render_bar(step: int, width: int = 18) -> str:
    filled = step % (width + 1)
    return f"{'█' * filled}{'·' * (width - filled)}"


def _render_percent_bar(percent: int, width: int = 18) -> str:
    clamped = max(0, min(100, percent))
    filled = int(round(clamped * width / 100))
    return f"{'█' * filled}{'·' * (width - filled)}"


def _extract_saved_video_id(saved_line: str) -> str | None:
    """
    Extract video id from VBench split-clip log line, e.g.:
    Saved .../split_clip/<video_id>/<video_id>_000.mp4
    """
    match = re.search(r"split_clip/([^/]+)/", saved_line)
    if match:
        return match.group(1)
    return None


# =============================================================================
# Callable wrappers with progress
# =============================================================================
def run_callable_with_progress(
    task_fn,
    title: str,
    prefix: str = "",
    refresh_sec: float = 1.0,
    status_mode: str = "events",
    enable_live: bool = True,
    expected_units: int | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[str, int]:
    """
    Run a callable with one-line progress display and captured output.

    Returns:
        Tuple of (captured_stdout_and_stderr, saved_clip_count)
    """
    monitor = StdoutMonitor()
    worker_error: list[Exception] = []

    def _worker():
        try:
            with contextlib.redirect_stdout(monitor), contextlib.redirect_stderr(monitor):
                task_fn()
        except Exception as e:
            worker_error.append(e)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    start = time.time()
    tick = 0

    while thread.is_alive():
        elapsed = int(time.time() - start)
        (
            saved_count,
            line_count,
            _,
            latest_percent,
            saved_video_count,
            latest_done,
            latest_total,
        ) = monitor.snapshot()
        display_percent = latest_percent
        if status_mode == "clips" and expected_units is not None and expected_units > 0:
            display_percent = min(99, int(saved_video_count * 100 / expected_units))

        # Never show 100% until the worker thread has actually finished.
        if display_percent is not None and display_percent >= 100:
            display_percent = 99

        if status_mode == "clips":
            if expected_units is not None and expected_units > 0:
                status_text = f"videos={saved_video_count}/{expected_units}, clips={saved_count}"
            else:
                status_text = f"clips={saved_count}"
        else:
            if latest_done is not None and latest_total is not None and latest_total > 0:
                status_text = f"progress={latest_done}/{latest_total}"
            else:
                status_text = f"events={line_count}"

        if progress_callback is not None:
            progress_callback(
                {
                    "title": title,
                    "percent": display_percent,
                    "status_text": status_text,
                    "elapsed_sec": elapsed,
                    "done": False,
                }
            )

        if enable_live:
            if display_percent is not None:
                bar = _render_percent_bar(display_percent)
                pct_text = f"{display_percent:>3d}%"
            else:
                bar = _render_bar(tick)
                pct_text = " --%"
            line = f"\r{prefix}{title:<24} | {bar} | {pct_text} | {status_text} | {elapsed:>4}s"
            print(line, end="", flush=True, file=sys.__stdout__)
            tick += 1
        thread.join(timeout=refresh_sec)

    thread.join()
    elapsed = int(time.time() - start)
    (
        saved_count,
        line_count,
        _,
        latest_percent,
        saved_video_count,
        latest_done,
        latest_total,
    ) = monitor.snapshot()
    final_percent = 100 if latest_percent is None else max(100, latest_percent)
    final_bar = _render_percent_bar(final_percent)
    if status_mode == "clips":
        if expected_units is not None and expected_units > 0:
            final_status = f"videos={saved_video_count}/{expected_units}, clips={saved_count}"
        else:
            final_status = f"clips={saved_count}"
    else:
        if latest_done is not None and latest_total is not None and latest_total > 0:
            final_status = f"progress={latest_done}/{latest_total}"
        else:
            final_status = f"events={line_count}"
    if progress_callback is not None:
        progress_callback(
            {
                "title": title,
                "percent": 100,
                "status_text": final_status,
                "elapsed_sec": elapsed,
                "done": True,
            }
        )
    if enable_live:
        print(
            f"\r{prefix}{title:<24} | {final_bar} | 100% | {final_status} | {elapsed:>4}s done",
            file=sys.__stdout__,
        )

    if worker_error:
        raise worker_error[0]
    return monitor.getvalue(), saved_count


def run_evaluate_with_progress(
    vbench: Any,
    eval_kwargs: dict,
    subtask: str,
    subtask_index: int,
    subtask_total: int,
    refresh_sec: float = 1.0,
    enable_live: bool = True,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[str, int]:
    """
    Run one VBench subtask with a lightweight manual progress bar.
    """
    return run_callable_with_progress(
        task_fn=lambda: vbench.evaluate(**eval_kwargs),
        title=subtask,
        prefix=f"[{subtask_index}/{subtask_total}] ",
        refresh_sec=refresh_sec,
        status_mode="events",
        enable_live=enable_live,
        progress_callback=progress_callback,
    )
