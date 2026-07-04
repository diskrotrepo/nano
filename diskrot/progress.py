"""Shared progress heartbeat for every data-pipeline stage.

A tiny, dependency-free reporter (stdlib ``time`` + ``threading`` + ``print`` only
— NO ``modal``, so the pure per-song modules ``pack_cache`` / ``key_detect`` /
``phonemize`` / ``filter_lyrics`` can use it too, and it stays importable in any
local context). ``modal_common`` re-exports ``ProgressReporter`` so the Modal
fan-out orchestrators can keep importing it from there.

Every stage prints the SAME line so the whole pipeline reads uniformly in
``modal app logs``:

    [<label>] progress N/T (P%) | R <unit>/min | elapsed Xm | ETA HhMMm

The heartbeat fires on TWO independent clocks:
  - update-driven: every ``every`` items or ``secs`` seconds, evaluated when
    ``update()`` is called (the common case — the result stream is flowing).
  - a background wall-clock KEEP-ALIVE thread: emits every ``secs`` even when the
    result stream STALLS (no ``update()`` calls at all). Without it the log goes
    silent whenever the fleet stops yielding — e.g. preempted GPU workers
    cold-restarting — which looks exactly like a hang even though the
    orchestrator is alive and waiting. The flat count + falling rate on those
    keep-alive lines make the stall self-evident.
"""
from __future__ import annotations

import threading
import time


class ProgressReporter:
    """Periodic progress heartbeat: prints ``[label] progress N/T (P%) | R/min |
    elapsed | ETA`` every ``every`` items OR every ``secs`` seconds, whichever comes
    first — so a short stage still updates on time and a long one isn't spammed. The
    live throughput + ETA give an at-a-glance sense of overall progress a bare count
    can't. Call ``update(k)`` per item/batch in the loop and ``done()`` once at the
    end. ``total <= 0`` (unknown up front) drops the %/ETA and shows count + rate only.

    A background keep-alive thread (daemon, started on the first ``update()``) also
    re-emits the heartbeat every ``secs`` when the stream stalls, so the log never
    goes silent while the stage is still alive; ``done()`` stops it.

    Usage:
        rep = ProgressReporter(len(pending), "tokenize", unit="files")
        for batch in results:
            ...
            rep.update(len(batch), extra=f"failed {n_failed}")
        rep.done()

    ``time.time()`` is real wall-clock (fine on Modal); ``flush=True`` so lines
    surface promptly in detached ``modal app logs``. A custom ``log`` sink (default
    ``print``) matches the codebase's injectable-logger convention.
    """

    def __init__(
        self, total: int, label: str, every: int = 2000, secs: float = 45.0,
        unit: str = "items", log=print,
    ):
        self.total = max(0, int(total))
        self.label = label
        self.every = max(1, int(every))
        self.secs = float(secs)
        self.unit = unit
        self._log = log
        self.n = 0
        self._t0 = time.time()
        self._last_t = self._t0
        self._next = self.every
        self._extra = ""
        # _lock serializes _emit + the shared counters between the update() caller
        # and the keep-alive thread. _stop ends the daemon; _watch is started
        # lazily on the first update() (so a reporter for an empty loop that never
        # updates spawns no thread).
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._watch: threading.Thread | None = None

    def update(self, k: int = 1, extra: str = "") -> None:
        """Advance the count by ``k``; print a heartbeat if the count or time
        threshold is crossed. ``extra`` appends a stage-specific suffix (e.g.
        ``"captioned 40k, failed 12"``)."""
        with self._lock:
            self.n += k
            self._extra = extra
            now = time.time()
            if self.n >= self._next or (now - self._last_t) >= self.secs:
                self._emit(now, extra)
                self._next = self.n + self.every
                self._last_t = now
            if self._watch is None:
                # Start the keep-alive daemon on first use. Launching it while
                # holding the lock is safe: the thread's first act is to block on
                # _stop.wait(secs), which touches no shared state.
                self._watch = threading.Thread(
                    target=self._watchdog, name=f"progress-{self.label}", daemon=True)
                self._watch.start()

    def done(self, extra: str = "") -> None:
        """Emit a final heartbeat regardless of cadence and stop the keep-alive
        thread (call once at the end)."""
        self._stop.set()
        with self._lock:
            self._emit(time.time(), extra)

    def _watchdog(self) -> None:
        """Daemon loop: wake every ``secs`` and emit a keep-alive if the stream has
        gone quiet that long, so a stall doesn't read as a hang."""
        while not self._stop.wait(self.secs):
            self._heartbeat_if_stalled()

    def _heartbeat_if_stalled(self) -> None:
        """Emit iff ``secs`` have passed since the last emit (any source). Split out
        from the thread loop so it can be unit-tested deterministically."""
        with self._lock:
            now = time.time()
            if now - self._last_t >= self.secs:
                self._emit(now, self._extra)
                self._last_t = now

    def _emit(self, now: float, extra: str = "") -> None:
        elapsed = max(1e-6, now - self._t0)
        rate = self.n / elapsed  # items/sec across the whole fleet
        if self.total:
            pct = 100.0 * self.n / self.total
            eta = int(max(0, self.total - self.n) / rate) if rate > 0 else 0
            h, r = divmod(eta, 3600)
            m, _ = divmod(r, 60)
            head, tail = f"{self.n:,}/{self.total:,} ({pct:.1f}%)", f" | ETA {h}h{m:02d}m"
        else:
            head, tail = f"{self.n:,}", ""
        extra_str = f" | {extra}" if extra else ""
        self._log(
            f"  [{self.label}] progress {head} | {rate * 60:,.0f} {self.unit}/min "
            f"| elapsed {int(elapsed // 60)}m{tail}{extra_str}",
            flush=True,
        )
