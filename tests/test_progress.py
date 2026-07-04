"""Tests for diskrot/progress.py — the shared pipeline progress heartbeat."""
from __future__ import annotations

from diskrot.progress import ProgressReporter


def _sink():
    lines: list[str] = []

    def log(msg, **kw):  # absorbs flush=True
        lines.append(msg)

    return lines, log


def test_count_trigger_emits_with_pct():
    lines, log = _sink()
    rep = ProgressReporter(100, "x", every=10, secs=1e9, unit="files", log=log)
    for _ in range(10):
        rep.update(1)
    assert len(lines) == 1
    assert "[x] progress" in lines[0]
    assert "10/100 (10.0%)" in lines[0]
    assert "files/min" in lines[0]
    assert "ETA" in lines[0]


def test_batch_update_accumulates():
    lines, log = _sink()
    rep = ProgressReporter(1000, "b", every=100, secs=1e9, log=log)
    rep.update(100)
    rep.update(100)
    assert len(lines) == 2
    assert "200/1,000" in lines[1]  # thousands separator + running total


def test_no_total_drops_pct_and_eta():
    lines, log = _sink()
    rep = ProgressReporter(0, "y", every=5, secs=1e9, log=log)
    rep.update(5)
    assert len(lines) == 1
    assert "%" not in lines[0]
    assert "ETA" not in lines[0]
    assert "5 |" in lines[0]  # bare count, then the rate field


def test_done_always_emits_even_below_threshold():
    lines, log = _sink()
    rep = ProgressReporter(10, "z", every=1000, secs=1e9, log=log)
    rep.update(3)  # below the count trigger and time trigger → no emit yet
    assert lines == []
    rep.done()
    assert len(lines) == 1
    assert "3/10" in lines[0]


def test_extra_suffix_appended():
    lines, log = _sink()
    rep = ProgressReporter(10, "e", every=1, secs=1e9, log=log)
    rep.update(1, extra="failed 2")
    assert lines[0].endswith("| failed 2")


def test_time_trigger_fires_without_count(monkeypatch):
    import diskrot.progress as pmod

    clock = {"t": 1000.0}
    monkeypatch.setattr(pmod.time, "time", lambda: clock["t"])
    lines, log = _sink()
    rep = ProgressReporter(100, "t", every=10_000, secs=45.0, log=log)
    rep.update(1)  # count below 10k, no time elapsed → no emit
    assert lines == []
    clock["t"] += 46.0  # cross the 45s time trigger
    rep.update(1)
    assert len(lines) == 1


def test_keepalive_emits_when_stream_stalls(monkeypatch):
    # The keep-alive fires when the result stream goes quiet (no update() calls) —
    # a preempted-worker / cold-restart stall must not read as a hang.
    import diskrot.progress as pmod

    clock = {"t": 1000.0}
    monkeypatch.setattr(pmod.time, "time", lambda: clock["t"])
    lines, log = _sink()
    rep = ProgressReporter(100, "k", every=10_000, secs=45.0, log=log)
    rep.update(5)  # below count + time triggers → no emit
    assert lines == []

    clock["t"] += 46.0  # stream stalls; only wall-clock advances
    rep._heartbeat_if_stalled()  # what the daemon loop calls each wake
    assert len(lines) == 1
    assert "5/100" in lines[0]  # flat count = the stall is visible

    rep._heartbeat_if_stalled()  # no time since last emit → no duplicate
    assert len(lines) == 1


def test_keepalive_quiet_while_stream_flows(monkeypatch):
    # If update() keeps emitting, the keep-alive must NOT add duplicate lines.
    import diskrot.progress as pmod

    clock = {"t": 1000.0}
    monkeypatch.setattr(pmod.time, "time", lambda: clock["t"])
    lines, log = _sink()
    rep = ProgressReporter(100, "f", every=1, secs=45.0, log=log)
    rep.update(1)  # every=1 → emits immediately, resets _last_t
    assert len(lines) == 1
    rep._heartbeat_if_stalled()  # 0s since the update emit → silent
    assert len(lines) == 1


def test_done_stops_keepalive_thread():
    # done() must set the stop event so the daemon exits promptly.
    rep = ProgressReporter(10, "s", every=1, log=lambda *a, **k: None)
    rep.update(1)  # starts the watchdog thread
    assert rep._watch is not None and rep._watch.is_alive()
    rep.done()
    rep._watch.join(timeout=5.0)
    assert not rep._watch.is_alive()
