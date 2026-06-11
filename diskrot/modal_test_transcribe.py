"""One-off diagnostic: run Transcriber on a sample of pending files and print
the in-band error strings that the full run only surfaces inside save_results
container logs. Reuses the real app/image/class from modal_transcribe."""
from diskrot.modal_transcribe import Transcriber, app, list_pending


@app.local_entrypoint()
def diagnose():
    pending = list_pending.remote()
    print(f"{len(pending)} pending")
    # Head of the sorted pending list (re-attempted first on every relaunch —
    # likely the perma-failing population) + a slice from the middle as control.
    sample = pending[:8] + pending[len(pending) // 2 : len(pending) // 2 + 8]
    t = Transcriber()
    for r in t.transcribe_file.map(sample, return_exceptions=True):
        if isinstance(r, Exception):
            print(f"MAP-EXC  {type(r).__name__}: {str(r)[:200]}")
            continue
        key, result, error = r
        status = "ok" if result else ("instrumental" if error is None else "ERROR")
        print(f"{status:12s} {key[:60]!r}  {('— ' + error[:200]) if error else ''}")
