"""Unit tests for the shared Modal-stage helpers in diskrot.modal_common."""

from diskrot.modal_common import list_wave_mp3s, wave_subdir


def _make_corpus(tmp_path):
    """A corpus root with a flat legacy file + two wave folders + decoys."""
    (tmp_path / "legacy_root.mp3").write_bytes(b"x")
    for wave, names in {"wave_base_0": ["a", "b"], "wave_7": ["c"]}.items():
        d = tmp_path / "waves" / wave
        d.mkdir(parents=True)
        for n in names:
            (d / f"{n}.mp3").write_bytes(b"x")
        (d / "notes.txt").write_bytes(b"x")  # non-mp3 ignored
    # non-wave_ folder under waves/ must not match the "all" sweep
    stray = tmp_path / "waves" / "scratch"
    stray.mkdir()
    (stray / "stray.mp3").write_bytes(b"x")
    return tmp_path


def test_wave_subdir():
    assert wave_subdir("") == ""
    assert wave_subdir("base_3") == "waves/wave_base_3"


def test_list_wave_mp3s_scoped(tmp_path):
    corpus = _make_corpus(tmp_path)
    got = list_wave_mp3s(corpus, "base_0")
    assert [p.stem for p in got] == ["a", "b"]


def test_list_wave_mp3s_legacy_root_is_nonrecursive(tmp_path):
    corpus = _make_corpus(tmp_path)
    got = list_wave_mp3s(corpus, "")
    assert [p.stem for p in got] == ["legacy_root"]


def test_list_wave_mp3s_all_sweeps_every_wave(tmp_path):
    corpus = _make_corpus(tmp_path)
    got = list_wave_mp3s(corpus, "all")
    # every waves/wave_*/ mp3, path-sorted ("wave_7" < "wave_base_0");
    # excludes the flat root and non-wave_ dirs
    assert [p.stem for p in got] == ["c", "a", "b"]
    assert all(p.parent.name.startswith("wave_") for p in got)
