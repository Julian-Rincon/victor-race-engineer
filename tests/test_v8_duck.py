"""Audio ducking: solo el stream del juego, restauración exacta, nunca EasyEffects."""
from __future__ import annotations

import json
import subprocess

import victor_duck


def _si(idx, name, binary, vols):
    return {"index": idx, "properties": {"application.name": name, "application.process.binary": binary},
            "volume": {f"ch{i}": {"value": v} for i, v in enumerate(vols)}}


class FakePactl:
    def __init__(self, streams):
        self.streams = streams
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd[1:])
        out = json.dumps(self.streams) if cmd[1:4] == ["-f", "json", "list"] else ""
        return subprocess.CompletedProcess(cmd, 0, out, "")

    def sets(self):
        return [c[1:] for c in self.calls if c[0] == "set-sink-input-volume"]


STREAMS = [
    _si(10, "acs.exe", "wine64-preloader", [65536, 60000]),
    _si(11, "EasyEffects", "easyeffects", [65536, 65536]),
    _si(12, "Firefox", "firefox", [65536, 65536]),
    _si(13, "paplay", "paplay", [65536]),
]


def test_ducks_only_the_game_and_restores_exact_values():
    fake = FakePactl(STREAMS)
    d = victor_duck.Ducker(level=0.5, runner=fake)
    assert d.duck() == 1
    assert fake.sets() == [["10", "32768", "30000"]]
    d.restore()
    assert fake.sets()[-1] == ["10", "65536", "60000"]
    assert not d.ducked


def test_never_touches_easyeffects_even_if_it_matches_app_list():
    fake = FakePactl([_si(11, "EasyEffects acs.exe", "easyeffects", [65536])])
    d = victor_duck.Ducker(runner=fake)
    assert d.duck() == 0 and fake.sets() == []


def test_double_duck_does_not_save_already_lowered_volume():
    fake = FakePactl(STREAMS)
    d = victor_duck.Ducker(level=0.5, runner=fake)
    d.duck(); d.duck()
    d.restore()
    assert fake.sets()[-1] == ["10", "65536", "60000"]


def test_level_one_disables_ducking():
    fake = FakePactl(STREAMS)
    assert victor_duck.Ducker(level=1.0, runner=fake).duck() == 0
    assert fake.calls == []


def test_pactl_failure_is_harmless():
    def boom(cmd, **kw):
        raise FileNotFoundError("pactl")
    d = victor_duck.Ducker(runner=boom)
    assert d.duck() == 0
    d.restore()


def test_custom_backend_goes_first_when_configured(monkeypatch):
    import victor_brain as vb
    monkeypatch.setattr(vb.cfg, "CUSTOM_LLM_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setattr(vb.cfg, "CUSTOM_LLM_MODEL", "qwen3-8b")
    monkeypatch.setattr(vb.cfg, "CUSTOM_LLM_API_KEY", "local")
    b = vb.VictorBrain().backends[0]
    assert (b.name, b.model, b.available()) == ("custom", "qwen3-8b", True)


def test_media_name_path_with_victor_does_not_block_game_ducking():
    s = _si(20, "acs.exe", "", [65536])
    s["properties"]["media.name"] = "/tmp/victor-bench/x.wav"
    fake = FakePactl([s])
    assert victor_duck.Ducker(runner=fake).duck() == 1
