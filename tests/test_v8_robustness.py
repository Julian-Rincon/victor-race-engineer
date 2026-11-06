"""v8 — robustez: keys, modelos retirados, STT local, comandos de voz y
respuestas sin LLM. Todo mockeado: nunca llama a una API real."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

import victor_brain as vb
import victor_config as cfg


# ─── Config: cadena de .env ──────────────────────────────────────────────────

def test_env_chain_first_file_wins_and_missing_files_are_skipped(tmp_path):
    a = tmp_path / "a.env"; a.write_text("GROQ_API_KEY=from_a\n")
    b = tmp_path / "b.env"; b.write_text("GROQ_API_KEY=from_b\nexport GEMINI_API_KEY='gem'\n")
    merged, used = cfg._load_env_chain([tmp_path / "nope.env", a, b])
    assert merged["GROQ_API_KEY"] == "from_a"
    assert merged["GEMINI_API_KEY"] == "gem"          # soporta `export` y comillas
    assert used == [a, b]


def test_get_key_accepts_short_and_nexus_prefixed_names(monkeypatch):
    monkeypatch.setattr(cfg, "_env", {"NEXUS_DEEPGRAM_API_KEY": "dg"})
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("NEXUS_DEEPGRAM_API_KEY", raising=False)
    assert cfg.get_key("DEEPGRAM_API_KEY") == "dg"
    monkeypatch.setattr(cfg, "_env", {"DEEPGRAM_API_KEY": "short"})
    assert cfg.get_key("NEXUS_DEEPGRAM_API_KEY") == "short"


def test_process_env_beats_dotenv(monkeypatch):
    monkeypatch.setattr(cfg, "_env", {"GROQ_API_KEY": "file"})
    monkeypatch.setenv("GROQ_API_KEY", "proc")
    assert cfg.get_key("GROQ_API_KEY") == "proc"


def test_victor_env_file_override_is_first_candidate(monkeypatch, tmp_path):
    monkeypatch.setenv("VICTOR_ENV_FILE", str(tmp_path / "x.env"))
    assert cfg.env_candidates()[0] == tmp_path / "x.env"


def test_default_models_are_not_the_retired_ones():
    retired = {"llama-3.3-70b-versatile", "gemini-2.0-flash", "meta/llama-4-maverick"}
    for model in (cfg.GROQ_MODEL, cfg.CEREBRAS_MODEL, cfg.NVIDIA_MODEL, cfg.GEMINI_MODEL):
        assert model not in retired


# ─── Brain: HTTP real simulado ───────────────────────────────────────────────

class _Resp:
    def __init__(self, status=200, body=None, text="", headers=None):
        self.status_code = status
        self._body = body if body is not None else {"choices": [{"message": {"content": "OK"}}]}
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _brain(**keys):
    with patch.object(vb.cfg, "GROQ_API_KEY", keys.get("groq", "")), \
         patch.object(vb.cfg, "CEREBRAS_API_KEY", keys.get("cerebras", "")), \
         patch.object(vb.cfg, "NVIDIA_API_KEY", keys.get("nvidia", "")), \
         patch.object(vb.cfg, "GEMINI_API_KEY", keys.get("gemini", "")), \
         patch.object(vb.cfg, "MISTRAL_API_KEY", keys.get("mistral", "")), \
         patch.object(vb.cfg, "OPENROUTER_API_KEY", keys.get("openrouter", "")), \
         patch.object(vb.cfg, "CUSTOM_LLM_BASE_URL", ""), \
         patch.object(vb.cfg, "LOCAL_LLM_MODEL", keys.get("local", "")):
        return vb.VictorBrain()


def test_session_sends_custom_user_agent():
    # Groq rechaza el UA por defecto de python-requests.
    assert "python-requests" not in vb._session.headers["User-Agent"]
    assert "victor" in vb._session.headers["User-Agent"].lower()


def test_retired_model_heals_from_models_endpoint_and_retries():
    brain = _brain(groq="k")
    g = brain.backends[0]
    g.model = "llama-3.3-70b-versatile"
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json["model"])
        if json["model"] == "llama-3.3-70b-versatile":
            return _Resp(400, text='{"error":{"message":"The model `llama-3.3-70b-versatile` has been decommissioned"}}')
        return _Resp()

    models = {"data": [{"id": "whisper-large-v3"}, {"id": "openai/gpt-oss-20b"},
                       {"id": "openai/gpt-oss-120b"}, {"id": "meta-llama/llama-guard-4-12b"}]}
    with patch.object(vb._session, "post", side_effect=fake_post), \
         patch.object(vb._session, "get", return_value=_Resp(body=models)):
        reply = brain.complete("sys", [{"role": "user", "content": "hola"}])
    assert reply == "OK"
    assert g.model == "openai/gpt-oss-120b"
    assert calls == ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"]


def test_heal_strips_gemini_models_prefix():
    brain = _brain(gemini="k")
    gem = brain.backends[3]
    body = {"data": [{"id": "models/gemini-3.6-flash"}, {"id": "models/text-embedding-004"}]}
    with patch.object(vb._session, "get", return_value=_Resp(body=body)):
        assert brain.list_models(gem) == ["gemini-3.6-flash", "text-embedding-004"]


def test_pick_model_skips_non_chat_models():
    assert vb.pick_model(["whisper-large-v3-turbo", "llama-guard-x", "qwen3-32b"], ["qwen3", "llama"]) == "qwen3-32b"
    assert vb.pick_model(["whisper-large-v3"], ["whisper"]) is None


def test_payment_error_benches_backend_for_the_session_and_falls_back():
    brain = _brain(cerebras="k", gemini="k")
    cer = brain.backends[1]

    def fake_post(url, headers, json, timeout):
        if "cerebras" in url:
            return _Resp(402, text="payment required")
        return _Resp(body={"choices": [{"message": {"content": "gemini ok"}}]})

    with patch.object(vb._session, "post", side_effect=fake_post):
        assert brain.complete("s", [{"role": "user", "content": "x"}]) == "gemini ok"
    assert not cer.available()
    assert cer.paused_until - time.time() > 3600     # no reintenta cada 5 min
    assert "402" in brain.status()["cerebras"]


def test_rate_limit_respects_retry_after_without_tripping_breaker():
    brain = _brain(groq="k")
    g = brain.backends[0]
    with patch.object(vb._session, "post", return_value=_Resp(429, headers={"retry-after": "42"})):
        assert brain.complete("s", [{"role": "user", "content": "x"}]) == ""
    assert g.fails == 0
    assert 30 < g.paused_until - time.time() <= 42


def test_reasoning_effort_low_sent_for_gpt_oss_on_groq_only():
    brain = _brain(groq="k", gemini="k")
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen[url] = json
        return _Resp(500) if "groq" in url else _Resp()

    brain.backends[0].model = "openai/gpt-oss-120b"
    with patch.object(vb._session, "post", side_effect=fake_post):
        brain.complete("s", [{"role": "user", "content": "x"}], max_tokens=100)
    groq_payload = next(v for k, v in seen.items() if "groq" in k)
    gem_payload = next(v for k, v in seen.items() if "googleapis" in k)
    assert groq_payload["reasoning_effort"] == "low"
    assert groq_payload["max_tokens"] == 100 + brain.backends[0].reasoning_reserve
    assert "reasoning_effort" not in gem_payload
    assert gem_payload["max_tokens"] == 100


def test_status_reports_missing_keys():
    assert _brain().status()["groq"] == "sin key"


# ─── STT: orden de respaldo ──────────────────────────────────────────────────

def test_stt_falls_back_to_local_when_cloud_unavailable(monkeypatch):
    monkeypatch.setattr(vb, "_transcribe_groq", lambda p: ("", False))
    monkeypatch.setattr(vb.cfg, "DEEPGRAM_API_KEY", "")
    monkeypatch.setattr(vb, "_transcribe_local", lambda p: "victor cuanto combustible")
    assert vb.transcribe("x.wav") == "victor cuanto combustible"


def test_stt_does_not_retry_locally_when_groq_rejected_as_hallucination(monkeypatch):
    monkeypatch.setattr(vb, "_transcribe_groq", lambda p: ("", True))
    local = MagicMock(return_value="alucinación")
    monkeypatch.setattr(vb, "_transcribe_local", local)
    assert vb.transcribe("x.wav") == ""
    local.assert_not_called()


def test_local_stt_can_be_disabled(monkeypatch):
    monkeypatch.setattr(vb.cfg, "LOCAL_STT_MODEL", "off")
    assert not vb.local_stt_enabled()


# ─── Comandos de voz ─────────────────────────────────────────────────────────

def test_kw_hit_matches_multiword_phrases_and_ignores_accents(eng):
    assert eng._kw_hit("Víctor, modo gap", eng._GAP_MODE_KW)
    assert eng._kw_hit("¿Cómo vamos?", {"como vamos"})
    assert eng._kw_hit("donde estoy", {"dónde estoy"})
    assert not eng._kw_hit("parabrisas roto", {"para"})


@pytest.fixture
def voice(eng, spoken, tmp_path, monkeypatch):
    """Ejecuta _handle_voice con un texto transcrito dado; devuelve lo hablado."""
    monkeypatch.setattr(eng, "_muted", False)
    monkeypatch.setattr(eng, "_gap_mode", False)
    llm = MagicMock(return_value="")
    monkeypatch.setattr(eng, "_groq", llm)
    monkeypatch.setattr(eng, "_in_braking_zone", lambda: False)

    def run(text):
        wav = tmp_path / "rec.wav"; wav.write_bytes(b"")
        monkeypatch.setattr(eng, "_transcribe", lambda p: text)
        spoken.clear()
        eng._handle_voice(str(wav))
        return [t for _, t in spoken]
    run.llm = llm
    return run


def test_para_los_gaps_disables_gap_mode_instead_of_muting(eng, voice):
    eng._gap_mode = True
    out = voice("Victor, para los gaps")
    assert eng._gap_mode is False
    assert eng._muted is False
    assert out == ["Gap mode desactivado."]


def test_modo_gap_activates_gap_mode(eng, voice):
    out = voice("Víctor, modo gap")
    assert eng._gap_mode is True and out[0].startswith("Gap mode activo")


def test_para_inside_a_question_does_not_mute(eng, voice):
    voice("Victor, ¿para cuántas vueltas me alcanza el combustible?")
    assert eng._muted is False
    assert voice.llm.called           # fue al flujo de combustible


def test_short_para_still_mutes(eng, voice):
    voice("Victor, para")
    assert eng._muted is True


def test_resume_la_carrera_is_status_not_unmute(eng, voice):
    eng._muted = True
    voice("Victor, resume la carrera completa")
    assert eng._muted is True         # no se reactivó por "resume"


def test_without_llm_fuel_question_still_gets_an_answer(eng, voice, monkeypatch):
    t = eng.Telemetry(); t.fuel = 23.4; t.fuel_laps = 7.2
    monkeypatch.setattr(eng, "_current", t)
    out = voice("Victor, cuánto combustible me queda")
    assert out and "23.4 litros" in out[0] and "7 vueltas" in out[0]


def test_local_stt_typo_neomaticos_routes_to_tyres(eng, voice, monkeypatch):
    t = eng.Telemetry(); t.tyre_fl = 85; t.tyre_fr = 88; t.tyre_rl = 80; t.tyre_rr = 81
    monkeypatch.setattr(eng, "_current", t)
    out = voice("Víctor, ¿cómo están los neomáticos?")
    assert "Delanteras 85 y 88" in out[0]


def test_local_reply_damage_and_sector(eng):
    t = eng.Telemetry()
    assert eng._local_reply("damage", t) == "Sin daños, coche limpio."
    t.dmg_front = 40
    assert "Considera boxes" in eng._local_reply("damage", t)
    t.last_splits = [30000, 31000, 29000]; t.best_splits = [29900, 30500, 29000]
    assert "sector 2" in eng._local_reply("sector", t)


def test_fmt_lap(eng):
    assert eng._fmt_lap(102345) == "1:42.345"
    assert eng._fmt_lap(59000) == "59.000"


# ─── Texto hablable (Piper leía mal símbolos) ────────────────────────────────

@pytest.mark.parametrize("raw, spoken", [
    ("4° de 16", "cuarto de 16"),
    ("Vas P3, cerca del podio", "Vas tercero, cerca del podio"),
    ("Neumático FL a 104°C — sobrecalentamiento.", "Neumático FL a 104 grados, sobrecalentamiento."),
    ("Desgaste 23% restante.", "Desgaste 23 por ciento restante."),
    ("Vuelta 5, 1:42.345.", "Vuelta 5, 1 minuto 42 con 345."),
    ("Vuelta proyectada 102.345s.", "Vuelta proyectada 1 minuto 42 con 345."),
    ("+0.32s vs tu mejor.", "más 0 coma 32 segundos contra tu mejor."),
    ("-1.20s", "menos 1 coma 20 segundos"),
    ("a 2.1s detrás", "a 2 coma 1 segundos detrás"),
    ("Consumo 2.45L/v, quedan 31.5L", "Consumo 2.45 litros por vuelta, quedan 31.5 litros"),
    ("3 sectores", "3 sectores"),
    ("15° de 20", "puesto 15 de 20"),
])
def test_speakable(eng, raw, spoken):
    assert eng._speakable(raw) == spoken


def test_gap_and_damage_natural_phrasings_route(eng, voice):
    voice("Victor, a cuánto está el de adelante")
    assert "gaps con rivales" in voice.llm.call_args[0][0]
    voice("Victor, tengo daños")
    assert "daños" in voice.llm.call_args[0][0]


def test_failures_after_an_expired_rate_limit_still_trip_breaker():
    brain = _brain(groq="k")
    g = brain.backends[0]
    g.pause(-1, "rate limit (429)")            # pausa vieja ya vencida
    with patch.object(vb._session, "post", return_value=_Resp(500)):
        for _ in range(3):
            brain.complete("s", [{"role": "user", "content": "x"}])
    assert g.fails == 3 and not g.available()


def test_openrouter_heal_never_picks_a_paid_model():
    brain = _brain(openrouter="k")
    orb = next(b for b in brain.backends if b.name == "openrouter")
    body = {"data": [{"id": "openai/gpt-oss-120b"}, {"id": "anthropic/some-paid"}]}
    with patch.object(vb._session, "get", return_value=_Resp(body=body)):
        assert brain._heal(orb) is False
    assert orb.model.endswith(":free")


def test_mistral_and_openrouter_are_in_the_chain_after_gemini():
    names = [b.name for b in _brain().backends]
    assert names[-2:] == ["mistral", "openrouter"]


def test_zero_quota_429_is_permanent_not_transient():
    brain = _brain(mistral="k")
    m = next(b for b in brain.backends if b.name == "mistral")
    resp = _Resp(429, headers={"x-ratelimit-limit-req-minute": "0"})
    with patch.object(vb._session, "post", return_value=resp):
        brain.complete("s", [{"role": "user", "content": "x"}])
    assert m.paused_until - time.time() > 3600


def test_openrouter_heal_picks_only_free_models():
    brain = _brain(openrouter="k")
    orb = next(b for b in brain.backends if b.name == "openrouter")
    body = {"data": [{"id": "nvidia/nemotron-3-ultra-550b-a55b"},
                     {"id": "nvidia/nemotron-3.5-content-safety:free"},
                     {"id": "google/gemma-4-31b-it:free"}]}
    orb.model = "dead:free"
    with patch.object(vb._session, "get", return_value=_Resp(body=body)):
        assert brain._heal(orb) is True
    assert orb.model == "google/gemma-4-31b-it:free"


def test_tyre_life_wording_is_remaining_life(eng):
    t = eng.Telemetry(); t.tyre_wear_fl = t.tyre_wear_fr = t.tyre_wear_rl = t.tyre_wear_rr = 96
    assert "96 por ciento de vida" in eng._local_reply("tyres", t)


def test_local_ollama_is_last_and_never_uses_gpu():
    brain = _brain(groq="k", local="gemma4:e2b")
    assert brain.backends[-1].name == "local"
    seen = {}

    def fake_post(url, json, timeout, headers=None):
        if "groq" in url:
            return _Resp(500)
        seen.update(url=url, json=json, timeout=timeout)
        return _Resp(body={"message": {"content": "Gomas bien."}})

    with patch.object(vb._session, "post", side_effect=fake_post):
        assert brain.complete("s", [{"role": "user", "content": "x"}]) == "Gomas bien."
    assert seen["url"].endswith("/api/chat")
    assert seen["json"]["options"]["num_gpu"] == 0
    assert seen["timeout"] >= 25
    assert brain.last_backend == "local"


def test_local_llm_skipped_when_allow_local_false():
    brain = _brain(local="gemma4:e2b")
    with patch.object(vb._session, "post") as post:
        assert brain.complete("s", [{"role": "user", "content": "x"}], allow_local=False) == ""
    post.assert_not_called()


def test_fuel_question_does_not_use_local_llm_but_free_question_does(eng, voice):
    voice("Victor, cuánto combustible me queda")
    assert voice.llm.call_args.kwargs["allow_local"] is False
    voice("Victor, cuéntame un chiste de ingenieros")
    assert voice.llm.call_args.kwargs["allow_local"] is True


def test_speakable_strips_markdown(eng):
    assert eng._speakable("detrás de **Torres** a 2.1s") == "detrás de Torres a 2 coma 1 segundos"


def test_system_prompt_forbids_inventing_data(eng):
    assert "NUNCA INVENTES DATOS" in eng._SYSTEM
    assert "USA SIEMPRE el nombre del rival" not in eng._SYSTEM
