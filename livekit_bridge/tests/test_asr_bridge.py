"""
Tests for livekit_bridge.asr_bridge.

Task 1 verified (livekit_bridge/format_probe.py, server/asr_server.py:71-117)
that the pipeline's real ASR entry point is the module-level function
``_run_inference(audio_float32, sample_rate, use_itn)`` in
``server/asr_server.py`` — not a method on some "model" object — and that it
returns a 3-tuple ``(text, inference_ms, audio_duration_s)``, not a bare
string. The task brief's draft (``AsrBridge(model=fake_model)`` mocking
``fake_model.transcribe``) does not match this real API, so these tests
inject/patch the real callable instead of a fictional ``model.transcribe``.
"""

import json
import subprocess
import sys

import numpy as np
import pytest

from livekit_bridge.asr_bridge import (
    SENSEVOICE_MARKER_GLYPHS,
    AsrBridge,
    strip_sensevoice_markers,
)


def test_successful_transcription_returns_text():
    def fake_infer_fn(audio_float32, sample_rate, use_itn):
        # Real _run_inference returns (text, inference_ms, audio_duration_s)
        # (server/asr_server.py:78-81, :117) — the fake must match that shape.
        return "你好", 42.0, 1.0

    bridge = AsrBridge(infer_fn=fake_infer_fn)

    result = bridge.transcribe(np.zeros(16000, dtype=np.float32))

    assert result == "你好"


def test_inference_exception_degrades_to_empty_string_not_crash():
    def fake_infer_fn(audio_float32, sample_rate, use_itn):
        raise RuntimeError("model inference failed")

    bridge = AsrBridge(infer_fn=fake_infer_fn)

    result = bridge.transcribe(np.zeros(16000, dtype=np.float32))

    assert result == ""


def test_default_infer_fn_is_the_real_pipeline_singleton_entry_point():
    # Guards against AsrBridge silently constructing a second model
    # instance instead of reusing the pipeline's lazy singleton: the
    # default callable must be the exact same function object imported
    # from server.asr_server, whose _run_inference internally calls
    # _load_sensevoice() (server/asr_server.py:85) to fetch the shared
    # singleton rather than instantiating a new AutoModel(...).
    from server.asr_server import _run_inference

    bridge = AsrBridge()

    assert bridge._infer_fn is _run_inference


# ─── SenseVoice non-speech marker filtering (final-review finding) ──────


def _bridge_returning(text):
    return AsrBridge(infer_fn=lambda audio, sr, itn: (text, 1.0, 1.0))


@pytest.mark.parametrize("glyph", sorted(SENSEVOICE_MARKER_GLYPHS))
def test_marker_only_result_is_treated_as_no_speech(glyph):
    # e.g. a cough -> "😷", background music -> "🎼",
    # <|nospeech|><|Event_UNK|> -> "❓" — none may drive an LLM call.
    assert _bridge_returning(glyph).transcribe(np.zeros(320, dtype=np.float32)) == ""


@pytest.mark.parametrize(
    "raw",
    ["🎼😀", " 😷 ", "❓😊", "<|Speech_Noise|>", "🤧<|GBG|>", "。", "?", "🎼 ...", ""],
)
def test_nothing_meaningful_left_is_no_speech(raw):
    assert strip_sensevoice_markers(raw) == ""


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("你好😊", "你好"),
        ("🎼今天天气怎么样？😔", "今天天气怎么样？"),
        ("😀hello there😮", "hello there"),
        ("plain text", "plain text"),
        ("123", "123"),
    ],
)
def test_real_speech_keeps_its_text_without_markers(raw, expected):
    assert strip_sensevoice_markers(raw) == expected
    assert _bridge_returning(raw).transcribe(np.zeros(320, dtype=np.float32)) == expected


_FUNASR_PROBE = r"""
import json
from funasr.utils import postprocess_utils as pu
glyphs = set()
for d in (pu.emoji_dict, pu.emo_dict, pu.event_dict):
    glyphs |= {v for v in d.values() if v and not v.startswith("<|")}
glyphs |= pu.emo_set | pu.event_set
tokens = [k for k in pu.emoji_dict if k.count("<|") == 1]
noise_inputs = ["<|nospeech|><|Event_UNK|>"]
noise_inputs += ["<|zh|><|NEUTRAL|>" + t + "<|woitn|>" for t in pu.event_dict]
noise_inputs += ["<|en|>" + e + "<|Speech|><|woitn|>" for e in pu.emo_dict]
noise_inputs += ["<|nospeech|><|EMO_UNKNOWN|>" + t + "<|withitn|>" for t in tokens]
print(json.dumps({
    "glyphs": sorted(glyphs),
    "postprocessed": [pu.rich_transcription_postprocess(s) for s in noise_inputs],
}, ensure_ascii=True))
"""


def test_marker_set_matches_installed_funasr():
    # SENSEVOICE_MARKER_GLYPHS is a literal copied from the installed funasr
    # (see asr_bridge.py's module docstring for why it is not imported at
    # runtime). This recomputes, from the real installed funasr source, every
    # non-empty glyph rich_transcription_postprocess can emit, and runs the
    # real postprocessing over noise-only SenseVoice token sequences — so a
    # funasr upgrade that adds or changes a marker fails here instead of
    # letting noise drive LLM calls again. Run in a subprocess: importing
    # funasr eagerly imports every funasr submodule (and torch), which must
    # not leak into this test process's sys.modules.
    completed = subprocess.run(
        [sys.executable, "-c", _FUNASR_PROBE],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    assert set(result["glyphs"]) == SENSEVOICE_MARKER_GLYPHS
    assert len(result["postprocessed"]) > 20
    for postprocessed in result["postprocessed"]:
        assert strip_sensevoice_markers(postprocessed) == "", postprocessed
