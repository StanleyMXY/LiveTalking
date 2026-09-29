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

import numpy as np

from livekit_bridge.asr_bridge import AsrBridge


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
