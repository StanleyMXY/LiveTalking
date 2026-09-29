"""
Task 1 verification results: LiveKit audio frame format and the pipeline's
ASR call convention. Later tasks (Task 2 audio resampling, Task 6 ASR bridge)
import the constants below instead of re-deriving these facts.

Verified by reading the *installed* ``livekit`` Python package's source
(v1.1.20, installed via ``pip install livekit`` into this project's venv —
see ``pip show livekit`` for the install location) and this repo's
``server/asr_server.py``. No live LiveKit dev server was needed or used;
per the controller's ruling, the wire format is a documented, static SDK
fact and reading the SDK source is the reliable way to establish it.

Run as a human-readable dump of these facts:
    python -m livekit_bridge.format_probe
"""

# ─── Fact 1: LiveKit audio frame format ────────────────────────────────────
#
# Sample encoding / byte order
# -----------------------------
# `rtc.AudioFrame.data` is documented and implemented as 16-bit signed
# integers (int16), interleaved by channel, in the platform's native byte
# order (the property casts the underlying buffer to ctypes/memoryview
# format "h" = native signed short — little-endian on every deployment
# target this pipeline runs on: x86-64 / ARM64):
#   - docstring: "The format of the audio data is 16-bit signed integers
#     (int16) interleaved by channel."
#     livekit/rtc/audio_frame.py:27 (installed package: livekit==1.1.20,
#     site-packages/livekit/rtc/audio_frame.py)
#   - `AudioFrame.data` property: `memoryview(self._data).cast("B").cast("h")`
#     livekit/rtc/audio_frame.py:118-126
# This matches `np.frombuffer(buf, dtype=np.int16)`, which is what
# server/asr_server.py already assumes for its own PCM16 input
# (server/asr_server.py:211), so no byte-order conversion is needed when
# handing LiveKit frame data to the pipeline's ASR path.
#
# Sample rate / channel count actually delivered to Python code
# ---------------------------------------------------------------
# This is the one non-obvious fact: the sample rate/channel count of the
# `AudioFrame` objects your code receives from a subscribed track is NOT
# simply "whatever the publishing client captured at". Audio frames are
# only obtained by constructing an `rtc.AudioStream` from the subscribed
# track (via the `room.on("track_subscribed")` callback -> a
# `RemoteAudioTrack`, confirmed at livekit/rtc/room.py:878-908, which emits
# `(track, publication, participant)`), and `AudioStream.__init__` /
# `.from_track()` / `.from_participant()` take explicit `sample_rate` and
# `num_channels` parameters that are sent to LiveKit's native (Rust) FFI
# engine, which resamples/re-channels internally before your Python code
# ever sees a frame:
#   - `sample_rate: int = 48000` and `num_channels: int = 1` are the
#     constructor defaults. livekit/rtc/audio_stream.py:64-65 (also
#     :154-155 on `from_participant`, :206-207 on `from_track`)
#   - those values are forwarded verbatim to the FFI request that creates
#     the native stream: `new_audio_stream.sample_rate = self._sample_rate`
#     / `.num_channels = self._num_channels`.
#     livekit/rtc/audio_stream.py:258-259 (see also :277-278 for the
#     participant-based constructor)
# Practical consequence for this bridge: if `livekit_bridge` constructs the
# `AudioStream` with the SDK's defaults (or omits the params), it receives
# 48 kHz mono frames — WebRTC/Opus's native rate, which is what browser/
# mobile LiveKit clients publish at. If it instead constructs the stream
# with `sample_rate=16000, num_channels=1` explicitly, LiveKit's native FFI
# layer performs the 48kHz -> 16kHz resampling *before* Python sees the
# frame, and no separate Python-side resampling step is strictly required.
# The constant below records the SDK's *default*/native format (the fact
# asked for by this task); which of the two paths Task 2 actually takes is
# that task's design decision, not re-decided here.
LIVEKIT_AUDIO_SAMPLE_RATE = 48000  # rtc.AudioStream default; see citations above
PIPELINE_ASR_SAMPLE_RATE = 16000  # already known: server/asr_server.py:122 (`SAMPLE_RATE = 16000`)
NEEDS_RESAMPLING = LIVEKIT_AUDIO_SAMPLE_RATE != PIPELINE_ASR_SAMPLE_RATE  # True: 48000 != 16000


# ─── Fact 2: pipeline ASR call convention ──────────────────────────────────
#
# Real inference function (not the websocket handshake code)
# -------------------------------------------------------------
# `_run_inference(audio_float32, sample_rate, use_itn)` in
# server/asr_server.py:71-117 is the function that actually runs the
# SenseVoice/FunASR model. It is a **blocking** call (must be invoked via
# `loop.run_in_executor`, exactly as the existing websocket handler already
# does at server/asr_server.py:218-224).
#
#   Input:
#     - `audio_float32`: 1-D `numpy.ndarray`, dtype float32, samples
#       normalized to [-1, 1], MONO (no channel dimension). The websocket
#       handler produces this from raw PCM16 bytes via
#       `np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0`
#       (server/asr_server.py:211-212) before calling `_run_inference`.
#     - `sample_rate`: int, Hz. The websocket handler always passes the
#       module constant `SAMPLE_RATE = 16000` (server/asr_server.py:122,
#       :222) — i.e. PIPELINE_ASR_SAMPLE_RATE above. `_run_inference` uses
#       it only to write a valid WAV header via `soundfile.write(...)`
#       (server/asr_server.py:88-89) so FunASR can read the rate back out;
#       it does not itself resample.
#     - `use_itn`: bool, passed straight through to `model.generate(...,
#       use_itn=use_itn)` (server/asr_server.py:98).
#
#   Output: `tuple[str, float, float]` = (transcribed_text, inference_ms,
#     audio_duration_s) (server/asr_server.py:78-81, :117). `transcribed_text`
#     is plain post-processed text with no timestamps — the websocket
#     handler always sends back `"timestamp": None` (server/asr_server.py:
#     196-201, :236-241); there is no per-word/segment timing available
#     from this call.
#
# Model instance lifecycle (singleton, not per-request)
# ---------------------------------------------------------
# The `funasr.AutoModel` instance is a **lazily-initialized module-level
# global singleton**, not recreated per request:
#   - `_sensevoice_model = None` module global (server/asr_server.py:28)
#   - `_load_sensevoice()` (server/asr_server.py:33-68) does a double-checked
#     lock (`_sensevoice_load_lock`, a `threading.Lock`) around
#     `global _sensevoice_model; ...; _sensevoice_model = AutoModel(...)` so
#     concurrent first callers share one initialization and the ~500MB
#     HuggingFace-hosted SenseVoiceSmall weights are loaded exactly once
#     per process.
#   - Every call to `_run_inference` fetches the same instance via
#     `model = _load_sensevoice()` (server/asr_server.py:85); actual
#     `model.generate(...)` calls are additionally serialized (not
#     parallelized) across concurrent requests via a second lock,
#     `_sensevoice_inference_lock` (server/asr_server.py:30, :93-100).
#   - Consequence for a later ASR bridge task: reuse this singleton by
#     importing `_run_inference` (and, if needed, `_load_sensevoice`)
#     directly from `server.asr_server` and calling it the same way the
#     websocket handler does (`run_in_executor` + the existing locks) —
#     never construct a second `AutoModel(...)`, which would reload the
#     model and duplicate GPU/CPU memory.
#   - Per this project's CLAUDE.md model/dataset rule: the model is already
#     loaded via `hub="hf"` (HuggingFace), not ModelScope
#     (server/asr_server.py:62) — nothing to change here.


def probe_livekit_audio_format():
    """
    Optional manual/live secondary check: connect to a real LiveKit room,
    subscribe to a published audio track, and print the actual AudioFrame
    fields observed at runtime, to double-check the source-reading above
    against a live server.

    Not required for Task 1 (no LiveKit dev server was available in this
    environment; the controller ruled that reading the installed SDK's
    source, done above, is the reliable way to establish this static wire
    format fact). Left unimplemented on purpose — this is optional
    tooling for a future manual spot-check, not part of Task 1's deliverable.
    """
    raise NotImplementedError(
        "Optional manual probe, not required by Task 1 — implement only if "
        "a live LiveKit dev server becomes available and a human wants to "
        "double-check the source-derived facts recorded above."
    )


if __name__ == "__main__":
    print("LIVEKIT_AUDIO_SAMPLE_RATE :", LIVEKIT_AUDIO_SAMPLE_RATE)
    print("PIPELINE_ASR_SAMPLE_RATE  :", PIPELINE_ASR_SAMPLE_RATE)
    print("NEEDS_RESAMPLING          :", NEEDS_RESAMPLING)
