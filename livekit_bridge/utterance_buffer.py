"""
UtteranceBuffer — energy-threshold VAD plus utterance-boundary buffering.

``feed()`` takes one 20ms float32 chunk at a time (``_FRAME_MS``; the
entrypoint makes that true by constructing its ``rtc.AudioStream`` with
``frame_size_ms=20``) and returns a completed utterance when either:

- speech has been followed by ``silence_threshold_ms`` of silence (the
  normal end of a turn), or
- the buffered speech has reached ``max_utterance_ms`` without such a gap
  (final-review finding). Without this cap, ambient noise that stays above
  ``_ENERGY_THRESHOLD`` means no silence gap is ever seen: the buffer never
  emits, grows for the whole life of the job, and the avatar can never
  answer. When the cap is hit the accumulated audio is emitted as a
  completed utterance and buffering starts over from the next chunk.

Default cap: 30s (``_DEFAULT_MAX_UTTERANCE_MS``). A conversational turn to
a voice assistant is normally a sentence or two — a few seconds — so 30s
is well past any realistic single turn while still bounding memory to
30s × 16kHz × 4 bytes ≈ 1.9MB per session. It is also the same ceiling
funasr uses for SenseVoice: its own ``sensevoice`` preset configures the
VAD with ``max_single_segment_time: 30000`` ms (installed
``funasr/cli.py:12``, also ``funasr/bin/_server_app.py:265``), so an
utterance cut at the cap is never longer than what the ASR model is
already set up to receive as one segment.

The energy threshold VAD itself is a deliberate, reviewed choice for this
branch; swapping it for a model-based VAD (e.g. ``livekit-plugins-silero``)
is a deferred follow-up — see ``livekit_bridge/__init__.py``'s "Known
limitations" list.
"""

import numpy as np

_FRAME_MS = 20
_ENERGY_THRESHOLD = 0.05
_DEFAULT_MAX_UTTERANCE_MS = 30_000


class UtteranceBuffer:
    def __init__(
        self,
        silence_threshold_ms: int = 500,
        max_utterance_ms: int = _DEFAULT_MAX_UTTERANCE_MS,
    ):
        self._silence_threshold_frames = silence_threshold_ms // _FRAME_MS
        self._max_utterance_frames = max(1, max_utterance_ms // _FRAME_MS)
        self._speech_chunks: list[np.ndarray] = []
        self._silence_frame_count = 0
        self._has_speech = False

    def feed(self, audio_chunk: np.ndarray) -> np.ndarray | None:
        energy = float(np.abs(audio_chunk).mean())
        is_speech = energy > _ENERGY_THRESHOLD

        if is_speech:
            self._speech_chunks.append(audio_chunk)
            self._has_speech = True
            self._silence_frame_count = 0
            if len(self._speech_chunks) >= self._max_utterance_frames:
                # Cap reached with no silence gap: force-emit (module
                # docstring). Counts buffered speech only — silence chunks
                # are never appended, so they never count toward the cap.
                return self._emit()
            return None

        if not self._has_speech:
            return None

        self._silence_frame_count += 1
        if self._silence_frame_count < self._silence_threshold_frames:
            return None

        return self._emit()

    def reset(self) -> None:
        self._speech_chunks = []
        self._silence_frame_count = 0
        self._has_speech = False

    def _emit(self) -> np.ndarray | None:
        result = np.concatenate(self._speech_chunks) if self._speech_chunks else None
        self.reset()
        return result
