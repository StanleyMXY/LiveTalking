"""
BargeInDetector — decides when above-threshold audio is sustained enough to
count as the user starting to talk (final-review finding).

The entrypoint used to treat a single 20ms frame above ``_ENERGY_THRESHOLD``
(while the avatar was speaking) as a barge-in, so one cough, click or noise
burst interrupted a correct answer — often with nothing to replace it,
since that noise then transcribes to a SenseVoice non-speech marker that
``AsrBridge`` now filters to ``""``. This class requires ``min_speech_ms``
of *consecutive* above-threshold frames first.

Scope: detection only. It does not decide what an utterance is — that is
``UtteranceBuffer``'s job — and it uses the same frame size and energy
threshold as ``UtteranceBuffer`` so both classify each frame identically.
It keeps the frames of the current run (at most ``min_speech_ms`` worth,
12 frames by default) only so the entrypoint can put them back into the
``UtteranceBuffer`` after resetting it on a confirmed barge-in; otherwise
that reset would throw away the first ``min_speech_ms`` of the very speech
that triggered it.

Default 240ms (12 × 20ms frames): inside the 200-300ms band chosen for this
fix — longer than a click or a typical short noise burst, while still about
one syllable of real speech, so a genuine interruption is detected quickly.
A sustained noise longer than this (a long cough, music) still passes; an
energy-only detector cannot tell those apart from speech. When that
happens, the interrupted answer is not replaced, because the noise's own
utterance is filtered to ``""`` after ASR. This is a known limitation of the
energy-threshold VAD (see ``livekit_bridge/__init__.py``'s "Known
limitations" list), not something this class claims to solve.
"""

import numpy as np

from livekit_bridge.utterance_buffer import _ENERGY_THRESHOLD, _FRAME_MS

_DEFAULT_MIN_SPEECH_MS = 240


class BargeInDetector:
    def __init__(self, min_speech_ms: int = _DEFAULT_MIN_SPEECH_MS):
        self._min_frames = max(1, min_speech_ms // _FRAME_MS)
        self._run: list[np.ndarray] = []
        self._fired_this_run = False

    def feed(self, chunk: np.ndarray) -> list[np.ndarray] | None:
        """Observe one 20ms chunk.

        Returns the run's chunks (oldest first, *including* ``chunk``)
        exactly once per run of consecutive above-threshold frames: on the
        frame that makes the run ``min_speech_ms`` long. Returns ``None``
        on every other frame. A below-threshold frame ends the run.
        """
        if float(np.abs(chunk).mean()) <= _ENERGY_THRESHOLD:
            self._run = []
            self._fired_this_run = False
            return None
        if self._fired_this_run:
            return None
        self._run.append(chunk)
        if len(self._run) < self._min_frames:
            return None
        run, self._run = self._run, []
        self._fired_this_run = True
        return run
