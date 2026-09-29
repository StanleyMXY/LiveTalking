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
``feed()`` returns the confirming run's own frames, but the entrypoint no
longer replays them into ``UtteranceBuffer`` (an earlier version of the
final-review fix did, to "restore" audio after resetting the buffer on a
confirmed barge-in — that reset was itself a bug, since the buffer never
holds genuinely stale audio in the first place; see ``entrypoint.py``'s
``_process_frame`` docstring). Returning the run is kept for callers that
may want it for logging/diagnostics.

Default 240ms (12 × 20ms frames): inside the 200-300ms band chosen for this
fix — longer than a click or a typical short noise burst, while still about
one syllable of real speech. This is a real detection latency, not an
instant one: a genuine interruption is confirmed only after ~240ms of
sustained energy, not on the first loud frame.

Two known false-negative/positive gaps of this energy-only design (see
``livekit_bridge/__init__.py``'s "Known limitations" list for the full
picture, this is intentionally not solved here):
- Real speech with any dip below ``_ENERGY_THRESHOLD`` inside a
  ``min_speech_ms`` window (a stop closure, a brief pause between
  syllables) resets ``_run`` and restarts the count, so a genuine
  interruption can take longer than 240ms to confirm, or in the worst
  case never confirm before the utterance-completing silence gap.
- A sustained noise longer than 240ms (a long cough, music) still passes,
  since an energy-only detector cannot tell it apart from speech — the
  interrupted answer is stopped with nothing to replace it, because the
  noise's own utterance is filtered to ``""`` after ASR.
- ``_fired_this_run`` means a run fires at most once regardless of whether
  the avatar is speaking at that instant — a loud run that confirms while
  the avatar is silent "uses up" that confirmation and does not re-fire
  later just because the avatar starts talking mid-run. In practice this
  is bounded by ``AsrBridge``'s empty-text filtering and by
  ``_drive_avatar_with_transcription``'s unconditional per-utterance
  supersede once the utterance itself completes.
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
