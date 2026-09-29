import numpy as np

_FRAME_MS = 20
_ENERGY_THRESHOLD = 0.05


class UtteranceBuffer:
    def __init__(self, silence_threshold_ms: int = 500):
        self._silence_threshold_frames = silence_threshold_ms // _FRAME_MS
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
            return None

        if not self._has_speech:
            return None

        self._silence_frame_count += 1
        if self._silence_frame_count < self._silence_threshold_frames:
            return None

        result = np.concatenate(self._speech_chunks) if self._speech_chunks else None
        self.reset()
        return result

    def reset(self) -> None:
        self._speech_chunks = []
        self._silence_frame_count = 0
        self._has_speech = False
