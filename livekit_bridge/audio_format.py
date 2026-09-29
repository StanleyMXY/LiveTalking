import numpy as np


def to_pipeline_format(raw_pcm_bytes: bytes) -> np.ndarray:
    """
    Convert int16 PCM bytes to normalized float32 audio array.

    Converts raw int16 PCM bytes (as received from LiveKit AudioStream)
    to float32 samples in the range [-1.0, 1.0], matching the convention
    used by server/asr_server.py for pipeline input.

    Parameters
    ----------
    raw_pcm_bytes : bytes
        Raw PCM audio data in int16 format (2 bytes per sample).

    Returns
    -------
    np.ndarray
        Float32 audio samples in range [-1.0, 1.0].
        Empty array if input is empty.
    """
    return np.frombuffer(raw_pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
