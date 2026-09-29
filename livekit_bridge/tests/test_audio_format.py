import numpy as np
from livekit_bridge.audio_format import to_pipeline_format


def test_converts_int16_to_normalized_float32():
    raw_pcm = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16).tobytes()

    result = to_pipeline_format(raw_pcm)

    assert result.dtype == np.float32
    assert len(result) == 5
    np.testing.assert_allclose(result[0], 0.0, atol=1e-4)
    np.testing.assert_allclose(result[1], 0.5, atol=1e-4)
    np.testing.assert_allclose(result[3], 1.0, atol=1e-4)


def test_empty_input_returns_empty_array():
    result = to_pipeline_format(b"")

    assert len(result) == 0
    assert result.dtype == np.float32
