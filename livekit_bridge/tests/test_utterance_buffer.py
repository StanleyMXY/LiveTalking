import numpy as np
from livekit_bridge.utterance_buffer import UtteranceBuffer


def make_chunk(is_silence: bool, sample_count: int = 320) -> np.ndarray:
    if is_silence:
        return np.zeros(sample_count, dtype=np.float32)
    return np.full(sample_count, 0.3, dtype=np.float32)  # 模拟有能量的语音


def test_continuous_speech_does_not_trigger_utterance_end():
    buffer = UtteranceBuffer(silence_threshold_ms=500)
    for _ in range(10):
        result = buffer.feed(make_chunk(is_silence=False))
        assert result is None


def test_silence_after_speech_triggers_utterance_end():
    buffer = UtteranceBuffer(silence_threshold_ms=200)
    buffer.feed(make_chunk(is_silence=False))
    buffer.feed(make_chunk(is_silence=False))

    # 喂够200ms的静音(20ms/块 => 10块)
    result = None
    for _ in range(15):
        result = buffer.feed(make_chunk(is_silence=True))
        if result is not None:
            break

    assert result is not None
    assert len(result) > 0  # 返回累积的语音音频，不含尾部静音


def test_reset_clears_accumulated_audio():
    buffer = UtteranceBuffer(silence_threshold_ms=200)
    buffer.feed(make_chunk(is_silence=False))

    buffer.reset()

    # reset后再喂纯静音，不应该触发"一句话结束"（因为reset前的语音已经被清空，没有内容可以返回）
    result = buffer.feed(make_chunk(is_silence=True))
    assert result is None
