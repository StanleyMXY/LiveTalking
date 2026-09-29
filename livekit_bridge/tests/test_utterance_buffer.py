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


# ─── max-utterance cap (final-review finding) ──────────────────────────


def test_continuous_speech_is_force_emitted_at_the_cap():
    # Noise that never drops below the threshold must not grow the buffer
    # forever: at the cap the accumulated audio is emitted without waiting
    # for a silence gap.
    buffer = UtteranceBuffer(silence_threshold_ms=500, max_utterance_ms=200)  # 10 frames
    results = [buffer.feed(make_chunk(is_silence=False)) for _ in range(10)]

    assert all(r is None for r in results[:9])
    assert results[9] is not None
    assert len(results[9]) == 10 * 320


def test_buffering_continues_after_a_forced_emit():
    buffer = UtteranceBuffer(silence_threshold_ms=500, max_utterance_ms=200)
    emitted = [
        r for r in (buffer.feed(make_chunk(is_silence=False)) for _ in range(25)) if r is not None
    ]

    assert [len(e) for e in emitted] == [10 * 320, 10 * 320]
    # The 5 frames after the second cap are still buffered and end normally
    # on silence.
    result = None
    for _ in range(25):
        result = buffer.feed(make_chunk(is_silence=True))
        if result is not None:
            break
    assert result is not None and len(result) == 5 * 320


def test_silence_frames_do_not_count_toward_the_cap():
    buffer = UtteranceBuffer(silence_threshold_ms=500, max_utterance_ms=200)
    for _ in range(9):
        assert buffer.feed(make_chunk(is_silence=False)) is None
    for _ in range(20):  # a pause shorter than the 500ms silence threshold
        assert buffer.feed(make_chunk(is_silence=True)) is None

    result = buffer.feed(make_chunk(is_silence=False))
    assert result is not None and len(result) == 10 * 320


def test_default_cap_is_30_seconds():
    buffer = UtteranceBuffer()
    speech = make_chunk(is_silence=False)
    for _ in range(1499):
        assert buffer.feed(speech) is None
    result = buffer.feed(speech)
    assert result is not None and len(result) == 1500 * 320  # 30s at 16kHz
