import numpy as np

from livekit_bridge.barge_in_detector import BargeInDetector


def _speech():
    return np.full(320, 0.3, dtype=np.float32)


def _silence():
    return np.zeros(320, dtype=np.float32)


def test_single_frame_does_not_confirm():
    detector = BargeInDetector()
    assert detector.feed(_speech()) is None
    assert detector.feed(_silence()) is None


def test_default_needs_12_consecutive_frames_and_returns_the_run():
    detector = BargeInDetector()
    frames = [np.full(320, 0.1 + i * 0.01, dtype=np.float32) for i in range(12)]
    results = [detector.feed(f) for f in frames]

    assert all(r is None for r in results[:11])
    assert results[11] is not None
    assert len(results[11]) == 12
    for returned, fed in zip(results[11], frames):
        assert returned is fed


def test_a_below_threshold_frame_restarts_the_count():
    detector = BargeInDetector(min_speech_ms=100)  # 5 frames
    for _ in range(4):
        assert detector.feed(_speech()) is None
    assert detector.feed(_silence()) is None
    for _ in range(4):
        assert detector.feed(_speech()) is None
    assert detector.feed(_speech()) is not None


def test_fires_once_per_run_then_again_after_silence():
    detector = BargeInDetector(min_speech_ms=100)
    fired = [detector.feed(_speech()) is not None for _ in range(20)]
    assert fired.count(True) == 1

    detector.feed(_silence())
    fired = [detector.feed(_speech()) is not None for _ in range(5)]
    assert fired == [False, False, False, False, True]
