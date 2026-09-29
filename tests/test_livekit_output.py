import asyncio

import numpy as np
import pytest

from streamout.livekit_output import LiveKitOutput


class FakeAudioSource:
    """Mimics the parts of the real livekit.rtc.AudioSource contract that
    LiveKitOutput relies on: sample_rate/num_channels attributes and an
    *async* capture_frame(frame) method.

    Verified against the installed livekit==1.1.20 package
    (.venv/Lib/site-packages/livekit/rtc/audio_source.py):
    `AudioSource.capture_frame` is declared `async def`, not sync, and takes
    an `rtc.AudioFrame` object (not a raw numpy array).
    """

    def __init__(self, sample_rate=16000, num_channels=1):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.captured_frames = []

    async def capture_frame(self, frame):
        self.captured_frames.append(frame)


@pytest.mark.asyncio
async def test_push_video_frame_is_noop():
    fake_source = FakeAudioSource()
    output = LiveKitOutput(audio_source=fake_source, loop=asyncio.get_running_loop())

    # push_video_frame's real BaseOutput signature takes only `frame`
    # (no userdata/eventpoint argument) — confirmed in streamout/base_output.py.
    # Should not raise, and should have no observable side effect.
    output.push_video_frame(np.zeros((256, 256, 3), dtype=np.uint8))

    assert len(fake_source.captured_frames) == 0


async def _wait_until(predicate, timeout=1.0):
    """Poll `predicate` until it is truthy, or raise after `timeout` seconds.

    Used instead of a fixed number of `asyncio.sleep(0)` yields: the exact
    number of loop iterations asyncio.run_coroutine_threadsafe needs before
    the scheduled coroutine actually runs is an implementation detail, so a
    magic-number yield count would be a flaky assumption baked into the test.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_push_audio_frame_forwards_to_audio_source():
    fake_source = FakeAudioSource()
    output = LiveKitOutput(audio_source=fake_source, loop=asyncio.get_running_loop())

    audio_chunk = np.zeros(320, dtype=np.int16)
    output.push_audio_frame(audio_chunk, eventpoint=None)

    # push_audio_frame is synchronous (matches BaseOutput's abstract signature)
    # but the real capture_frame is a coroutine, so LiveKitOutput must schedule
    # it onto the event loop via asyncio.run_coroutine_threadsafe without
    # blocking the caller — it does not await the result itself.
    await _wait_until(lambda: len(fake_source.captured_frames) == 1)

    captured = fake_source.captured_frames[0]
    assert captured.sample_rate == 16000
    assert captured.num_channels == 1
    assert captured.samples_per_channel == 320


def test_push_audio_frame_survives_closed_event_loop():
    """asyncio.run_coroutine_threadsafe() raises RuntimeError('Event loop is
    closed') SYNCHRONOUSLY (from loop.call_soon_threadsafe -> _check_closed)
    when the target loop is already closed — this can happen for real if a
    LiveKit disconnect/shutdown closes the loop while the pipeline's
    synchronous render thread is still calling push_audio_frame. That
    RuntimeError must be caught and logged, not propagated into the caller
    (which would otherwise crash the whole render loop, not just this
    output). No mocking needed: a real closed event loop reproduces the
    exact real failure.
    """
    fake_source = FakeAudioSource()
    closed_loop = asyncio.new_event_loop()
    closed_loop.close()

    output = LiveKitOutput(audio_source=fake_source, loop=closed_loop)

    audio_chunk = np.zeros(320, dtype=np.int16)
    # Must not raise.
    output.push_audio_frame(audio_chunk, eventpoint=None)

    # The frame is dropped, not queued or retried.
    assert fake_source.captured_frames == []
