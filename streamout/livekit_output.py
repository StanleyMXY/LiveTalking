###############################################################################
#  Output — LiveKit 房间音频输出
###############################################################################

import asyncio
from typing import TYPE_CHECKING, Optional

import numpy as np
from livekit import rtc
from numpy.typing import NDArray

from streamout.base_output import BaseOutput
from utils.logger import logger

if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar

# avatars/base_avatar.py's render() throttle loop (base_avatar.py:529-532)
# treats get_buffer_size()'s return value as a count of queued *video*
# frames, each RENDER_LOOP_FRAME_SECONDS long: `if buffer_size >= 5:
# time.sleep(0.04 * buffer_size * 0.8)`. server/webrtc.py's own
# get_buffer_size() (server/webrtc.py:202-203) returns exactly that: a
# literal video-frame queue depth (`self.__video._queue.qsize()`).
# LiveKitOutput drops video entirely (push_video_frame is a no-op below),
# so there is no video queue to report — it reports the *audio* backlog
# instead (AudioSource.queued_duration, in seconds — audio_source.py:83-86),
# converted into the same frame-duration units so the shared ">= 5"
# threshold means the same amount of real backlog regardless of which
# output implementation is active.
_RENDER_LOOP_FRAME_SECONDS = 0.04


class LiveKitOutput(BaseOutput):
    """把管线合成的音频推到LiveKit房间的本地音频track；不处理视频。

    视频帧被丢弃(no-op) —— 数字人渲染循环仍会照常运行并消耗对应的GPU算力，
    这是现有渲染循环耦合决定的既定成本，不由本类负责节省。

    Code-review finding, fixed here: this class previously did not override
    ``get_buffer_size()`` at all, so it inherited ``BaseOutput``'s always-``0``
    default (streamout/base_output.py:45-47) — unlike every other existing
    output (webrtc/rtmp/virtualcam), which all implement real throttling
    here. Concretely: ``push_audio_frame`` below fires an unbounded number of
    ``run_coroutine_threadsafe(capture_frame(...))`` calls without ever
    waiting for any of them (see its own docstring) — ``capture_frame``
    itself blocks once ``rtc.AudioSource``'s native 1000ms queue fills
    (audio_source.py:106-109: "If the size of the audio frame exceeds the
    internal queue size, the method will wait"), but that wait happens on
    the *event loop* thread inside the scheduled coroutine, not on the
    render thread calling ``push_audio_frame`` — so nothing was slowing the
    render thread down to match, and pending frames/coroutines could grow
    without bound, with speech ending up queued behind an ever-growing
    silence backlog.
    """

    def __init__(
        self,
        audio_source: "rtc.AudioSource",
        loop: asyncio.AbstractEventLoop,
        opt=None,
        parent: Optional['BaseAvatar'] = None,
        **kwargs,
    ):
        super().__init__(opt, parent)
        self._audio_source = audio_source
        # push_audio_frame() is called synchronously from the pipeline's
        # (non-async) render thread, but livekit.rtc.AudioSource.capture_frame
        # is a coroutine (verified in the installed livekit==1.1.20 package,
        # livekit/rtc/audio_source.py: `async def capture_frame(...)`).
        # We therefore need the event loop that owns the LiveKit room/track
        # so the coroutine can be scheduled from this thread without blocking
        # the caller.
        self._loop = loop

    def start(self) -> None:
        pass

    def push_video_frame(self, frame) -> None:
        # 视频不转发到LiveKit —— 只是不转发画面数据，不代表节省了渲染循环
        # 对应的GPU计算（该耦合是既有渲染循环的既定成本，非本类范畴）。
        pass

    def push_audio_frame(self, frame: NDArray[np.int16], eventpoint=None) -> None:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return

        pcm16 = frame if frame.dtype == np.int16 else frame.astype(np.int16)

        num_channels = self._audio_source.num_channels
        sample_rate = self._audio_source.sample_rate
        samples_per_channel = pcm16.size // num_channels

        audio_frame = rtc.AudioFrame(
            data=pcm16.tobytes(),
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=samples_per_channel,
        )

        # capture_frame() is async; run_coroutine_threadsafe schedules it on
        # the room's event loop and returns immediately, so this call does
        # not block the (synchronous) caller.
        capture_coro = self._audio_source.capture_frame(audio_frame)
        try:
            future = asyncio.run_coroutine_threadsafe(capture_coro, self._loop)
        except RuntimeError as exc:
            # run_coroutine_threadsafe schedules via loop.call_soon_threadsafe(),
            # which calls self._check_closed() and raises
            # RuntimeError('Event loop is closed') SYNCHRONOUSLY, before any
            # scheduling happens (verified in CPython's
            # asyncio.base_events.BaseEventLoop.call_soon_threadsafe /
            # _check_closed). This is a realistic race: push_audio_frame runs on
            # the pipeline's synchronous render thread
            # (avatars/base_avatar.py's process_frames loop), and a LiveKit
            # disconnect/shutdown can close self._loop while that thread is
            # still mid-flight. Drop this frame instead of letting the
            # RuntimeError propagate into and crash the whole render loop.
            capture_coro.close()  # avoid a "coroutine was never awaited" warning
            logger.error(
                f"LiveKitOutput: audio event loop is closed, dropping frame: {exc}"
            )
            return
        future.add_done_callback(self._log_capture_error)

        if self.parent:
            self.parent.notify(eventpoint)

    def get_buffer_size(self) -> int:
        """How backed up the LiveKit-side audio queue is, in the same
        video-frame-duration units every other output's ``get_buffer_size()``
        uses — see the module-level ``_RENDER_LOOP_FRAME_SECONDS`` comment
        for why raw ``queued_duration`` seconds cannot be returned directly.
        """
        return int(self._audio_source.queued_duration / _RENDER_LOOP_FRAME_SECONDS)

    def clear_queue(self) -> None:
        """Discard whatever synthesized audio is still queued in
        ``rtc.AudioSource``'s own internal buffer.

        ``flush_talk()`` (the pipeline's canonical interrupt handler,
        ``avatars/base_avatar.py``) only clears the pipeline's own upstream
        ASR/TTS/render queues — it has no generic hook into a specific
        ``BaseOutput`` implementation's own buffer, and none of the other
        existing outputs need one cleared this way. Called explicitly by
        ``livekit_bridge.entrypoint`` on a detected barge-in (see that
        module's docstring), so stale queued speech does not keep playing
        out of a LiveKit session after an interrupt just because
        ``AudioSource`` itself buffers up to a second ahead
        (audio_source.py:38, ``queue_size_ms: int = 1000``).
        """
        self._audio_source.clear_queue()

    def stop(self) -> None:
        pass

    @staticmethod
    def _log_capture_error(future) -> None:
        exc = future.exception()
        if exc is not None:
            logger.error(f"LiveKitOutput: audio capture_frame failed: {exc}")
