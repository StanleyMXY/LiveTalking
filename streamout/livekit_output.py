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


class LiveKitOutput(BaseOutput):
    """把管线合成的音频推到LiveKit房间的本地音频track；不处理视频。

    视频帧被丢弃(no-op) —— 数字人渲染循环仍会照常运行并消耗对应的GPU算力，
    这是现有渲染循环耦合决定的既定成本，不由本类负责节省。
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

    def stop(self) -> None:
        pass

    @staticmethod
    def _log_capture_error(future) -> None:
        exc = future.exception()
        if exc is not None:
            logger.error(f"LiveKitOutput: audio capture_frame failed: {exc}")
