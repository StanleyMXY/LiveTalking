"""
Tests for livekit_bridge.entrypoint.

Task 8 is an integration task wiring together Tasks 2-7 plus a real,
external LiveKit connection — there is no local LiveKit dev server or test
client available in this environment (see the Task 8 report), so nothing
here talks to a real `rtc.Room`/`JobContext`. What *is* genuinely
unit-testable without live infrastructure, and is covered below:

- the small helper functions entrypoint() is built from, each verified
  against the real installed livekit/livekit-agents API it wraps (see
  entrypoint.py's module docstring for citations) — with everything from
  the real SDK that would require a live connection replaced by a minimal
  fake/mock;
- the interrupt-detection gating logic in `_process_frame`, including the
  case the brief's "unconditional" suggestion would have gotten wrong
  (a mid-utterance energy dip while the avatar is NOT speaking), and that a
  detected barge-in also clears LiveKitOutput's own queued audio (code
  review C2);
- that the two documented-blocking calls (`asr_bridge.transcribe`,
  `pipeline_bridge.drive_with_text`) are actually executed off the calling
  thread via `run_in_executor`, not merely moved around, and that they are
  scheduled as a background task rather than awaited inline so the frame
  loop keeps draining while the LLM streams (code review I1);
- that `entrypoint()`'s `finally` block runs (and calls
  `session_manager.remove_session`, releases the room registry, and calls
  `ctx.shutdown()`) on every exit path: normal stream completion, an
  exception propagating out of the frame loop, task cancellation, the "no
  audio track ever arrived" timeout path, and even when `aclose()` itself
  raises (code review I2/I3); and that the duplicate-dispatch early return
  also calls `ctx.shutdown()` (code review I2);
- that at most one *tracked* `drive_with_text`-driving task is ever left
  in `pending_tasks` per session: a new utterance completing while a
  previous one's response is still being generated supersedes it (cancels
  the stale task, calls `pipeline_bridge.interrupt()`) rather than letting
  both run concurrently (code review follow-up finding to I1). What this
  file does *not* prove on its own — because it genuinely cannot, without
  reaching into `PipelineBridge`'s own generation-fencing — is that an
  already-executing `llm_response` call (the case where cancelling the
  asyncio task has no effect, since that call is running in its own
  executor thread with no cancellation checkpoint) is actually stopped
  from reaching the real session; that guarantee is tested directly in
  `test_pipeline_bridge.py` instead (see its
  `test_superseded_call_cannot_reach_real_session_after_new_drive_call`).
"""

import asyncio
import contextlib
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from livekit import rtc

from livekit_bridge import entrypoint as ep
from livekit_bridge.session_registry import BridgeSessionRegistry
from streamout.livekit_output import LiveKitOutput


def _make_chunk(is_speech: bool, n: int = 320) -> np.ndarray:
    if is_speech:
        return np.full(n, 0.3, dtype=np.float32)
    return np.zeros(n, dtype=np.float32)


def _fake_frame_event(is_speech: bool = False, n: int = 320):
    amplitude = 16000 if is_speech else 0
    data = np.full(n, amplitude, dtype=np.int16).tobytes()
    return SimpleNamespace(frame=SimpleNamespace(data=data))


# ─── _room_name_from_job ────────────────────────────────────────────────


def test_room_name_from_job_reads_pre_connect_safe_field():
    # JobContext.room's own docstring says the room is not connected yet
    # when the entrypoint starts; ctx.job.room.name is the documented,
    # dispatch-time value this function relies on instead (see
    # entrypoint.py's module docstring for the full citation, including a
    # correction about ctx.room.name's actual pre-connect state).
    ctx = MagicMock()
    ctx.job.room.name = "expected-room"

    assert ep._room_name_from_job(ctx) == "expected-room"


# ─── _register_first_audio_track_waiter ─────────────────────────────────


class _FakeRoom:
    def __init__(self):
        self._handlers = {}

    def on(self, event, callback):
        self._handlers[event] = callback

    def emit(self, event, *args):
        self._handlers[event](*args)


@pytest.mark.asyncio
async def test_track_waiter_ignores_video_and_resolves_on_first_audio_track():
    room = _FakeRoom()
    future = ep._register_first_audio_track_waiter(room)

    fake_video_track = MagicMock(spec=rtc.RemoteVideoTrack)
    room.emit("track_subscribed", fake_video_track, MagicMock(), MagicMock())
    assert not future.done()

    fake_audio_track = MagicMock(spec=rtc.RemoteAudioTrack)
    room.emit("track_subscribed", fake_audio_track, MagicMock(), MagicMock())

    assert future.done()
    assert future.result() is fake_audio_track


@pytest.mark.asyncio
async def test_track_waiter_resolves_only_once():
    room = _FakeRoom()
    future = ep._register_first_audio_track_waiter(room)

    first = MagicMock(spec=rtc.RemoteAudioTrack)
    second = MagicMock(spec=rtc.RemoteAudioTrack)
    room.emit("track_subscribed", first, MagicMock(), MagicMock())
    room.emit("track_subscribed", second, MagicMock(), MagicMock())  # must not raise

    assert future.result() is first


# ─── _make_audio_stream ──────────────────────────────────────────────────


def test_make_audio_stream_uses_verified_sample_rate_and_frame_size():
    fake_track = MagicMock()
    with patch.object(rtc, "AudioStream") as fake_audio_stream_cls:
        ep._make_audio_stream(fake_track)

        # sample_rate/num_channels: Task 1's verified fact. frame_size_ms=20:
        # this task's own addition, matching UtteranceBuffer's _FRAME_MS
        # assumption (see entrypoint.py module docstring).
        fake_audio_stream_cls.assert_called_once_with(
            fake_track, sample_rate=16000, num_channels=1, frame_size_ms=20
        )


# ─── _publish_livekit_audio_track ────────────────────────────────────────


@pytest.mark.asyncio
async def test_publish_livekit_audio_track_wires_source_track_and_options():
    fake_source = MagicMock()
    fake_track = MagicMock()
    ctx = MagicMock()
    ctx.room.local_participant.publish_track = AsyncMock()

    with (
        patch.object(rtc, "AudioSource", return_value=fake_source) as fake_source_cls,
        patch.object(
            rtc.LocalAudioTrack, "create_audio_track", return_value=fake_track
        ) as fake_create_track,
    ):
        result = await ep._publish_livekit_audio_track(ctx)

    fake_source_cls.assert_called_once_with(16000, 1)
    fake_create_track.assert_called_once_with("avatar-voice", fake_source)
    ctx.room.local_participant.publish_track.assert_awaited_once()
    published_track_arg, options_arg = ctx.room.local_participant.publish_track.call_args.args
    assert published_track_arg is fake_track
    # SOURCE_MICROPHONE: the same convention livekit-agents' own built-in
    # avatar-worker output uses for a synthesized voice track
    # (livekit/agents/voice/avatar/_runner.py:109).
    assert options_arg.source == rtc.TrackSource.SOURCE_MICROPHONE
    assert result is fake_source


# ─── _create_avatar_session ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_avatar_session_wires_output_and_starts_render_thread():
    fake_avatar_session = MagicMock()
    fake_audio_source = MagicMock()
    loop = asyncio.get_running_loop()

    with (
        patch.object(ep.session_manager, "create_session", new=AsyncMock()) as fake_create,
        patch.object(ep.session_manager, "get_session", return_value=fake_avatar_session),
        patch.object(ep.threading, "Thread") as fake_thread_cls,
    ):
        fake_thread = MagicMock()
        fake_thread_cls.return_value = fake_thread

        result = await ep._create_avatar_session("room-1", fake_audio_source, loop)

    fake_create.assert_awaited_once_with({}, sessionid="room-1")
    assert result is fake_avatar_session
    # quit_event must be assigned before the thread starts, not left for
    # render() to assign on its own first line (see entrypoint.py's
    # docstring on this function for why).
    assert isinstance(fake_avatar_session.quit_event, threading.Event)
    assert isinstance(fake_avatar_session.output, LiveKitOutput)
    # parent=avatar_session (code review minor finding): without it,
    # LiveKitOutput.push_audio_frame's self.parent.notify(eventpoint) is
    # silently dropped, unlike every other existing output.
    assert fake_avatar_session.output.parent is fake_avatar_session

    fake_thread_cls.assert_called_once_with(
        target=fake_avatar_session.render,
        args=(fake_avatar_session.quit_event,),
        daemon=True,
    )
    fake_thread.start.assert_called_once()


# ─── _process_frame: interrupt-detection gating ─────────────────────────


@pytest.mark.asyncio
async def test_process_frame_no_interrupt_when_avatar_not_speaking():
    # A silence->speech transition while the avatar is NOT talking must not
    # fire interrupt()/reset() — this is the exact case the brief's
    # "unconditional" fallback would have mishandled if it had also
    # unconditionally reset the buffer (see entrypoint.py module docstring's
    # "Interrupt-handling design" section).
    avatar_session = MagicMock()
    avatar_session.is_speaking.return_value = False
    pipeline_bridge = MagicMock()
    utterance_buffer = MagicMock()
    utterance_buffer.feed.return_value = None
    asr_bridge = MagicMock()
    loop = asyncio.get_running_loop()

    was_speech = await ep._process_frame(
        _make_chunk(is_speech=True),
        was_speech=False,
        utterance_buffer=utterance_buffer,
        asr_bridge=asr_bridge,
        pipeline_bridge=pipeline_bridge,
        avatar_session=avatar_session,
        loop=loop,
        pending_tasks=set(),
    )

    assert was_speech is True
    pipeline_bridge.interrupt.assert_not_called()
    utterance_buffer.reset.assert_not_called()
    utterance_buffer.feed.assert_called_once()


@pytest.mark.asyncio
async def test_process_frame_interrupts_on_barge_in_while_avatar_speaking():
    avatar_session = MagicMock()
    avatar_session.is_speaking.return_value = True
    pipeline_bridge = MagicMock()
    utterance_buffer = MagicMock()
    utterance_buffer.feed.return_value = None
    asr_bridge = MagicMock()
    loop = asyncio.get_running_loop()

    was_speech = await ep._process_frame(
        _make_chunk(is_speech=True),
        was_speech=False,
        utterance_buffer=utterance_buffer,
        asr_bridge=asr_bridge,
        pipeline_bridge=pipeline_bridge,
        avatar_session=avatar_session,
        loop=loop,
        pending_tasks=set(),
    )

    assert was_speech is True
    pipeline_bridge.interrupt.assert_called_once_with(avatar_session)
    utterance_buffer.reset.assert_called_once()


@pytest.mark.asyncio
async def test_process_frame_clears_output_queue_on_barge_in():
    # Code review C2: flush_talk() cannot reach LiveKit's own queued audio.
    # A detected barge-in must also call avatar_session.output.clear_queue()
    # when the current output exposes one.
    avatar_session = MagicMock()
    avatar_session.is_speaking.return_value = True
    avatar_session.output = MagicMock(spec=LiveKitOutput)
    pipeline_bridge = MagicMock()
    utterance_buffer = MagicMock()
    utterance_buffer.feed.return_value = None
    asr_bridge = MagicMock()
    loop = asyncio.get_running_loop()

    await ep._process_frame(
        _make_chunk(is_speech=True),
        was_speech=False,
        utterance_buffer=utterance_buffer,
        asr_bridge=asr_bridge,
        pipeline_bridge=pipeline_bridge,
        avatar_session=avatar_session,
        loop=loop,
        pending_tasks=set(),
    )

    avatar_session.output.clear_queue.assert_called_once()


@pytest.mark.asyncio
async def test_process_frame_barge_in_survives_output_without_clear_queue():
    # Other outputs (webrtc/rtmp/virtualcam) do not implement clear_queue()
    # at all — the hasattr-style guard must not raise for them.
    avatar_session = MagicMock()
    avatar_session.is_speaking.return_value = True
    avatar_session.output = object()  # deliberately has no clear_queue
    pipeline_bridge = MagicMock()
    utterance_buffer = MagicMock()
    utterance_buffer.feed.return_value = None
    asr_bridge = MagicMock()
    loop = asyncio.get_running_loop()

    # Must not raise.
    await ep._process_frame(
        _make_chunk(is_speech=True),
        was_speech=False,
        utterance_buffer=utterance_buffer,
        asr_bridge=asr_bridge,
        pipeline_bridge=pipeline_bridge,
        avatar_session=avatar_session,
        loop=loop,
        pending_tasks=set(),
    )

    pipeline_bridge.interrupt.assert_called_once_with(avatar_session)


@pytest.mark.asyncio
async def test_process_frame_no_interrupt_without_silence_to_speech_transition():
    # Already speaking on the previous chunk -> no rising edge -> no
    # interrupt, even though the avatar is "speaking". Guards against a
    # naive per-chunk (not edge-triggered) implementation.
    avatar_session = MagicMock()
    avatar_session.is_speaking.return_value = True
    pipeline_bridge = MagicMock()
    utterance_buffer = MagicMock()
    utterance_buffer.feed.return_value = None
    asr_bridge = MagicMock()
    loop = asyncio.get_running_loop()

    await ep._process_frame(
        _make_chunk(is_speech=True),
        was_speech=True,
        utterance_buffer=utterance_buffer,
        asr_bridge=asr_bridge,
        pipeline_bridge=pipeline_bridge,
        avatar_session=avatar_session,
        loop=loop,
        pending_tasks=set(),
    )

    pipeline_bridge.interrupt.assert_not_called()
    utterance_buffer.reset.assert_not_called()


@pytest.mark.asyncio
async def test_process_frame_empty_chunk_is_noop():
    avatar_session = MagicMock()
    utterance_buffer = MagicMock()
    pipeline_bridge = MagicMock()
    asr_bridge = MagicMock()
    loop = asyncio.get_running_loop()

    was_speech = await ep._process_frame(
        np.zeros(0, dtype=np.float32),
        was_speech=False,
        utterance_buffer=utterance_buffer,
        asr_bridge=asr_bridge,
        pipeline_bridge=pipeline_bridge,
        avatar_session=avatar_session,
        loop=loop,
        pending_tasks=set(),
    )

    assert was_speech is False
    utterance_buffer.feed.assert_not_called()
    avatar_session.is_speaking.assert_not_called()


# ─── _process_frame: executor offloading + background scheduling ───────


@pytest.mark.asyncio
async def test_process_frame_offloads_blocking_calls_to_executor_not_event_loop():
    # Both AsrBridge.transcribe and PipelineBridge.drive_with_text are
    # documented as deliberately synchronous/blocking; this proves they
    # actually run off the event-loop thread (via run_in_executor), not just
    # that the code "looks" offloaded.
    main_thread_id = threading.get_ident()
    call_threads = {}

    def fake_transcribe(audio):
        call_threads["transcribe"] = threading.get_ident()
        return "hello"

    def fake_drive_with_text(avatar_session, text):
        call_threads["drive_with_text"] = threading.get_ident()

    avatar_session = MagicMock()
    avatar_session.is_speaking.return_value = False
    asr_bridge = MagicMock()
    asr_bridge.transcribe.side_effect = fake_transcribe
    pipeline_bridge = MagicMock()
    pipeline_bridge.drive_with_text.side_effect = fake_drive_with_text
    utterance_buffer = MagicMock()
    utterance_buffer.feed.return_value = np.zeros(320, dtype=np.float32)
    loop = asyncio.get_running_loop()
    pending_tasks: set = set()

    await ep._process_frame(
        _make_chunk(is_speech=False),
        was_speech=False,
        utterance_buffer=utterance_buffer,
        asr_bridge=asr_bridge,
        pipeline_bridge=pipeline_bridge,
        avatar_session=avatar_session,
        loop=loop,
        pending_tasks=pending_tasks,
    )

    # The drive-with-transcription sequence is now a scheduled background
    # task (code review I1), not awaited inline — give it a moment to run.
    assert len(pending_tasks) == 1
    await asyncio.gather(*pending_tasks)

    assert call_threads.get("transcribe") not in (None, main_thread_id)
    assert call_threads.get("drive_with_text") not in (None, main_thread_id)
    pipeline_bridge.drive_with_text.assert_called_once_with(avatar_session, "hello")


@pytest.mark.asyncio
async def test_process_frame_skips_asr_when_no_utterance_completed():
    avatar_session = MagicMock()
    avatar_session.is_speaking.return_value = False
    asr_bridge = MagicMock()
    pipeline_bridge = MagicMock()
    utterance_buffer = MagicMock()
    utterance_buffer.feed.return_value = None
    loop = asyncio.get_running_loop()
    pending_tasks: set = set()

    await ep._process_frame(
        _make_chunk(is_speech=True),
        was_speech=False,
        utterance_buffer=utterance_buffer,
        asr_bridge=asr_bridge,
        pipeline_bridge=pipeline_bridge,
        avatar_session=avatar_session,
        loop=loop,
        pending_tasks=pending_tasks,
    )

    assert pending_tasks == set()
    asr_bridge.transcribe.assert_not_called()
    pipeline_bridge.drive_with_text.assert_not_called()


@pytest.mark.asyncio
async def test_process_frame_does_not_block_caller_on_a_slow_drive_call():
    # Code review I1's core claim: _process_frame must return promptly even
    # if the transcribe/drive sequence is slow, so entrypoint()'s frame loop
    # keeps draining (and keeps detecting interrupts) while it runs. A real
    # (short) blocking sleep inside the fake transcribe, run for real in the
    # executor thread pool, is the most direct way to prove this without
    # mocking time itself.
    import time

    def slow_transcribe(audio):
        time.sleep(0.3)
        return "text"

    avatar_session = MagicMock()
    avatar_session.is_speaking.return_value = False
    asr_bridge = MagicMock()
    asr_bridge.transcribe.side_effect = slow_transcribe
    pipeline_bridge = MagicMock()
    utterance_buffer = MagicMock()
    utterance_buffer.feed.return_value = np.zeros(320, dtype=np.float32)
    loop = asyncio.get_running_loop()
    pending_tasks: set = set()

    call_started_at = loop.time()
    await ep._process_frame(
        _make_chunk(is_speech=False),
        was_speech=False,
        utterance_buffer=utterance_buffer,
        asr_bridge=asr_bridge,
        pipeline_bridge=pipeline_bridge,
        avatar_session=avatar_session,
        loop=loop,
        pending_tasks=pending_tasks,
    )
    elapsed = loop.time() - call_started_at

    # _process_frame itself must return almost immediately: the transcribe
    # call (which sleeps for 0.3s) runs in a background task, not awaited
    # here — this call returning well under that proves it.
    assert elapsed < 0.15
    assert len(pending_tasks) == 1

    await asyncio.gather(*pending_tasks)


@pytest.mark.asyncio
async def test_process_frame_supersedes_stale_drive_task_on_new_utterance():
    # Code review follow-up to I1: scheduling every completed utterance as
    # an independent background task let two drive_with_text calls run
    # concurrently with no ordering, interleaving their TTS output. A new
    # utterance completing while a previous one's response is still being
    # generated must supersede it (cancel the stale task + interrupt the
    # avatar + clear the output queue), not queue behind it — see
    # entrypoint.py's module docstring's "at most one *tracked* drive task
    # per session" section for why supersede was chosen over queueing.
    #
    # This test only exercises the case where the stale task is still
    # awaiting transcribe() (stage 1) when superseded — cancellation alone
    # is sufficient there, since the coroutine never resumes to reach
    # drive_with_text. It deliberately does NOT prove the harder case (a
    # stale task already executing llm_response in its own executor
    # thread, where cancelling the asyncio task has no effect) is handled
    # — that guarantee comes from PipelineBridge's own generation fencing,
    # not from anything in this module, and is proven directly in
    # test_pipeline_bridge.py's
    # test_superseded_call_cannot_reach_real_session_after_new_drive_call.
    avatar_session = MagicMock()
    avatar_session.is_speaking.return_value = False
    avatar_session.output = MagicMock(spec=LiveKitOutput)
    pipeline_bridge = MagicMock()
    utterance_buffer = MagicMock()
    utterance_buffer.feed.return_value = np.zeros(320, dtype=np.float32)
    loop = asyncio.get_running_loop()
    pending_tasks: set = set()

    # The first utterance's transcribe() blocks (in a real executor thread)
    # until the test releases it, simulating a response that is still being
    # generated when the second utterance completes.
    first_call_release = threading.Event()
    call_count = {"n": 0}

    def blocking_transcribe(audio):
        call_count["n"] += 1
        if call_count["n"] == 1:
            first_call_release.wait(timeout=5)
            return "first"
        return "second"

    asr_bridge = MagicMock()
    asr_bridge.transcribe.side_effect = blocking_transcribe

    await ep._process_frame(
        _make_chunk(is_speech=False),
        was_speech=False,
        utterance_buffer=utterance_buffer,
        asr_bridge=asr_bridge,
        pipeline_bridge=pipeline_bridge,
        avatar_session=avatar_session,
        loop=loop,
        pending_tasks=pending_tasks,
    )
    await asyncio.sleep(0.05)  # let the task actually start running transcribe
    assert len(pending_tasks) == 1
    first_task = next(iter(pending_tasks))
    assert not first_task.done()

    # Second utterance completes while the first is still in flight.
    await ep._process_frame(
        _make_chunk(is_speech=False),
        was_speech=False,
        utterance_buffer=utterance_buffer,
        asr_bridge=asr_bridge,
        pipeline_bridge=pipeline_bridge,
        avatar_session=avatar_session,
        loop=loop,
        pending_tasks=pending_tasks,
    )

    # The stale task must have had cancellation requested, and the avatar
    # told to interrupt (flush_talk + clear_queue), before the second task
    # started — proven by cancelling()/pending state rather than waiting
    # for the blocked executor thread to actually unwind.
    assert first_task.cancelling() > 0
    pipeline_bridge.interrupt.assert_called_once_with(avatar_session)
    avatar_session.output.clear_queue.assert_called_once()
    assert len(pending_tasks) == 1
    second_task = next(iter(pending_tasks))
    assert second_task is not first_task

    first_call_release.set()  # release the blocked executor thread
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.gather(first_task, return_exceptions=True)
    assert first_task.cancelled()
    # The cancelled first task's transcribe result ("first") must never
    # have reached drive_with_text.
    pipeline_bridge.drive_with_text.assert_not_called()

    await asyncio.gather(second_task)
    pipeline_bridge.drive_with_text.assert_called_once_with(avatar_session, "second")


# ─── entrypoint(): dedup + cleanup on every exit path ───────────────────


class _FakeAudioStream:
    def __init__(self, frame_events):
        self._frame_events = list(frame_events)
        self.aclose = AsyncMock()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._frame_events:
            raise StopAsyncIteration
        return self._frame_events.pop(0)


class _FailingAudioStream(_FakeAudioStream):
    async def __anext__(self):
        raise RuntimeError("simulated LiveKit disconnect")


def _make_ctx(room_name: str):
    ctx = MagicMock()
    ctx.job.room.name = room_name
    ctx.job.metadata = ""
    ctx.connect = AsyncMock()
    return ctx


@pytest.mark.asyncio
async def test_entrypoint_skips_duplicate_dispatch_for_same_room():
    registry = BridgeSessionRegistry()
    registry.acquire("room-dup")  # simulate an in-flight job for this room

    with patch.object(ep, "_registry", registry):
        ctx = _make_ctx("room-dup")
        await ep.entrypoint(ctx)
        ctx.connect.assert_not_called()
        # Code review I2: the room is never connected on this path, so
        # nothing else will ever end the job on its own — entrypoint()
        # must tell the framework explicitly.
        ctx.shutdown.assert_called_once()

    registry.release("room-dup")


@pytest.mark.asyncio
async def test_entrypoint_cleans_up_on_normal_stream_completion():
    registry = BridgeSessionRegistry()
    fake_avatar_session = MagicMock()
    fake_avatar_session.is_speaking.return_value = False
    fake_track = MagicMock()
    fake_stream = _FakeAudioStream([_fake_frame_event(), _fake_frame_event()])

    resolved_future = asyncio.get_running_loop().create_future()
    resolved_future.set_result(fake_track)

    ctx = _make_ctx("room-1")

    with (
        patch.object(ep, "_registry", registry),
        patch.object(ep, "_register_first_audio_track_waiter", return_value=resolved_future),
        patch.object(ep, "_publish_livekit_audio_track", new=AsyncMock(return_value=MagicMock())),
        patch.object(
            ep, "_create_avatar_session", new=AsyncMock(return_value=fake_avatar_session)
        ),
        patch.object(ep, "_make_audio_stream", return_value=fake_stream),
        patch.object(ep.session_manager, "remove_session") as fake_remove,
    ):
        await ep.entrypoint(ctx)

    ctx.connect.assert_awaited_once()
    fake_stream.aclose.assert_awaited_once()
    fake_remove.assert_called_once_with("room-1")
    ctx.shutdown.assert_called_once()
    # Released, not left held:
    assert registry.acquire("room-1") is True
    registry.release("room-1")


@pytest.mark.asyncio
async def test_entrypoint_cleans_up_when_exception_raised_mid_loop():
    registry = BridgeSessionRegistry()
    fake_avatar_session = MagicMock()
    fake_avatar_session.is_speaking.return_value = False
    fake_track = MagicMock()
    fake_stream = _FailingAudioStream([_fake_frame_event()])

    resolved_future = asyncio.get_running_loop().create_future()
    resolved_future.set_result(fake_track)

    ctx = _make_ctx("room-2")

    with (
        patch.object(ep, "_registry", registry),
        patch.object(ep, "_register_first_audio_track_waiter", return_value=resolved_future),
        patch.object(ep, "_publish_livekit_audio_track", new=AsyncMock(return_value=MagicMock())),
        patch.object(
            ep, "_create_avatar_session", new=AsyncMock(return_value=fake_avatar_session)
        ),
        patch.object(ep, "_make_audio_stream", return_value=fake_stream),
        patch.object(ep.session_manager, "remove_session") as fake_remove,
    ):
        with pytest.raises(RuntimeError, match="simulated LiveKit disconnect"):
            await ep.entrypoint(ctx)

    # finally must still run on the exception path: stream closed, session
    # removed, room released, job told to shut down — this is the "process
    # crash/disconnect resource leak" Review Focus item.
    fake_stream.aclose.assert_awaited_once()
    fake_remove.assert_called_once_with("room-2")
    ctx.shutdown.assert_called_once()
    assert registry.acquire("room-2") is True
    registry.release("room-2")


@pytest.mark.asyncio
async def test_entrypoint_cleans_up_when_aclose_itself_raises():
    # Code review I3: an earlier version awaited audio_stream.aclose()
    # directly in `finally`, ahead of the session/registry cleanup — if
    # aclose() raised, everything after it in that finally block was
    # skipped, leaking the GPU render thread and permanently sticking the
    # room to this worker. aclose() failing must not prevent the rest of
    # cleanup from running.
    registry = BridgeSessionRegistry()
    fake_avatar_session = MagicMock()
    fake_avatar_session.is_speaking.return_value = False
    fake_track = MagicMock()
    fake_stream = _FakeAudioStream([_fake_frame_event()])
    fake_stream.aclose = AsyncMock(side_effect=RuntimeError("aclose blew up"))

    resolved_future = asyncio.get_running_loop().create_future()
    resolved_future.set_result(fake_track)

    ctx = _make_ctx("room-5")

    with (
        patch.object(ep, "_registry", registry),
        patch.object(ep, "_register_first_audio_track_waiter", return_value=resolved_future),
        patch.object(ep, "_publish_livekit_audio_track", new=AsyncMock(return_value=MagicMock())),
        patch.object(
            ep, "_create_avatar_session", new=AsyncMock(return_value=fake_avatar_session)
        ),
        patch.object(ep, "_make_audio_stream", return_value=fake_stream),
        patch.object(ep.session_manager, "remove_session") as fake_remove,
    ):
        # Must not raise/propagate the aclose() failure.
        await ep.entrypoint(ctx)

    fake_stream.aclose.assert_awaited_once()
    fake_remove.assert_called_once_with("room-5")
    ctx.shutdown.assert_called_once()
    assert registry.acquire("room-5") is True
    registry.release("room-5")


class _HangingAudioStream(_FakeAudioStream):
    async def __anext__(self):
        await asyncio.sleep(3600)  # never completes on its own; must be cancelled


@pytest.mark.asyncio
async def test_entrypoint_cleans_up_when_task_is_cancelled():
    registry = BridgeSessionRegistry()
    fake_avatar_session = MagicMock()
    fake_avatar_session.is_speaking.return_value = False
    fake_track = MagicMock()
    fake_stream = _HangingAudioStream([_fake_frame_event()])

    resolved_future = asyncio.get_running_loop().create_future()
    resolved_future.set_result(fake_track)

    ctx = _make_ctx("room-4")

    with (
        patch.object(ep, "_registry", registry),
        patch.object(ep, "_register_first_audio_track_waiter", return_value=resolved_future),
        patch.object(ep, "_publish_livekit_audio_track", new=AsyncMock(return_value=MagicMock())),
        patch.object(
            ep, "_create_avatar_session", new=AsyncMock(return_value=fake_avatar_session)
        ),
        patch.object(ep, "_make_audio_stream", return_value=fake_stream),
        patch.object(ep.session_manager, "remove_session") as fake_remove,
    ):
        task = asyncio.create_task(ep.entrypoint(ctx))
        await asyncio.sleep(0.05)  # let it reach the hanging await inside the frame loop
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    # A plain try/finally runs during cancellation propagation too — this
    # is the third exit path the self-review calls out alongside normal
    # completion and an ordinary exception.
    fake_stream.aclose.assert_awaited_once()
    fake_remove.assert_called_once_with("room-4")
    ctx.shutdown.assert_called_once()
    assert registry.acquire("room-4") is True
    registry.release("room-4")


@pytest.mark.asyncio
async def test_entrypoint_cleans_up_when_no_audio_track_ever_arrives():
    registry = BridgeSessionRegistry()
    fake_avatar_session = MagicMock()
    fake_avatar_session.is_speaking.return_value = False
    never_resolved_future = asyncio.get_running_loop().create_future()

    ctx = _make_ctx("room-3")

    with (
        patch.object(ep, "_registry", registry),
        patch.object(ep, "_TRACK_WAIT_TIMEOUT_S", 0.01),
        patch.object(
            ep, "_register_first_audio_track_waiter", return_value=never_resolved_future
        ),
        patch.object(ep, "_publish_livekit_audio_track", new=AsyncMock(return_value=MagicMock())),
        patch.object(
            ep, "_create_avatar_session", new=AsyncMock(return_value=fake_avatar_session)
        ),
        patch.object(ep.session_manager, "remove_session") as fake_remove,
    ):
        await ep.entrypoint(ctx)

    fake_remove.assert_called_once_with("room-3")
    ctx.shutdown.assert_called_once()
    assert registry.acquire("room-3") is True
    registry.release("room-3")
