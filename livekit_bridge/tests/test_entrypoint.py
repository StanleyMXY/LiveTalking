"""
Tests for livekit_bridge.entrypoint.

entrypoint.py wires the other livekit_bridge components to a real,
external LiveKit connection — there is no local LiveKit dev server or test
client available in this environment (see livekit_bridge/__init__.py's
"Known limitations"), so nothing here talks to a real
`rtc.Room`/`JobContext`. What *is* genuinely unit-testable without live
infrastructure, and is covered below:

- the small helper functions entrypoint() is built from, each verified
  against the real installed livekit/livekit-agents API it wraps (see
  entrypoint.py's module docstring for citations) — with everything from
  the real SDK that would require a live connection replaced by a minimal
  fake/mock;
- the barge-in logic in `_process_frame`: it needs 240ms of sustained
  speech (a single noise frame or a short burst does not interrupt), only
  fires while the avatar is speaking, fires once per run, clears
  LiveKitOutput's own queued audio (code review C2), and keeps the new
  speech's first frames when it resets the buffer;
- that every meaningful utterance supersedes the previous response — even
  once that response's llm_response call has returned — while a
  noise-only (filtered-to-empty) utterance does not, and an older utterance
  whose ASR finishes last is dropped;
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
- that an already-executing `llm_response` call is stopped from reaching
  the real session once superseded: covered end to end here with the real
  `PipelineBridge` (`test_new_utterance_cuts_off_already_enqueued_answer_end_to_end`,
  `test_interrupt_between_dispatch_and_worker_start_fences_the_real_bridge`),
  and the fence itself, with real threads, in `test_pipeline_bridge.py`.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from livekit import rtc

from livekit_bridge import entrypoint as ep
from livekit_bridge.barge_in_detector import BargeInDetector
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


# ─── _process_frame: barge-in detection ─────────────────────────────────

# BargeInDetector's default: 240ms of consecutive above-threshold 20ms frames.
_BARGE_IN_FRAMES = 12


def _frame_kwargs(**overrides):
    avatar_session = MagicMock()
    avatar_session.is_speaking.return_value = False
    utterance_buffer = MagicMock()
    utterance_buffer.feed.return_value = None
    kwargs = dict(
        barge_in_detector=BargeInDetector(),
        utterance_buffer=utterance_buffer,
        utterance_order=ep._UtteranceOrder(),
        asr_bridge=MagicMock(),
        pipeline_bridge=MagicMock(),
        avatar_session=avatar_session,
        loop=asyncio.get_running_loop(),
        pending_tasks=set(),
    )
    kwargs.update(overrides)
    return kwargs


async def _feed_frames(chunks, kwargs):
    for chunk in chunks:
        await ep._process_frame(chunk, **kwargs)


@pytest.mark.asyncio
async def test_process_frame_single_noise_frame_does_not_barge_in():
    # Final-review finding: one 20ms frame above the threshold (a click, a
    # cough onset) while the avatar is speaking must not interrupt it.
    kwargs = _frame_kwargs()
    kwargs["avatar_session"].is_speaking.return_value = True

    await _feed_frames([_make_chunk(is_speech=True), _make_chunk(is_speech=False)], kwargs)

    kwargs["pipeline_bridge"].interrupt.assert_not_called()
    kwargs["utterance_buffer"].reset.assert_not_called()


@pytest.mark.asyncio
async def test_process_frame_burst_shorter_than_threshold_does_not_barge_in():
    kwargs = _frame_kwargs()
    kwargs["avatar_session"].is_speaking.return_value = True

    burst = [_make_chunk(is_speech=True)] * (_BARGE_IN_FRAMES - 1)
    await _feed_frames(burst + [_make_chunk(is_speech=False)] + burst, kwargs)

    kwargs["pipeline_bridge"].interrupt.assert_not_called()
    kwargs["utterance_buffer"].reset.assert_not_called()


@pytest.mark.asyncio
async def test_process_frame_sustained_speech_barges_in_once_while_avatar_speaking():
    kwargs = _frame_kwargs()
    avatar_session = kwargs["avatar_session"]
    avatar_session.is_speaking.return_value = True
    avatar_session.output = MagicMock(spec=LiveKitOutput)
    stale_task = MagicMock()
    stale_task.done.return_value = False
    kwargs["pending_tasks"].add(stale_task)

    speech = [_make_chunk(is_speech=True)] * _BARGE_IN_FRAMES
    await _feed_frames(speech[:-1], kwargs)
    kwargs["pipeline_bridge"].interrupt.assert_not_called()

    await _feed_frames(speech[-1:], kwargs)
    kwargs["pipeline_bridge"].interrupt.assert_called_once_with(avatar_session)
    avatar_session.output.clear_queue.assert_called_once()
    kwargs["utterance_buffer"].reset.assert_called_once()
    stale_task.cancel.assert_called_once()
    assert kwargs["pending_tasks"] == set()

    # Speech continuing past the threshold is the same run: no re-fire.
    await _feed_frames([_make_chunk(is_speech=True)] * 30, kwargs)
    kwargs["pipeline_bridge"].interrupt.assert_called_once()


@pytest.mark.asyncio
async def test_process_frame_new_run_after_silence_can_barge_in_again():
    kwargs = _frame_kwargs()
    kwargs["avatar_session"].is_speaking.return_value = True
    run = [_make_chunk(is_speech=True)] * _BARGE_IN_FRAMES

    await _feed_frames(run + [_make_chunk(is_speech=False)] + run, kwargs)

    assert kwargs["pipeline_bridge"].interrupt.call_count == 2


@pytest.mark.asyncio
async def test_process_frame_no_barge_in_when_avatar_not_speaking():
    # Sustained speech while the avatar is NOT talking must not
    # interrupt/reset — see entrypoint.py's "Interrupt-handling design".
    kwargs = _frame_kwargs()

    await _feed_frames([_make_chunk(is_speech=True)] * (_BARGE_IN_FRAMES + 5), kwargs)

    kwargs["pipeline_bridge"].interrupt.assert_not_called()
    kwargs["utterance_buffer"].reset.assert_not_called()
    assert kwargs["utterance_buffer"].feed.call_count == _BARGE_IN_FRAMES + 5


@pytest.mark.asyncio
async def test_process_frame_barge_in_survives_output_without_clear_queue():
    # Other outputs (webrtc/rtmp/virtualcam) do not implement clear_queue()
    # at all — the hasattr-style guard must not raise for them.
    kwargs = _frame_kwargs()
    kwargs["avatar_session"].is_speaking.return_value = True
    kwargs["avatar_session"].output = object()  # deliberately has no clear_queue

    await _feed_frames([_make_chunk(is_speech=True)] * _BARGE_IN_FRAMES, kwargs)

    kwargs["pipeline_bridge"].interrupt.assert_called_once_with(kwargs["avatar_session"])


@pytest.mark.asyncio
async def test_process_frame_barge_in_keeps_the_new_speech_and_drops_stale_audio():
    # On a confirmed barge-in the buffer is reset to drop audio from before
    # the onset, and the run's earlier frames are fed back in — otherwise
    # the reset would cut the first 220ms off the very speech that
    # triggered it. Uses the real UtteranceBuffer.
    from livekit_bridge.utterance_buffer import UtteranceBuffer

    captured = []

    def capture_transcribe(audio):
        captured.append(audio)
        return ""

    asr_bridge = MagicMock()
    asr_bridge.transcribe.side_effect = capture_transcribe
    kwargs = _frame_kwargs(utterance_buffer=UtteranceBuffer(silence_threshold_ms=500), asr_bridge=asr_bridge)
    kwargs["avatar_session"].is_speaking.return_value = True

    stale = np.full(320, 0.9, dtype=np.float32)
    new_speech = np.full(320, 0.3, dtype=np.float32)
    frames = [stale] * 3 + [_make_chunk(is_speech=False)] * 2  # stale partial, not ended
    frames += [new_speech] * (_BARGE_IN_FRAMES + 3)
    await _feed_frames(frames, kwargs)
    kwargs["avatar_session"].is_speaking.return_value = False
    await _feed_frames([_make_chunk(is_speech=False)] * 25, kwargs)
    await asyncio.gather(*kwargs["pending_tasks"])

    assert len(captured) == 1
    np.testing.assert_array_equal(
        captured[0], np.concatenate([new_speech] * (_BARGE_IN_FRAMES + 3))
    )


@pytest.mark.asyncio
async def test_process_frame_empty_chunk_is_noop():
    kwargs = _frame_kwargs()

    await ep._process_frame(np.zeros(0, dtype=np.float32), **kwargs)

    kwargs["utterance_buffer"].feed.assert_not_called()
    kwargs["avatar_session"].is_speaking.assert_not_called()


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

    def fake_drive_with_text(avatar_session, text, generation):
        call_threads["drive_with_text"] = threading.get_ident()

    kwargs = _frame_kwargs()
    kwargs["asr_bridge"].transcribe.side_effect = fake_transcribe
    kwargs["pipeline_bridge"].begin_generation.return_value = 7
    kwargs["pipeline_bridge"].drive_with_text.side_effect = fake_drive_with_text
    kwargs["utterance_buffer"].feed.return_value = np.zeros(320, dtype=np.float32)

    await ep._process_frame(_make_chunk(is_speech=False), **kwargs)

    # The drive-with-transcription sequence is a scheduled background task
    # (code review I1), not awaited inline — give it a moment to run.
    assert len(kwargs["pending_tasks"]) == 1
    await asyncio.gather(*kwargs["pending_tasks"])

    assert call_threads.get("transcribe") not in (None, main_thread_id)
    assert call_threads.get("drive_with_text") not in (None, main_thread_id)
    kwargs["pipeline_bridge"].drive_with_text.assert_called_once_with(
        kwargs["avatar_session"], "hello", 7
    )


@pytest.mark.asyncio
async def test_process_frame_skips_asr_when_no_utterance_completed():
    kwargs = _frame_kwargs()

    await ep._process_frame(_make_chunk(is_speech=True), **kwargs)

    assert kwargs["pending_tasks"] == set()
    kwargs["asr_bridge"].transcribe.assert_not_called()
    kwargs["pipeline_bridge"].drive_with_text.assert_not_called()


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

    kwargs = _frame_kwargs()
    kwargs["asr_bridge"].transcribe.side_effect = slow_transcribe
    kwargs["utterance_buffer"].feed.return_value = np.zeros(320, dtype=np.float32)
    loop = kwargs["loop"]

    call_started_at = loop.time()
    await ep._process_frame(_make_chunk(is_speech=False), **kwargs)
    elapsed = loop.time() - call_started_at

    # _process_frame itself must return almost immediately: the transcribe
    # call (which sleeps for 0.3s) runs in a background task, not awaited
    # here — this call returning well under that proves it.
    assert elapsed < 0.15
    assert len(kwargs["pending_tasks"]) == 1

    await asyncio.gather(*kwargs["pending_tasks"])


# ─── _process_frame: every meaningful utterance supersedes ──────────────


def _ordered_mocks():
    """pipeline_bridge and avatar_session.output as children of one parent
    mock, so the relative order of their calls can be asserted."""
    parent = MagicMock()
    avatar_session = MagicMock()
    avatar_session.is_speaking.return_value = False
    avatar_session.output = parent.output
    return parent, parent.pipeline_bridge, avatar_session


@pytest.mark.asyncio
async def test_meaningful_utterance_supersedes_even_after_previous_drive_task_finished():
    # Final-review finding: llm_response returns as soon as it has enqueued
    # its sentences, so the previous utterance's drive task is done() while
    # its answer is still playing. The next meaningful utterance must still
    # interrupt it, not queue behind it.
    parent, pipeline_bridge, avatar_session = _ordered_mocks()
    pipeline_bridge.begin_generation.side_effect = [1, 2]
    asr_bridge = MagicMock()
    asr_bridge.transcribe.side_effect = ["first", "second"]
    kwargs = _frame_kwargs(
        pipeline_bridge=pipeline_bridge, avatar_session=avatar_session, asr_bridge=asr_bridge
    )
    kwargs["utterance_buffer"].feed.return_value = np.zeros(320, dtype=np.float32)

    await ep._process_frame(_make_chunk(is_speech=False), **kwargs)
    await asyncio.gather(*kwargs["pending_tasks"])
    assert kwargs["pending_tasks"] == set()  # nothing "in flight" any more

    parent.reset_mock()
    await ep._process_frame(_make_chunk(is_speech=False), **kwargs)
    await asyncio.gather(*kwargs["pending_tasks"])

    # interrupt + clear_queue, then a fresh generation, then the drive — in
    # that order.
    assert [c[0] for c in parent.mock_calls] == [
        "pipeline_bridge.interrupt",
        "output.clear_queue",
        "pipeline_bridge.begin_generation",
        "pipeline_bridge.drive_with_text",
    ]
    pipeline_bridge.drive_with_text.assert_called_once_with(avatar_session, "second", 2)


@pytest.mark.asyncio
async def test_noise_only_utterance_does_not_interrupt_current_response():
    # AsrBridge returns "" for no speech and for SenseVoice's non-speech
    # markers (a cough -> "😷"); such an utterance must leave the answer
    # that is currently playing alone.
    kwargs = _frame_kwargs()
    kwargs["avatar_session"].output = MagicMock(spec=LiveKitOutput)
    kwargs["asr_bridge"].transcribe.return_value = ""
    kwargs["utterance_buffer"].feed.return_value = np.zeros(320, dtype=np.float32)

    await ep._process_frame(_make_chunk(is_speech=False), **kwargs)
    await asyncio.gather(*kwargs["pending_tasks"])

    kwargs["pipeline_bridge"].interrupt.assert_not_called()
    kwargs["avatar_session"].output.clear_queue.assert_not_called()
    kwargs["pipeline_bridge"].begin_generation.assert_not_called()
    kwargs["pipeline_bridge"].drive_with_text.assert_not_called()


@pytest.mark.asyncio
async def test_older_utterance_whose_asr_finishes_last_is_dropped():
    # Transcriptions run in the executor and can finish out of order. The
    # newest utterance must win even if an older one's ASR returns later.
    first_call_release = threading.Event()

    def transcribe(audio):
        if audio[0] == 1.0:
            first_call_release.wait(timeout=5)
            return "first"
        return "second"

    kwargs = _frame_kwargs()
    kwargs["asr_bridge"].transcribe.side_effect = transcribe
    kwargs["pipeline_bridge"].begin_generation.return_value = 5

    kwargs["utterance_buffer"].feed.return_value = np.ones(320, dtype=np.float32)
    await ep._process_frame(_make_chunk(is_speech=False), **kwargs)
    first_task = next(iter(kwargs["pending_tasks"]))
    await asyncio.sleep(0.05)  # let it start its (blocked) transcribe

    kwargs["utterance_buffer"].feed.return_value = np.full(320, 2.0, dtype=np.float32)
    await ep._process_frame(_make_chunk(is_speech=False), **kwargs)
    second_task = next(t for t in kwargs["pending_tasks"] if t is not first_task)
    await asyncio.gather(second_task)
    kwargs["pipeline_bridge"].drive_with_text.assert_called_once_with(
        kwargs["avatar_session"], "second", 5
    )

    first_call_release.set()
    await asyncio.gather(first_task)

    # "first" finished ASR after "second" had committed: dropped, and it
    # did not interrupt "second"'s answer either.
    kwargs["pipeline_bridge"].drive_with_text.assert_called_once()
    kwargs["pipeline_bridge"].interrupt.assert_called_once()


@pytest.mark.asyncio
async def test_superseding_does_not_reset_the_utterance_buffer():
    # By the time an utterance's ASR returns, the buffer holds only audio
    # that arrived after it completed — the start of the user's *next*
    # utterance. Resetting it on the supersede path would cut that off.
    kwargs = _frame_kwargs()
    kwargs["asr_bridge"].transcribe.return_value = "hello"
    kwargs["utterance_buffer"].feed.return_value = np.zeros(320, dtype=np.float32)

    await ep._process_frame(_make_chunk(is_speech=False), **kwargs)
    await asyncio.gather(*kwargs["pending_tasks"])

    kwargs["pipeline_bridge"].interrupt.assert_called_once()
    kwargs["utterance_buffer"].reset.assert_not_called()


@pytest.mark.asyncio
async def test_new_utterance_cuts_off_already_enqueued_answer_end_to_end():
    # End to end with the real PipelineBridge: the first answer's
    # llm_response has already returned (both sentences enqueued, task
    # done). The next utterance must flush it before its own answer is
    # enqueued.
    from livekit_bridge.pipeline_bridge import PipelineBridge

    session = MagicMock()
    session.is_speaking.return_value = False
    events = []
    session.put_msg_txt.side_effect = lambda msg, datainfo: events.append(("put", msg))
    session.flush_talk.side_effect = lambda: events.append(("flush",))

    def fake_llm_response(message, proxy, datainfo):
        proxy.put_msg_txt(f"{message}-1", {})
        proxy.put_msg_txt(f"{message}-2", {})

    asr_bridge = MagicMock()
    asr_bridge.transcribe.side_effect = ["a", "b"]
    kwargs = _frame_kwargs(
        pipeline_bridge=PipelineBridge(), avatar_session=session, asr_bridge=asr_bridge
    )
    kwargs["utterance_buffer"].feed.return_value = np.zeros(320, dtype=np.float32)

    with patch("livekit_bridge.pipeline_bridge.llm_response", side_effect=fake_llm_response):
        await ep._process_frame(_make_chunk(is_speech=False), **kwargs)
        await asyncio.gather(*kwargs["pending_tasks"])
        await ep._process_frame(_make_chunk(is_speech=False), **kwargs)
        await asyncio.gather(*kwargs["pending_tasks"])

    assert events == [
        ("flush",),
        ("put", "a-1"),
        ("put", "a-2"),
        ("flush",),
        ("put", "b-1"),
        ("put", "b-2"),
    ]


# ─── _drive_avatar_with_transcription: generation allocated on the loop ─


@pytest.mark.asyncio
async def test_drive_generation_is_allocated_on_event_loop_before_executor_dispatch():
    # Round 4, Gap 2 (pipeline_bridge.py module docstring): the drive
    # call's generation must be allocated on the event-loop thread, before
    # run_in_executor dispatches drive_with_text — not on the worker — and
    # the pre-allocated value must be what drive_with_text receives.
    loop_thread_id = threading.get_ident()
    events = []

    def fake_begin_generation():
        events.append(("begin_generation", threading.get_ident()))
        return 41

    def fake_drive_with_text(avatar_session, text, generation):
        events.append(("drive_with_text", threading.get_ident()))

    pipeline_bridge = MagicMock()
    pipeline_bridge.begin_generation.side_effect = fake_begin_generation
    pipeline_bridge.drive_with_text.side_effect = fake_drive_with_text
    asr_bridge = MagicMock()
    asr_bridge.transcribe.return_value = "hi"
    avatar_session = MagicMock()

    order = ep._UtteranceOrder()
    await ep._drive_avatar_with_transcription(
        asyncio.get_running_loop(),
        asr_bridge,
        pipeline_bridge,
        avatar_session,
        np.zeros(320, dtype=np.float32),
        utterance_order=order,
        seq=order.issue(),
    )

    assert [name for name, _ in events] == ["begin_generation", "drive_with_text"]
    assert events[0][1] == loop_thread_id
    assert events[1][1] != loop_thread_id
    pipeline_bridge.drive_with_text.assert_called_once_with(avatar_session, "hi", 41)


class _DispatchGatedLoop:
    """Wraps the real running loop so the test can hold a drive_with_text
    call in the "dispatched, but no worker has started it yet" state — the
    exact window Gap 2 is about — without replacing the executor itself."""

    def __init__(self, real_loop):
        self._real_loop = real_loop
        self.drive_dispatched = asyncio.Event()
        self.release_drive = asyncio.Event()

    async def run_in_executor(self, executor, func, *args):
        if getattr(func, "__name__", None) == "drive_with_text":
            self.drive_dispatched.set()
            await self.release_drive.wait()
        return await self._real_loop.run_in_executor(executor, func, *args)


@pytest.mark.asyncio
async def test_interrupt_between_dispatch_and_worker_start_fences_the_real_bridge():
    # Gap 2 end to end, with the real PipelineBridge: a drive call is
    # dispatched, interrupt() lands on the event loop before the worker
    # runs it, and only then does the worker run drive_with_text. The
    # llm_response it runs must not reach the real session. When
    # drive_with_text allocated its generation on the worker (pre-fix), the
    # allocation came after the interrupt's bump and this fragment got
    # through.
    from livekit_bridge.pipeline_bridge import PipelineBridge

    bridge = PipelineBridge()
    session = MagicMock()
    asr_bridge = MagicMock()
    asr_bridge.transcribe.return_value = "hi"
    gated_loop = _DispatchGatedLoop(asyncio.get_running_loop())
    order = ep._UtteranceOrder()

    def fake_llm_response(message, proxy, datainfo):
        proxy.put_msg_txt("fragment from a superseded call", {})

    with patch("livekit_bridge.pipeline_bridge.llm_response", side_effect=fake_llm_response):
        drive = asyncio.ensure_future(
            ep._drive_avatar_with_transcription(
                gated_loop,
                asr_bridge,
                bridge,
                session,
                np.zeros(320, dtype=np.float32),
                utterance_order=order,
                seq=order.issue(),
            )
        )
        await asyncio.wait_for(gated_loop.drive_dispatched.wait(), timeout=5)

        bridge.interrupt(session)  # event loop: lands before the worker runs
        gated_loop.release_drive.set()
        await asyncio.wait_for(drive, timeout=5)

    # Twice: once by the drive's own supersede (every meaningful utterance
    # supersedes, before allocating its generation), once by the test's
    # interrupt() above.
    assert session.flush_talk.call_count == 2
    session.put_msg_txt.assert_not_called()


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
