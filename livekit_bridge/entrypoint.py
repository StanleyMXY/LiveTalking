"""
livekit_bridge.entrypoint — Task 8: the LiveKit Agent entrypoint that wires
together Tasks 2-7 (``audio_format``, ``utterance_buffer``, ``asr_bridge``,
``pipeline_bridge``, ``session_registry``) and ``streamout.livekit_output``
into one per-room job.

The brief's draft code (``task-8-brief.md``) is stale in several places it
itself flags with a warning marker. Every correction below was verified by
reading the *installed* ``livekit-agents==1.8.3`` / ``livekit==1.1.18``
source (this repo's ``requirements.txt`` now pins both; see the comment
there for why the previously-pinned ``livekit==1.1.20`` had to move — a
downgrade that was checked byte-for-byte against Task 1/4's dependencies
before accepting it, not merely allowed to happen).

Verified facts, with citations
-------------------------------
- **The room is not connected when the entrypoint starts, and
  ``ctx.room.name`` is not the documented way to read the room name before
  that.** ``JobContext.room``'s docstring says explicitly: "When the
  entrypoint is called, the worker has not connected to the Room yet."
  (``livekit/agents/job.py:505-511``). Correction from an earlier version
  of this docstring, made during code review: a bare, freshly constructed
  ``rtc.Room()`` does have ``.name == ''``, but that is not proof about the
  *real* ``ctx.room`` object — the framework actually pre-populates
  ``self._room._info.name`` from the dispatch info before the entrypoint
  ever runs (``livekit/agents/ipc/job_proc_lazy_main.py:307``), so
  ``ctx.room.name`` may already happen to be correct by the time the
  entrypoint starts. That is an internal implementation detail, not a
  documented contract, so this module still does not rely on it. The
  documented, pre-connect-safe room name is ``ctx.job.room.name``:
  ``ctx.job`` is a ``livekit.protocol.agent.Job`` protobuf message
  (``JobContext.job`` property, ``job.py:494-497``), whose ``room`` field
  is a ``livekit.protocol.models.Room`` message that does carry a ``name``
  field (verified against both messages' protobuf descriptors) and is
  populated by the dispatch/assignment before the entrypoint ever runs.
  See ``_room_name_from_job`` below.
- **Dispatch metadata**: ``agent.Job`` really does have a top-level
  ``metadata`` string field (verified against the ``Job`` descriptor), so
  the brief's field name (``ctx.job.metadata``) was right. Its *contents*
  (does the Java dispatcher put a bare account_id there, or a JSON blob?)
  are not established anywhere in this repo, this task's brief, or the
  connection-shell spec (``docs/superpowers/specs/2026-09-29-app-shell-cloud-
  integration-design.md`` §5/6, as linked from the plan) — grepping this
  repo and the plan for ``account_id`` finds nothing. Parsing an unverified
  cross-project contract would be fabricating one, so this module logs the
  raw string and threads it no further (see ``entrypoint``'s body).
- **Connecting**: ``await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)``
  (``job.py:672-735``; ``AutoSubscribe`` enum at ``job.py:135-139``). Once
  connected, ``_apply_auto_subscribe_opts`` (``job.py:990-1006``, called
  from inside ``connect()``) subscribes every already-published audio
  track and registers a ``"track_published"`` listener for future ones;
  subscription completion (for both cases) fires the SDK's own
  ``"track_subscribed"`` event on the room
  (``livekit/rtc/room.py:871-901``: callback receives
  ``(track, publication, participant)`` — this part of Task 1's
  ``format_probe.py`` citation is re-confirmed here). We register our own
  ``"track_subscribed"`` listener *before* calling ``ctx.connect()`` so we
  cannot race that internal auto-subscribe flow.
- **``rtc.AudioStream(track, sample_rate=16000, num_channels=1,
  frame_size_ms=20)``** — ``sample_rate``/``num_channels`` are Task 1's
  already-verified fact. ``frame_size_ms=20`` is a fact *this* task adds:
  left at the SDK default (``None``), the field is simply omitted from the
  FFI request (``livekit/rtc/audio_stream.py:261-262:
  if self._frame_size_ms: new_audio_stream.frame_size_ms = ...``), so the
  native layer picks whatever frame size it wants. ``UtteranceBuffer``
  (Task 3) hardcodes ``_FRAME_MS = 20`` and divides
  ``silence_threshold_ms // _FRAME_MS`` to get a frame count — that
  arithmetic is only correct if every ``feed()`` call really is a 20ms
  frame. Passing ``frame_size_ms=20`` explicitly is what makes that true,
  instead of leaving it an unverified coincidence.
- ``audio_stream.aclose()`` releases the native stream/FFI handle
  (``livekit/rtc/audio_stream.py:318``) — called in ``entrypoint``'s
  ``finally``.
- **Publishing the synthesized voice back to the room**: the brief guessed
  ``ctx.room.local_participant.create_audio_source()`` — no such method
  exists on ``LocalParticipant`` or ``Room`` in the installed SDK.
  ``rtc.AudioSource`` is a plain constructor
  (``livekit/rtc/audio_source.py:27-41``:
  ``AudioSource(sample_rate, num_channels, ...)``\\ ),
  ``rtc.LocalAudioTrack.create_audio_track(name, source)`` wraps it into a
  publishable track (``livekit/rtc/track.py:157-170``), and
  ``await ctx.room.local_participant.publish_track(track, options)``
  publishes it (``livekit/rtc/participant.py:1002-1004``).
  ``rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)`` is
  exactly what livekit-agents' own built-in avatar-worker output uses for a
  synthesized voice track
  (``livekit/agents/voice/avatar/_runner.py:109``), so this bridge follows
  that convention instead of inventing one.
- **Worker registration**: ``WorkerOptions`` is a plain alias for the real
  dataclass ``ServerOptions`` (``livekit/agents/worker.py:306:
  WorkerOptions = ServerOptions``) and takes ``entrypoint_fnc``
  (``worker.py:199-201``); it is run via
  ``livekit.agents.cli.run_app(WorkerOptions(...))``
  (``livekit/agents/cli/cli.py:471-480``) — the same
  ``if __name__ == "__main__": cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))``
  shape used by every livekit-agents quickstart. ``run_app``'s own
  docstring calls this "the (deprecated) rich Python CLI" being phased out
  in favor of the ``lk agent`` CLI / newer ``AgentServer`` class — it is
  still fully functional in 1.8.3 and is the simplest correct entry for
  this task's "minimal launch script" requirement (see
  ``livekit_bridge/worker_app.py``); a human productionizing this later may
  want to look at the newer ``AgentServer`` path instead.

Interrupt-handling design (not fully spelled out by the brief)
----------------------------------------------------------------
The brief suggests, as a simplest-correct-option fallback, calling
``pipeline_bridge.interrupt()`` "unconditionally" on every silence→speech
transition, reasoning that ``flush_talk()`` is documented as a safe no-op
when nothing is playing. That reasoning is fine for ``interrupt()`` itself,
but this module does **not** also call ``utterance_buffer.reset()``
unconditionally on every such transition, because that part *would* be a
real bug: ordinary speech has many brief energy dips below any fixed
threshold (unvoiced consonants, the pause between two words in the same
sentence), so a per-chunk energy-only rising-edge detector fires far more
often than genuine "user started talking again" events. Wiping
``utterance_buffer``'s accumulated audio on every such micro-transition
would mean it could never accumulate a complete utterance.

The task's own wording actually already names the correct gating
condition: "when the user starts speaking again *while the avatar is still
talking*". "The avatar is still talking" is exactly
``avatar_session.is_speaking()`` (``avatars/base_avatar.py:232-233``, a
real, pre-existing method — not something invented for this task). Gating
the interrupt+reset pair on ``is_speech and not was_speech and
avatar_session.is_speaking()`` implements the requirement as literally
stated, and avoids the false-trigger bug above: a genuine mid-utterance
pause essentially never coincides with the avatar *also* happening to be
mid-speech, and once a real barge-in has fired once, the avatar stops
speaking almost immediately (``flush_talk()``), so ``is_speaking()`` goes
false and further micro-pauses in the same (new) utterance no longer
re-trigger it. See ``_process_frame`` below.

Interrupt must also clear LiveKit's own audio backlog (code review finding)
------------------------------------------------------------------------------
``flush_talk()`` (``PipelineBridge.interrupt``) only clears the pipeline's
own upstream ASR/TTS/render queues — it has no way to reach into a specific
``BaseOutput`` implementation's own buffer, and none of the other existing
outputs (webrtc/rtmp/virtualcam) need one cleared this way, so nothing
wires this up generically. ``streamout.livekit_output.LiveKitOutput``
forwards synthesized audio into ``rtc.AudioSource``, which keeps its own
internal queue (default up to 1000ms — ``livekit/rtc/audio_source.py:38``,
``queue_size_ms: int = 1000``) independent of anything ``flush_talk()``
touches; without clearing it too, an interrupt would still let up to a
second of stale, already-queued speech play out before new speech starts.
``AudioSource.clear_queue()`` (``audio_source.py:88-100``) exists exactly
for this. On a detected barge-in, ``_process_frame`` calls
``avatar_session.output.clear_queue()`` if the current output exposes one
(the same optional-capability ``hasattr`` pattern
``avatars/base_avatar.py:529`` already uses for ``get_buffer_size()``), in
addition to ``pipeline_bridge.interrupt()``/``utterance_buffer.reset()``.

The frame-iteration loop must not stall waiting for the LLM (code review finding)
--------------------------------------------------------------------------------------
An earlier version of this module ``await``ed both
``asr_bridge.transcribe`` and ``pipeline_bridge.drive_with_text`` inline,
sequentially, inside the ``async for frame_event in audio_stream`` loop.
Since ``llm_response`` streams synchronously to completion
(``llm.py:45``), that ``await`` chain does not return until the whole LLM
response has streamed — during which the loop pulls no new frames from
``audio_stream`` at all. Frames still arrive (buffered inside
``AudioStream``'s own queue) but are only processed in a burst afterward,
which also delays interrupt detection for the entire duration of the LLM
call — exactly the tail-latency problem this plan's Review Focus calls
out, just moved to a different queue. The existing ``/human`` route already
avoids this for the same reason: it fires ``run_in_executor`` for
``llm_response`` without awaiting it (``server/routes.py:71-76``). This
module now does the same: ``_process_frame`` schedules the
transcribe-then-drive sequence as a tracked background task
(``loop.create_task``, kept in a ``pending_tasks`` set so it is not
garbage-collected mid-flight and so its exceptions are logged via a
done-callback instead of silently dropped) and returns immediately,
letting the frame loop keep draining and keep detecting interrupts while
the LLM streams. A known, separate, out-of-scope gap (not fixed here, per
code review guidance not to make it worse without a real fix): there is no
LLM cancellation token anywhere in ``llm.py``, so after ``flush_talk()`` an
already-in-flight ``llm_response`` call still runs to completion and still
calls ``avatar_session.put_msg_txt`` for the old answer once it finishes —
``entrypoint.py``'s ``finally`` block cancels any *not-yet-started*
background task (preventing a queued-but-not-running
``drive_with_text`` call from firing against a torn-down session) but
cannot stop a call that is already executing inside the executor thread.

At most one drive task in flight per session (code review follow-up finding)
----------------------------------------------------------------------------------
Scheduling every completed utterance as its own independent background
task (the fix directly above) introduced a new problem: nothing stopped
two of them from running concurrently. ``UtteranceBuffer`` cuts an
utterance at 500ms of silence, and a user pausing mid-thought — or simply
an LLM response taking longer than 500ms to *start* producing audio, which
is common — routinely produces a second completed utterance while the
first one's ``drive_with_text`` task is still running. The existing
barge-in check (``is_speech and not was_speech and
avatar_session.is_speaking()``) does not catch this: it fires on a
*speech-onset* transition, but ``utterance_buffer.feed()`` only returns a
completed utterance on a *silence* chunk, and at that point
``avatar_session.is_speaking()`` may still be ``False`` (the previous
task's LLM call may not have produced its first audio chunk yet) even
though its task is very much still running. Left unfixed, both
``llm_response`` calls run in parallel and both call
``avatar_session.put_msg_txt`` (``llm.py:62-87``), interleaving two
answers' TTS fragments into the same session.

Two designs were possible: queue completed utterances and drive them one
at a time (FIFO), or treat a new utterance completing mid-response as an
implicit interrupt of that response (supersede). Supersede was chosen:
this plan's whole interrupt model is built around "the user's most recent
utterance is the one that matters" (the barge-in check above already
embodies exactly that policy for the mid-speech case), and a real-time
conversational avatar answering an ordinary mid-thought pause with a
disjointed reply to just the fragment before the pause — then, once that
finishes, *also* replying to the fragment after it — is not better
behavior than cutting the first (likely-incomplete) response short and
answering the newer one; unbounded queueing would also let a user who
keeps pausing accumulate an ever-growing backlog of answers. Neither
design can fix the deeper cause (``UtteranceBuffer``'s segmentation has
already split what may have been one continuous thought into two ASR
calls by the time either sees it) — that is unrelated to this fix.

``_supersede_in_flight_response`` is now the single implementation shared
by both trigger points (mid-speech barge-in, and a new utterance
completing while ``pending_tasks`` still holds an unfinished one): it calls
``pipeline_bridge.interrupt()``, cancels every not-yet-finished pending
task (``_cancel_pending_drive_tasks`` — same limitation as above: this can
only stop a task that has not yet reached its ``drive_with_text`` call,
not one already executing inside the executor thread), and clears the
output's own queue if it exposes one. ``_process_frame`` calls it before
starting a new drive task whenever ``pending_tasks`` still contains an
unfinished one, which — since this is now the only way a task is ever
added — means at most one drive task is ever in flight per session by
construction.

The job does not end on its own — ``ctx.shutdown()`` must be called (code review finding)
---------------------------------------------------------------------------------------------
Read ``livekit/agents/ipc/job_proc_lazy_main.py:280-420``: the job task
``await``s a single ``self._shutdown_fut``
(``job_proc_lazy_main.py:400``), which is resolved only by the room's
``"disconnected"`` event (``:289-294``), an explicit ``ctx.shutdown()``
call (``:299-305``), or the entrypoint task raising *before*
``ctx.connect()`` was ever called (``:376-387``, a crash-before-connect
safety net). If the entrypoint coroutine simply *returns* — no exception,
and ``ctx.connect()`` was already called — none of those three fire, and
the job sits waiting on ``_shutdown_fut`` forever, holding a thread/executor
slot (see ``livekit_bridge/worker_app.py``'s module docstring for how one
job-executor is spawned per job). This is a real leak on two of this
module's own paths that previously just ``return``ed: the duplicate-
dispatch early return (the room is never even connected on that path, so
``"disconnected"`` can never fire either) and the "no audio track arrived
within the timeout" path (the room stays connected — the participant just
never published audio — so ``"disconnected"`` may never fire). Both paths,
and the main ``finally`` block (covering normal completion/exception/
cancellation), now call ``ctx.shutdown(reason=...)``
(``job.py:871-872`` — synchronous, safe to call redundantly: the framework
suppresses ``asyncio.InvalidStateError`` if the future is already resolved,
``job_proc_lazy_main.py:302``).

``aclose()`` is now the *last* thing ``finally`` does (code review follow-up finding)
------------------------------------------------------------------------------------------
An earlier version awaited ``audio_stream.aclose()`` first in ``finally``,
ahead of the session/registry cleanup, wrapped only in
``except Exception`` — a ``CancelledError`` delivered to this task while
it was specifically suspended inside that ``await`` would not be caught by
that ``except Exception`` (``CancelledError`` is a ``BaseException``, not
an ``Exception``, since Python 3.8), and would propagate out of
``finally`` before ``session_manager.remove_session``, ``_registry.release``,
or ``ctx.shutdown()`` ever ran — the same resource-leak risk the
``aclose()``-try/except fix above was meant to close, just narrowed to one
specific timing window. Reordered so every synchronous, load-bearing
cleanup step runs first; ``audio_stream.aclose()`` runs last, still
wrapped in ``try/except Exception`` for its ordinary-failure case, but
nothing afterward depends on it succeeding (or, now, on it even being
reached) any more.

What is verified vs. not — see the Task 8 report
---------------------------------------------------
Everything above was checked against real installed SDK/package source
(citations inline and in the report). Live end-to-end behavior against a
real LiveKit dev server + client, and the runtime behavior of
``JobExecutorType.PROCESS`` on Linux, were **not** exercised in this
environment — see ``livekit_bridge/worker_app.py``'s module docstring and
the Task 8 report for what a human must still verify before shipping.
"""

from __future__ import annotations

import asyncio
import logging
import threading

import numpy as np
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext

from livekit_bridge.asr_bridge import AsrBridge
from livekit_bridge.audio_format import to_pipeline_format
from livekit_bridge.pipeline_bridge import PipelineBridge
from livekit_bridge.session_registry import BridgeSessionRegistry
from livekit_bridge.utterance_buffer import _ENERGY_THRESHOLD, UtteranceBuffer
from server.session_manager import session_manager
from streamout.livekit_output import LiveKitOutput

logger = logging.getLogger(__name__)

# LiveKit's native FFI layer resamples/downmixes to whatever we ask for when
# constructing the AudioStream (Task 1, format_probe.py) — this is the
# pipeline's own internal convention (server/asr_server.py SAMPLE_RATE=16000),
# not a LiveKit default.
_LIVEKIT_AUDIO_SAMPLE_RATE = 16000
_LIVEKIT_AUDIO_NUM_CHANNELS = 1
# Matches UtteranceBuffer's own _FRAME_MS=20 assumption — see module
# docstring's "frame_size_ms=20" paragraph for why this must be explicit.
_FRAME_SIZE_MS = 20
# Safety net only: this job is dispatched *because* a specific room needs
# handling, so waiting is normally the right thing to do. A generous but
# finite timeout just prevents one dispatch whose target participant never
# publishes audio (e.g. a client-side bug) from hanging this job forever.
_TRACK_WAIT_TIMEOUT_S = 60.0

_registry = BridgeSessionRegistry()


def _room_name_from_job(ctx: JobContext) -> str:
    """The dispatch-time room name, safe to read before ``ctx.connect()``.

    ``JobContext.room``'s own docstring says the room is not connected yet
    when the entrypoint starts; ``ctx.job.room.name`` is the documented,
    dispatch-time value that does not depend on that (see module
    docstring's room-name paragraph for the full citation, including a
    correction: whether ``ctx.room.name`` also happens to already be
    populated at this point is an internal implementation detail, not a
    documented guarantee, so this function does not rely on it either way).
    """
    return ctx.job.room.name


def _register_first_audio_track_waiter(
    room: rtc.Room,
) -> "asyncio.Future[rtc.RemoteAudioTrack]":
    """Register a ``"track_subscribed"`` listener and return a Future that
    resolves with the first *audio* track the SDK subscribes to (a video
    track, if one is ever subscribed, is ignored).

    Must be called before ``ctx.connect()`` — see module docstring's
    "Connecting" paragraph for why registering after would risk missing the
    event for a track that gets auto-subscribed during ``connect()`` itself.
    """
    loop = asyncio.get_event_loop()
    future: "asyncio.Future[rtc.RemoteAudioTrack]" = loop.create_future()

    def _on_track_subscribed(track, publication, participant) -> None:
        if future.done():
            return
        if isinstance(track, rtc.RemoteAudioTrack):
            future.set_result(track)

    room.on("track_subscribed", _on_track_subscribed)
    return future


def _make_audio_stream(track: rtc.Track) -> rtc.AudioStream:
    return rtc.AudioStream(
        track,
        sample_rate=_LIVEKIT_AUDIO_SAMPLE_RATE,
        num_channels=_LIVEKIT_AUDIO_NUM_CHANNELS,
        frame_size_ms=_FRAME_SIZE_MS,
    )


async def _publish_livekit_audio_track(ctx: JobContext) -> rtc.AudioSource:
    """Create and publish the local audio track that carries the pipeline's
    synthesized speech back into the room; return the ``AudioSource`` so it
    can be handed to ``LiveKitOutput``.
    """
    audio_source = rtc.AudioSource(_LIVEKIT_AUDIO_SAMPLE_RATE, _LIVEKIT_AUDIO_NUM_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("avatar-voice", audio_source)
    await ctx.room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )
    return audio_source


async def _create_avatar_session(
    room_name: str,
    audio_source: rtc.AudioSource,
    loop: asyncio.AbstractEventLoop,
):
    """Build the pipeline's avatar session for this room, inject
    ``LiveKitOutput`` as its output, and start its render loop.

    Uses ``session_manager.create_session`` (not
    ``app.build_avatar_session`` directly) so the session is tracked the
    same way the existing HTTP path tracks its sessions — that tracking is
    exactly what makes ``session_manager.remove_session`` in
    ``entrypoint``'s ``finally`` block able to clean it up.

    ``avatar_session.output`` is only ever read once ``render()``'s
    ``process_frames`` loop starts (``avatars/base_avatar.py:430:
    self.output.start()``), and ``BaseAvatar.__init__`` never constructs an
    output object for an unrecognized ``opt.transport`` (see
    ``worker_app.py``'s ``_load_and_wire_pipeline`` for why ``opt.transport``
    is deliberately set to ``'livekit'``, a value ``_output_modules`` does
    not recognize) — so overriding ``.output`` here, before starting the
    render thread, races nothing.

    ``parent=avatar_session`` is passed to ``LiveKitOutput`` so
    ``push_audio_frame``'s ``self.parent.notify(eventpoint)``
    (``streamout/livekit_output.py:95-96``) actually fires, matching every
    other existing output (``registry.create("streamout", ..., parent=self)``,
    ``avatars/base_avatar.py:151``) instead of silently dropping status
    events.

    ``quit_event`` is assigned onto the session *before* starting the
    thread (not left for ``render()`` to assign on its own first line) so
    that even if cleanup runs before the render thread's first line
    executes, ``session_manager.remove_session``'s
    ``getattr(avatar_session, 'quit_event', None)`` check always finds it.
    """
    await session_manager.create_session({}, sessionid=room_name)
    avatar_session = session_manager.get_session(room_name)

    quit_event = threading.Event()
    avatar_session.quit_event = quit_event
    avatar_session.output = LiveKitOutput(audio_source, loop, parent=avatar_session)

    render_thread = threading.Thread(
        target=avatar_session.render, args=(quit_event,), daemon=True
    )
    render_thread.start()
    return avatar_session


async def _drive_avatar_with_transcription(
    loop: asyncio.AbstractEventLoop,
    asr_bridge: AsrBridge,
    pipeline_bridge: PipelineBridge,
    avatar_session,
    audio: np.ndarray,
) -> None:
    """Transcribe one completed utterance and drive the avatar with it.

    Both steps are deliberately synchronous/blocking (``asr_bridge.py``,
    ``pipeline_bridge.py`` docstrings) — offloaded here via
    ``loop.run_in_executor``, exactly like the pipeline's own existing
    ``/human`` route and websocket ASR handler already do for the same
    underlying calls (``server/routes.py:71-76``,
    ``server/asr_server.py:218-224``). This whole coroutine is meant to be
    scheduled as a background task (see ``_process_frame`` and module
    docstring's "frame-iteration loop must not stall" section), not
    awaited inline in the frame loop — calling it directly would block
    LiveKit's own async event handling for the duration of each
    inference/LLM call, and would stall frame draining (hence interrupt
    detection) for the whole LLM response on top of that.
    """
    text = await loop.run_in_executor(None, asr_bridge.transcribe, audio)
    await loop.run_in_executor(None, pipeline_bridge.drive_with_text, avatar_session, text)


def _on_drive_task_done(task: "asyncio.Task[None]", pending_tasks: set) -> None:
    pending_tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("drive_with_text background task failed", exc_info=exc)


def _cancel_pending_drive_tasks(pending_tasks: set) -> None:
    """Cancel and stop tracking every not-yet-finished drive task.

    Can only stop a task that has not yet reached its
    ``drive_with_text`` call (or is still awaiting the executor for
    ``transcribe``) — it cannot interrupt a blocking call already
    executing inside the executor thread. See module docstring's "at most
    one drive task in flight" section.
    """
    for task in list(pending_tasks):
        if not task.done():
            task.cancel()
        pending_tasks.discard(task)


def _supersede_in_flight_response(
    pipeline_bridge: PipelineBridge, avatar_session, pending_tasks: set
) -> None:
    """Stop whatever response is currently being generated/played for a
    previous utterance, because the user's latest utterance supersedes it.

    Shared by both trigger points that mean the same thing from this
    connection shell's point of view: a mid-speech barge-in
    (``avatar_session.is_speaking()`` was true when new speech started),
    and a new utterance completing while a previous one's drive task is
    still running (``pending_tasks`` still holds an unfinished one). See
    module docstring's "at most one drive task in flight" section for why
    the second case is treated the same as the first rather than queued.
    """
    pipeline_bridge.interrupt(avatar_session)
    _cancel_pending_drive_tasks(pending_tasks)
    output = getattr(avatar_session, "output", None)
    clear_queue = getattr(output, "clear_queue", None)
    if clear_queue is not None:
        clear_queue()


async def _process_frame(
    chunk: np.ndarray,
    *,
    was_speech: bool,
    utterance_buffer: UtteranceBuffer,
    asr_bridge: AsrBridge,
    pipeline_bridge: PipelineBridge,
    avatar_session,
    loop: asyncio.AbstractEventLoop,
    pending_tasks: set,
) -> bool:
    """Process one already-format-converted 20ms audio chunk.

    Returns the new ``was_speech`` state for the caller to carry into the
    next call. Factored out of ``entrypoint``'s main loop so it is
    unit-testable with a mocked ``avatar_session``/bridges/buffer and no
    real LiveKit connection at all.

    On a completed utterance, the transcribe-then-drive sequence is
    scheduled as a tracked background task (added to ``pending_tasks``, a
    set owned by the caller so the task is never garbage-collected
    mid-flight and so ``entrypoint``'s ``finally`` block can cancel any
    still-pending ones) rather than awaited here — see module docstring's
    "frame-iteration loop must not stall" section for why. At most one
    such task is ever left in flight: if a previous one has not finished
    by the time a new utterance completes, it is superseded first — see
    module docstring's "at most one drive task in flight" section.
    """
    if chunk.size == 0:
        return was_speech

    is_speech = float(np.abs(chunk).mean()) > _ENERGY_THRESHOLD
    if is_speech and not was_speech and avatar_session.is_speaking():
        # Barge-in: the user started talking again while the avatar was
        # still talking. See module docstring's "Interrupt-handling design"
        # section for why this is gated on is_speaking() rather than firing
        # unconditionally.
        _supersede_in_flight_response(pipeline_bridge, avatar_session, pending_tasks)
        utterance_buffer.reset()

    completed_utterance = utterance_buffer.feed(chunk)
    if completed_utterance is not None:
        if any(not task.done() for task in pending_tasks):
            # A previous utterance's response is still being generated —
            # treat this new one as superseding it rather than queueing
            # behind it.
            _supersede_in_flight_response(pipeline_bridge, avatar_session, pending_tasks)

        task = loop.create_task(
            _drive_avatar_with_transcription(
                loop, asr_bridge, pipeline_bridge, avatar_session, completed_utterance
            )
        )
        pending_tasks.add(task)
        task.add_done_callback(lambda t: _on_drive_task_done(t, pending_tasks))

    return is_speech


async def entrypoint(ctx: JobContext) -> None:
    """The livekit-agents worker entrypoint for this connection shell.

    One job == one LiveKit room == (at most) one LiveTalking avatar session,
    deduplicated across concurrent/retried dispatches for the same room by
    ``_registry`` (Task 7).
    """
    room_name = _room_name_from_job(ctx)
    if not _registry.acquire(room_name):
        logger.info(
            "room '%s' is already being handled by this worker process; "
            "skipping duplicate dispatch",
            room_name,
        )
        # The room is never connected on this path, so nothing will ever
        # resolve the framework's shutdown future on its own (see module
        # docstring's "the job does not end on its own" section) — without
        # this, the job would sit idle forever holding an executor slot.
        ctx.shutdown(reason="duplicate dispatch for room already in progress")
        return

    avatar_session = None
    audio_stream = None
    pending_drive_tasks: "set[asyncio.Task[None]]" = set()
    try:
        # ctx.job.metadata really is a real field on the Job proto (verified
        # against its descriptor) — but its contents are an unestablished
        # cross-project contract (see module docstring). Logged, not parsed.
        metadata = ctx.job.metadata
        if metadata:
            logger.info("room '%s' dispatched with metadata=%r", room_name, metadata)

        loop = asyncio.get_running_loop()

        # Registered before connect() so we cannot race the SDK's own
        # auto-subscribe flow that starts inside connect() itself.
        track_future = _register_first_audio_track_waiter(ctx.room)

        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        audio_source = await _publish_livekit_audio_track(ctx)
        avatar_session = await _create_avatar_session(room_name, audio_source, loop)

        pipeline_bridge = PipelineBridge()
        asr_bridge = AsrBridge()
        utterance_buffer = UtteranceBuffer(silence_threshold_ms=500)

        try:
            track = await asyncio.wait_for(track_future, timeout=_TRACK_WAIT_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning(
                "room '%s': no remote audio track published within %.0fs, giving up",
                room_name,
                _TRACK_WAIT_TIMEOUT_S,
            )
            return

        audio_stream = _make_audio_stream(track)

        was_speech = False
        async for frame_event in audio_stream:
            # frame_event.frame.data is a memoryview cast to native int16
            # (format_probe.py Fact 1) — np.frombuffer (inside
            # to_pipeline_format) accepts any buffer-protocol object, not
            # just bytes, so no extra copy/conversion is needed here.
            chunk = to_pipeline_format(frame_event.frame.data)
            was_speech = await _process_frame(
                chunk,
                was_speech=was_speech,
                utterance_buffer=utterance_buffer,
                asr_bridge=asr_bridge,
                pipeline_bridge=pipeline_bridge,
                avatar_session=avatar_session,
                loop=loop,
                pending_tasks=pending_drive_tasks,
            )
    finally:
        # Review Focus "连接壳进程异常退出/房间意外断开": this must run on
        # every exit path — normal completion, an exception propagating out
        # of the try body, or this task being cancelled by the framework —
        # and a plain try/finally guarantees that in all three cases.
        #
        # Every synchronous, load-bearing cleanup step below runs *before*
        # audio_stream.aclose() (moved to be the very last thing this block
        # does — code review follow-up finding): an earlier version awaited
        # aclose() first, wrapped only in `except Exception`, and a
        # CancelledError delivered to this task specifically while it was
        # suspended inside that await (CancelledError is a BaseException,
        # not an Exception, since Python 3.8 — that except clause would not
        # have caught it) would have propagated out of `finally` before any
        # of the cleanup below ever ran. See module docstring's "aclose() is
        # now the last thing finally does" section.

        # Best-effort: stop any not-yet-started background drive task from
        # firing against a session that's about to be torn down. Cannot
        # stop one that is already running inside the executor thread (see
        # module docstring's "frame-iteration loop must not stall" section
        # for the known, separate, out-of-scope llm.py cancellation gap).
        _cancel_pending_drive_tasks(pending_drive_tasks)

        # Unconditional (not gated on `avatar_session is not None`):
        # session_manager.remove_session is a no-op when room_name was never
        # tracked (server/session_manager.py:91: `if sessionid in
        # self.sessions`), so calling it unconditionally also cleans up the
        # rare case where session_manager.create_session's own placeholder
        # (`self.sessions[sessionid] = None`, session_manager.py:76) got left
        # behind by an exception raised between that placeholder write and
        # the real assignment — a pre-existing gap in session_manager.py
        # itself, not something this task can fix without editing that file,
        # but calling remove_session unconditionally here at least reclaims
        # the dict entry instead of leaking it.
        session_manager.remove_session(room_name)
        # Released next (not first, unlike the brief's draft ordering): a
        # session for this room is still being torn down until this line, so
        # releasing the registry after tearing it down — rather than before —
        # shrinks (though, since remove_session is fire-and-forget with
        # respect to the render thread actually exiting, cannot fully close)
        # the window in which a retried dispatch for the same room could
        # start building a second session before the first one's resources
        # are actually released.
        _registry.release(room_name)
        # See module docstring's "the job does not end on its own" section:
        # the entrypoint coroutine returning is not, by itself, enough to
        # end the job — this call is what actually does. Placed before
        # aclose() too: the framework does not act on this until the
        # current task (i.e. this whole finally block) finishes anyway, so
        # there is no ordering hazard in calling it early.
        ctx.shutdown(reason="connection shell job finished")

        # Last: nothing below this point is load-bearing for cleanup
        # anymore, so aclose() failing (or a CancelledError landing here
        # specifically) no longer risks skipping anything that matters.
        if audio_stream is not None:
            try:
                await audio_stream.aclose()
            except Exception:
                logger.exception("room '%s': audio_stream.aclose() failed", room_name)
