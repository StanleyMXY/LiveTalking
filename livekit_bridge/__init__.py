"""
livekit_bridge — connection shell that lets a LiveKit-based client talk to
the LiveTalking avatar pipeline over a LiveKit room, instead of through
LiveTalking's own aiortc WebRTC transport. It is a separate worker process
(``worker_app.py``) that loads the pipeline once and runs one job per room.

What is implemented
-------------------
- ``format_probe.py`` — verified facts about LiveKit's audio frame format
  and the pipeline's ASR call convention, as constants the other modules
  import.
- ``audio_format.py`` — converts LiveKit's int16 PCM frames to the
  pipeline's float32 format.
- ``utterance_buffer.py`` — energy-threshold VAD and utterance buffering:
  emits an utterance after 500ms of silence, or force-emits at a 30s cap.
- ``barge_in_detector.py`` — reports a speech onset only after 240ms of
  sustained above-threshold audio, so a single noise frame cannot interrupt
  the avatar.
- ``asr_bridge.py`` — runs the pipeline's own SenseVoice singleton
  (``server/asr_server._run_inference``) and filters SenseVoice's
  non-speech emoji markers, returning ``""`` for noise-only results.
- ``pipeline_bridge.py`` — drives the pipeline's real ``llm_response`` with
  transcribed text and implements interrupt/supersede with a generation
  fence, so a superseded LLM call can no longer reach the session.
- ``session_registry.py`` — deduplicates avatar sessions by room name
  (per worker process only — see below).
- ``streamout/livekit_output.py`` (outside this package) — the
  ``BaseOutput`` that publishes synthesized audio to the LiveKit room.
- ``entrypoint.py`` — the LiveKit Agent job entrypoint that wires all of
  the above together: connect, publish the avatar's audio track, create the
  avatar session, run the frame loop (barge-in detection, utterance
  buffering, ASR, most-recent-utterance-wins supersede), and clean up on
  every exit path.
- ``worker_app.py`` — the worker launch script.

Known limitations / must verify before production
-------------------------------------------------
Nothing below is fixed on this branch. Each item is either unverified or
deliberately deferred; none may be assumed to work.

1. **No live end-to-end test has been run.** No LiveKit server, test
   client, or running LiveTalking service was available in the environment
   this was built in. Every SDK interaction was verified by reading the
   installed ``livekit==1.1.18`` / ``livekit-agents==1.8.3`` source and
   covered by unit tests with fakes; none of it has run against a real
   room. A human must do a live end-to-end pass (join, speak, get an
   answer, barge in, reconnect, leave) before this ships.

2. **Superseded LLM calls still run to completion.** ``llm.py`` has no
   cancellation token. When a response is superseded, the generation fence
   (``pipeline_bridge.py``) makes every further ``put_msg_txt`` from the old
   ``llm_response`` call a no-op, but the call itself keeps streaming from
   the LLM provider until it finishes, consuming tokens and an executor
   thread. Fixing this needs a cancellation checkpoint inside
   ``llm_response``'s streaming loop, i.e. a change to ``llm.py``, which is
   outside this branch's scope.

3. **``JobExecutorType.THREAD`` vs ``PROCESS`` needs human sign-off.**
   ``worker_app.py`` pins ``THREAD`` so every job shares the one model
   loaded in ``__main__``. Cost: livekit-agents' per-job process isolation is
   lost (a crash in one job can take down the others in the process), and
   pipeline globals that were written for one event loop are now shared by
   several job threads — see item 6. ``PROCESS`` would need its own
   per-subprocess model-loading strategy. The choice must be confirmed by a
   human before shipping; it is not settled.

4. **The one-time model load only works when ``worker_app.py`` is run
   directly** (``python -m livekit_bridge.worker_app <cmd>`` or
   ``python livekit_bridge/worker_app.py <cmd>``). It lives in the file's
   ``if __name__ == "__main__":`` block, so a launcher that imports the
   module instead (the ``lk agent`` CLI, ``importlib.import_module``) skips
   it, and the first job fails in ``_create_avatar_session``.

5. **``ctx.job.metadata`` is an unestablished cross-project contract.** The
   field exists on the ``Job`` proto, but what the dispatcher puts in it
   (a bare ``account_id``? a JSON object?) is not defined anywhere, and the
   Java-side dispatcher that would define it does not exist yet.
   ``entrypoint.py`` only logs the raw string; nothing parses or uses it,
   so no ``account_id`` reaches the pipeline. This must be specified jointly
   with the Java backend and then wired through.

6. **Deferred: worker capacity signaling.** ``worker_app.py`` sets no
   ``load_fnc`` or ``request_fnc`` on ``WorkerOptions``, so the LiveKit
   dispatcher cannot know this worker is full. The only capacity limit is
   ``session_manager.max_session``, and it is checked inside
   ``session_manager.create_session`` — which ``entrypoint.py`` calls only
   *after* ``ctx.connect()`` and after publishing the avatar's audio track.
   A job that arrives when the worker is full therefore joins the room,
   publishes a track, then fails with ``MaxSessionError`` and leaves (the
   ``finally`` cleanup): the client sees the agent appear and vanish, and
   the dispatch is lost instead of going to a worker with free capacity.
   In addition, that check is not synchronized: the
   count skips the ``None`` placeholder written for a session still being
   built (``server/session_manager.py:68-76``), and the build itself is
   awaited in an executor (``:79-81``), so creates that overlap in time can
   all pass the check. Under ``THREAD`` mode (item 3) each job runs its own
   event loop on its own thread against the same ``session_manager``
   singleton, so concurrent dispatches genuinely overlap and can admit more
   sessions than ``max_session`` — each holding GPU memory, which can
   exhaust the GPU. The fix is a capacity-aware ``load_fnc``/``request_fnc``
   on the worker plus a lock-guarded capacity reservation taken *before*
   ``ctx.connect()``. Deferred to its own follow-up because it is
   deployment capacity management, not connection wiring — but it must be
   decided together with item 3 before shipping.

7. **Deferred: session lifetime is tied to one audio track.** The
   entrypoint subscribes to the *first* remote audio track and runs the
   session for as long as that track's ``AudioStream`` lasts. When that
   stream ends — which also happens on a client network reconnect, or when
   the client unpublishes and republishes its microphone — the entrypoint
   tears the avatar session down and calls ``ctx.shutdown()``, permanently.
   It does not wait for, or relink to, a new track from the same
   participant. So an ordinary network blip ends the conversation instead
   of resuming it, contrary to this project's expectation that reconnection
   is transparent. The fix is to scope the session to the room/participant
   rather than to one track subscription: on end-of-stream, check
   ``participant.disconnect_reason`` to tell a deliberate leave from a
   transient drop, and on a transient drop wait a bounded time for a newly
   subscribed audio track from the same participant identity and relink
   the frame loop to it. Deferred to its own task and design because it
   restructures the entrypoint's whole session lifecycle.

8. **Room dedup is per worker process only.** ``BridgeSessionRegistry`` is
   an in-memory ``set``. With more than one worker process or host (the
   normal way LiveKit workers are scaled), or after a worker restart, a
   retried dispatch for a room that lands on a different process is not
   detected and can create a duplicate avatar session in the same room.
   Guaranteeing no duplicates at scale needs a shared registry (e.g. Redis,
   or the Java backend) or idempotent dispatch on the dispatching side
   (check for an existing dispatch before ``CreateDispatch``).

9. **Deferred: the VAD is an energy threshold, not a speech model.**
   ``utterance_buffer.py`` and ``barge_in_detector.py`` classify frames by
   mean absolute amplitude (``_ENERGY_THRESHOLD = 0.05``). This was a
   deliberate, reviewed choice for this branch, but it cannot tell speech
   from other loud sound: sustained noise longer than 240ms (a long cough,
   music) still counts as a barge-in and interrupts the avatar, and the
   interrupted answer is not replaced because the noise transcribes to a
   filtered-out marker. Steady noise above the threshold also delays every
   utterance boundary until the 30s cap. The connection-shell plan (AvatarApp
   repo, ``docs/superpowers/plans/2026-09-29-livekit-connection-shell.md``,
   Task 3) already anticipates swapping in LiveKit's Silero VAD
   (``livekit.plugins.silero``) behind the same ``feed``/``reset``
   interface; that swap is the deferred follow-up.
"""
