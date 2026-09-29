"""
livekit_bridge.worker_app — launch script for this connection shell's
LiveKit Agent worker process.

Why a separate process at all
------------------------------
``app.py`` starts the pipeline's existing aiohttp/Flask HTTP service and a
livekit-agents worker are two different long-running programs (one serves
``/offer`` and friends forever via ``loop.run_forever()`` — app.py:217 —
the other registers with the LiveKit server and waits for job dispatch via
``cli.run_app``); neither returns control to the other, so they cannot live
in the same call stack. They *can* share the same Python venv/environment
(a requirement of the plan), which is all this script relies on: it imports
``app``, ``server.session_manager``, and ``config`` exactly as app.py does,
just without ever calling ``app.main()`` (which would also start the HTTP
server this worker does not need).

The model must be loaded exactly once, in ``__main__`` — not in
``prewarm_fnc`` (code review finding, corrected here)
--------------------------------------------------------------------------------
An earlier version of this file loaded the avatar model inside a
``prewarm_fnc`` registered on ``WorkerOptions``, reasoning that
``ServerOptions.prewarm_fnc`` (``worker.py:206``, "A function to perform
any necessary initialization before the job starts") sounded like the
framework's "once per worker process" hook. Reading the actual dispatch
machinery (``livekit/agents/ipc/proc_pool.py`` and
``livekit/agents/ipc/job_thread_executor.py``) shows that assumption was
wrong: there is no single long-lived "worker process" that runs every job.
Instead:

- ``ProcPool._proc_spawn_task`` (``proc_pool.py:204-260``) creates a *new*
  ``ThreadJobExecutor``/``ProcJobExecutor`` and calls its
  ``initialize()`` — which is what actually invokes ``prewarm_fnc``
  (``job_thread_executor.py:175-178``: ``initialize()`` sends an
  ``InitializeRequest`` that a freshly spawned thread's
  ``job_proc_lazy_main.thread_main`` receives and answers by calling
  ``self._initialize_process_fnc(self._job_proc)``,
  ``job_proc_lazy_main.py:230``) — for *every* executor it spawns, not once
  total.
- ``ThreadJobExecutor.launch_job`` (``job_thread_executor.py:263-266``)
  raises if the executor already has a running job — i.e. one executor
  instance (one spawned thread, one ``prewarm_fnc`` call) handles exactly
  one job, ever, then is discarded (``ProcPool._monitor_process_task``,
  ``proc_pool.py:274-280``, removes it from ``self._executors`` once its
  ``join()`` resolves).
- ``num_idle_processes`` only controls how many *pre-warmed, not-yet-
  assigned* executors are kept queued for lower dispatch latency
  (``ProcPool._acquire_proc``/``_main_task``,
  ``proc_pool.py:114-163,282-298``) — it defaults to 0 in dev mode
  (``worker.py:227-229``), and even in "production" mode
  (default up to 4) a burst of concurrent jobs beyond that count still
  spawns fresh executors on demand. This is true for both
  ``JobExecutorType.THREAD`` and ``JobExecutorType.PROCESS`` — the THREAD/
  PROCESS choice only changes *where* each job-scoped executor runs, not
  whether one gets created (and prewarmed) per job.

Consequence: registering the avatar-model load as ``prewarm_fnc`` reloads
the full multi-GB GPU model, and re-runs ``warm_up``, on *every single job
dispatch* — not once. Concretely, this would have: raced concurrent jobs'
reads of the shared ``app.opt``/``app.model``/``app.global_avatars``
globals against a new job's in-progress overwrite of the same globals; grown
GPU memory per concurrent/sequential job, since a job's live session still
references the model instance that was current when its session was built,
even after a later job's prewarm overwrites ``app.model`` with a fresh
instance; and very likely exceeded ``initialize_process_timeout`` (default
10s, ``worker.py:235``), since a real model load + ``warm_up`` call
realistically takes much longer than that, failing job initialization
outright.

The fix: load the model exactly once, in ``__main__``, *before*
``cli.run_app(...)`` is ever called — not through any per-job hook. Under
``JobExecutorType.THREAD`` (still pinned explicitly below — see that
option's own comment for why), every job's executor is a thread spawned in
this *same* OS process (``job_thread_executor.py:145-150``:
``threading.Thread(target=job_proc_lazy_main.thread_main, ...)``), and
Python's module cache (``sys.modules``) is shared across every thread in
one process — so the ``app.opt``/``app.model``/etc. this script sets once,
here, remain valid module-global state for every job's ``entrypoint()``
call afterward; nothing re-imports ``app`` fresh per job the way a genuine
subprocess under ``JobExecutorType.PROCESS`` would.

What was not exercised in this environment
----------------------------------------------
This dev machine is Windows. ``JobExecutorType.THREAD`` is pinned
explicitly below (not left at the platform default), so the thread-per-job
behavior described above is established by reading the source, not by
actually dispatching a real job — see ``livekit_bridge/__init__.py``'s
"Known limitations".
The ``JobExecutorType.PROCESS`` path (a genuine subprocess per job,
relevant if a future change ever needs it) was verified only by reading
``job_proc_executor.py``/``proc_pool.py``'s source, not by running it.

This script must be launched by running it directly (code review note)
------------------------------------------------------------------------------
``_load_and_wire_pipeline()`` only runs because it is called from this
file's own ``if __name__ == "__main__":`` block, which only executes when
this file is run directly (``python -m livekit_bridge.worker_app dev`` /
``python livekit_bridge/worker_app.py dev``). The framework's other
launchers (the ``lk agent`` CLI, or anything that does
``importlib.import_module("livekit_bridge.worker_app")`` instead of
running it as ``__main__``) would import this module without ever
executing that block, silently skipping the one-time model load — the
first job dispatched under such a launcher would then fail loudly inside
``entrypoint()``/``_create_avatar_session`` (``app.opt``/``app.model`` are
never set, so ``app.build_avatar_session`` would raise ``AttributeError``
or use stale/`None` globals), not silently produce a broken session. This
is an accepted constraint, not fixed here: a human deploying this worker
must launch it by running this file directly, not through a launcher that
imports it as a plain module.
"""

from __future__ import annotations

import importlib
import sys

from dotenv import load_dotenv
from livekit.agents import JobExecutorType, WorkerOptions, cli

import app
from livekit_bridge.entrypoint import entrypoint
from server.session_manager import session_manager
from utils.logger import logger

_AVATAR_MODULES = {
    "musetalk": "avatars.musetalk_avatar",
    "wav2lip": "avatars.wav2lip_avatar",
    "ultralight": "avatars.ultralight_avatar",
}


def _load_pipeline_config():
    """Parse LiveTalking's own CLI/YAML config (``config.parse_args()``)
    without letting it collide with livekit-agents' own CLI parsing of
    ``sys.argv`` (``python -m livekit_bridge.worker_app dev`` /
    ``start`` / ...).

    ``config.parse_args()`` calls argparse's *strict* ``parser.parse_args()``
    (``config.py:118``), which raises/exits on any argv token it does not
    recognize — including livekit-agents' own subcommand and flags. Both
    parsers read the same process-global ``sys.argv``, and neither was
    written with the other in mind, so this function temporarily substitutes
    a minimal argv (just the program name — every LiveTalking flag then
    falls back to its config.yaml/argparse default, matching how this
    worker is meant to be configured) while calling ``parse_args()``, then
    restores the real argv so ``livekit.agents.cli`` can parse its own
    subcommand afterwards.
    """
    from config import parse_args

    original_argv = sys.argv
    try:
        sys.argv = original_argv[:1]
        return parse_args()
    finally:
        sys.argv = original_argv


def _load_avatar_pipeline(opt):
    """Mirrors app.py's model-loading block (``app.py:130-154``) — this
    worker process needs its own loaded avatar model exactly like app.py's
    HTTP service does, since (see module docstring) it never runs
    ``app.main()`` itself.
    """
    if opt.model not in _AVATAR_MODULES:
        raise ValueError(f"unknown avatar model {opt.model!r}")

    avatar_mod = importlib.import_module(_AVATAR_MODULES[opt.model])
    load_model = avatar_mod.load_model
    load_avatar = avatar_mod.load_avatar
    warm_up = avatar_mod.warm_up

    global_avatars = {}
    if opt.model == "musetalk":
        model = load_model()
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size, model)
    elif opt.model == "wav2lip":
        model = load_model("./models/wav2lip.pth")
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size, model, 256)
    else:  # ultralight
        model = load_model(opt)
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size, global_avatars[opt.avatar_id], 160)

    return model, global_avatars, load_avatar


def _load_and_wire_pipeline() -> None:
    """Load the avatar model exactly once, before this worker ever starts
    accepting job dispatch, and wire ``app.build_avatar_session`` /
    ``session_manager`` for ``entrypoint()`` to use later.

    Must be called from ``__main__`` before ``cli.run_app(...)`` — see
    module docstring's "the model must be loaded exactly once" section for
    why this cannot be a ``prewarm_fnc`` registered on ``WorkerOptions``.

    ``app.build_avatar_session`` (``app.py:76-100``) reads
    ``opt``/``model``/``global_avatars``/``load_avatar`` as bare module
    globals of ``app.py``, resolved at call time — so setting them as
    attributes on the already-imported ``app`` module object here is what
    makes ``app.build_avatar_session`` callable correctly from
    ``entrypoint()`` for every subsequent job.
    """
    opt = _load_pipeline_config()
    # Deliberately not one of BaseAvatar's _output_modules keys
    # (avatars/base_avatar.py:140-145: webrtc/rtcpush/rtmp/virtualcam only),
    # so BaseAvatar.__init__ does not construct any of the existing
    # streamout.* output objects for a LiveKit-bridged session
    # (avatars/base_avatar.py:148-155) — entrypoint.py's
    # _create_avatar_session always overrides avatar_session.output with
    # streamout.livekit_output.LiveKitOutput itself, before the render
    # thread starts. The unavoidable cost: BaseAvatar.__init__ logs one
    # ERROR-level "Output transport livekit not found in map." line per
    # session. Harmless (self.output is simply left unset, and we set it
    # ourselves immediately after) but noisy; it cannot be silenced without
    # editing avatars/base_avatar.py, which this branch deliberately does
    # not touch (new code is confined to livekit_bridge/ and
    # streamout/livekit_output.py).
    opt.transport = "livekit"

    model, global_avatars, load_avatar = _load_avatar_pipeline(opt)

    app.opt = opt
    app.model = model
    app.global_avatars = global_avatars
    app.load_avatar = load_avatar

    session_manager.set_max_session(opt.max_session)
    session_manager.init_builder(app.build_avatar_session)

    logger.info(
        "livekit_bridge worker pipeline loaded: model=%s avatar_id=%s max_session=%d",
        opt.model,
        opt.avatar_id,
        opt.max_session,
    )


def _build_worker_options() -> WorkerOptions:
    return WorkerOptions(
        entrypoint_fnc=entrypoint,
        # Pinned explicitly rather than left at the platform default
        # (THREAD on Windows, PROCESS elsewhere — worker.py:146-150). This
        # pipeline already caps concurrency with a single in-process
        # capacity gate (session_manager.max_session) around one
        # GPU-resident model instance per process (the same singleton
        # discipline server/asr_server.py's _load_sensevoice() and
        # app.py's global_avatars caching already use), and (see module
        # docstring) the model is loaded exactly once, directly in
        # __main__, before this call. Under JobExecutorType.PROCESS, each
        # job's executor is a genuine subprocess with no shared memory
        # with this one-time load — it would need its own per-subprocess
        # loading strategy (e.g. its own prewarm_fnc, accepting that
        # tradeoff's redundant-load cost per idle-pool slot) to work at
        # all. THREAD keeps every job in this one process, sharing the one
        # model already loaded above, at the cost of losing livekit-agents'
        # normal per-job process-crash isolation (an unhandled crash in
        # one job's code can now affect others in the same process), and
        # it makes session_manager's unlocked capacity check-then-increment
        # a real cross-thread race. This tradeoff must be confirmed by a
        # human before shipping — see livekit_bridge/__init__.py's "Known
        # limitations" (THREAD vs PROCESS, and worker capacity signaling).
        job_executor_type=JobExecutorType.THREAD,
    )


if __name__ == "__main__":
    import torch.multiprocessing as mp

    # Mirrors app.py's own __main__ guard (app.py:232) — required by the
    # same avatar-model loading code path this script reuses.
    mp.set_start_method("spawn")
    load_dotenv()

    _load_and_wire_pipeline()

    cli.run_app(_build_worker_options())
