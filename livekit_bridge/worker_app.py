"""
livekit_bridge.worker_app — launch script for this connection shell's
LiveKit Agent worker process (Task 8's "启动脚本" deliverable).

Why a separate process at all
------------------------------
``app.py`` starts the pipeline's existing aiohttp/Flask HTTP service and a
livekit-agents worker are two different long-running programs (one serves
``/offer`` and friends forever via ``loop.run_forever()`` — app.py:217 —
the other registers with the LiveKit server and waits for job dispatch via
``cli.run_app``); neither returns control to the other, so they cannot live
in the same call stack. They *can* share the same Python venv/environment
(the brief's requirement), which is all this script relies on: it imports
``app``, ``server.session_manager``, and ``config`` exactly as app.py does,
just without ever calling ``app.main()`` (which would also start the HTTP
server this worker does not need).

Two real API facts, verified against the installed livekit-agents==1.8.3
source, drove this file's structure — both are explained in detail at their
point of use below:

1. ``ServerOptions.job_executor_type`` (``livekit/agents/worker.py:210``)
   defaults to ``JobExecutorType.THREAD`` on Windows and
   ``JobExecutorType.PROCESS`` on every other platform
   (``worker.py:146-150``) — i.e. on Linux, a real production deployment,
   each job normally runs in a genuine OS subprocess, not this script's own
   process. This script pins ``JobExecutorType.THREAD`` explicitly instead
   of taking the platform default — see ``prewarm``'s docstring for why.
2. ``ServerOptions.prewarm_fnc`` (``worker.py:206``, "A function to perform
   any necessary initialization before the job starts") is the framework's
   documented hook for exactly this task's "load the GPU model once, reuse
   it across jobs" requirement, and runs inside whichever process will
   actually execute jobs — so it is the correct (and, under
   ``JobExecutorType.PROCESS``, the *only* correct) place to do the
   avatar-model loading that ``app.py:main()`` normally does before
   starting its HTTP server.

What was not exercised in this environment
----------------------------------------------
This dev machine is Windows, so the THREAD executor path (this script's own
process runs both prewarm and every job) is the only one actually run here.
The PROCESS-executor path (a genuine subprocess per job-executor slot,
relevant to a Linux production deployment) was verified only by reading
worker.py's source, not by running it — see the Task 8 report.
"""

from __future__ import annotations

import importlib
import sys

from dotenv import load_dotenv
from livekit.agents import JobExecutorType, JobProcess, WorkerOptions, cli

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


def prewarm(proc: JobProcess) -> None:
    """Load the avatar model once per worker process and wire up
    ``app.build_avatar_session`` — see module docstring point 2.

    ``app.build_avatar_session`` (``app.py:76-100``) reads
    ``opt``/``model``/``global_avatars``/``load_avatar`` as bare module
    globals of ``app.py``, resolved at call time — so setting them as
    attributes on the already-imported ``app`` module object here, inside
    whichever process this hook runs in, is what makes
    ``app.build_avatar_session`` callable correctly from ``entrypoint()``
    later, regardless of executor type.
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
    # ourselves immediately after) but noisy — see the Task 8 report for why
    # this could not be silenced without editing avatars/base_avatar.py,
    # which is out of this task's scope.
    opt.transport = "livekit"

    model, global_avatars, load_avatar = _load_avatar_pipeline(opt)

    app.opt = opt
    app.model = model
    app.global_avatars = global_avatars
    app.load_avatar = load_avatar

    session_manager.set_max_session(opt.max_session)
    session_manager.init_builder(app.build_avatar_session)

    proc.userdata["opt"] = opt
    logger.info(
        "livekit_bridge worker prewarm complete: model=%s avatar_id=%s max_session=%d",
        opt.model,
        opt.avatar_id,
        opt.max_session,
    )


if __name__ == "__main__":
    import torch.multiprocessing as mp

    # Mirrors app.py's own __main__ guard (app.py:232) — required by the
    # same avatar-model loading code path this script reuses.
    mp.set_start_method("spawn")
    load_dotenv()

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            # Pinned explicitly rather than left at the platform default
            # (THREAD on Windows, PROCESS elsewhere — worker.py:146-150).
            # This pipeline already caps concurrency with a single
            # in-process capacity gate (session_manager.max_session) around
            # one GPU-resident model instance per process (the same
            # singleton discipline server/asr_server.py's
            # _load_sensevoice() and app.py's global_avatars caching already
            # use). JobExecutorType.PROCESS's own idle-process pool
            # (num_idle_processes, default up to 4) would instead prewarm
            # several independent subprocesses, each loading its own full
            # copy of the same GPU model — for a pipeline architected around
            # exactly one loaded model per process, that risks redundant
            # multi-GB model loads competing for the same GPU's VRAM.
            # THREAD keeps every job in this one process, sharing the one
            # model prewarm() already loaded, at the cost of losing
            # livekit-agents' normal per-job process-crash isolation (an
            # unhandled crash in one job's code can now affect others in the
            # same process). See the Task 8 report: this tradeoff should be
            # confirmed by a human before shipping, not treated as settled
            # by this comment alone.
            job_executor_type=JobExecutorType.THREAD,
        )
    )
