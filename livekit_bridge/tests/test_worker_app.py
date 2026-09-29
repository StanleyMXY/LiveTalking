"""
Tests for livekit_bridge.worker_app.

Code review finding C1 (fixed here): an earlier version of worker_app.py
registered the avatar-model load as a ``prewarm_fnc`` on ``WorkerOptions``,
which — per ``livekit.agents.ipc.proc_pool``/``job_thread_executor`` (see
worker_app.py's module docstring for the full citation trail) — actually
runs once per *job-scoped executor*, not once per worker process, since
``ProcPool`` spawns a fresh executor (and calls its ``initialize()``, which
invokes ``prewarm_fnc``) for essentially every job. The fix moves the
one-time model load to ``__main__``, before ``cli.run_app(...)`` is ever
called, and stops registering any ``prewarm_fnc`` at all.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from livekit.agents import JobExecutorType
from livekit.agents.worker import _default_setup_fnc

from livekit_bridge import worker_app
from livekit_bridge.entrypoint import entrypoint


def test_build_worker_options_registers_no_prewarm_fnc():
    # This is the direct regression guard for C1: if model loading were
    # ever moved back into a prewarm_fnc, this assertion would catch it —
    # WorkerOptions.prewarm_fnc must stay at the framework's own no-op
    # default, because model loading now happens once in __main__ instead.
    options = worker_app._build_worker_options()

    assert options.prewarm_fnc is _default_setup_fnc
    assert options.entrypoint_fnc is entrypoint
    assert options.job_executor_type == JobExecutorType.THREAD


def test_load_and_wire_pipeline_sets_app_globals_and_session_manager():
    fake_opt = SimpleNamespace(
        model="wav2lip", avatar_id="avatar-1", max_session=3, transport="webrtc"
    )
    fake_model = MagicMock()
    fake_global_avatars = {"avatar-1": MagicMock()}
    fake_load_avatar = MagicMock()

    with (
        patch.object(worker_app, "_load_pipeline_config", return_value=fake_opt),
        patch.object(
            worker_app,
            "_load_avatar_pipeline",
            return_value=(fake_model, fake_global_avatars, fake_load_avatar),
        ) as fake_load_avatar_pipeline,
        patch.object(worker_app.session_manager, "set_max_session") as fake_set_max,
        patch.object(worker_app.session_manager, "init_builder") as fake_init_builder,
    ):
        worker_app._load_and_wire_pipeline()

    # opt.transport is forced to a value BaseAvatar's _output_modules does
    # not recognize (see worker_app.py's _load_and_wire_pipeline docstring)
    # so it never constructs a webrtc/rtmp/virtualcam output for a
    # LiveKit-bridged session.
    assert fake_opt.transport == "livekit"
    fake_load_avatar_pipeline.assert_called_once_with(fake_opt)

    assert worker_app.app.opt is fake_opt
    assert worker_app.app.model is fake_model
    assert worker_app.app.global_avatars is fake_global_avatars
    assert worker_app.app.load_avatar is fake_load_avatar

    fake_set_max.assert_called_once_with(3)
    fake_init_builder.assert_called_once_with(worker_app.app.build_avatar_session)


def test_load_pipeline_config_restores_sys_argv_around_the_strict_parse():
    import sys

    original_argv = ["worker_app.py", "dev", "--room", "some-room"]
    with patch.object(sys, "argv", original_argv):
        opt = worker_app._load_pipeline_config()

        # config.parse_args()'s final parser.parse_args() call is strict
        # (config.py:118) and would raise on 'dev'/'--room' if they were
        # still in sys.argv while it ran — reaching this line at all means
        # they were not.
        assert opt is not None
        assert sys.argv == original_argv
