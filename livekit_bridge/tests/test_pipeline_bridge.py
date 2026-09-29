import threading
from unittest.mock import MagicMock, patch

from livekit_bridge.pipeline_bridge import PipelineBridge


def test_empty_text_does_not_drive_avatar():
    bridge = PipelineBridge()
    fake_session = MagicMock()

    with patch("livekit_bridge.pipeline_bridge.llm_response") as fake_llm_response:
        bridge.drive_with_text(fake_session, "", bridge.begin_generation())
        bridge.drive_with_text(fake_session, "   ", bridge.begin_generation())

        fake_llm_response.assert_not_called()
    fake_session.assert_not_called()


def test_nonempty_text_drives_avatar():
    bridge = PipelineBridge()
    fake_session = MagicMock()

    # Code review follow-up finding: drive_with_text no longer passes the
    # raw avatar_session straight through to llm_response — it passes a
    # thin per-call proxy instead (see pipeline_bridge.py's module
    # docstring's "Generation fencing" section for why). Assert the real
    # contract precisely: llm_response is called with the text, *some*
    # object (not necessarily fake_session itself) whose .opt/.put_msg_txt
    # forward to the real session, and datainfo={} (server/routes.py:72-76
    # for type=='chat'; llm.py:45's real signature).
    with patch("livekit_bridge.pipeline_bridge.llm_response") as fake_llm_response:
        bridge.drive_with_text(fake_session, "你好", bridge.begin_generation())

        fake_llm_response.assert_called_once()
        call_args = fake_llm_response.call_args.args
        assert call_args[0] == "你好"
        proxy = call_args[1]
        assert call_args[2] == {}

    fake_session.opt = "the-real-opt"
    assert proxy.opt == "the-real-opt"

    proxy.put_msg_txt("hello", {"tts": {}})
    fake_session.put_msg_txt.assert_called_once_with("hello", {"tts": {}})


def test_interrupt_calls_flush_talk():
    bridge = PipelineBridge()
    fake_session = MagicMock()

    bridge.interrupt(fake_session)

    fake_session.flush_talk.assert_called_once()


# ─── Generation fencing: an orphaned, still-executing llm_response call ──
# must not be able to reach the real avatar_session once superseded.


def test_superseded_call_cannot_reach_real_session_after_new_drive_call():
    # This is the load-bearing case the previous fix (asyncio Task
    # cancellation alone) could not cover: llm_response has already
    # reached its streaming loop and is calling put_msg_txt directly
    # (simulating an orphaned executor thread that keeps running after the
    # asyncio-level task that dispatched it was cancelled) — a *second*
    # drive_with_text call (the superseding one) must make every further
    # put_msg_txt call from the *first* llm_response invocation a no-op.
    bridge = PipelineBridge()
    fake_session = MagicMock()
    captured_proxies = []

    def fake_llm_response(message, proxy, datainfo):
        captured_proxies.append(proxy)

    with patch("livekit_bridge.pipeline_bridge.llm_response", side_effect=fake_llm_response):
        bridge.drive_with_text(fake_session, "first utterance", bridge.begin_generation())
        first_proxy = captured_proxies[0]

        # The first call's put_msg_txt still works before anything
        # supersedes it.
        first_proxy.put_msg_txt("fragment before supersede", {})
        fake_session.put_msg_txt.assert_called_once_with("fragment before supersede", {})

        # A second, superseding utterance completes and starts its own
        # drive_with_text call — simulating entrypoint.py's
        # _supersede_in_flight_response detecting an in-flight response and
        # driving a new one.
        bridge.drive_with_text(fake_session, "second utterance", bridge.begin_generation())
        second_proxy = captured_proxies[1]

    # The first (now-stale) proxy's put_msg_txt must be a no-op from here
    # on, even though nothing about the first llm_response call itself was
    # ever told to stop (it is, by construction, still "running" from the
    # proxy's point of view — this is exactly the orphaned-executor-thread
    # scenario).
    fake_session.put_msg_txt.reset_mock()
    first_proxy.put_msg_txt("fragment after supersede", {})
    fake_session.put_msg_txt.assert_not_called()

    # The second (current) proxy's put_msg_txt still reaches the real
    # session normally.
    second_proxy.put_msg_txt("new response fragment", {})
    fake_session.put_msg_txt.assert_called_once_with("new response fragment", {})


def test_interrupt_alone_also_supersedes_the_current_generation():
    # Every interrupt() — not just a superseding drive_with_text call —
    # must invalidate whatever generation is currently in flight (the
    # mid-speech barge-in path in entrypoint.py calls interrupt() without
    # necessarily starting a new drive_with_text call in the same breath).
    bridge = PipelineBridge()
    fake_session = MagicMock()
    captured_proxies = []

    def fake_llm_response(message, proxy, datainfo):
        captured_proxies.append(proxy)

    with patch("livekit_bridge.pipeline_bridge.llm_response", side_effect=fake_llm_response):
        bridge.drive_with_text(fake_session, "utterance", bridge.begin_generation())
        proxy = captured_proxies[0]

    bridge.interrupt(fake_session)

    proxy.put_msg_txt("fragment after interrupt", {})
    fake_session.put_msg_txt.assert_not_called()


def test_proxy_put_msg_txt_defaults_datainfo_to_empty_dict():
    # Matches the real avatar_session.put_msg_txt's own default
    # (avatars/base_avatar.py: datainfo:dict={}) and the fact that
    # put_msg_txt's implementation immediately does
    # datainfo.get('tts', {})... — passing None through would crash it.
    bridge = PipelineBridge()
    fake_session = MagicMock()
    captured_proxies = []

    def fake_llm_response(message, proxy, datainfo):
        captured_proxies.append(proxy)

    with patch("livekit_bridge.pipeline_bridge.llm_response", side_effect=fake_llm_response):
        bridge.drive_with_text(fake_session, "utterance", bridge.begin_generation())
        proxy = captured_proxies[0]

    proxy.put_msg_txt("msg", None)

    fake_session.put_msg_txt.assert_called_once_with("msg", {})


# ─── Round 4: cross-thread races in the generation fence ────────────────
#
# Every test above runs the proxy and interrupt() on one thread in strict
# sequence, so none of them could observe a race between an executor
# thread's put_msg_txt and the event-loop thread's interrupt(). The tests
# below force each specific interleaving with threading.Events.


class _QueueLikeSession:
    """Minimal stand-in for the real avatar_session whose put_msg_txt can
    be paused *on entry*, before it enqueues — i.e. at the exact point the
    proxy has already decided to forward but the fragment has not yet
    landed in the queue. flush_talk() clears that queue, like the real
    BaseTTS.flush_talk (tts/base_tts.py:30-32)."""

    def __init__(self):
        self.queue = []
        self.forward_entered = threading.Event()
        self.release_forward = threading.Event()
        self.pause_next_forward = False
        self.flush_calls = 0

    def put_msg_txt(self, msg, datainfo):
        if self.pause_next_forward:
            self.pause_next_forward = False
            self.forward_entered.set()
            assert self.release_forward.wait(timeout=5), "test never released forward"
        self.queue.append(msg)

    def flush_talk(self):
        self.flush_calls += 1
        self.queue.clear()


def test_interrupt_cannot_interleave_between_generation_check_and_forward():
    # Gap 1: the generation check and the real forward must be one atomic
    # critical section. Otherwise a stale executor thread whose check
    # passed just before interrupt() can have its forward land *after*
    # interrupt()'s bump-and-flush, i.e. in the freshly flushed queue,
    # ahead of the next answer. Against the pre-fix code (check under the
    # lock, forward outside it), interrupt() here runs to completion while
    # the forward is paused, then the paused forward appends — leaving
    # "stale fragment" in the queue — so this test fails.
    bridge = PipelineBridge()
    session = _QueueLikeSession()
    captured_proxies = []

    def fake_llm_response(message, proxy, datainfo):
        captured_proxies.append(proxy)

    with patch("livekit_bridge.pipeline_bridge.llm_response", side_effect=fake_llm_response):
        bridge.drive_with_text(session, "utterance", bridge.begin_generation())
    proxy = captured_proxies[0]

    # Stale executor thread: its generation is still current when it calls
    # put_msg_txt, so it passes the check and enters the real forward —
    # where it pauses.
    session.pause_next_forward = True
    stale_thread = threading.Thread(target=proxy.put_msg_txt, args=("stale fragment", {}))
    stale_thread.start()
    assert session.forward_entered.wait(timeout=5)

    # interrupt() fires from another thread (the event loop, in production)
    # while that forward is in progress.
    interrupt_done = threading.Event()

    def run_interrupt():
        bridge.interrupt(session)
        interrupt_done.set()

    interrupt_thread = threading.Thread(target=run_interrupt)
    interrupt_thread.start()

    # interrupt() must not be able to bump + flush while the forward it
    # would race with is still in progress.
    interrupt_blocked = not interrupt_done.wait(timeout=0.2)

    session.release_forward.set()
    stale_thread.join(timeout=5)
    interrupt_thread.join(timeout=5)
    assert interrupt_done.is_set()

    # The user-visible symptom first: the forward must have completed
    # before interrupt()'s flush, so the flush removed it; nothing stale
    # survives into the next answer's queue.
    assert session.queue == []
    assert session.flush_calls == 1
    # And the mechanism: interrupt() was held off while the forward ran.
    assert interrupt_blocked, "interrupt() bumped+flushed while a forward was mid-flight"

    # And any further attempt from the stale call is fenced off.
    proxy.put_msg_txt("later stale fragment", {})
    assert session.queue == []


def test_generation_allocated_before_dispatch_is_invalidated_by_interrupt_before_worker_runs():
    # Gap 2: the generation must be decided on the event-loop thread
    # *before* drive_with_text is dispatched to an executor worker. The
    # dangerous interleaving is: dispatch decided -> interrupt() on the
    # event loop -> worker thread only then starts drive_with_text. When
    # drive_with_text allocated its own generation on the worker (pre-fix),
    # it came out *newer* than the interrupt's bump and ran unfenced. With
    # the generation pre-allocated, the interrupt marks it stale before the
    # worker ever uses it. (The entrypoint-side half — that allocation
    # really happens on the event-loop thread, before dispatch — is
    # test_entrypoint.py's
    # test_drive_generation_is_allocated_on_event_loop_before_executor_dispatch.)
    bridge = PipelineBridge()
    session = _QueueLikeSession()

    def fake_llm_response(message, proxy, datainfo):
        proxy.put_msg_txt("fragment from a call superseded before it started", {})

    generation = bridge.begin_generation()  # event loop: dispatch decided
    bridge.interrupt(session)  # event loop: interrupt lands before the worker runs

    with patch("livekit_bridge.pipeline_bridge.llm_response", side_effect=fake_llm_response):
        worker = threading.Thread(
            target=bridge.drive_with_text, args=(session, "utterance", generation)
        )
        worker.start()
        worker.join(timeout=5)

    assert session.queue == []


def test_older_generation_running_on_another_thread_is_fenced_by_newer_one():
    # Cross-thread version of the supersede case: the older call's
    # put_msg_txt runs on its own thread after a newer generation was
    # allocated; only the newer call's fragments reach the session.
    bridge = PipelineBridge()
    session = _QueueLikeSession()
    captured_proxies = []

    def fake_llm_response(message, proxy, datainfo):
        captured_proxies.append(proxy)

    with patch("livekit_bridge.pipeline_bridge.llm_response", side_effect=fake_llm_response):
        bridge.drive_with_text(session, "first", bridge.begin_generation())
        bridge.drive_with_text(session, "second", bridge.begin_generation())
    old_proxy, new_proxy = captured_proxies

    old_thread = threading.Thread(target=old_proxy.put_msg_txt, args=("old", {}))
    old_thread.start()
    old_thread.join(timeout=5)
    new_proxy.put_msg_txt("new", {})

    assert session.queue == ["new"]
