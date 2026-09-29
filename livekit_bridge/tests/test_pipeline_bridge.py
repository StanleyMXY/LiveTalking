from unittest.mock import MagicMock, patch

from livekit_bridge.pipeline_bridge import PipelineBridge


def test_empty_text_does_not_drive_avatar():
    bridge = PipelineBridge()
    fake_session = MagicMock()

    with patch("livekit_bridge.pipeline_bridge.llm_response") as fake_llm_response:
        bridge.drive_with_text(fake_session, "")
        bridge.drive_with_text(fake_session, "   ")

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
        bridge.drive_with_text(fake_session, "你好")

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
        bridge.drive_with_text(fake_session, "first utterance")
        first_proxy = captured_proxies[0]

        # The first call's put_msg_txt still works before anything
        # supersedes it.
        first_proxy.put_msg_txt("fragment before supersede", {})
        fake_session.put_msg_txt.assert_called_once_with("fragment before supersede", {})

        # A second, superseding utterance completes and starts its own
        # drive_with_text call — simulating entrypoint.py's
        # _supersede_in_flight_response detecting an in-flight response and
        # driving a new one.
        bridge.drive_with_text(fake_session, "second utterance")
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
        bridge.drive_with_text(fake_session, "utterance")
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
        bridge.drive_with_text(fake_session, "utterance")
        proxy = captured_proxies[0]

    proxy.put_msg_txt("msg", None)

    fake_session.put_msg_txt.assert_called_once_with("msg", {})
