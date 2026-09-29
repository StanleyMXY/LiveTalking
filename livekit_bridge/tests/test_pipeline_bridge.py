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

    # 精确断言：真实调用是 llm.llm_response(message, avatar_session, datainfo)
    # (server/routes.py:72-76 对 type=='chat' 分支的真实调用；llm.py:45 的真实签名)
    with patch("livekit_bridge.pipeline_bridge.llm_response") as fake_llm_response:
        bridge.drive_with_text(fake_session, "你好")

        fake_llm_response.assert_called_once_with("你好", fake_session, {})


def test_interrupt_calls_flush_talk():
    bridge = PipelineBridge()
    fake_session = MagicMock()

    bridge.interrupt(fake_session)

    fake_session.flush_talk.assert_called_once()
