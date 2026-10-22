import threading
import unittest
from unittest.mock import MagicMock, patch

from tts.qwentts import PRESET_VOICES, QwenTTS


class _ImmediateEvent:
    """Stands in for the threading.Event the real class waits on for response.done."""

    def __init__(self):
        self.cleared = 0

    def clear(self):
        self.cleared += 1

    def set(self):
        pass

    def wait(self, timeout=None):
        return True


def make_tts(voice="Cherry", preset_only=True):
    """A QwenTTS without the network connection: only the state txt_to_audio uses."""
    tts = object.__new__(QwenTTS)
    tts.voice = voice
    tts._last_good_voice = voice
    tts._preset_only = preset_only
    tts._remainder = None
    tts._first_chunk = True
    tts._current_text = ""
    tts._current_textevent = {}
    tts._response_event = _ImmediateEvent()
    tts._tts_client = MagicMock()
    tts.parent = MagicMock()
    tts._open_connection = MagicMock()
    return tts


class VoiceListTest(unittest.TestCase):
    def test_voices_rejected_by_the_service_are_not_offered(self):
        # Rejected with "Invalid voice specified" by qwen3-tts-flash-realtime on 2026-10-15.
        for name in ("Daniel", "Aidan", "Vivienne", "Aria", "Alya", "Ravi"):
            self.assertNotIn(name, PRESET_VOICES)

    def test_default_voice_is_offered(self):
        self.assertIn("Cherry", PRESET_VOICES)


class TxtToAudioTest(unittest.TestCase):
    def test_unknown_voice_is_ignored_and_the_connection_kept(self):
        tts = make_tts()
        tts.txt_to_audio(("你好", {"tts": {"ref_file": "Alya"}}))
        self.assertEqual(tts.voice, "Cherry")
        tts._tts_client.close.assert_not_called()
        tts._tts_client.append_text.assert_called_once_with("你好")
        tts._tts_client.commit.assert_called_once()

    def test_valid_voice_switch_reopens_the_connection(self):
        tts = make_tts()
        tts.txt_to_audio(("你好", {"tts": {"ref_file": "Ethan"}}))
        self.assertEqual(tts.voice, "Ethan")
        tts._tts_client.close.assert_called_once()
        tts._open_connection.assert_called_once()
        tts._tts_client.append_text.assert_called_once_with("你好")

    def test_closed_connection_is_reopened_and_the_sentence_is_not_lost(self):
        tts = make_tts()
        tts._tts_client.append_text.side_effect = [ConnectionError("closed"), None]
        tts.txt_to_audio(("你好", {}))
        tts._tts_client.close.assert_called_once()
        tts._open_connection.assert_called_once()
        self.assertEqual(tts._tts_client.append_text.call_count, 2)
        tts._tts_client.commit.assert_called_once()
        tts.parent.send_msg.assert_not_called()

    def test_failure_that_survives_a_reconnect_is_reported_to_the_page(self):
        tts = make_tts()
        tts._tts_client.append_text.side_effect = ConnectionError("closed")
        tts.txt_to_audio(("你好", {}))
        tts.parent.send_msg.assert_called_once()
        self.assertIn("tts_error", tts.parent.send_msg.call_args[0][0])

    def test_clone_model_does_not_apply_the_preset_list(self):
        tts = make_tts(voice="cloned-voice-id", preset_only=False)
        tts.txt_to_audio(("你好", {"tts": {"ref_file": "another-cloned-id"}}))
        self.assertEqual(tts.voice, "another-cloned-id")


class OpenConnectionRetryTest(unittest.TestCase):
    def make(self):
        tts = object.__new__(QwenTTS)
        tts.voice = "Cherry"
        tts._tts_client = MagicMock()
        return tts

    def test_slow_handshake_is_retried_until_it_succeeds(self):
        tts = self.make()
        tts._tts_client.connect.side_effect = [TimeoutError("5s"), TimeoutError("5s"), None]
        with patch("tts.qwentts.time.sleep"):
            tts._open_connection()
        self.assertEqual(tts._tts_client.connect.call_count, 3)
        tts._tts_client.update_session.assert_called_once()

    def test_gives_up_after_the_last_attempt_with_the_original_error(self):
        tts = self.make()
        tts._tts_client.connect.side_effect = TimeoutError("5s")
        with patch("tts.qwentts.time.sleep"):
            with self.assertRaises(TimeoutError):
                tts._open_connection()
        self.assertEqual(tts._tts_client.connect.call_count, 3)

    def test_other_errors_are_not_retried(self):
        tts = self.make()
        tts._tts_client.connect.side_effect = ValueError("bad config")
        with self.assertRaises(ValueError):
            tts._open_connection()
        self.assertEqual(tts._tts_client.connect.call_count, 1)


if __name__ == "__main__":
    unittest.main()
