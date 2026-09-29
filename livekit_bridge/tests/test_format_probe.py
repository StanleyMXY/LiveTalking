"""
Tests for livekit_bridge.format_probe.

These are not feature tests — they exist to (a) fail loudly if the Task 1
constants are ever left unresolved (None) again, and (b) act as a
regression guard: if a future `livekit` package upgrade or an edit to
server/asr_server.py changes the facts recorded in format_probe.py, these
tests catch the drift instead of letting a stale, silently-wrong constant
be relied on by Task 2 / Task 6.

Run with:
    python -m unittest livekit_bridge.tests.test_format_probe
"""

import inspect
import unittest

from livekit.rtc.audio_frame import AudioFrame
from livekit.rtc.audio_stream import AudioStream

from livekit_bridge import format_probe
from server import asr_server


class FormatProbeConstantsTestCase(unittest.TestCase):
    """The three constants must be real verified values, never None."""

    def test_constants_are_not_none(self):
        self.assertIsNotNone(format_probe.LIVEKIT_AUDIO_SAMPLE_RATE)
        self.assertIsNotNone(format_probe.PIPELINE_ASR_SAMPLE_RATE)
        self.assertIsNotNone(format_probe.NEEDS_RESAMPLING)

    def test_constants_have_expected_types(self):
        self.assertIsInstance(format_probe.LIVEKIT_AUDIO_SAMPLE_RATE, int)
        self.assertIsInstance(format_probe.PIPELINE_ASR_SAMPLE_RATE, int)
        self.assertIsInstance(format_probe.NEEDS_RESAMPLING, bool)

    def test_needs_resampling_is_consistent_with_the_two_rates(self):
        # NEEDS_RESAMPLING must not be an independently-typed-in guess: it
        # has to agree with the two rates it is derived from.
        expected = format_probe.LIVEKIT_AUDIO_SAMPLE_RATE != format_probe.PIPELINE_ASR_SAMPLE_RATE
        self.assertEqual(format_probe.NEEDS_RESAMPLING, expected)

    def test_pipeline_rate_matches_asr_server_constant(self):
        # Guards against format_probe.py and server/asr_server.py drifting
        # apart on what "the pipeline's ASR sample rate" actually is.
        self.assertEqual(format_probe.PIPELINE_ASR_SAMPLE_RATE, asr_server.SAMPLE_RATE)


class LiveKitAudioStreamDefaultTestCase(unittest.TestCase):
    """
    Regression guard on the installed `livekit` package: if a future SDK
    version changes AudioStream's default sample_rate, this fails instead
    of leaving LIVEKIT_AUDIO_SAMPLE_RATE silently wrong.
    """

    def test_audio_stream_default_sample_rate_matches_recorded_constant(self):
        sig = inspect.signature(AudioStream.__init__)
        default_sample_rate = sig.parameters["sample_rate"].default
        self.assertEqual(default_sample_rate, format_probe.LIVEKIT_AUDIO_SAMPLE_RATE)

    def test_audio_stream_default_num_channels_is_mono(self):
        sig = inspect.signature(AudioStream.__init__)
        default_num_channels = sig.parameters["num_channels"].default
        self.assertEqual(default_num_channels, 1)


class LiveKitAudioFrameFormatTestCase(unittest.TestCase):
    """Exercises the real installed SDK to confirm the documented int16 format."""

    def test_audio_frame_data_is_int16_samples(self):
        frame = AudioFrame.create(sample_rate=16000, num_channels=1, samples_per_channel=10)
        # AudioFrame.data is documented as memoryview cast to native signed
        # short ("h" == int16); confirm the actual format code, not just
        # the docstring.
        self.assertEqual(frame.data.format, "h")
        self.assertEqual(len(frame.data), 10)

    def test_audio_frame_exposes_requested_rate_and_channels(self):
        frame = AudioFrame.create(sample_rate=16000, num_channels=1, samples_per_channel=160)
        self.assertEqual(frame.sample_rate, 16000)
        self.assertEqual(frame.num_channels, 1)
        self.assertEqual(frame.samples_per_channel, 160)


class AsrInferenceCallConventionTestCase(unittest.TestCase):
    """
    Confirms the real inference function's signature and the
    singleton-model-holder pattern, without ever triggering an actual
    model load (this test must not download or run SenseVoice).
    """

    def test_run_inference_signature_matches_documented_convention(self):
        sig = inspect.signature(asr_server._run_inference)
        self.assertEqual(list(sig.parameters.keys()), ["audio_float32", "sample_rate", "use_itn"])

    def test_model_is_held_as_a_module_level_singleton(self):
        # Static check, deliberately avoids calling _load_sensevoice()
        # (which would load/download the real model).
        self.assertTrue(hasattr(asr_server, "_sensevoice_model"))
        source = inspect.getsource(asr_server._load_sensevoice)
        self.assertIn("global _sensevoice_model", source)
        self.assertIn("_sensevoice_load_lock", source)

    def test_run_inference_reuses_the_singleton_loader(self):
        source = inspect.getsource(asr_server._run_inference)
        self.assertIn("_load_sensevoice()", source)


if __name__ == "__main__":
    unittest.main()
