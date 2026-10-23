import io
import os
import subprocess
import tempfile
import unittest

import numpy as np
import soundfile as sf

from server import media_utils
from server.media_utils import MediaDecodeError, limit_image_side, to_wav_bytes

FFMPEG = media_utils._ffmpeg_exe()


def tone_wav_bytes(seconds=1.0, rate=24000):
    t = np.arange(int(seconds * rate)) / rate
    buf = io.BytesIO()
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 220 * t)).astype('float32'), rate, format='WAV')
    return buf.getvalue()


def encode(wav: bytes, ext: str, *codec_args) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, 'in.wav'), os.path.join(tmp, f'out.{ext}')
        open(src, 'wb').write(wav)
        subprocess.run([FFMPEG, '-y', '-loglevel', 'error', '-i', src, *codec_args, dst], check=True)
        return open(dst, 'rb').read()


class ToWavBytesTest(unittest.TestCase):
    def test_audio_soundfile_can_read_is_returned_untouched(self):
        wav = tone_wav_bytes()
        self.assertIs(to_wav_bytes(wav), wav)

    @unittest.skipIf(FFMPEG is None, "ffmpeg not available")
    def test_m4a_is_converted_to_a_wav_soundfile_can_read(self):
        m4a = encode(tone_wav_bytes(2.0), 'm4a', '-c:a', 'aac')
        with self.assertRaises(Exception):
            sf.read(io.BytesIO(m4a))  # the premise: soundfile alone cannot read it
        data, rate = sf.read(io.BytesIO(to_wav_bytes(m4a)))
        self.assertEqual(rate, 24000)
        self.assertAlmostEqual(len(data) / rate, 2.0, delta=0.15)

    @unittest.skipIf(FFMPEG is None, "ffmpeg not available")
    def test_mp3_is_accepted(self):
        mp3 = encode(tone_wav_bytes(1.5), 'mp3', '-c:a', 'libmp3lame')
        data, rate = sf.read(io.BytesIO(to_wav_bytes(mp3)))
        self.assertAlmostEqual(len(data) / rate, 1.5, delta=0.2)

    @unittest.skipIf(FFMPEG is None, "ffmpeg not available")
    def test_garbage_is_rejected_with_a_clear_error(self):
        with self.assertRaises(MediaDecodeError):
            to_wav_bytes(b'this is not audio at all' * 100)

    def test_without_ffmpeg_an_unreadable_file_is_reported_not_crashed(self):
        original = media_utils._ffmpeg_exe
        media_utils._ffmpeg_exe = lambda: None
        try:
            with self.assertRaises(MediaDecodeError):
                to_wav_bytes(b'not audio')
        finally:
            media_utils._ffmpeg_exe = original


class LimitReferenceAudioTest(unittest.TestCase):
    def speech_with_pause(self, seconds, pause_at, rate=16000):
        """A tone that is silent for 0.2 s starting at `pause_at`."""
        t = np.arange(int(seconds * rate)) / rate
        sig = (0.3 * np.sin(2 * np.pi * 220 * t)).astype('float32')
        sig[int(pause_at * rate):int((pause_at + 0.2) * rate)] = 0
        buf = io.BytesIO()
        sf.write(buf, sig, rate, format='WAV')
        return buf.getvalue()

    def test_short_audio_is_returned_untouched(self):
        wav = tone_wav_bytes(5.0)
        out, cut = media_utils.limit_reference_audio(wav)
        self.assertIs(out, wav)
        self.assertFalse(cut)

    def test_audio_of_exactly_the_limit_is_not_cut(self):
        _, cut = media_utils.limit_reference_audio(tone_wav_bytes(media_utils.MAX_REFERENCE_SECONDS), media_utils.MAX_REFERENCE_SECONDS)
        self.assertFalse(cut)

    def test_a_34_second_recording_is_cut_below_the_limit_at_the_pause(self):
        out, cut = media_utils.limit_reference_audio(self.speech_with_pause(34.7, pause_at=9.0))
        self.assertTrue(cut)
        data, rate = sf.read(io.BytesIO(out))
        self.assertEqual(rate, 16000)
        self.assertLessEqual(len(data) / rate, media_utils.MAX_REFERENCE_SECONDS)
        self.assertAlmostEqual(len(data) / rate, 9.1, delta=0.15)   # inside the pause, not mid-tone

    def test_stereo_is_kept_stereo(self):
        rate = 8000
        buf = io.BytesIO()
        sf.write(buf, np.full((35 * rate, 2), 0.1, 'float32'), rate, format='WAV')
        out, cut = media_utils.limit_reference_audio(buf.getvalue())
        data, _ = sf.read(io.BytesIO(out))
        self.assertTrue(cut)
        self.assertEqual(data.shape[1], 2)
        self.assertLessEqual(len(data) / rate, media_utils.MAX_REFERENCE_SECONDS)


class LimitImageSideTest(unittest.TestCase):
    def write_image(self, tmp, name, w, h):
        import cv2
        path = os.path.join(tmp, name)
        cv2.imwrite(path, (np.random.rand(h, w, 3) * 255).astype('uint8'))
        return path

    def test_large_photo_is_scaled_down_keeping_the_aspect_ratio(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_image(tmp, 'big.jpg', 1136, 2690)
            out = limit_image_side(path)
            self.assertNotEqual(out, path)
            h, w = cv2.imread(out).shape[:2]
            self.assertEqual(max(h, w), 768)
            self.assertAlmostEqual(w / h, 1136 / 2690, places=2)
            self.assertTrue(os.path.isfile(path))  # the original is kept

    def test_small_photo_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_image(tmp, 'small.png', 576, 768)
            self.assertEqual(limit_image_side(path), path)

    def test_unreadable_file_is_returned_as_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'broken.jpg')
            open(path, 'wb').write(b'not an image')
            self.assertEqual(limit_image_side(path), path)


if __name__ == '__main__':
    unittest.main()
