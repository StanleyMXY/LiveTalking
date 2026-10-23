###############################################################################
#  Helpers for the files users upload (reference audio, avatar photos)
###############################################################################

import io
import os
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf

# CosyVoice accepts a reference of at most 30 s (the product's voice sample rule is 1-30 s as well); the
# demo cuts a longer one just below that.
MAX_REFERENCE_SECONDS = 29.5
CUT_SEARCH_SECONDS = 3.0
CUT_FRAME_SECONDS = 0.05

# The working avatars are 576x768 frames. A larger photo only makes the "paste the face back onto the
# photo" step of SadTalker slower (measured on the DGX Spark: a 2690x1136 photo needed 5.1 s per frame,
# 150 frames, 12.8 minutes, while the avatar looks the same at 768).
MAX_AVATAR_IMAGE_SIDE = 768


class MediaDecodeError(Exception):
    """The uploaded file could not be read as audio."""


def _ffmpeg_exe():
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def to_wav_bytes(data: bytes) -> bytes:
    """Uploaded reference audio as bytes that both LiveTalking's and CosyVoice's soundfile can read.

    soundfile (libsndfile) reads wav, flac, ogg and most mp3 but not m4a/aac, which is what phones record.
    Anything it cannot read is converted to a 16 bit WAV with ffmpeg (sample rate and channels kept).
    """
    try:
        sf.read(io.BytesIO(data), dtype='float32')
        return data
    except Exception:
        pass

    exe = _ffmpeg_exe()
    if exe is None:
        raise MediaDecodeError("this audio format cannot be read here (ffmpeg is not installed); use wav")
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, 'upload')
        dst = os.path.join(tmp, 'converted.wav')
        with open(src, 'wb') as f:
            f.write(data)
        try:
            proc = subprocess.run(
                [exe, '-nostdin', '-loglevel', 'error', '-y', '-i', src, '-vn', '-acodec', 'pcm_s16le', '-f', 'wav', dst],
                capture_output=True, timeout=60,
            )
        except subprocess.TimeoutExpired:
            raise MediaDecodeError("converting the audio took too long")
        if proc.returncode != 0 or not os.path.isfile(dst) or os.path.getsize(dst) == 0:
            detail = proc.stderr.decode('utf-8', 'replace').strip().splitlines()[-1:] or ['unknown error']
            raise MediaDecodeError(f"not a readable audio file ({detail[0][:160]})")
        with open(dst, 'rb') as f:
            return f.read()


def limit_reference_audio(wav: bytes, max_seconds: float = MAX_REFERENCE_SECONDS):
    """(wav bytes, was_cut) with the reference audio cut to at most `max_seconds`.

    CosyVoice refuses a reference longer than 30 s, so a longer recording is cut instead of rejected. The
    cut is made at the quietest moment of the last CUT_SEARCH_SECONDS before the limit, so it does not fall
    in the middle of a word. The result is a 16 bit WAV with the original sample rate and channels.
    """
    data, rate = sf.read(io.BytesIO(wav), dtype='float32')
    limit = int(max_seconds * rate)
    if len(data) <= limit:
        return wav, False
    mono = data if data.ndim == 1 else data.mean(axis=1)
    frame = max(1, int(CUT_FRAME_SECONDS * rate))
    start = max(0, limit - int(CUT_SEARCH_SECONDS * rate))
    cut, quietest = limit, None
    for pos in range(start, limit - frame + 1, frame):
        energy = float(np.square(mono[pos:pos + frame]).mean())
        if quietest is None or energy < quietest:
            quietest, cut = energy, pos + frame // 2
    out = io.BytesIO()
    sf.write(out, data[:cut], rate, format='WAV', subtype='PCM_16')
    return out.getvalue(), True


def limit_image_side(path: str, max_side: int = MAX_AVATAR_IMAGE_SIDE) -> str:
    """Path of an image whose longest side is at most `max_side`.

    A smaller copy is written next to the original (PNG, aspect ratio kept); the original is not changed.
    An image that is already small enough, or one OpenCV cannot read, is returned as it is.
    """
    import cv2

    img = cv2.imread(path)
    if img is None:
        return path
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return path
    scale = max_side / longest
    small = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    out = f"{os.path.splitext(path)[0]}_{max_side}.png"
    if not cv2.imwrite(out, small):
        return path
    return out
