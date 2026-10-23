###############################################################################
#  Helpers for the files users upload (reference audio, avatar photos)
###############################################################################

import io
import os
import shutil
import subprocess
import tempfile

import soundfile as sf

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
