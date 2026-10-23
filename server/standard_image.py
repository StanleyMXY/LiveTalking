###############################################################################
#  Standard avatar image: photo -> one clean, front-facing portrait
#
#  Follows the product's avatar generation (AvatarApp design
#  docs/superpowers/specs/2026-10-11-avatar-rendering-design.md section 3.2-3.3, implemented by
#  AvatarEngine): check that the photo has a face, ask the image-edit model for a standard image of the
#  same person (waist up, face and gaze to the camera, scene background, natural warm light), check that
#  the generated image still has a face, retry with a new random seed otherwise. The user looks at the
#  result and decides; there is no same-person check.
#
#  Only the realistic style is offered in this demo (no cartoon). The age check and the mouth-opening
#  check of the product are not part of the demo: the first is a compliance rule of the product, the
#  second is a requirement of the phone renderer (the lip-sync model here redraws the mouth itself).
###############################################################################

import base64
import math
import os
import secrets
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests

from utils.logger import logger

# ---- prompt: AvatarEngine avatarengine/prompts.py (prompt v4), REALISTIC style, same wording ----------
_FRAMING = (
    "A medium shot from a slightly greater distance: the person's whole upper body down to the waist is in the "
    "frame, with some space around them; the head and face are turned straight towards the camera and looking "
    "into the camera. Natural, warm light from the scene."
)
_MOUTH = (
    "The person has a broad, warm, friendly smile: the lips are parted and the upper front teeth are clearly "
    "visible, bright and white, with the jaw relaxed so only a little of the inside of the mouth shows; not "
    "laughing, not surprised."
)
_KEEP_FACE = (
    "Keep the exact eye colour of the photo and keep the eyes, nose and mouth in the same positions as in the "
    "photo."
)
_REST = (
    "The person is wearing {outfit_spec}; only the clothes on the torso and arms change, everything on the "
    "head, face and neck stays exactly as in the photo. Keep the person's exact skin tone; do not lighten the "
    "skin. Background: {background_spec}. This must be the same person: preserve every facial feature "
    "exactly as it appears in the photo, neither adding nor removing anything."
)
PROMPT_TEMPLATE = (
    "Turn this photo into a natural, photorealistic portrait of the same person. "
    + _FRAMING + " " + _MOUTH + " " + _KEEP_FACE + " " + _REST
)
NEGATIVE_PROMPT = (
    "cartoon, illustration, 3d render, closed mouth, open-mouthed laugh, surprised expression, gaping "
    "mouth, changed eye colour, distorted face, duplicated eyes, extra eyes, extra limbs, text, watermark, "
    "blurry, lightened skin, grain, noise"
)
# The outfit and background texts AvatarEngine's live end-to-end run uses (avatarengine/tools/e2e_live.py:61,63).
DEFAULT_OUTFIT = "a dark blue suit jacket over a plain white crew-neck T-shirt"
DEFAULT_BACKGROUND = "a cozy living room with a sofa, plants and warm evening window light"

# ---- image-edit API: AvatarEngine avatarengine/providers/image_edit.py -------------------------------
EDIT_PATH = "/services/aigc/multimodal-generation/generation"
DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1"
DEFAULT_MODEL = "qwen-image-3.0"
EDIT_MIN_SIDE = 384
EDIT_MAX_SIDE = 2048
EDIT_MAX_RATIO = 10.0
EDIT_MAX_DATA_URI_BYTES = 10_000_000
OUTPUT_TARGET_AREA = 1024 * 1024
OUTPUT_MIN_AREA = 512 * 512
OUTPUT_MAX_AREA = 2048 * 2048
OUTPUT_MAX_ASPECT = 8
OUTPUT_SIDE_MULTIPLE = 16
SEED_MAX = 2_147_483_647

MAX_ATTEMPTS = 3
FACE_CHECK_MAX_SIDE = 1024
# the head-and-shoulders window cut around the face before generating
CROP_WIDTH_IN_FACES = 3.2          # window width in face widths
CROP_ASPECT = 0.75                 # width / height = 3:4
CROP_FACE_CENTRE_FROM_TOP = 0.35   # where the middle of the face sits in the window (fraction of its height)
CROP_MIN_MARGIN_IN_FACES = 1.5     # a window narrower than this many face widths is not worth cutting
CROP_SKIP_COVERAGE = 0.85          # a window covering this much of the photo: it is already framed like that
MAX_STORED_JOBS = 50
JOB_KEEP_SECONDS = 3600


class StandardImageError(Exception):
    """Generation failed; `code` is one of the codes below, `str(error)` is meant for the user."""

    def __init__(self, message, code="generation_failed"):
        super().__init__(message)
        self.code = code


NO_FACE = "no_face"
NOT_CONFIGURED = "not_configured"
BAD_PHOTO = "bad_photo"
GENERATION_FAILED = "generation_failed"


# ---------------------------------------------------------------------------------------------------
# request preparation
# ---------------------------------------------------------------------------------------------------
def fit_for_edit(img):
    """(data URI, width, height) of the photo as it is sent to the image-edit API: long side at most 2048,
    short side at least 384 where that is possible, proportions at most 10:1, whole data URI under 10 MB."""
    h, w = img.shape[:2]
    if min(h, w) < 1 or max(h, w) / min(h, w) > EDIT_MAX_RATIO:
        raise StandardImageError("The photo's proportions are too extreme; use a normal photo.", BAD_PHOTO)
    scale = 1.0
    if max(h, w) > EDIT_MAX_SIDE:
        scale = EDIT_MAX_SIDE / max(h, w)
    elif min(h, w) < EDIT_MIN_SIDE:
        scale = min(EDIT_MIN_SIDE / min(h, w), EDIT_MAX_SIDE / max(h, w))
    if scale != 1.0:
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interpolation)
    out_h, out_w = img.shape[:2]
    for mime, ext, params in (("image/png", ".png", []), ("image/jpeg", ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 92])):
        ok, buf = cv2.imencode(ext, img, params)
        if not ok:
            continue
        uri = f"data:{mime};base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"
        if len(uri) <= EDIT_MAX_DATA_URI_BYTES:
            return uri, out_w, out_h
    raise StandardImageError("The photo is too large to send; use a smaller one.", BAD_PHOTO)


def _round_to_multiple(value):
    return max(OUTPUT_SIDE_MULTIPLE, int(value / OUTPUT_SIDE_MULTIPLE + 0.5) * OUTPUT_SIDE_MULTIPLE)


def output_size(width, height, target_area=OUTPUT_TARGET_AREA):
    """The `size` of the edited image: about one megapixel at the photo's aspect ratio, both sides a
    multiple of 16, inside the API's limits (same rule as AvatarEngine's `output_size`)."""
    area = min(max(target_area, OUTPUT_MIN_AREA), OUTPUT_MAX_AREA)
    ratio = min(max(width / height, 1 / OUTPUT_MAX_ASPECT), OUTPUT_MAX_ASPECT)
    out_h = math.sqrt(area / ratio)
    out_w, out_h = _round_to_multiple(ratio * out_h), _round_to_multiple(out_h)
    step = OUTPUT_SIDE_MULTIPLE
    while True:
        if out_w * out_h > OUTPUT_MAX_AREA:
            if out_w >= out_h:
                out_w -= step
            else:
                out_h -= step
        elif out_w * out_h < OUTPUT_MIN_AREA:
            if out_w <= out_h:
                out_w += step
            else:
                out_h += step
        elif out_w > OUTPUT_MAX_ASPECT * out_h:
            out_w = OUTPUT_MAX_ASPECT * out_h
        elif out_h > OUTPUT_MAX_ASPECT * out_w:
            out_h = OUTPUT_MAX_ASPECT * out_w
        else:
            return out_w, out_h


def build_prompt(outfit=DEFAULT_OUTFIT, background=DEFAULT_BACKGROUND):
    # str.format on the template only: braces inside the texts are not interpreted
    return PROMPT_TEMPLATE.format(outfit_spec=outfit.strip(), background_spec=background.strip())


# ---------------------------------------------------------------------------------------------------
# the image-edit call
# ---------------------------------------------------------------------------------------------------
class ImageEditor:
    """One synchronous DashScope image-edit call per attempt.

    A request is repeated only when it certainly did not run (connect timeout, HTTP 429): a read timeout
    or any other failure after the request was sent may already have been billed, so it is not repeated
    here (the generation loop starts a new attempt instead, which is a new, deliberate edit).
    """

    def __init__(self, api_key, base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL, timeout_s=180):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout_s = timeout_s

    def edit(self, data_uri, width, height, prompt, negative_prompt, seed):
        size_w, size_h = output_size(width, height)
        body = {
            "model": self.model,
            "input": {"messages": [{"role": "user", "content": [{"image": data_uri}, {"text": prompt}]}]},
            "parameters": {"n": 1, "watermark": False, "negative_prompt": negative_prompt,
                           "prompt_extend": False, "seed": seed, "size": f"{size_w}*{size_h}"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = None
        for retry in range(3):
            try:
                response = requests.post(self.base_url + EDIT_PATH, headers=headers, json=body,
                                         timeout=(10, self.timeout_s))
            except requests.ConnectTimeout:
                if retry == 2:
                    raise StandardImageError("Could not reach the image generation service; try again.")
                time.sleep(1 + retry)
                continue
            except requests.RequestException as e:
                raise StandardImageError(f"The image generation service did not answer ({type(e).__name__}); try again.")
            if response.status_code == 429 and retry < 2:
                time.sleep(2 + retry * 2)
                continue
            break
        if response.status_code != 200:
            raise StandardImageError(_provider_error(response))
        url = _result_url(response)
        if url is None:
            raise StandardImageError("The image generation service returned no image; try again.")
        return self._download(url)

    def _download(self, url):
        last = None
        for retry in range(3):
            try:
                r = requests.get(url, timeout=(10, 60))
                if r.status_code == 200:
                    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        return img
                    last = "the returned image could not be read"
                else:
                    last = f"download answered HTTP {r.status_code}"
            except requests.RequestException as e:
                last = type(e).__name__
            time.sleep(1 + retry)
        raise StandardImageError(f"The generated image could not be downloaded ({last}); try again.")


def _result_url(response):
    try:
        url = response.json()["output"]["choices"][0]["message"]["content"][0]["image"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    return url if isinstance(url, str) and url else None


def _provider_error(response):
    code = message = ""
    try:
        body = response.json()
        code, message = str(body.get("code", "")), str(body.get("message", ""))
    except ValueError:
        pass
    if "DataInspection" in code or "inappropriate" in message.lower():
        return "The photo or the generated image was rejected by the content safety check; use another photo."
    return f"The image generation service refused the request (HTTP {response.status_code} {code}); try again."


# ---------------------------------------------------------------------------------------------------
# face check
# ---------------------------------------------------------------------------------------------------
_detector = None
_detector_lock = threading.Lock()


def find_face(img):
    """(x1, y1, x2, y2) of the face the project's detector (the one avatar creation uses) finds, in the
    coordinates of `img`, or None."""
    global _detector
    h, w = img.shape[:2]
    scale = 1.0
    small = img
    if max(h, w) > FACE_CHECK_MAX_SIDE:
        scale = FACE_CHECK_MAX_SIDE / max(h, w)
        small = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    with _detector_lock:
        if _detector is None:
            import torch
            from avatars.wav2lip import face_detection
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            _detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)
        box = _detector.get_detections_for_batch(np.array([small]))[0]
    if box is None:
        return None
    return tuple(int(round(v / scale)) for v in box)


def crop_around_face(img, box):
    """The head-and-shoulders part of the photo, 3:4, around the face box.

    The model keeps the proportions of what it is given, so a phone screenshot (status bar, black bars, a
    small face in a tall frame) would come back as a very narrow portrait with a small face. The window is
    CROP_WIDTH_IN_FACES face widths wide, 3:4, the middle of the face a little above the middle of the
    window, moved inside the photo when it sticks out. A photo that is already framed about like that (the
    window would cover most of it), and one where the window would not even leave a margin around the face,
    is returned unchanged.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    face_w = max(x2 - x1, 1)
    crop_w = face_w * CROP_WIDTH_IN_FACES
    crop_h = crop_w / CROP_ASPECT
    shrink = min(1.0, w / crop_w, h / crop_h)
    crop_w, crop_h = crop_w * shrink, crop_h * shrink
    if crop_w < face_w * CROP_MIN_MARGIN_IN_FACES:
        return img
    if crop_w * crop_h >= CROP_SKIP_COVERAGE * w * h:
        return img
    left = min(max((x1 + x2) / 2 - crop_w / 2, 0), w - crop_w)
    top = min(max((y1 + y2) / 2 - CROP_FACE_CENTRE_FROM_TOP * crop_h, 0), h - crop_h)
    left, top = int(round(left)), int(round(top))
    return img[top:top + int(round(crop_h)), left:left + int(round(crop_w))].copy()


def has_face(img):
    return find_face(img) is not None


# ---------------------------------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------------------------------
def generate_standard_image(photo_path, editor, face_finder=find_face, max_attempts=MAX_ATTEMPTS,
                            outfit=DEFAULT_OUTFIT, background=DEFAULT_BACKGROUND, on_attempt=None):
    """PNG bytes of the standard image generated from the photo. Raises StandardImageError."""
    photo = cv2.imread(photo_path)
    if photo is None:
        raise StandardImageError("This file cannot be read as an image.", BAD_PHOTO)
    # Early, free rejection: generating costs money and needs a face (design 3.1)
    box = face_finder(photo)
    if box is None:
        raise StandardImageError("No face was found in the photo; use a photo with a clearly visible face.", NO_FACE)
    photo = crop_around_face(photo, box)

    data_uri, width, height = fit_for_edit(photo)
    prompt = build_prompt(outfit, background)
    for attempt in range(1, max_attempts + 1):
        if on_attempt:
            on_attempt(attempt)
        # random seed per attempt: without one the service answers the same input with the same image
        edited = editor.edit(data_uri, width, height, prompt, NEGATIVE_PROMPT, secrets.randbelow(SEED_MAX + 1))
        if face_finder(edited) is not None:
            ok, png = cv2.imencode(".png", edited)
            if ok:
                return png.tobytes()
        logger.warning("standard image attempt %d/%d: no face in the generated image", attempt, max_attempts)
    raise StandardImageError(
        f"No usable image after {max_attempts} attempts (the generated image had no clear face); try another photo.",
        GENERATION_FAILED)


class StandardImageJobs:
    """Background generation: one job per uploaded photo, polled by the page."""

    def __init__(self, directory, editor_factory, face_finder=find_face, max_workers=2, max_attempts=MAX_ATTEMPTS):
        self.directory = directory
        self.editor_factory = editor_factory
        self.face_finder = face_finder
        self.max_attempts = max_attempts
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="standard-image")
        self._jobs = {}
        self._lock = threading.Lock()

    def start(self, photo_path):
        self._forget_old()
        job_id = uuid.uuid4().hex
        job = {"job_id": job_id, "status": "running", "attempt": 0, "max_attempts": self.max_attempts,
               "error": "", "error_code": "", "path": None, "created": time.time()}
        with self._lock:
            self._jobs[job_id] = job
        self._pool.submit(self._run, job, photo_path)
        return job_id

    def get(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def _run(self, job, photo_path):
        try:
            editor = self.editor_factory()
            png = generate_standard_image(photo_path, editor, self.face_finder, self.max_attempts,
                                          on_attempt=lambda n: job.update(attempt=n))
            path = os.path.join(self.directory, f"standard_{job['job_id']}.png")
            with open(path, "wb") as f:
                f.write(png)
            job.update(path=path, status="done")
        except StandardImageError as e:
            job.update(status="failed", error=str(e), error_code=e.code)
        except Exception as e:
            logger.exception("standard image job failed:")
            job.update(status="failed", error=f"Unexpected error ({type(e).__name__}).", error_code=GENERATION_FAILED)
        finally:
            try:
                os.remove(photo_path)   # the uploaded photo is not kept after the attempt
            except OSError:
                pass

    def _forget_old(self):
        now = time.time()
        with self._lock:
            old = [j for j, v in self._jobs.items() if now - v["created"] > JOB_KEEP_SECONDS and v["status"] != "running"]
            while len(self._jobs) - len(old) > MAX_STORED_JOBS:
                oldest = min((j for j in self._jobs if j not in old and self._jobs[j]["status"] != "running"),
                             key=lambda j: self._jobs[j]["created"], default=None)
                if oldest is None:
                    break
                old.append(oldest)
            for j in old:
                path = self._jobs.pop(j).get("path")
                if path:
                    try:
                        os.remove(path)
                    except OSError:
                        pass


def editor_from_environment():
    """The editor configured by the environment: DASHSCOPE_API_KEY (as the rest of the demo), and optional
    AVATAR_IMAGE_EDIT_MODEL / AVATAR_IMAGE_EDIT_BASE_URL."""
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise StandardImageError("Image generation is not configured on this server (no DashScope key).", NOT_CONFIGURED)
    return ImageEditor(key,
                       os.environ.get("AVATAR_IMAGE_EDIT_BASE_URL", DEFAULT_BASE_URL),
                       os.environ.get("AVATAR_IMAGE_EDIT_MODEL", DEFAULT_MODEL))
