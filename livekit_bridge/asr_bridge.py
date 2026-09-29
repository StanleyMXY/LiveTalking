"""
AsrBridge — 语音转写桥接（复用现有管线的 ASR 模型单例，不新建模型，不走
websocket）。

真实调用路径核实（不是规划阶段的猜测；详见 livekit_bridge/format_probe.py
"Fact 2" 部分，那里是本任务的权威依据，比 brief 草稿代码更可信）：

- 真实的推理入口是 ``server/asr_server.py`` 里的**模块级函数**
  ``_run_inference(audio_float32, sample_rate, use_itn)``
  (server/asr_server.py:71-117)——不是某个"model"对象上的
  ``model.transcribe(audio)`` 方法（brief 草稿里的那个签名是没有核实过的
  占位形状，实际代码里不存在这个方法）。
- 它是**阻塞**调用，必须由异步调用方通过 ``loop.run_in_executor`` 调用，
  就像现有 websocket handler 已经做的那样 (server/asr_server.py:218-224)。
  但根据本任务的接口定义，``AsrBridge.transcribe`` 本身是同步方法，
  返回 ``str``——把 executor 调度留给上层调用方（``entrypoint.py``），这里不做。
- 它的真实返回值是**三元组** ``(transcribed_text, inference_ms,
  audio_duration_s)`` (server/asr_server.py:78-81, :117)，不是裸字符串，
  所以这里必须解包后只取第一项文本。
- 模型本身是**懒加载的模块级单例**，通过 ``_load_sensevoice()``
  (server/asr_server.py:33-68) 访问；``_run_inference`` 内部已经调用了它
  (server/asr_server.py:85)，这里不需要也不应该再调用
  ``_load_sensevoice()`` 或构造第二个 ``AutoModel(...)``——否则会重复加载
  模型、重复占用 GPU/CPU 内存。

因为真实的复用单元是一个模块级函数而不是一个对象方法，构造函数注入的是这个
真实可调用对象本身（默认为 ``server.asr_server._run_inference``），不是
brief 草稿里虚构的 ``model`` 参数——测试直接替换这个可调用对象来验证正常
转写和异常降级，而不是 mock 一个不存在的 ``model.transcribe`` 接口。

Non-speech marker filtering (final-review finding, Review Focus (c))
----------------------------------------------------------------------
``_run_inference`` returns the output of funasr's
``rich_transcription_postprocess`` (server/asr_server.py:83, :105), not
raw text. That postprocessing (installed ``funasr==1.4.16``,
``funasr/utils/postprocess_utils.py:338-477``) rewrites SenseVoice's special
tokens into emoji: ``<|nospeech|><|Event_UNK|>`` becomes ``❓`` (:458);
``format_str_v2`` deletes every ``emoji_dict`` token (:418-420), then
prepends the ``event_dict`` glyph for each audio event present
(``🎼 👏 😀 😭 🤧`` — BGM, applause, laughter, cry, sneeze *and* cough,
:348-357, :425-427) and appends the dominant emotion's ``emo_dict`` glyph
(``😊 😔 😡 😰 🤢 😮``, :338-346, :428); language tags, ``<|NEUTRAL|>``,
``<|Speech|>``, ``<|Breath|>``, ``<|withitn|>`` etc. map to ``""``. So pure
noise — a cough, background music, an unrecognisable sound — comes back as
a *non-empty* string such as ``"🤧"`` or ``"❓"``, which passes
``drive_with_text``'s ``text.strip()`` check and would drive a real LLM
call on nothing.

The filtered set is every non-empty glyph in ``emoji_dict``/``emo_dict``/
``event_dict`` plus ``emo_set``/``event_set`` (:368-408) — a superset of
what the code path above can emit today (``😷``, ``emoji_dict``'s cough
glyph, is listed there but ``format_str_v2`` never inserts ``emoji_dict``
values), kept so that no glyph funasr declares as a marker can pass.

``strip_sensevoice_markers`` removes exactly that set of emoji (plus, as a
defensive second net, any raw ``<|...|>`` token ``format_str_v2`` would have
left untouched because it is not an ``emoji_dict`` key — :418-420 only
removes keys it knows), and returns ``""`` unless at least one letter/digit
(``str.isalnum()`` — true for CJK characters) is left, so a result that is
only punctuation (``"?"``, ``"。"``) is also treated as "no speech". ``""`` is
the existing "no speech" contract every caller already handles.

The emoji set is a literal copied from the installed funasr source, not
imported from it at runtime: funasr is not pinned in ``requirements.txt``
(it is commented out there), its dictionaries are module internals rather
than a public API, and importing ``funasr.utils.postprocess_utils`` runs
funasr's package ``__init__``, which eagerly imports every funasr submodule
(``funasr/__init__.py:48-78``; ~28s, pulls in torch). Instead,
``test_asr_bridge.py``'s ``test_marker_set_matches_installed_funasr``
recomputes the set from the installed funasr in a subprocess and fails if
the two ever diverge, so a funasr upgrade that adds a marker cannot slip
through silently.
"""

import logging
import re
from typing import Callable, Tuple

import numpy as np

from server.asr_server import _run_inference
from livekit_bridge.format_probe import PIPELINE_ASR_SAMPLE_RATE

logger = logging.getLogger(__name__)

InferFn = Callable[[np.ndarray, int, bool], Tuple[str, float, float]]

# Every non-empty marker glyph funasr==1.4.16's postprocess_utils declares
# for a SenseVoice special token (a superset of what
# rich_transcription_postprocess actually emits) — see module docstring for
# the source lines. Kept in sync with the installed funasr by
# test_asr_bridge.test_marker_set_matches_installed_funasr.
SENSEVOICE_MARKER_GLYPHS = frozenset(
    {
        "❓",  # <|nospeech|><|Event_UNK|>
        # events
        "🎼", "👏", "😀", "😭", "🤧", "😷",
        # emotions
        "😊", "😔", "😡", "😰", "🤢", "😮",
    }
)
_MARKER_GLYPH_RE = re.compile("|".join(re.escape(g) for g in sorted(SENSEVOICE_MARKER_GLYPHS)))
_RAW_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|<>]*\|>")
_WHITESPACE_RUN_RE = re.compile(r"\s+")


def strip_sensevoice_markers(text: str) -> str:
    """Remove SenseVoice's emoji/event markers from postprocessed ASR text.

    Returns ``""`` when nothing meaningful (no letter or digit) remains, so
    noise-only results fall into the existing "no speech" contract. See the
    module docstring's "Non-speech marker filtering" section.
    """
    if not text:
        return ""
    # Replaced with "" (not " "), matching format_str_v2's own removal
    # (postprocess_utils.py:420), which already strips the spaces around
    # each glyph (:430-432) — so this restores the adjacency the text had.
    stripped = _RAW_SPECIAL_TOKEN_RE.sub("", text)
    stripped = _MARKER_GLYPH_RE.sub("", stripped)
    stripped = _WHITESPACE_RUN_RE.sub(" ", stripped).strip()
    if not any(ch.isalnum() for ch in stripped):
        return ""
    return stripped


class AsrBridge:
    """把 LiveKit 侧收到的音频交给现有管线的 ASR 单例做识别。"""

    def __init__(self, infer_fn: InferFn = _run_inference):
        self._infer_fn = infer_fn

    def transcribe(self, audio: np.ndarray) -> str:
        """
        同步、阻塞地对一段单声道 float32 音频做识别。

        识别失败（模型抛任何异常）时降级返回空字符串，绝不向上抛出异常——
        本方法被连接壳的音频处理路径直接调用，一次识别失败不能让整条
        LiveKit 连接崩溃。下游必须把空字符串当作"没有识别到语音"处理，
        而不是当作错误。

        识别结果只含 SenseVoice 的非语音标记（咳嗽、背景音乐、❓ 等，见模块
        docstring 的 "Non-speech marker filtering" 一节）时同样返回空字符串；
        有真实文字时返回去掉这些标记后的文字。
        """
        try:
            text, _inference_ms, _audio_duration_s = self._infer_fn(
                audio, PIPELINE_ASR_SAMPLE_RATE, False
            )
        except Exception:
            logger.exception("ASR transcription failed, degrading to empty text")
            return ""
        filtered = strip_sensevoice_markers(text)
        if text and not filtered:
            logger.info("ASR result %r is non-speech markers only; treating as no speech", text)
        return filtered
