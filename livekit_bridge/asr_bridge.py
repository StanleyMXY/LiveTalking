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
  返回 ``str``——把 executor 调度留给上层调用方（Task 8），这里不做。
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
"""

import logging
from typing import Callable, Tuple

import numpy as np

from server.asr_server import _run_inference
from livekit_bridge.format_probe import PIPELINE_ASR_SAMPLE_RATE

logger = logging.getLogger(__name__)

InferFn = Callable[[np.ndarray, int, bool], Tuple[str, float, float]]


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
        """
        try:
            text, _inference_ms, _audio_duration_s = self._infer_fn(
                audio, PIPELINE_ASR_SAMPLE_RATE, False
            )
            return text
        except Exception:
            logger.exception("ASR transcription failed, degrading to empty text")
            return ""
