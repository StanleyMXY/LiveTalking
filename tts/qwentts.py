import os
import json
import base64
import time
import threading
import numpy as np
import resampy

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

try:
    import dashscope
    from dashscope.audio.qwen_tts_realtime import (
        QwenTtsRealtime,
        QwenTtsRealtimeCallback,
        AudioFormat,
    )
except ImportError:
    logger.error("QwenTTS 需要安装 dashscope SDK: pip install dashscope>=1.25.11")
    raise


SRC_SR = 24000   # Qwen TTS 只支持 24kHz 输出
DST_SR = 16000   # 项目标准采样率

# 预置音色名单来自阿里云百炼官方音色表（qwen-tts-voice-list）。2026-10-15 逐个用
# qwen3-tts-flash-realtime 合成过一个词，下面 48 个全部通过；原先写在这里的
# Daniel / Aidan / Vivienne / Aria / Alya / Ravi 被服务端拒绝（Invalid voice specified）。
PRESET_VOICES = {
    'Cherry', 'Serena', 'Ethan', 'Chelsie', 'Momo', 'Vivian', 'Moon', 'Maia', 'Kai', 'Nofish',
    'Bella', 'Jennifer', 'Ryan', 'Katerina', 'Aiden', 'Eldric Sage', 'Mia', 'Mochi', 'Bellona',
    'Vincent', 'Bunny', 'Neil', 'Elias', 'Arthur', 'Nini', 'Seren', 'Pip', 'Stella', 'Bodega',
    'Sonrisa', 'Alek', 'Dolce', 'Sohee', 'Ono Anna', 'Lenn', 'Emilien', 'Andre', 'Radio Gol',
    'Jada', 'Dylan', 'Li', 'Marcus', 'Roy', 'Peter', 'Sunny', 'Eric', 'Rocky', 'Kiki',
}


@register("tts", "qwentts")
class QwenTTS(BaseTTS):
    """
    阿里云通义千问实时语音合成 (Qwen TTS Realtime)
    基于 DashScope Python SDK (dashscope >= 1.25.11)
    使用 commit 模式：只建立一次 WebSocket 连接，每次合成通过 append_text + commit 触发。

    需要设置环境变量 DASHSCOPE_API_KEY。
    用法:
        python app.py --tts qwentts --REF_FILE Cherry
    其中 REF_FILE 用作音色名称 (voice)，如 Cherry / Ethan 等系统音色。
    """

    def __init__(self, opt, parent):
        super().__init__(opt, parent)

        # 音色名, 复用 REF_FILE 参数
        self.voice = opt.REF_FILE if opt.REF_FILE else 'Cherry'
        # 模型名: 克隆声音(voice_id 较长)自动切换到 vc-realtime 模型
        default_model = 'qwen3-tts-flash-realtime'
        if self.voice and self.voice not in PRESET_VOICES:
            default_model = 'qwen3-tts-vc-realtime-2026-01-15'
            logger.info(f"QwenTTS: 检测到克隆声音 ID，切换到 VC 模型: {default_model}")
        self.model = getattr(opt, 'qwen_tts_model', default_model)
        # WebSocket URL
        self.ws_url = getattr(opt, 'qwen_tts_url',
                              'wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime')

        # 设置 DashScope API Key
        api_key = getattr(opt, 'dashscope_api_key', None) or os.environ.get('DASHSCOPE_API_KEY')
        if api_key:
            dashscope.api_key = api_key
        else:
            logger.warning("QwenTTS: DASHSCOPE_API_KEY 未设置，请设置环境变量或通过参数传入")

        # ---------- 内部状态 ----------
        self._remainder = np.array([], dtype=np.float32)  # 上次重采样后不足一 chunk 的 16kHz 样本
        self._response_event = threading.Event()
        self._first_chunk = True          # 当前合成的一句话里的第一个音频包
        self._current_text = ''
        self._current_textevent = {}

        # ---------- 回调类 ----------
        tts_ref = self

        class _Callback(QwenTtsRealtimeCallback):
            def on_open(self) -> None:
                logger.info("QwenTTS WebSocket 连接已建立")

            def on_close(self, close_status_code, close_msg) -> None:
                logger.info(f"QwenTTS WebSocket 关闭: code={close_status_code}, msg={close_msg}")
                tts_ref._response_event.set()

            def on_event(self, response: dict) -> None:
                try:
                    event_type = response.get('type', '')

                    if event_type == 'session.created':
                        logger.info(f"QwenTTS session: {response.get('session', {}).get('id', '')}")

                    elif event_type == 'response.audio.delta':
                        audio_b64 = response.get('delta', '')
                        if audio_b64:
                            pcm_data = base64.b64decode(audio_b64)
                            tts_ref._on_audio_data(pcm_data)

                    elif event_type == 'response.done':
                        logger.info("QwenTTS response done")
                        tts_ref._last_good_voice = tts_ref.voice
                        tts_ref._flush_remainder()
                        tts_ref._response_event.set()

                    elif event_type == 'error':
                        logger.error(f"QwenTTS 错误: {response}")
                        # 服务端拒绝某个音色后会关闭连接；退回上一个可用音色，下一句话重连时使用
                        tts_ref.voice = tts_ref._last_good_voice
                        message = (response.get('error') or {}).get('message', '')
                        tts_ref._report_error(f"语音合成失败: {message}")
                        tts_ref._response_event.set()

                except Exception as e:
                    logger.exception(f"QwenTTS 回调处理异常: {e}")

        # ---------- 建立唯一连接 ----------
        self._callback = _Callback()
        self._tts_client = QwenTtsRealtime(
            model=self.model,
            callback=self._callback,
            url=self.ws_url,
        )
        # 预置音色模型只接受 PRESET_VOICES 里的名字；克隆音色模型不做这个限制
        self._preset_only = self.model == 'qwen3-tts-flash-realtime'
        self._last_good_voice = self.voice
        self._open_connection()
        logger.info(f"QwenTTS 初始化完成: model={self.model}, voice={self.voice}")

    # ========================== 连接管理 ==========================

    def _open_connection(self):
        self._tts_client.connect()
        self._tts_client.update_session(
            voice=self.voice,
            response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,  # Qwen TTS 只支持 24kHz 输出
            sample_rate=16000,
            mode='commit',
        )

    def _reopen_connection(self):
        try:
            self._tts_client.close()
        except Exception:
            pass  # 服务端已经关闭时 close() 可能抛错，重连照常进行
        self._open_connection()

    def _send_text(self, text: str):
        self._tts_client.append_text(text)
        self._tts_client.commit()

    def _report_error(self, text: str):
        """把合成失败推给页面（走 /sse），否则测试的人只会看到没有声音"""
        try:
            self.parent.send_msg(json.dumps({'status': 'tts_error', 'text': text}, ensure_ascii=False))
        except Exception:
            logger.exception("QwenTTS 上报错误失败")

    # ========================== 核心方法 ==========================

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        t_start = time.perf_counter()

        ref_file = textevent.get('tts', {}).get('ref_file') or self.voice

        # 重置状态
        self._remainder = np.array([], dtype=np.float32)
        self._first_chunk = True
        self._current_text = text
        self._current_textevent = textevent
        self._response_event.clear()

        try:
            #logger.info(f"QwenTTS 发送文本: {text[:80]}...")
            if ref_file != self.voice:
                if self._preset_only and ref_file not in PRESET_VOICES:
                    # 不能把会话切到一个服务端不认的音色：它会关闭连接，之后每句话都失败
                    logger.warning(f"QwenTTS: 音色 {ref_file!r} 不在可用列表中，继续使用 {self.voice!r}")
                else:
                    logger.info(f'ref_file:{ref_file},self.voice:{self.voice}')
                    self.voice = ref_file
                    self._reopen_connection()
            try:
                self._send_text(text)
            except ConnectionError:
                # 连接空闲超时或被服务端关闭：重连后补发这一句，而不是让之后每句话都失败
                logger.warning("QwenTTS: 连接已断开，正在重连")
                self._reopen_connection()
                self._send_text(text)

            # 等待 response.done（音频在回调中流式处理）
            self._response_event.wait(timeout=60)

            t_end = time.perf_counter()
            logger.info(f"QwenTTS 合成完成，耗时: {t_end - t_start:.2f}s")

        except Exception as e:
            logger.exception(f"QwenTTS txt_to_audio 异常: {e}")
            self._report_error(f"语音合成失败: {e}")

    # ========================== 流式音频处理（回调中调用）==========================

    def _on_audio_data(self, pcm_data: bytes):
        """收到 PCM 24kHz 16bit mono 音频，一次性 resample 到 16kHz 后分块推送"""
        if self.state != State.RUNNING:
            self._remainder = np.array([], dtype=np.float32)
            return

        # 整段 24kHz PCM -> float32 -> 一次性 resample 到 16kHz
        samples_16k = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        #samples_16k = resampy.resample(x=samples_24k, sr_orig=SRC_SR, sr_new=DST_SR)

        # 拼接上次剩余
        if self._remainder.shape[0] > 0:
            samples_16k = np.concatenate([self._remainder, samples_16k])

        # 按 self.chunk (320 samples = 20ms @16kHz) 分块推送
        idx = 0
        total = samples_16k.shape[0]
        while total - idx >= self.chunk and self.state == State.RUNNING:
            frame = samples_16k[idx:idx + self.chunk]
            eventpoint = {}
            if self._first_chunk:
                eventpoint = {'status': 'start', 'text': self._current_text}
                self._first_chunk = False
            eventpoint.update(**self._current_textevent)
            self.parent.put_audio_frame(frame, eventpoint)
            idx += self.chunk

        # 不足一 chunk 的留到下次
        self._remainder = samples_16k[idx:] if idx < total else np.array([], dtype=np.float32)

    def _flush_remainder(self):
        """合成完毕，推送剩余样本并发送 end 事件"""
        if self.state != State.RUNNING:
            self._remainder = np.array([], dtype=np.float32)
            return

        # 推送剩余完整 chunk
        if self._remainder.shape[0] >= self.chunk:
            idx = 0
            total = self._remainder.shape[0]
            while total - idx >= self.chunk and self.state == State.RUNNING:
                frame = self._remainder[idx:idx + self.chunk]
                eventpoint = {}
                if self._first_chunk:
                    eventpoint = {'status': 'start', 'text': self._current_text}
                    self._first_chunk = False
                eventpoint.update(**self._current_textevent)
                self.parent.put_audio_frame(frame, eventpoint)
                idx += self.chunk

        self._remainder = np.array([], dtype=np.float32)

        # 发送 end 事件
        eventpoint = {'status': 'end', 'text': self._current_text}
        eventpoint.update(**self._current_textevent)
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

    def stop_tts(self):
        self._tts_client.close()
        logger.info("QwenTTS 已关闭")