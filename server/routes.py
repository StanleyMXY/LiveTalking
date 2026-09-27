###############################################################################
#  服务器路由 — 统一异常处理的 API 路由
###############################################################################

import json
import re
import asyncio
import requests
from aiohttp import web

from utils.logger import logger

_SAFE_NAME_RE = re.compile(r'^[A-Za-z0-9_-]{1,64}$')
# SenseVoice 的 rich_transcription_postprocess 会在文字前后附带情绪/事件表情符号，
# 作为 TTS 参考文字稿要去掉，否则会被当成待合成文本的一部分。
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF❤️]", flags=re.UNICODE
)


# ─── 路由工具函数 ──────────────────────────────────────────────────────────

def json_ok(data=None):
    """返回成功 JSON 响应"""
    body = {"code": 0, "msg": "ok"}
    if data is not None:
        body["data"] = data
    return web.Response(
        content_type="application/json",
        text=json.dumps(body),
    )


def json_error(msg: str, code: int = -1):
    """返回错误 JSON 响应"""
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": code, "msg": str(msg)}),
    )


from server.session_manager import session_manager
from server.avatar_routes import setup_avatar_routes

def get_session(request, sessionid: str):
    """从 app 中获取 session 实例"""
    return session_manager.get_session(sessionid)


# ─── 路由处理函数 ──────────────────────────────────────────────────────────

async def human(request):
    """文本输入（echo/chat 模式），支持 voice/emotion 参数"""
    try:
        params: dict = await request.json()

        sessionid: str = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        if params.get('interrupt'):
            avatar_session.flush_talk()

        datainfo = {}
        if params.get('tts'):  # tts 参数透传（voice, emotion 等）
            datainfo['tts'] = params.get('tts')

        if params['type'] == 'echo':
            avatar_session.put_msg_txt(params['text'], datainfo)
        elif params['type'] == 'chat':
            llm_response = request.app.get("llm_response")
            if llm_response:
                asyncio.get_event_loop().run_in_executor(
                    None, llm_response, params['text'], avatar_session, datainfo
                )

        return json_ok()
    except Exception as e:
        logger.exception('human route exception:')
        return json_error(str(e))


async def interrupt_talk(request):
    """打断当前说话"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.flush_talk()
        return json_ok()
    except Exception as e:
        logger.exception('interrupt_talk exception:')
        return json_error(str(e))


async def humanaudio(request):
    """上传音频文件"""
    try:
        form = await request.post()
        sessionid = str(form.get('sessionid', ''))
        fileobj = form["file"]
        filebytes = fileobj.file.read()

        datainfo = {}

        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.put_audio_file(filebytes, datainfo)
        return json_ok()
    except Exception as e:
        logger.exception('humanaudio exception:')
        return json_error(str(e))


async def set_audiotype(request):
    """设置自定义状态（动作编排）"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.set_custom_state(params['audiotype'])
        return json_ok()
    except Exception as e:
        logger.exception('set_audiotype exception:')
        return json_error(str(e))


async def record(request):
    """录制控制"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        if params['type'] == 'start_record':
            avatar_session.start_recording()
        elif params['type'] == 'end_record':
            avatar_session.stop_recording()
        return json_ok()
    except Exception as e:
        logger.exception('record exception:')
        return json_error(str(e))


async def is_speaking(request):
    """查询是否正在说话"""
    params = await request.json()
    sessionid = params.get('sessionid', '')
    avatar_session = get_session(request, sessionid)
    if avatar_session is None:
        return json_error("session not found")
    return json_ok(data=avatar_session.is_speaking())

async def sse_handler(request):
    """SSE 事件流，推送服务器状态更新到客户端"""
    sessionid = request.query.get('sessionid', '')
    avatar_session = session_manager.get_session(sessionid)
    if avatar_session is None:
        return json_error("session not found")

    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
        }
    )
    await response.prepare(request)

    import queue
    msgqueue = queue.Queue()
    avatar_session.add_msgqueue(msgqueue)

    try:
        while True:
            try:
                msg = msgqueue.get_nowait()
                await response.write(f"data: {msg}\n\n".encode('utf-8'))
            except queue.Empty:
                await asyncio.sleep(0.01)
    except (asyncio.CancelledError, ConnectionResetError):
        logger.info('SSE connection closed for session: %s', sessionid)
    finally:
        if msgqueue in avatar_session.msgqueues:
            avatar_session.msgqueues.remove(msgqueue)

    return response


async def admin_config(request):
    """Admin: 获取全局配置参数"""
    try:
        opt = request.app.get("opt")
        if opt:
            return json_ok(data={"config": vars(opt)})
        return json_error("Config not found")
    except Exception as e:
        logger.exception('admin_config exception:')
        return json_error(str(e))


async def admin_sessions(request):
    """Admin: 获取活跃的会话及其配置"""
    try:
        sessions_info = []
        for sid, avatar_session in session_manager.sessions.items():
            if avatar_session:
                s_opt = getattr(avatar_session, 'opt', None)
                s_data = {
                    "sessionid": sid,
                    "speaking": avatar_session.is_speaking() if hasattr(avatar_session, 'is_speaking') else False,
                    "recording": getattr(avatar_session, 'recording', False),
                }
                if s_opt:
                    s_data.update({
                        "model": getattr(s_opt, "model", ""),
                        "avatar_id": getattr(s_opt, "avatar_id", ""),
                        "REF_FILE": getattr(s_opt, "REF_FILE", ""),
                        "transport": getattr(s_opt, "transport", ""),
                        "batch_size": getattr(s_opt, "batch_size", 0),
                        "customopt": getattr(s_opt, "customopt", []),
                    })
                sessions_info.append(s_data)
        return json_ok(data={"sessions": sessions_info})
    except Exception as e:
        logger.exception('admin_sessions exception:')
        return json_error(str(e))


def _find_tts_backend(opt, backend_id: str):
    """在 opt.tts_backends 里按 id 找条目；找不到返回 None"""
    for entry in (getattr(opt, 'tts_backends', None) or []):
        if entry.get('id') == backend_id:
            return entry
    return None


async def get_tts_voices(request):
    """GET /api/tts/voices?backend=<id> — 列出该 backend 当前可用的音色"""
    try:
        opt = request.app.get("opt")
        backend_id = request.query.get('backend', '')
        entry = _find_tts_backend(opt, backend_id)
        if entry is None:
            return json_error(f"backend '{backend_id}' not found")

        module = entry.get('module')
        if module == 'qwentts':
            from tts.qwentts import PRESET_VOICES
            return json_ok(data={"voices": sorted(PRESET_VOICES)})
        elif module == 'omnitts':
            server_url = entry.get('TTS_SERVER') or getattr(opt, 'TTS_SERVER', '')
            try:
                res = requests.get(f"{server_url.rstrip('/')}/v1/voices", timeout=5)
                res.raise_for_status()
                voices = [v['name'] for v in res.json().get('voices', [])]
            except Exception:
                logger.exception(f'get_tts_voices: failed to reach {server_url}')
                voices = []
            return json_ok(data={"voices": voices})
        else:
            return json_ok(data={"voices": []})
    except Exception as e:
        logger.exception('get_tts_voices exception:')
        return json_error(str(e))


async def post_tts_voices(request):
    """POST /api/tts/voices — multipart 上传参考音频，克隆出一个新音色

    字段: audio_file (上传), name, backend, transcript (可选)
    没给 transcript 时用本地 SenseVoice 自动转写。
    """
    try:
        reader = await request.multipart()
        params = {}
        audio_bytes = None

        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == 'audio_file':
                audio_bytes = await part.read()
            else:
                params[part.name] = await part.text()

        name = params.get('name', '')
        backend_id = params.get('backend', '')
        transcript = params.get('transcript', '').strip()

        if not audio_bytes:
            return json_error("audio_file is required")
        if not _SAFE_NAME_RE.match(name):
            return json_error("name must match ^[A-Za-z0-9_-]{1,64}$")

        opt = request.app.get("opt")
        entry = _find_tts_backend(opt, backend_id)
        if entry is None or entry.get('module') != 'omnitts':
            return json_error(f"backend '{backend_id}' does not support voice cloning")
        server_url = (entry.get('TTS_SERVER') or getattr(opt, 'TTS_SERVER', '')).rstrip('/')

        if not transcript:
            import io
            import soundfile as sf
            from server.asr_server import _run_inference

            audio_float32, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
            if audio_float32.ndim > 1:
                audio_float32 = audio_float32.mean(axis=1)

            loop = asyncio.get_event_loop()
            transcript, _, _ = await loop.run_in_executor(
                None, _run_inference, audio_float32, sr, True
            )
            transcript = _EMOJI_RE.sub('', transcript).strip()
            if not transcript:
                return json_error("auto-transcription produced empty text, please supply transcript manually")

        res = requests.post(
            f"{server_url}/v1/voices",
            files={"audio_file": ("clip.wav", audio_bytes)},
            data={"name": name, "transcript": transcript},
            timeout=30,
        )
        if res.status_code != 200:
            return json_error(f"voice registration failed: {res.text}")

        return json_ok(data=res.json())
    except Exception as e:
        logger.exception('post_tts_voices exception:')
        return json_error(str(e))


# ─── 路由注册 ──────────────────────────────────────────────────────────────

async def index(request):
    """默认首页重定向"""
    opt = request.app.get("opt")
    pagename = 'index.html'
    if opt and opt.transport == 'rtmp':
        pagename = 'rtmpapi.html'
    elif opt and opt.transport == 'rtcpush':
        pagename = 'rtcpushapi.html'
    raise web.HTTPFound(f'/{pagename}')


def setup_routes(app):
    """注册所有路由到 aiohttp app"""
    app.router.add_get("/", index)
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/record", record)
    app.router.add_post("/interrupt_talk", interrupt_talk)
    app.router.add_post("/is_speaking", is_speaking)
    app.router.add_get("/api/admin/config", admin_config)
    app.router.add_get("/api/admin/sessions", admin_sessions)
    app.router.add_get("/api/tts/voices", get_tts_voices)
    app.router.add_post("/api/tts/voices", post_tts_voices)
    app.router.add_get('/sse', sse_handler)

    # ── Local ASR endpoint (SenseVoice/FunASR) ── Issue #604 ──
    try:
        from server.asr_server import asr_websocket_handler, is_funasr_available
        if is_funasr_available():
            app.router.add_get("/api/asr", asr_websocket_handler)
            logger.info("[ASR] Local SenseVoice ASR endpoint enabled at /api/asr")
        else:
            logger.info("[ASR] funasr not installed — local ASR endpoint disabled "
                        "(pip install funasr modelscope)")
    except Exception as e:
        logger.warning(f"[ASR] Failed to register ASR endpoint: {e}")

    # 注册 avatar 生成相关的路由
    setup_avatar_routes(app)

    app.router.add_static('/', path='web')
