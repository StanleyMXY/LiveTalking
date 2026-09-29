"""
PipelineBridge — 文字驱动桥接。

把 ASR 转写出的文字交给现有管线驱动数字人，复用管线现成的
文字->LLM->TTS->视频流程，而不是重新实现一遍。

真实调用路径核实（不是规划阶段的猜测）：
- ``server/routes.py`` 的 ``/human`` 路由，``type == 'chat'`` 分支
  (server/routes.py:71-76)：
      llm_response = request.app.get("llm_response")
      ...
      asyncio.get_event_loop().run_in_executor(
          None, llm_response, params['text'], avatar_session, datainfo
      )
  这里的 ``llm_response`` 是 ``app.py:40`` 里 ``from llm import llm_response``
  导入后，在 ``app.py:175`` 注册进 aiohttp app (``appasync["llm_response"] = llm_response``)
  的同一个函数对象——不是路由自己定义的另一个函数。
- 真实函数签名在 ``llm.py:45``：
      def llm_response(message, avatar_session: 'BaseAvatar', datainfo: dict = {}):
  第三个参数 ``datainfo`` 的默认值是空字典 ``{}``，不是 ``None``。这不是随意的：
  ``llm_response`` 内部把 ``datainfo`` 原样转发给
  ``avatar_session.put_msg_txt(result, datainfo)``，而 ``put_msg_txt`` 的实现
  (avatars/base_avatar.py:158-159) 一进来就执行
  ``datainfo.get('tts', {}).get('backend', ...)``——如果传 ``None`` 进去，
  这里会直接 ``AttributeError: 'NoneType' object has no attribute 'get'``。
  规划阶段草稿里的 ``datainfo=None`` 是没有核实过的错误猜测，会在真实调用链里崩溃。
- 本桥接类没有 aiohttp 的 ``request.app`` 注册表可用，且 brief 要求的接口是
  ``drive_with_text(avatar_session, text) -> None``（没有 tts 透传参数），
  所以直接 ``from llm import llm_response`` 导入同一个函数对象，并传入空字典
  ``{}`` 作为 ``datainfo``——这与 routes.py 在没有 ``tts`` 参数时构造的
  ``datainfo = {}`` 完全一致。
- ``interrupt()`` 对应 ``/human`` 路由里 ``params.get('interrupt')`` 分支
  (server/routes.py:62-63) 和独立的 ``/interrupt_talk`` 路由
  (server/routes.py:84-96)，两处都是直接调用 ``avatar_session.flush_talk()``。
"""

from llm import llm_response


class PipelineBridge:
    """把 LiveKit 侧收到的文字交给现有管线驱动数字人。"""

    def drive_with_text(self, avatar_session, text: str) -> None:
        if not text or not text.strip():
            return
        llm_response(text, avatar_session, {})

    def interrupt(self, avatar_session) -> None:
        avatar_session.flush_talk()
