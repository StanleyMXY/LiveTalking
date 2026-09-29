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

Generation fencing (code review follow-up finding — why ``drive_with_text``
no longer passes the raw ``avatar_session`` straight through)
-----------------------------------------------------------------------------------------
``entrypoint.py``'s connection shell can supersede an in-flight response
when a new utterance completes before the previous one has finished (see
that module's "at most one *tracked* drive task per session" docstring
section). The first attempt at this cancelled the *asyncio* ``Task`` that
was awaiting
``drive_with_text`` — which only helps if cancellation lands while that
task is still awaiting ``asr_bridge.transcribe`` (stage 1). Once a task has
already reached ``llm_response`` (stage 2) and is executing inside its own
executor thread, cancelling the asyncio coroutine that dispatched it does
nothing to that thread: ``llm_response``'s streaming loop
(``llm.py:63-87``) has no cancellation checkpoint anywhere in it and keeps
calling ``avatar_session.put_msg_txt(...)`` — once per sentence-terminator-
delimited chunk while streaming, not just once at the end — regardless of
what happens on the asyncio side. Calling ``avatar_session.flush_talk()``
does not stop this either: it clears the TTS queue *once* and pauses
playback, but does not block *future* ``put_msg_txt`` calls — each new
call re-enqueues unconditionally and flips the TTS backend's state back to
running (``tts/base_tts.py:34-46``). So an orphaned, still-executing
``llm_response`` call kept pushing its stale sentence fragments into the
same queue as the new response's, interleaving both — the original bug,
just narrowed to "only once the old call had already reached its LLM
streaming stage," which given this plan's own timing (a 500ms silence cut
vs. realistic LLM response latency) is the common case, not a rare one.

The fix, verified to require no change to ``llm.py``: read the whole of
``llm_response`` and confirmed it only ever touches two things on whatever
object it is given — ``.opt`` (``llm.py:47``) and ``.put_msg_txt(...)``
(``llm.py:81``, ``:87``). ``drive_with_text`` now passes a thin per-call
proxy, ``_SupersedableAvatarProxy``, instead of the raw ``avatar_session``.
The proxy forwards ``.opt`` straight through, but every ``put_msg_txt``
call first checks whether *this specific call's* generation — a
monotonically increasing counter on the bridge, incremented once per
``drive_with_text`` call and, critically, also once per ``interrupt()``
call — is still the bridge's current one; if a newer utterance has since
superseded it, the call becomes a silent no-op instead of reaching the
real ``avatar_session``. This actually stops the interleaving at its
source (each individual enqueue attempt), rather than only ever having a
chance to stop it at the asyncio-task boundary, which cannot reach an
executor thread already running someone else's Python code.

The asyncio-level task cancellation in ``entrypoint.py`` is kept
unchanged alongside this — it is still correct and useful for the
stage-1 (still-in-``transcribe``) case, and for not wasting executor
capacity even starting a ``drive_with_text`` call that would immediately
no-op anyway.
"""

import threading

from llm import llm_response


class PipelineBridge:
    """把 LiveKit 侧收到的文字交给现有管线驱动数字人。"""

    def __init__(self) -> None:
        self._generation_lock = threading.Lock()
        self._current_generation = 0

    def drive_with_text(self, avatar_session, text: str) -> None:
        if not text or not text.strip():
            return
        with self._generation_lock:
            self._current_generation += 1
            generation = self._current_generation
        proxy = _SupersedableAvatarProxy(avatar_session, self, generation)
        llm_response(text, proxy, {})

    def interrupt(self, avatar_session) -> None:
        # Also invalidates whatever generation is currently in flight (see
        # module docstring's "Generation fencing" section) — every
        # interrupt(), not just a superseding drive_with_text call, must
        # stop an orphaned llm_response call's future put_msg_txt calls
        # from reaching the real session.
        with self._generation_lock:
            self._current_generation += 1
        avatar_session.flush_talk()

    def _is_current_generation(self, generation: int) -> bool:
        with self._generation_lock:
            return generation == self._current_generation


class _SupersedableAvatarProxy:
    """Thin per-``drive_with_text``-call stand-in for the real
    ``avatar_session``, passed to ``llm_response`` instead of it.

    See the module docstring's "Generation fencing" section for why this
    exists. ``llm_response`` only ever reads ``.opt`` and calls
    ``.put_msg_txt(...)`` on whatever object it is given — nothing else on
    this proxy needs to exist.
    """

    __slots__ = ("_avatar_session", "_bridge", "_generation")

    def __init__(self, avatar_session, bridge: "PipelineBridge", generation: int) -> None:
        self._avatar_session = avatar_session
        self._bridge = bridge
        self._generation = generation

    @property
    def opt(self):
        return self._avatar_session.opt

    def put_msg_txt(self, msg, datainfo=None) -> None:
        if not self._bridge._is_current_generation(self._generation):
            return
        self._avatar_session.put_msg_txt(msg, {} if datainfo is None else datainfo)
