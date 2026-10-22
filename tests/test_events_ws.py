import asyncio
import json
import queue
import unittest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from server import routes


class FakeSession:
    def __init__(self):
        self.msgqueues = []

    def add_msgqueue(self, q):
        self.msgqueues.append(q)


class EventsWebSocketTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.session = FakeSession()
        patcher = patch.object(routes.session_manager, "get_session",
                               side_effect=lambda sid: self.session if sid == "s1" else None)
        patcher.start()
        self.addCleanup(patcher.stop)
        app = web.Application()
        app.router.add_get("/ws/events", routes.events_ws)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    async def test_events_are_delivered_and_the_queue_is_removed_on_close(self):
        ws = await self.client.ws_connect("/ws/events?sessionid=s1")
        for _ in range(100):
            if self.session.msgqueues:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(len(self.session.msgqueues), 1)
        self.session.msgqueues[0].put(json.dumps({"status": "reply", "text": "你好"}, ensure_ascii=False))
        msg = await asyncio.wait_for(ws.receive_json(), 3)
        self.assertEqual(msg, {"status": "reply", "text": "你好"})
        await ws.close()
        for _ in range(200):
            if not self.session.msgqueues:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(self.session.msgqueues, [])

    async def test_unknown_session_is_closed_with_a_reason(self):
        ws = await self.client.ws_connect("/ws/events?sessionid=nope")
        msg = await asyncio.wait_for(ws.receive(), 3)
        self.assertEqual(msg.type.name, "CLOSE")
        self.assertEqual(msg.data, 4404)


if __name__ == "__main__":
    unittest.main()
