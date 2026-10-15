import asyncio
import os
import unittest
from unittest.mock import patch

from server import ice

MINTED = [
    {"urls": ["stun:stun.cloudflare.com:3478", "stun:stun.cloudflare.com:53"]},
    {
        "urls": [
            "turn:turn.cloudflare.com:3478?transport=udp",
            "turn:turn.cloudflare.com:53?transport=udp",
            "turns:turn.cloudflare.com:443?transport=tcp",
        ],
        "username": "u",
        "credential": "c",
    },
]


def run(coro):
    return asyncio.run(coro)


class IceServersTest(unittest.TestCase):
    def setUp(self):
        ice._state.update(servers=None, minted_at=0.0, failed_at=None)
        self.env = patch.dict(os.environ, {"CF_TURN_KEY_ID": "kid", "CF_TURN_API_TOKEN": "tok"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_port_53_urls_are_dropped_and_credentials_kept(self):
        servers = ice.drop_port_53(MINTED)
        self.assertEqual(servers[0]["urls"], ["stun:stun.cloudflare.com:3478"])
        self.assertEqual(
            servers[1]["urls"],
            ["turn:turn.cloudflare.com:3478?transport=udp", "turns:turn.cloudflare.com:443?transport=tcp"],
        )
        self.assertEqual(servers[1]["username"], "u")
        self.assertEqual(servers[1]["credential"], "c")

    def test_a_port_that_merely_ends_in_53_is_kept(self):
        servers = ice.drop_port_53([{"urls": ["turn:h:3453?transport=udp", "turn:h:53"]}])
        self.assertEqual(servers[0]["urls"], ["turn:h:3453?transport=udp"])

    def test_without_key_serves_stun_only_and_never_calls_the_api(self):
        with patch.dict(os.environ, {"CF_TURN_KEY_ID": "", "CF_TURN_API_TOKEN": ""}), \
                patch.object(ice, "_mint_blocking") as mint:
            self.assertEqual(run(ice.get_ice_servers()), ice._FALLBACK)
            mint.assert_not_called()

    def test_minted_credentials_are_cached(self):
        with patch.object(ice, "_mint_blocking", return_value=ice.drop_port_53(MINTED)) as mint:
            first = run(ice.get_ice_servers())
            second = run(ice.get_ice_servers())
        self.assertEqual(first, second)
        self.assertEqual(mint.call_count, 1)

    def test_api_failure_falls_back_and_is_not_retried_immediately(self):
        with patch.object(ice, "_mint_blocking", side_effect=RuntimeError("boom")) as mint:
            self.assertEqual(run(ice.get_ice_servers()), ice._FALLBACK)
            self.assertEqual(run(ice.get_ice_servers()), ice._FALLBACK)
        self.assertEqual(mint.call_count, 1)

    def test_api_failure_after_success_keeps_serving_the_previous_credentials(self):
        good = ice.drop_port_53(MINTED)
        with patch.object(ice, "_mint_blocking", return_value=good):
            run(ice.get_ice_servers())
        ice._state["minted_at"] = -ice._CACHE_S - 1  # expired
        with patch.object(ice, "_mint_blocking", side_effect=RuntimeError("boom")):
            self.assertEqual(run(ice.get_ice_servers()), good)


if __name__ == "__main__":
    unittest.main()
