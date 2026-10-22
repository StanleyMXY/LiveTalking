###############################################################################
#  ICE (STUN/TURN) server list for the browser
#
#  The browser must gather relay candidates itself: a viewer behind a VPN or a
#  strict firewall can only reach the server through a relay it contacts over
#  TCP/TLS. Credentials are minted here with the long-term key from the
#  environment, so the key never reaches the page.
###############################################################################

import asyncio
import os
import re
import time

import requests

from utils.logger import logger

_API_URL = "https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}/credentials/generate-ice-servers"
_CREDENTIAL_TTL_S = 86400     # lifetime of the minted username/credential
_CACHE_S = 3600               # reuse one minted set this long instead of calling the API per page load
_RETRY_AFTER_FAILURE_S = 30   # do not hammer the API (and block page loads) while it is failing
_REQUEST_TIMEOUT_S = 10

# STUN only: used when no TURN key is configured or the API is unreachable.
_FALLBACK = [{"urls": ["stun:stun.cloudflare.com:3478"]}]

# Port 53 is blocked by browsers and its URL would only add a timeout while the
# page waits for ICE gathering to finish (documented by Cloudflare).
_PORT_53_RE = re.compile(r":53(\?|$)")

_state = {"servers": None, "minted_at": 0.0, "failed_at": None}
_lock = asyncio.Lock()


def drop_port_53(servers):
    kept = []
    for server in servers:
        urls = [u for u in server.get("urls", []) if not _PORT_53_RE.search(u)]
        if urls:
            kept.append({**server, "urls": urls})
    return kept


def _mint_blocking(key_id: str, api_token: str):
    resp = requests.post(
        _API_URL.format(key_id=key_id),
        headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"},
        json={"ttl": _CREDENTIAL_TTL_S},
        timeout=_REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    servers = drop_port_53(resp.json()["iceServers"])
    if not servers:
        raise ValueError("credential response contained no usable ICE server")
    return servers


async def get_ice_servers():
    # Set ICE_SERVERS_DISABLED=1 where every viewer reaches the server on a private network
    # (for example over a VPN): external STUN/TURN servers would only delay candidate gathering.
    if os.environ.get("ICE_SERVERS_DISABLED", "").strip().lower() in ("1", "true", "yes"):
        return []

    key_id = os.environ.get("CF_TURN_KEY_ID", "").strip()
    api_token = os.environ.get("CF_TURN_API_TOKEN", "").strip()
    if not key_id or not api_token:
        return _FALLBACK

    async with _lock:
        now = time.monotonic()
        if _state["servers"] and now - _state["minted_at"] < _CACHE_S:
            return _state["servers"]
        if _state["failed_at"] is not None and now - _state["failed_at"] < _RETRY_AFTER_FAILURE_S:
            return _state["servers"] or _FALLBACK
        try:
            servers = await asyncio.get_running_loop().run_in_executor(
                None, _mint_blocking, key_id, api_token
            )
        except Exception as e:
            # Log the type only: the message of a requests error can carry the URL (key id).
            _state["failed_at"] = now
            logger.warning("TURN credential request failed (%s); serving %s",
                           type(e).__name__, "the previous credentials" if _state["servers"] else "STUN only")
            return _state["servers"] or _FALLBACK
        _state.update(servers=servers, minted_at=now, failed_at=None)
        logger.info("Minted TURN credentials (%d ICE servers)", len(servers))
        return servers
