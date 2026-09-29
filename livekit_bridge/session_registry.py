import threading


class BridgeSessionRegistry:
    """Deduplicates avatar sessions by LiveKit room name — within ONE worker
    process only.

    ``acquire(room)`` returns ``False`` if this process is already handling
    ``room``, so a concurrent or retried dispatch for the same room that
    lands on this same process does not build a second avatar session. The
    lock makes that check-and-add atomic across the job threads of one
    process (``JobExecutorType.THREAD``, see ``worker_app.py``).

    Limitation — per-process, not distributed: the state is a plain Python
    ``set`` in this process's memory. It does not see rooms handled by any
    other worker process or host, and it is empty again after a restart. If
    this connection shell runs as more than one worker process or on more
    than one host (the standard way LiveKit agent workers are scaled), or a
    worker restarts while a room is live, a retried dispatch for a room
    that lands on a *different* process is not caught here and can create a
    duplicate avatar session in the same room. Guaranteeing no duplicates
    at that scale needs either a shared/distributed registry (e.g. backed
    by Redis, or by the Java backend that will own dispatch) in place of
    this class, or idempotent dispatch on the dispatching side (check for
    an existing dispatch for the room before calling ``CreateDispatch``).
    Listed in ``livekit_bridge/__init__.py``'s "Known limitations".
    """

    def __init__(self):
        self._active_rooms: set[str] = set()
        self._lock = threading.Lock()

    def acquire(self, room_name: str) -> bool:
        with self._lock:
            if room_name in self._active_rooms:
                return False
            self._active_rooms.add(room_name)
            return True

    def release(self, room_name: str) -> None:
        with self._lock:
            self._active_rooms.discard(room_name)
