import threading


class BridgeSessionRegistry:
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
