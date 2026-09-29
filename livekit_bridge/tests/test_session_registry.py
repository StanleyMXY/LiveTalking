from livekit_bridge.session_registry import BridgeSessionRegistry


def test_second_acquire_on_same_room_fails_until_released():
    registry = BridgeSessionRegistry()

    assert registry.acquire("room-1") is True
    assert registry.acquire("room-1") is False

    registry.release("room-1")

    assert registry.acquire("room-1") is True


def test_different_rooms_do_not_conflict():
    registry = BridgeSessionRegistry()

    assert registry.acquire("room-1") is True
    assert registry.acquire("room-2") is True
