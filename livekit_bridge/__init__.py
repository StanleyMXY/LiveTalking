"""
livekit_bridge — connection shell that lets a LiveKit-based client talk to
the LiveTalking avatar pipeline (send-only aiortc session + in-process ASR)
over a LiveKit room instead of LiveTalking's own aiortc WebRTC transport.

This package is being built incrementally across a multi-task plan. Task 1
only records verified facts about the LiveKit audio wire format and the
pipeline's ASR call convention (see ``format_probe.py``); it does not yet
implement the bridge itself.
"""
