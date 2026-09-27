"""
Voice enrollment utility for Qwen3-TTS voice cloning.

Usage:
    python -m tools.enroll_voice <audio_file> [--name myvoice]

The audio file should be:
  - WAV (16bit) / MP3 / M4A
  - 10-20 seconds of clear speech, no background noise
  - Sample rate >= 24kHz recommended
  - Mono channel

After enrollment, copy the returned voice_id to config.yaml as REF_FILE.
Then set tts_model to qwen3-tts-vc-realtime in config.yaml (or leave empty for auto-detect).
"""

import argparse
import base64
import json
import mimetypes
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ENROLLMENT_URL = "https://dashscope-intl.aliyuncs.com/api/v1/services/audio/tts/customization"
TARGET_MODEL = "qwen3-tts-vc-realtime-2026-01-15"

MIME_MAP = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
}


def enroll(audio_path: str, name: str, api_key: str) -> str:
    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    mime = MIME_MAP.get(p.suffix.lower())
    if not mime:
        raise ValueError(f"Unsupported format: {p.suffix}. Use .wav, .mp3, or .m4a")

    file_size = p.stat().st_size
    if file_size > 10 * 1024 * 1024:
        raise ValueError(f"File too large: {file_size / 1024 / 1024:.1f}MB (max 10MB)")

    b64 = base64.b64encode(p.read_bytes()).decode()
    data_uri = f"data:{mime};base64,{b64}"

    print(f"Enrolling voice from: {p.name} ({file_size / 1024:.0f}KB)")
    print(f"Target model: {TARGET_MODEL}")

    resp = requests.post(
        ENROLLMENT_URL,
        json={
            "model": "qwen-voice-enrollment",
            "input": {
                "action": "create",
                "target_model": TARGET_MODEL,
                "preferred_name": name,
                "audio": {"data": data_uri},
            },
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=60,
    )

    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)

    result = resp.json()
    voice_id = result.get("output", {}).get("voice", "")
    if not voice_id:
        print(f"Unexpected response: {json.dumps(result, indent=2)}", file=sys.stderr)
        sys.exit(1)

    return voice_id


def list_voices(api_key: str):
    resp = requests.post(
        ENROLLMENT_URL,
        json={
            "model": "qwen-voice-enrollment",
            "input": {"action": "list", "page_index": 0, "page_size": 50},
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}", file=sys.stderr)
        return
    result = resp.json()
    voices = result.get("output", {}).get("voice_list", [])
    if not voices:
        print("No enrolled voices found.")
    else:
        print(f"Found {len(voices)} voice(s):")
        for v in voices:
            print(f"  - {v.get('voice_id', v.get('voice', '?'))}: {v}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Enroll a voice for Qwen3-TTS voice cloning")
    parser.add_argument("audio", nargs="?", help="Path to audio file (.wav/.mp3/.m4a)")
    parser.add_argument("--name", default="myvoice", help="Voice name prefix (lowercase, <10 chars)")
    parser.add_argument("--list", action="store_true", help="List existing enrolled voices")
    args = parser.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: DASHSCOPE_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    if args.list:
        list_voices(api_key)
        return

    if not args.audio:
        parser.print_help()
        sys.exit(1)

    voice_id = enroll(args.audio, args.name, api_key)
    print(f"\nVoice enrolled successfully!")
    print(f"Voice ID: {voice_id}")
    print(f"\nTo use this voice, update config.yaml:")
    print(f"  REF_FILE: '{voice_id}'")


if __name__ == "__main__":
    main()
