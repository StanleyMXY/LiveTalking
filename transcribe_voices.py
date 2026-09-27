import io
import os
import re
os.environ.setdefault("MODELSCOPE_CACHE", r"D:\model_cache\modelscope")
os.environ.setdefault("HF_HOME", r"D:\model_cache\huggingface")
import soundfile as sf
import numpy as np
import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

SRC_DIR = r"D:\duix_avatar_data\CosyVoice\Voices"
DST_DIR = r"D:\Projects\CosyVoice\voices"
FILES = {
    "Li Yunlong.wav": "li_yunlong",
    "Ma Baoguo.wav": "ma_baoguo",
    "Zhizunbao.wav": "zhizunbao",
}

# Strip SenseVoice's emotion/event emoji markers, keep only spoken text + normal punctuation
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\u2764\ufe0f]", flags=re.UNICODE
)

def clean_text(s: str) -> str:
    return _EMOJI_RE.sub("", s).strip()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Loading SenseVoiceSmall on {device}...")
model = AutoModel(
    model="FunAudioLLM/SenseVoiceSmall",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device=device,
    trust_remote_code=True,
    hub="hf",
)
print("Model loaded.\n")

os.makedirs(DST_DIR, exist_ok=True)
results = {}
for fname, voice_id in FILES.items():
    path = os.path.join(SRC_DIR, fname)
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix stereo to mono
    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio, sr, format="WAV")
    wav_buf.seek(0)

    res = model.generate(input=wav_buf, cache={}, language="auto", use_itn=True, batch_size_s=60)
    raw_text = rich_transcription_postprocess(res[0]["text"]) if res and res[0].get("text") else ""
    text = clean_text(raw_text)
    results[voice_id] = text

    # Write mono 16kHz-or-native wav + matching transcript straight into CosyVoice's voices/
    dst_wav = os.path.join(DST_DIR, voice_id + ".wav")
    sf.write(dst_wav, audio, sr, subtype="PCM_16")
    dst_txt = os.path.join(DST_DIR, voice_id + ".txt")
    with open(dst_txt, "w", encoding="utf-8") as f:
        f.write(text)

with open(r"D:\Projects\PycharmProjects\AI-Avatar\LiveTalking\transcribe_result.txt", "w", encoding="utf-8") as f:
    for voice_id, text in results.items():
        f.write(f"=== {voice_id} ===\n{text}\n\n")
