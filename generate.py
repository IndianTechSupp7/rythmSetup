import sys
import os
import json
import numpy as np
import librosa
from scipy.signal import butter, lfilter
from demucs.separate import main as demucs_main

# === Tunable parameters ===
ONSET_SENSITIVITY = 0.15  # ↑ = fewer detections, ↓ = more sensitive
GRAPH_SIZE = 80
MAX_TAIL_LENGTH = 200

# === Step 0: Read input file from cmd ===
if len(sys.argv) < 2:
    print("Usage: py generate.py <song_file>")
    sys.exit(1)

RAW_SONG_PATH = sys.argv[1]
song_name = os.path.splitext(os.path.basename(RAW_SONG_PATH))[0]

# Create output folder
OUTPUT_FOLDER = os.path.join("songs", song_name)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
destination = os.path.join(OUTPUT_FOLDER, os.path.basename(RAW_SONG_PATH))

os.rename(RAW_SONG_PATH, destination)
# === Step 1: Separate drums using Demucs ===
print(f"🎵 Separating drums from {RAW_SONG_PATH}...")

# Demucs CLI arguments
demucs_args = [
    "--mp3",  # export as MP3
    "-n",
    "htdemucs",  # model name
    "-o",
    "separated",  # temp folder
    "--two-stems",
    "drums",  # extract only drums
    destination,
]

demucs_main(demucs_args)

# Demucs output path
DRUMS_PATH = os.path.join("separated", "htdemucs", song_name, "drums.mp3")

# Copy to output folder
final_drums_path = os.path.join(OUTPUT_FOLDER, "drums.mp3")
os.replace(DRUMS_PATH, final_drums_path)
print(f"✅ Drums exported to {final_drums_path}")


# === Step 2: Bandpass filter function ===
def bandpass_filter(data, sr, lowcut, highcut):
    nyquist = 0.5 * sr
    low = max(0.001, lowcut / nyquist)
    high = min(0.999, highcut / nyquist)
    if high <= low:
        high = low + 0.001
    b, a = butter(4, [low, high], btype="band")
    return lfilter(b, a, data)


# === Step 3: Load drums and extract onsets ===
y, sr = librosa.load(final_drums_path)
print(f"Loaded {final_drums_path} (sr={sr})")

bands = {
    "Kick": bandpass_filter(y, sr, 30, 120),
    "Snare": bandpass_filter(y, sr, 120, 2500),
    "Tom": bandpass_filter(y, sr, 200, 1000),
    "Cymbal": bandpass_filter(y, sr, 4000, 10000),
}

onsets = {}
strengths = {}

for name, data in bands.items():
    y_h, y_p = librosa.effects.hpss(data)
    onset_env = librosa.onset.onset_strength(y=y_p, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        units="frames",
        delta=ONSET_SENSITIVITY,
        backtrack=False,
        pre_max=10,
        post_max=10,
        pre_avg=30,
        post_avg=30,
        wait=0,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    s_values = np.take(onset_env, onset_frames) if len(onset_frames) else np.array([])
    if len(s_values) > 0:
        s_values = (s_values - np.min(s_values)) / (np.ptp(s_values) + 1e-6)
    onsets[name] = onset_times
    strengths[name] = s_values
    print(f"{name}: {len(onset_times)} hits detected")

# === Step 4: Save beatmap JSON ===
beatmap = {
    "song": RAW_SONG_PATH,
    "bpm": float(librosa.beat.tempo(y=y, sr=sr, aggregate=np.mean)[0]),
    "sample_rate": int(sr),
    "tracks": {},
}

for name in onsets.keys():
    beatmap["tracks"][name] = []
    for i, t in enumerate(onsets[name]):
        s = float(strengths[name][i]) if len(strengths[name]) > i else 1.0
        beatmap["tracks"][name].append(
            {"time": round(float(t), 3), "strength": round(s, 3)}
        )
    beatmap["tracks"][name] = sorted(beatmap["tracks"][name], key=lambda x: x["time"])

beatmap_path = os.path.join(OUTPUT_FOLDER, f"{song_name}.json")
with open(beatmap_path, "w", encoding="utf-8") as f:
    json.dump(beatmap, f, indent=4)

print(f"✅ Beatmap exported to {beatmap_path}")
