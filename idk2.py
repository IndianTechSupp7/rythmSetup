import pygame
import librosa
import numpy as np
import time
from scipy.signal import butter, lfilter
import json

SONG_PATH = "separated/htdemucs/greedy/drums.mp3"
RAW_SONG_PATH = "greedy.mp3"

# === Tunable parameters ===
ONSET_SENSITIVITY = 0.15  # ↑ = fewer detections, ↓ = more sensitive
GRAPH_SIZE = 80
MAX_TAIL_LENGTH = 200


# === Safe bandpass filter ===
def bandpass_filter(data, sr, lowcut, highcut):
    nyquist = 0.5 * sr
    low = max(0.001, lowcut / nyquist)
    high = min(0.999, highcut / nyquist)
    if high <= low:
        high = low + 0.001
    b, a = butter(4, [low, high], btype="band")
    return lfilter(b, a, data)


# === Step 1: Load and separate frequency bands ===
y, sr = librosa.load(SONG_PATH)
print(f"Loaded {SONG_PATH} (sr={sr})")

bands = {
    "Kick": bandpass_filter(y, sr, 30, 120),
    "Snare": bandpass_filter(y, sr, 120, 2500),
    "Tom": bandpass_filter(y, sr, 200, 1000),
    "Cymbal": bandpass_filter(y, sr, 4000, 10000),
}

# === Step 2: Detect onsets and strengths with HPSS filtering ===
onsets = {}
strengths = {}

for name, data in bands.items():
    # Separate percussive part only
    y_h, y_p = librosa.effects.hpss(data)

    # Get onset envelope (percussive only)
    onset_env = librosa.onset.onset_strength(y=y_p, sr=sr)

    # Onset detection with sensitivity control
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

    # Normalize strengths
    if len(s_values) > 0:
        s_values = (s_values - np.min(s_values)) / (np.ptp(s_values) + 1e-6)

    onsets[name] = onset_times
    strengths[name] = s_values
    print(f"{name}: {len(onset_times)} hits detected")

# === Step 2: Export to a JSON beatmap ===
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

# Optional: sort by time for each track
for name in beatmap["tracks"]:
    beatmap["tracks"][name] = sorted(beatmap["tracks"][name], key=lambda x: x["time"])

# Save to file
with open("beatmap.json", "w", encoding="utf-8") as f:
    json.dump(beatmap, f, indent=4)

print("✅ Beatmap exported to beatmap.json")

# === Step 3: Pygame setup ===
pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Improved Drum Strength Visualizer")

pygame.mixer.music.load(RAW_SONG_PATH)
pygame.mixer.music.play()

clock = pygame.time.Clock()
start_time = time.time()
running = True

lanes = list(onsets.keys())
colors = [(255, 120, 120), (255, 255, 120), (100, 200, 255), (180, 120, 255)]
positions = [150, 250, 350, 450]
pulses = [0] * len(lanes)
indices = [0] * len(lanes)
tails = [[] for _ in positions]
font = pygame.font.SysFont(None, 36)

# === Step 4: Main loop ===
while running:
    dt = clock.tick(60) / 1000
    t = time.time() - start_time

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((15, 15, 25))

    # Draw trails
    for i, group in enumerate(tails):
        for tail in group:
            tail[0] -= 100 * dt
        group[:] = group[:MAX_TAIL_LENGTH]
        if len(group) > 2:
            pygame.draw.lines(
                screen,
                colors[i],
                False,
                [
                    (tail[0], (positions[i] + GRAPH_SIZE / 2) - GRAPH_SIZE * tail[2])
                    for tail in group
                ],
            )

    # Draw each lane
    for i, lane in enumerate(lanes):
        times = onsets[lane]
        s_vals = strengths[lane]

        # Trigger when hitting next onset
        if indices[i] < len(times) and t >= times[indices[i]]:
            strength = s_vals[indices[i]] if len(s_vals) > indices[i] else 1.0
            pulses[i] = 0.3 + strength * 1.0
            indices[i] += 1

        # Decay pulse
        pulses[i] = max(0, pulses[i] - dt * 4)

        # Draw lane line + pulse circle
        y = positions[i]
        pygame.draw.line(screen, (50, 50, 70), (100, y), (700, y), 3)
        radius = int(40 * (1 + pulses[i]))
        base_color = np.array(colors[i])
        color = np.clip(base_color * (0.6 + 0.8 * pulses[i]), 0, 255).astype(int)
        pygame.draw.circle(screen, color, (400, y), radius)
        tails[i].insert(0, [400, y, pulses[i]])

        # Label
        label = font.render(lane, True, (200, 200, 200))
        screen.blit(label, (720, y - 18))

    pygame.display.flip()

pygame.quit()
