import json
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

BEATMAP_FILE = "beatmap.json"
SONG_PATH = "greedy.mp3"

# === Load audio and beatmap ===
y, sr = librosa.load(SONG_PATH)
duration = librosa.get_duration(y=y, sr=sr)

with open(BEATMAP_FILE, "r", encoding="utf-8") as f:
    beatmap = json.load(f)

tracks = beatmap["tracks"]
colors = {
    "Kick": "red",
    "Snare": "orange",
    "Tom": "blue",
    "Cymbal": "purple",
}

# === Plot setup ===
fig, ax = plt.subplots(figsize=(12, 6))
librosa.display.waveshow(y, sr=sr, color="gray", alpha=0.5, ax=ax)
ax.set_xlim(0, duration)
ax.set_ylim(-1, 1)
ax.set_title("Beatmap Editor (Drag notes, press S to save)")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Amplitude")

# Store all scatter plots
plots = {}
for name, events in tracks.items():
    times = [e["time"] for e in events]
    strengths = [e["strength"] for e in events]
    plots[name] = ax.scatter(
        times, [0.8 - 0.2 * list(tracks.keys()).index(name)] * len(times),
        s=np.array(strengths) * 80 + 30,
        color=colors.get(name, "white"),
        label=name, picker=True
    )

ax.legend()
selected_point = None
selected_track = None

# === Interactivity ===
def on_pick(event):
    global selected_point, selected_track
    for name, plot in plots.items():
        if event.artist == plot:
            ind = event.ind[0]
            selected_point = ind
            selected_track = name
            print(f"Selected {name} point #{ind}")
            break

def on_motion(event):
    global selected_point, selected_track
    if selected_point is None or selected_track is None:
        return
    if event.xdata is None:
        return
    # Move note horizontally
    x = float(np.clip(event.xdata, 0, duration))
    tracks[selected_track][selected_point]["time"] = x
    plots[selected_track].set_offsets([
        (e["time"], 0.8 - 0.2 * list(tracks.keys()).index(selected_track))
        for e in tracks[selected_track]
    ])
    fig.canvas.draw_idle()

def on_release(event):
    global selected_point, selected_track
    if selected_point is not None:
        print(f"Released {selected_track} point #{selected_point}")
    selected_point = None
    selected_track = None

def on_key(event):
    if event.key.lower() == "s":
        with open(BEATMAP_FILE, "w", encoding="utf-8") as f:
            json.dump(beatmap, f, indent=4)
        print("✅ Beatmap saved.")

fig.canvas.mpl_connect("pick_event", on_pick)
fig.canvas.mpl_connect("motion_notify_event", on_motion)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()
