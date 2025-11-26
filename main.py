import pygame
import librosa
import numpy as np
import time

RAW_SONG_PATH = "One Kiss.mp3"
SONG_PATH = "separated/htdemucs/One Kiss/drums.mp3"


# === Step 1: Load song and detect onsets ===
y, sr = librosa.load(SONG_PATH)
onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=False, units="frames")
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

print(f"Detected {len(onset_times)} onsets")

# === Step 2: Pygame setup ===
pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((600, 600))
pygame.display.set_caption("Beat Onset Visualizer")

pygame.mixer.music.load(RAW_SONG_PATH)
pygame.mixer.music.play()

clock = pygame.time.Clock()
running = True
start_time = time.time()

circle_radius = 80
pulse = 0
onset_index = 0

# === Step 3: Loop ===
while running:
    dt = clock.tick(60) / 1000
    t = time.time() - start_time

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Trigger when reaching an onset
    if onset_index < len(onset_times) and t >= onset_times[onset_index]:
        pulse = 1.0
        onset_index += 1

    # Fade the pulse over time
    pulse = max(0, pulse - dt * 5)

    # Draw
    screen.fill((15, 10, 30))
    radius = int(circle_radius * (1 + 0.4 * pulse))
    color = (200, 80 + int(100 * pulse), 255 - int(100 * pulse))
    pygame.draw.circle(screen, color, (300, 300), radius)

    pygame.display.flip()

pygame.quit()
