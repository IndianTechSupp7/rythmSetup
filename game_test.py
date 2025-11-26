import json
import time
import pygame
from window import Window
import numpy as np


class Node:
    def __init__(self, spawn_time, beat_time, pos):
        self.pos = np.array(list(pos), np.float32)
        self.spawn_time = spawn_time
        self.beat_time = beat_time
        self.r = 10
        self.color = [70] * 3
        self.triggered = False

    def update(self, dir, current_time):
        if current_time >= self.spawn_time:
            self.pos += dir
            if self.pos[0] > 0 and not self.triggered:
                self.collide()
                self.triggered = True
            self.r = max(self.r - 1, 10)
            self.color = [max(i - 10, 70) for i in self.color]

    def collide(self):
        self.r = 20
        self.color = [255] * 3

    def render(self, surf, offset):
        pygame.draw.circle(surf, self.color, self.pos - offset, self.r)


class Game(Window):
    def setup(self):

        pygame.mixer.init()
        pygame.mixer.music.load("Routines In The Night.mp3")
        pygame.mixer.music.play()

        self.beatmap = get_js("beatmap.json")

        self.center = np.array((self.w / 2, self.h / 2))

        self.dir = np.array((1.0, 0.0))

        self.note_speed = 200
        self.hit_line_y = 0
        self.x_start = -self.center[0]

        self.nodes = []
        for beat in self.beatmap["tracks"]["Kick"]:
            travel_time = (self.hit_line_y - self.x_start) / self.note_speed
            spawn_time = beat["time"] - travel_time
            self.nodes.append(
                Node(spawn_time, beat["time"], pos=(self.x_start, self.center[1]))
            )

        self.start_time = time.time()
        return {}

    def update(self):
        self.center = np.array((self.w / 2, self.h / 2))
        current_time = time.time() - self.start_time

        pygame.draw.line(
            self.display, "white", (self.center[0], 0), (self.center[0], self.h)
        )

        for node in self.nodes:
            print(node.pos)
            node.update(self.dir * self.note_speed * self.dt, current_time)
        for node in self.nodes[::-1]:
            node.render(self.display, (-self.center[0], 0))


def get_js(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data


if __name__ == "__main__":
    Game().run()
