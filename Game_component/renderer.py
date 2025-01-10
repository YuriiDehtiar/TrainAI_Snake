# Game_component/renderer.py

import matplotlib.pyplot as plt
import numpy as np

class Renderer:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.im = None
        print("Renderer initialized.")

    def render(self, state):
        # дндюмн оепеб╡пйс:
        if np.isnan(state).any() or np.isinf(state).any():
            raise ValueError("NaN or Inf found in `state` before rendering.")

        color_map = {
            0: [1, 1, 1],  # EMPTY
            1: [0, 1, 0],  # BODY
            2: [0, 0, 1],  # HEAD
            3: [1, 0, 0],  # APPLE
            4: [0, 0, 0]   # WALL
        }

        h, w, c = state.shape
        rgb_array = np.zeros((h, w, 3))
        for i in range(h):
            for j in range(w):
                cell_value = int(state[i, j, 0])
                rgb_array[i, j] = color_map.get(cell_value, [1, 1, 1])

        if self.im is None:
            self.im = self.ax.imshow(rgb_array, interpolation='nearest')
        else:
            self.im.set_data(rgb_array)

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        plt.pause(0.1)

    def close(self):
        print("Closing renderer.")
        plt.ioff()
        plt.close(self.fig)

