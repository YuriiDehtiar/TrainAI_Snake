"""Seeded Snake environment with explicit body order and bounded episodes."""

from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class Rewards:
    step: float = -0.01
    apple: float = 1.0
    collision: float = -1.0
    starvation: float = -1.0
    win_bonus: float = 5.0


class SnakeEnv(gym.Env):
    """Actions: 0=straight, 1=left, 2=right; reversing is not an action.

    Channels: wall, head, body, apple, neck, tail, body vacancy time,
    hunger, episode progress, and four direction planes (up/right/down/left).
    All observations are float32 in [0, 1].
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
    TURNS = (0, -1, 1)
    CHANNELS = 13

    def __init__(self, config=None, *, render_mode=None):
        super().__init__()
        config = dict(config or {})
        self.width = int(config.get("width", 10))
        self.height = int(config.get("height", 10))
        if self.width < 5 or self.height < 5:
            raise ValueError("The board must be at least 5 by 5.")
        self.capacity = (self.width - 2) * (self.height - 2)
        self.initial_length = int(config.get("initial_length", 3))
        if not 2 <= self.initial_length < self.capacity:
            raise ValueError("initial_length must be between 2 and capacity - 1.")
        self.max_steps_without_apple = int(config.get("max_steps_without_apple", 4 * self.capacity))
        self.max_episode_steps = int(config.get("max_episode_steps", 100 * self.capacity))
        if min(self.max_steps_without_apple, self.max_episode_steps) < 1:
            raise ValueError("Episode limits must be positive.")
        self.rewards = Rewards(**config.get("rewards", {}))
        if self.rewards.step >= 0 or self.rewards.apple <= 0:
            raise ValueError("A step must cost reward and an apple must earn reward.")
        if self.rewards.collision >= 0 or self.rewards.starvation >= 0:
            raise ValueError("Collision and starvation must have negative rewards.")
        self.render_mode = render_mode or config.get("render_mode")
        if self.render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render mode: {self.render_mode}")
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0.0, 1.0, (self.height, self.width, self.CHANNELS), np.float32)
        self.snake = []
        self.apple = None
        self.direction = 1
        self.episode_steps = self.steps_since_apple = self.apples = 0
        self._finished = False
        self._figure = self._axis = self._image = None
        self._render_closed = False

    def settings(self):
        return {
            "width": self.width, "height": self.height,
            "initial_length": self.initial_length,
            "max_steps_without_apple": self.max_steps_without_apple,
            "max_episode_steps": self.max_episode_steps,
            "rewards": asdict(self.rewards),
        }

    def _inside(self, position):
        row, col = position
        return 1 <= row < self.height - 1 and 1 <= col < self.width - 1

    def _spawn_snake(self):
        # Bounded random walks avoid the unbounded DFS used in v8.
        for _ in range(100):
            path = [(int(self.np_random.integers(1, self.height - 1)),
                     int(self.np_random.integers(1, self.width - 1)))]
            while len(path) < self.initial_length:
                row, col = path[-1]
                candidates = [(row + dr, col + dc) for dr, dc in self.DIRECTIONS]
                candidates = [p for p in candidates if self._inside(p) and p not in path]
                if not candidates:
                    break
                path.append(candidates[int(self.np_random.integers(len(candidates)))])
            if len(path) == self.initial_length:
                head = path[0]
                if any(self._inside((head[0]+dr, head[1]+dc))
                       and (head[0]+dr, head[1]+dc) not in path[:-1]
                       for dr, dc in self.DIRECTIONS):
                    return path
        # A connected fallback also supports deliberately long initial snakes.
        path = []
        for row in range(1, self.height - 1):
            columns = list(range(1, self.width - 1))
            if row % 2 == 0:
                columns.reverse()
            path.extend((row, col) for col in columns)
        if self.np_random.integers(2):
            path.reverse()
        start = int(self.np_random.integers(len(path) - self.initial_length + 1))
        return path[start:start + self.initial_length]

    def _place_apple(self):
        occupied = set(self.snake)
        empty = [(r, c) for r in range(1, self.height - 1)
                 for c in range(1, self.width - 1) if (r, c) not in occupied]
        self.apple = empty[int(self.np_random.integers(len(empty)))] if empty else None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = self._spawn_snake()
        head, neck = self.snake[:2]
        vector = (head[0] - neck[0], head[1] - neck[1])
        self.direction = self.DIRECTIONS.index(vector)
        self.episode_steps = self.steps_since_apple = self.apples = 0
        self._finished = False
        self._place_apple()
        return self._observation(), self._info()

    def _info(self, reason="running"):
        return {
            "apples": self.apples, "length": len(self.snake),
            "steps": self.episode_steps, "steps_since_apple": self.steps_since_apple,
            "reason": reason, "won": reason == "win",
        }

    def _observation(self):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[0, :, 0] = obs[-1, :, 0] = 1
        obs[:, 0, 0] = obs[:, -1, 0] = 1
        if self.snake:
            positions = np.asarray(self.snake, dtype=np.intp)
            rows, cols = positions[:, 0], positions[:, 1]
            obs[rows[0], cols[0], 1] = 1
            obs[rows[1:], cols[1:], 2] = 1
            # Same vacancy-time encoding, assigned as one NumPy operation.
            obs[rows, cols, 6] = np.arange(len(self.snake), 0, -1, dtype=np.float64) / self.capacity
        if self.apple is not None:
            obs[self.apple[0], self.apple[1], 3] = 1
        if len(self.snake) > 1:
            obs[self.snake[1][0], self.snake[1][1], 4] = 1
        if self.snake:
            obs[self.snake[-1][0], self.snake[-1][1], 5] = 1
        obs[:, :, 7] = min(self.steps_since_apple / self.max_steps_without_apple, 1.0)
        obs[:, :, 8] = min(self.episode_steps / self.max_episode_steps, 1.0)
        obs[:, :, 9 + self.direction] = 1
        return obs

    def step(self, action):
        if self._finished:
            raise RuntimeError("Call reset() after the episode ends.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        self.episode_steps += 1
        self.direction = (self.direction + self.TURNS[int(action)]) % 4
        dr, dc = self.DIRECTIONS[self.direction]
        head = (self.snake[0][0] + dr, self.snake[0][1] + dc)
        eating = head == self.apple
        # Moving into the departing tail is legal when not growing.
        occupied = self.snake if eating else self.snake[:-1]
        if not self._inside(head) or head in occupied:
            reason = "wall" if not self._inside(head) else "body"
            self._finished = True
            return self._observation(), self.rewards.collision, True, False, self._info(reason)

        self.snake.insert(0, head)
        if eating:
            self.apples += 1
            self.steps_since_apple = 0
            reward = self.rewards.apple
            self._place_apple()
        else:
            self.snake.pop()
            self.steps_since_apple += 1
            reward = self.rewards.step

        terminated = truncated = False
        reason = "running"
        if len(self.snake) == self.capacity:
            terminated, reason = True, "win"
            reward += self.rewards.win_bonus
        elif self.steps_since_apple >= self.max_steps_without_apple:
            terminated, reason = True, "starvation"
            reward = self.rewards.starvation
        elif self.episode_steps >= self.max_episode_steps:
            truncated, reason = True, "time_limit"
        self._finished = terminated or truncated
        return self._observation(), float(reward), terminated, truncated, self._info(reason)

    @property
    def window_is_open(self):
        if self._figure is None:
            return not self._render_closed
        import matplotlib.pyplot as plt
        return plt.fignum_exists(self._figure.number)

    def render(self):
        rgb = np.ones((self.height, self.width, 3), dtype=np.uint8) * 245
        rgb[0, :] = rgb[-1, :] = 30
        rgb[:, 0] = rgb[:, -1] = 30
        for row, col in self.snake[1:]:
            rgb[row, col] = (40, 175, 90)
        if self.snake:
            rgb[self.snake[-1]] = (245, 185, 40)
            rgb[self.snake[0]] = (45, 100, 230)
        if self.apple is not None:
            rgb[self.apple] = (230, 55, 60)
        if self.render_mode == "rgb_array":
            return rgb
        if self.render_mode != "human" or self._render_closed:
            return None
        import matplotlib.pyplot as plt
        if self._figure is None:
            plt.ion()
            self._figure, self._axis = plt.subplots(figsize=(7, 7))
            self._figure.canvas.manager.set_window_title("SnakeAI v9")
            self._figure.canvas.mpl_connect("close_event", lambda _: setattr(self, "_render_closed", True))
            self._axis.set_xticks([])
            self._axis.set_yticks([])
            self._image = self._axis.imshow(rgb, interpolation="nearest")
            self._figure.text(0.5, 0.025, "Blue: head | Green: body | Yellow: tail | Red: apple", ha="center")
            plt.show(block=False)
        if not self.window_is_open:
            self._render_closed = True
            return None
        self._image.set_data(rgb)
        self._axis.set_title(f"SnakeAI v9 | Apples: {self.apples} | Length: {len(self.snake)} | Steps: {self.episode_steps}")
        plt.pause(1 / self.metadata["render_fps"])
        return None

    def close(self):
        self._render_closed = True
        if self._figure is not None:
            import matplotlib.pyplot as plt
            plt.close(self._figure)
