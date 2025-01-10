# Game_component/snake_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .game_board import GameBoard
from .reward_function import RewardFunction
from .renderer import Renderer

class SnakeEnv(gym.Env):
    """
    Snake environment compatible with Gym (Gymnasium).
    Used for reinforcement learning.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, width=10, height=10, render=False):
        """
        Initializing the snake environment:
        - width and height set the dimensions of the playing field (in cells).
        - render determines whether a renderer will be used for visualization.
        """
        super(SnakeEnv, self).__init__()
        self.width = width
        self.height = height

        # Доступні дії: 0 - вгору, 1 - вниз, 2 - вліво, 3 - вправо
        self.action_space = spaces.Discrete(4)

        # Спостереження: поле розміром (height, width, 1) зі значеннями від 0 до 4
        self.observation_space = spaces.Box(
            low=0,
            high=4,
            shape=(self.height, self.width, 1),
            dtype=np.float32
        )

        # Ігрова логіка
        self.game = GameBoard(self.width, self.height)
        self.reward_function = RewardFunction()
        self.last_state = None

        # Рендерер (для візуалізації), якщо render=True
        self.renderer = Renderer() if render else None

    def reset(self, *, seed=None, options=None):
        """
        Resets the environment to its initial state:
        - Calls reset() on the GameBoard.
        - Gets the initial state as a matrix (height x width) and transforms it.
        """
        super().reset(seed=seed)

        # Стандартний reset
        self.game.reset()
        curr_state = self.game.get_state().astype(np.float32)
        curr_state = np.expand_dims(curr_state, axis=-1)
        self.last_state = curr_state

        # Перевірка на NaN або Inf
        if np.isnan(self.last_state).any() or np.isinf(self.last_state).any():
            raise ValueError("NaN or Inf detected in the initial state after reset().")

        # Повертаємо початковий стан і порожній словник info
        return self.last_state, {}

    def step(self, action):
        """
        Performs a game step:
        1. Sets the direction of the snake's movement (game.set_direction(action)).
        2. Calls game.step(), which returns (done, eaten_apple, self_collision_180).
        3. Forms the current state (curr_state) and calculates the reward (reward).
        4. Returns (curr_state, reward, done, truncated, info).
        """
        # Встановлення напряму руху
        self.game.set_direction(action)

        # Крок гри
        done, eaten_apple, self_collision_180 = self.game.step()

        # Формуємо новий стан
        curr_state = self.game.get_state().astype(np.float32)
        curr_state = np.expand_dims(curr_state, axis=-1)

        # Перевірка на NaN або Inf
        if np.isnan(curr_state).any() or np.isinf(curr_state).any():
            raise ValueError("NaN or Inf detected in 'curr_state' after game.step().")

        # Обчислення винагороди
        reward = self.reward_function.compute_reward(
            prev_state=self.last_state,
            curr_state=curr_state,
            done=done,
            eaten_apple=eaten_apple,
            self_collision_180=self_collision_180
        )

        # Перевірка на NaN або Inf у винагороді
        if np.isnan(reward) or np.isinf(reward):
            raise ValueError(f"NaN or Inf detected in reward after step. reward={reward}")

        # Оновлюємо останній стан
        self.last_state = curr_state

        # У Gymnasium повертаємо: (obs, reward, done, truncated, info)
        info = {}
        truncated = False
        return curr_state, reward, done, truncated, info

    def render(self, mode='human'):
        if mode == 'human' and self.renderer is not None:
            self.renderer.render(self.last_state)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

