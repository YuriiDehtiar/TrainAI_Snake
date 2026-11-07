import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from snake_v9.env import SnakeEnv


class SnakeEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = SnakeEnv()
        self.env.reset(seed=42)

    def tearDown(self):
        self.env.close()

    def state(self, snake, apple, direction=1):
        self.env.snake = list(snake)
        self.env.apple = apple
        self.env.direction = direction
        self.env.episode_steps = self.env.steps_since_apple = self.env.apples = 0
        self.env._finished = False

    def test_gymnasium_contract_and_seed(self):
        check_env(self.env, skip_render_check=True)

    def test_seed_reproduces_initial_state(self):
        first, _ = self.env.reset(seed=123)
        second, _ = self.env.reset(seed=123)
        np.testing.assert_array_equal(first, second)

    def test_observation_marks_neck_tail_order_and_direction(self):
        self.state([(4, 4), (4, 3), (4, 2)], (2, 2))
        obs = self.env._observation()
        self.assertTrue(self.env.observation_space.contains(obs))
        self.assertEqual(obs[4, 3, 4], 1)
        self.assertEqual(obs[4, 2, 5], 1)
        self.assertGreater(obs[4, 3, 6], obs[4, 2, 6])
        self.assertTrue(np.all(obs[:, :, 10] == 1))

    def test_relative_actions_never_reverse_into_neck(self):
        for action in range(3):
            self.state([(4, 4), (4, 3), (4, 2)], (2, 2))
            _, _, terminated, _, _ = self.env.step(action)
            self.assertFalse(terminated)
            self.assertNotEqual(self.env.snake[0], (4, 3))

    def test_apple_grows_snake_and_resets_hunger(self):
        self.state([(4, 4), (4, 3), (4, 2)], (4, 5))
        self.env.steps_since_apple = 20
        _, reward, done, _, info = self.env.step(0)
        self.assertEqual(reward, 1)
        self.assertFalse(done)
        self.assertEqual(len(self.env.snake), 4)
        self.assertEqual(info['apples'], 1)
        self.assertEqual(info['steps_since_apple'], 0)
        self.assertNotIn(self.env.apple, self.env.snake)

    def test_move_into_vacating_tail_is_legal(self):
        self.state([(3, 3), (3, 2), (4, 2), (4, 3)], (1, 1), direction=1)
        _, _, done, _, _ = self.env.step(2)
        self.assertFalse(done)
        self.assertEqual(self.env.snake[0], (4, 3))
        self.assertEqual(len(set(self.env.snake)), 4)

    def test_collision_with_non_tail_body_ends_episode(self):
        self.state([(3, 3), (3, 2), (4, 2), (4, 3), (4, 4)], (1, 1), direction=1)
        _, reward, done, _, info = self.env.step(2)
        self.assertTrue(done)
        self.assertEqual(reward, -1)
        self.assertEqual(info['reason'], 'body')

    def test_wall_collision_is_terminal(self):
        self.state([(1, 3), (2, 3), (3, 3)], (5, 5), direction=0)
        obs, reward, done, truncated, info = self.env.step(0)
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertEqual(info['reason'], 'wall')
        self.assertTrue(self.env.observation_space.contains(obs))

    def test_survival_after_200_steps_has_no_positive_reward(self):
        self.state([(4, 4), (4, 3), (4, 2)], (2, 2))
        self.env.episode_steps = 201
        self.env.steps_since_apple = 30
        _, reward, _, _, _ = self.env.step(0)
        self.assertLess(reward, 0)

    def test_starvation_ends_episode_and_reset_clears_counters(self):
        self.state([(4, 4), (4, 3), (4, 2)], (2, 2))
        self.env.steps_since_apple = self.env.max_steps_without_apple - 1
        _, reward, terminated, truncated, info = self.env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['reason'], 'starvation')
        self.assertLess(reward, 0)
        self.env.reset()
        self.assertEqual(self.env.steps_since_apple, 0)
        self.assertEqual(self.env.episode_steps, 0)

    def test_episode_step_limit_truncates(self):
        self.state([(4, 4), (4, 3), (4, 2)], (2, 2))
        self.env.episode_steps = self.env.max_episode_steps - 1
        _, _, terminated, truncated, info = self.env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info['reason'], 'time_limit')

    def test_full_board_is_a_win(self):
        self.env.close()
        self.env = SnakeEnv({'width': 5, 'height': 5})
        self.env.reset(seed=1)
        self.state([(1, 2), (1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3), (2, 2)], (1, 3), direction=1)
        obs, reward, done, truncated, info = self.env.step(0)
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertTrue(info['won'])
        self.assertIsNone(self.env.apple)
        self.assertGreater(reward, 1)
        self.assertTrue(self.env.observation_space.contains(obs))

    def test_random_episodes_preserve_invariants(self):
        self.env.action_space.seed(7)
        obs, _ = self.env.reset(seed=7)
        for _ in range(3000):
            obs, reward, terminated, truncated, info = self.env.step(self.env.action_space.sample())
            self.assertTrue(self.env.observation_space.contains(obs))
            self.assertTrue(np.isfinite(reward))
            self.assertEqual(len(self.env.snake), len(set(self.env.snake)))
            self.assertNotIn(self.env.apple, self.env.snake)
            if terminated or truncated:
                self.env.reset()


if __name__ == '__main__':
    unittest.main()
