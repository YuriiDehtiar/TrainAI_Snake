class RewardFunction:
    """
    Updated reward function with "grace periods":
    - Steps_in_episode tracks total steps in current episode.
    - steps_since_apple tracks steps since last apple eaten.
    - First 25 steps of the episode: no penalty.
    - Also first 25 steps after eating an apple: no penalty.
    Other logic:
    - If done: -100, reset counters.
    - If apple eaten: +40, steps_since_apple=0
    - If no apple for >250 steps_since_apple: -10
    - Else if >150 steps_since_apple: -5
    - Else: -0.1
    Bonus based on steps_in_episode:
    - If steps_in_episode >250: +2
    - Else if steps_in_episode >150: +1
    - Else 0
    """

    def __init__(self):
        self.steps_since_apple = 0
        self.steps_in_episode = 0

    def compute_reward(self, 
                       prev_state, 
                       curr_state, 
                       done, 
                       eaten_apple,
                       self_collision_180=False):
        self.steps_in_episode += 1

        if done:
            # Спочатку перевіряємо спеціальний випадок 180-градусного самозідання
            if self_collision_180:
                reward = -500.0
            else:
                reward = -100.0

            # Скидаємо лічильники
            self.steps_since_apple = 0
            self.steps_in_episode = 0
            return reward

        if eaten_apple:
            self.steps_since_apple = 0
            base_reward = 40.0
        else:
            self.steps_since_apple += 1
            # Перевірка на "безштрафний" період
            # Якщо steps_in_episode <= 25 або steps_since_apple <= 25
            # не застосовуємо штраф, ставимо мінімальний базовий штраф = 0
            if self.steps_in_episode <= 25 or self.steps_since_apple <= 25:
                base_reward = -0.1  # Без штрафу
            else:
                # Застосовуємо логіку штрафів згідно кроків без яблука
                if self.steps_since_apple > 200:
                    base_reward = -10.0
                elif self.steps_since_apple > 100:
                    base_reward = -5.0
                else:
                    base_reward = -0.2

        # Бонус за довгий епізод
        if self.steps_in_episode > 200:
            bonus = 2
        elif self.steps_in_episode > 100:
            bonus = 1
        else:
            bonus = 0

        return base_reward + bonus

