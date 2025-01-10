# Game_component/game_board.py
import numpy as np
import random

class GameBoard:
    """
    Class GameBoard is responsible for controlling the playing field, the snake, the apple, and the walls.
    """

    EMPTY = 0       # Порожня клітинка
    SNAKE_BODY = 1  # Сегмент тіла змійки (окрім голови)
    SNAKE_HEAD = 2  # Голова змійки
    APPLE = 3       # Яблуко
    WALL = 4        # Стіна

    def __init__(self, width=10, height=10, initial_tail_length=1, random_tail_length_range=None):
        """
        Initializes the playing field.
        
        :param width: the width of the field (number of columns)
        :param height: the height of the field (number of rows)
        :param initial_tail_length: the initial snake tail length
        :param random_tail_length_range: a tuple (min_len, max_len) for random tail length
        """
        self.width = width
        self.height = height
        self.initial_tail_length = initial_tail_length
        self.random_tail_length_range = random_tail_length_range
        self.reset()

    def reset(self):
        """
        Resets the playing field to its initial state:
        - Clears the field and places walls around the perimeter
        - Determines the tail length (if a range is specified, it is chosen randomly)
        - Places the snake's head in the center
        - Adds the snake's tail (if possible)
        - Sets the direction of movement (to the right)
        - Places an apple
        """
        self.state = np.zeros((self.height, self.width), dtype=np.int8)

        # Додаємо стіни по периметру
        self.state[0, :] = self.WALL
        self.state[-1, :] = self.WALL
        self.state[:, 0] = self.WALL
        self.state[:, -1] = self.WALL

        # Визначаємо довжину хвоста (random_tail_length_range або initial_tail_length)
        if self.random_tail_length_range is not None:
            min_len, max_len = self.random_tail_length_range
            tail_length = random.randint(min_len, max_len)
        else:
            tail_length = self.initial_tail_length

        # Розміщуємо голову змійки в центрі
        start_x = self.height // 2
        start_y = self.width // 2
        self.snake = [(start_x, start_y)]
        self.state[start_x, start_y] = self.SNAKE_HEAD

        # Розміщуємо хвіст, якщо це можливо
        placed_tail = self._try_place_tail(start_x, start_y, tail_length)
        # placed_tail — фактично покладена довжина хвоста (без голови).

        # Початковий напрямок руху (вправо)
        self.direction = (0, 1)

        # Розміщуємо яблуко
        self._place_apple()

    def _place_apple(self):
        """
        Places an apple in a random empty cell if one is available.
        """
        empty_positions = np.argwhere(self.state == self.EMPTY)
        if len(empty_positions) > 0:
            apple_pos = empty_positions[random.randint(0, len(empty_positions) - 1)]
            self.state[apple_pos[0], apple_pos[1]] = self.APPLE

    def set_direction(self, action):
        """
        Sets the snake's direction based on the player's/AI's action.

        :param action:
            0 - up (dx, dy) = (-1, 0)
            1 - down (dx, dy) = (1, 0)
            2 - left (dx, dy) = (0, -1)
            3 - right (dx, dy) = (0, 1)
        """
        directions = {
            0: (-1, 0),  # вгору
            1: (1, 0),   # вниз
            2: (0, -1),  # вліво
            3: (0, 1)    # вправо
        }
        dx, dy = directions[action]
        self.direction = (dx, dy)

    def step(self):


        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_x = head_x + dx
        new_y = head_y + dy

        # Перевіряємо, чи не відбулося 180°-зіткнення
        self_collision_180 = False
        if len(self.snake) > 1:
            second_x, second_y = self.snake[1]
            if (new_x, new_y) == (second_x, second_y):
                self_collision_180 = True
                return True, False, self_collision_180

        next_cell = self.state[new_x, new_y]

        # Якщо зіткнулися зі стіною чи тілом
        if next_cell == self.WALL or next_cell == self.SNAKE_BODY:
            return True, False, False

        eaten_apple = (next_cell == self.APPLE)

        # Додаємо нову голову
        self.snake.insert(0, (new_x, new_y))
        self.state[new_x, new_y] = self.SNAKE_HEAD

        # Оновлюємо колишню голову до стану тіла
        if len(self.snake) > 1:
            old_head_x, old_head_y = self.snake[1]
            self.state[old_head_x, old_head_y] = self.SNAKE_BODY

        if eaten_apple:
            self._place_apple()
        else:
            # Видаляємо хвіст, якщо яблуко не було з’їдено
            tail_x, tail_y = self.snake.pop()
            self.state[tail_x, tail_y] = self.EMPTY

        return False, eaten_apple, False

    def get_state(self):
        """
        Returns a copy of the current state of the playing field (a matrix with labels).
        """
        return self.state.copy()

    def _try_place_tail(self, head_x, head_y, tail_length):
        """
        Attempts to place a tail of length tail_length, starting from the head (head_x, head_y).

        :param head_x: the row where the head is located
        :param head_y: the column where the head is located
        :param tail_length: the desired tail length
        :return: the actual placed tail length (excluding the head)
        """
        if tail_length <= 0:
            return 0

        visited = set(self.snake)  # голова вже зайнята
        path = []

        if self._place_tail_dfs(head_x, head_y, tail_length, visited, path, head_x, head_y):
            for (x, y) in path:
                self.snake.append((x, y))
                self.state[x, y] = self.SNAKE_BODY
            return tail_length
        else:
            return 0

    def _place_tail_dfs(self, x, y, segments_left, visited, path, head_x, head_y):
        """
        Recursively attempts to place 'segments_left' tail segments starting from (x, y).

        :param x: current row
        :param y: current column
        :param segments_left: how many more segments need to be placed
        :param visited: set of already occupied (or checked) cells
        :param path: the path along which we are "laying" the tail
        :param head_x: the row coordinate of the forbidden zone (head)
        :param head_y: the column coordinate of the forbidden zone (head)
        :return: True if all segments were successfully placed; otherwise False
        """
        if segments_left == 0:
            return True

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.height and 0 <= ny < self.width:
                if self.state[nx, ny] == self.EMPTY and (nx, ny) not in visited:
                    # Перевірка "забороненої зони" після 5-го сегмента
                    if len(path) >= 5:
                        dist = abs(nx - head_x) + abs(ny - head_y)
                        if dist < 4:
                            continue

                    visited.add((nx, ny))
                    path.append((nx, ny))

                    if self._place_tail_dfs(nx, ny, segments_left - 1, visited, path, head_x, head_y):
                        return True

                    path.pop()
                    visited.remove((nx, ny))

        return False


