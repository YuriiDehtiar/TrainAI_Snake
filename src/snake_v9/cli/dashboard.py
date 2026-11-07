"""Visible training charts, episode records, a live Snake game."""

import os

# This process displays one game. The separate trainer owns the GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk

import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from snake_v9.io import write_json
from snake_v9.live_demo import LiveGame
from snake_v9.live_stats import JsonlTail

BG, PANEL, TEXT, MUTED = '#101827', '#172337', '#e7edf7', '#a6b5cc'
BLUE, GREEN, GOLD, RED = '#65adff', '#53dba3', '#ffd166', '#ff778b'
REASON_UA = {'wall': 'стіна', 'body': 'тіло', 'starvation': 'без яблук',
             'win': 'перемога', 'time_limit': 'ліміт ходів', 'running': 'гра триває'}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return default if default is not None else {}


def number(value, digits=1):
    return '—' if value is None else f'{value:,.{digits}f}'.replace(',', ' ')


class Dashboard:
    def __init__(self, root, run_dir):
        self.root, self.run_dir = root, Path(run_dir).resolve()
        self.tail = JsonlTail(self.run_dir / 'metrics.jsonl')
        self.history, self.status = [], {}
        self.game = None
        self.closed = False
        self.paused = tk.BooleanVar(value=False)
        self.fps = tk.DoubleVar(value=30)
        self.next_game_at = 0.0
        self.last_graph_iteration = -1
        self.title_text = tk.StringVar(value='Підключення до нового навчання…')
        self.details_text = tk.StringVar()
        self.message_text = tk.StringVar(value='Графіки та жива гра оновлюються незалежно від навчання.')
        self.game_text = tk.StringVar(value='Очікування перших ваг…')
        self.game_footer = tk.StringVar()
        self.record_text = tk.StringVar()
        self.end_text = tk.StringVar()
        root.title('SnakeAI v9 — графіки навчання та жива гра')
        root.configure(bg=BG)
        root.geometry('1720x1000')
        root.minsize(1280, 800)
        try:
            root.state('zoomed')
        except tk.TclError:
            pass
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.build_ui()
        root.after(100, self.refresh)
        root.after(200, self.game_tick)

    def label(self, parent, text=None, variable=None, size=11, color=TEXT, **kwargs):
        return tk.Label(parent, text=text, textvariable=variable, font=('Segoe UI', size),
                        bg=parent.cget('bg'), fg=color, justify='left', **kwargs)

    def build_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Treeview', background=PANEL, fieldbackground=PANEL, foreground=TEXT,
                        rowheight=27, font=('Segoe UI', 11), borderwidth=0)
        style.configure('Treeview.Heading', background='#24344e', foreground=TEXT,
                        font=('Segoe UI', 10, 'bold'))
        self.label(self.root, variable=self.title_text, size=18).pack(anchor='w', padx=20, pady=(12, 2))
        self.label(self.root, variable=self.details_text, size=11, color=MUTED).pack(anchor='w', padx=20)
        self.label(self.root, variable=self.message_text, size=11, color=GREEN).pack(anchor='w', padx=20, pady=(4, 8))
        content = tk.Frame(self.root, bg=BG)
        content.pack(fill='both', expand=True, padx=14)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=0, minsize=420)
        content.rowconfigure(0, weight=1)
        left = tk.Frame(content, bg=BG)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 12))
        right = tk.Frame(content, bg=PANEL, padx=12, pady=10)
        right.grid(row=0, column=1, sticky='nsew')
        self.figure = Figure(figsize=(10.5, 5.3), dpi=100, facecolor=BG)
        self.axes = self.figure.subplots(2, 2).ravel()
        self.figure.subplots_adjust(left=.075, right=.98, bottom=.09, top=.92, hspace=.48, wspace=.25)
        self.plot_canvas = FigureCanvasTkAgg(self.figure, master=left)
        self.plot_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.paint_graphs()
        self.label(left, text='Навчальні партії • середнє / мінімум / максимум за останні 100 завершених ігор', size=11).pack(anchor='w', pady=(6, 4))
        columns = ('metric', 'mean', 'min', 'max', 'worst', 'best')
        self.table = ttk.Treeview(left, columns=columns, show='headings', height=3, selectmode='none')
        for key, title, width in zip(columns,
                                    ('Показник', 'Середнє (100)', 'Мін. (100)', 'Макс. (100)', 'Мін. за весь запуск', 'Макс. за весь запуск'),
                                    (125, 125, 110, 110, 160, 160)):
            self.table.heading(key, text=title)
            self.table.column(key, width=width, minwidth=75, anchor='w' if key == 'metric' else 'center')
        for metric in ('apples', 'reward', 'steps'):
            self.table.insert('', 'end', iid=metric, values=('',) * 6)
        self.table.pack(fill='x')
        self.label(left, variable=self.record_text, size=10, color=GOLD, anchor='w').pack(fill='x', pady=(7, 2))
        self.label(left, variable=self.end_text, size=10, color=MUTED, anchor='w').pack(fill='x', pady=(0, 10))
        self.label(right, text='БЕЗПЕРЕРВНА ГРА МЕРЕЖІ', size=15).pack(anchor='w')
        self.label(right, text='Нові ваги — перед початком кожної партії', color=MUTED, size=10).pack(anchor='w', pady=(3, 8))
        self.board = tk.Canvas(right, width=390, height=390, bg=BG, highlightthickness=0)
        self.board.pack(fill='x')
        self.label(right, text='● голова    ● тіло    ● хвіст    ● яблуко', color=MUTED, size=10).pack(anchor='w', pady=(6, 8))
        self.label(right, variable=self.game_text, size=12, anchor='w').pack(fill='x')
        self.label(right, variable=self.game_footer, size=10, color=MUTED, anchor='w').pack(fill='x', pady=(10, 8))
        controls = tk.Frame(right, bg=PANEL)
        controls.pack(fill='x', pady=4)
        tk.Checkbutton(controls, text='Пауза гри', variable=self.paused,
                       bg=PANEL, fg=TEXT, selectcolor=BG, activebackground=PANEL,
                       activeforeground=TEXT, font=('Segoe UI', 11)).pack(side='left')
        self.label(controls, text='Ходів/с', size=10).pack(side='left', padx=(12, 0))
        tk.Scale(controls, from_=5, to=120, orient='horizontal', variable=self.fps,
                 resolution=1, length=145, bg=PANEL, fg=TEXT, troughcolor=BG,
                 highlightthickness=0, font=('Segoe UI', 9)).pack(side='left')
        self.stop_button = tk.Button(right, text='Зупинити навчання зі збереженням', command=self.stop_training,
                                    bg='#34425c', fg=TEXT, activebackground='#495c80',
                                    activeforeground=TEXT, relief='flat', padx=12, pady=8,
                                    font=('Segoe UI', 10))
        self.stop_button.pack(fill='x', pady=(12, 6))
        self.label(right, text='Пауза або закриття вікна не зупиняють навчання.\nГра обирає дію з найбільшим виходом мережі.\nПоказники гри й навчання рахуються окремо.',
                   color=MUTED, size=9).pack(anchor='w', pady=(2, 4))
        metadata = read_json(self.run_dir / 'run.json')
        initialization = ('відновлено зі збереженої контрольної точки' if metadata.get('resumed_from')
                          else 'початкові ваги з готової моделі' if metadata.get('initialization') == 'weights'
                          else 'випадкові початкові ваги')
        self.label(self.root, text=f'Запуск: {self.run_dir.name}  •  {initialization}  •  попередні моделі збережені',
                   color=MUTED, size=9).pack(anchor='w', padx=20, pady=(5, 8))

    def refresh(self):
        if self.closed:
            return
        try:
            self.status = read_json(self.run_dir / 'status.json', self.status)
            self.history.extend(self.tail.read())
            status = self.status
            metrics = status.get('metrics', {})
            phase = status.get('phase', 'starting')
            translated = {'starting': 'Ініціалізація', 'training': 'Навчання триває',
                          'completed': 'Навчання завершено', 'stopped': 'Навчання зупинено', 'failed': 'Помилка навчання'}
            iteration = status.get('iteration', 0)
            self.title_text.set(f"{translated.get(phase, phase)}  •  ітерація {iteration} / {status.get('total_iterations', 2000)}  •  ігор завершено: {metrics.get('episodes_total', 0):,}".replace(',', ' '))
            environments = status.get('environment_count', 1024)
            batch_size = status.get('batch_size', 65536)
            self.details_text.set(f"{environments} ігор × {batch_size // environments} ходів = {batch_size:,} нових переходів  |  мінібатч {status.get('minibatch_size', 16384):,}  |  {status.get('epochs', 10)} епох  |  переходів: {status.get('environment_steps', 0):,}".replace(',', ' '))
            if status.get('error'):
                self.message_text.set(status['error'])
            if iteration != self.last_graph_iteration and self.history:
                self.paint_graphs()
                self.last_graph_iteration = iteration
            self.update_table(metrics)
            if self.game is not None:
                try:
                    self.game.pointer()
                except (OSError, ValueError):
                    pass
                self.update_game_labels()
            write_json(self.run_dir / 'live_dashboard_status.json', {
                'pid': os.getpid(), 'updated_at': time.time(), 'training_iteration': iteration,
                'loaded_iteration': self.game.loaded_iteration if self.game else -1,
                'latest_iteration': self.game.latest_iteration if self.game else -1,
                'demo_games': self.game.games_finished if self.game else 0,
                'demo_steps': self.game.env.episode_steps if self.game else 0,
                'demo_paused': self.paused.get(),
                'error': self.game.error if self.game else None})
            if phase in ('completed', 'stopped', 'failed'):
                self.stop_button.configure(state='disabled')
        except Exception as error:
            self.message_text.set(f'Вікно: {type(error).__name__}: {error}')
        self.root.after(1500, self.refresh)

    def paint_graphs(self):
        titles = ('Яблука за гру • останні 100 ігор', 'Перемоги • частка завершених ігор',
                  'Винагорода за гру • останні 100 ігор', 'PPO • похибка під час оновлення')
        for axis, title in zip(self.axes, titles):
            axis.clear()
            axis.set_facecolor(PANEL)
            axis.set_title(title, color=TEXT, fontsize=11, loc='left', pad=10)
            axis.set_xlabel('Ітерація навчання', color=MUTED, fontsize=9)
            axis.tick_params(colors=MUTED, labelsize=8)
            axis.grid(alpha=.13, color=MUTED)
            for spine in axis.spines.values():
                spine.set_color('#354058')
        if self.history:
            x = [row['iteration'] for row in self.history]
            for axis, key, color in [(self.axes[0], 'apples', GREEN), (self.axes[2], 'reward', BLUE)]:
                mean_key = 'apples_mean' if key == 'apples' else 'episode_return_mean'
                axis.fill_between(x, [r.get(f'{key}_min', float('nan')) for r in self.history],
                                  [r.get(f'{key}_max', float('nan')) for r in self.history], color=color, alpha=.16, label='мін. — макс.')
                axis.plot(x, [r.get(mean_key, float('nan')) for r in self.history], color=color, linewidth=1.8, label='середнє')
            self.axes[1].plot(x, [100 * r.get('win_rate', 0) for r in self.history], color=GOLD, label='останні 100 ігор')
            self.axes[1].plot(x, [100 * r.get('win_rate_total', 0) for r in self.history], color=BLUE, alpha=.8, label='за весь запуск')
            self.axes[1].set_ylim(-2, 102)
            self.axes[1].set_ylabel('%', color=MUTED, fontsize=9)
            for key, label, color in [('policy_loss', 'актор', BLUE), ('vf_loss', 'критик', GOLD), ('total_loss', 'загальна', GREEN)]:
                if any(key in r for r in self.history):
                    self.axes[3].plot(x, [r.get(key, float('nan')) for r in self.history], label=label, color=color)
            for axis in self.axes:
                if axis.get_legend_handles_labels()[0]:
                    axis.legend(facecolor=PANEL, edgecolor=PANEL, labelcolor=MUTED, fontsize=8, loc='best')
        self.plot_canvas.draw_idle()

    def update_table(self, metrics):
        for key, label, mean_key, digits in [('apples', 'Яблука', 'apples_mean', 1),
                                            ('reward', 'Винагорода', 'episode_return_mean', 2),
                                            ('steps', 'Ходи', 'episode_len_mean', 0)]:
            self.table.item(key, values=(label, number(metrics.get(mean_key), digits),
                                        number(metrics.get(f'{key}_min'), digits), number(metrics.get(f'{key}_max'), digits),
                                        number(metrics.get(f'{key}_worst_all'), digits), number(metrics.get(f'{key}_best_all'), digits)))
        records = metrics.get('records', {})
        lines = []
        for key, label in [('best_reward', 'Найвища винагорода'), ('worst_reward', 'Найнижча винагорода')]:
            if key in records:
                row = records[key]
                lines.append(f"{label}: {row['reward']:.2f} • {row['apples']} яблук • {row['steps']} ходів • ітерація {row['iteration']}")
        self.record_text.set('\n'.join(lines))
        reasons = metrics.get('end_reasons_total', {})
        self.end_text.set(f"Перемоги: {metrics.get('wins_total', 0)}  •  за останні 100: {100 * metrics.get('win_rate', 0):.1f}%  •  за весь запуск: {100 * metrics.get('win_rate_total', 0):.1f}%\n" +
                          'Завершення: ' + '   |   '.join(f'{REASON_UA.get(key, key)}: {value:,}'.replace(',', ' ') for key, value in reasons.items()))

    def game_tick(self):
        if self.closed:
            return
        delay = max(8, int(1000 / self.fps.get()))
        try:
            if self.game is None:
                if (self.run_dir / 'run.json').exists() and (self.run_dir / 'live/latest.json').exists():
                    self.game = LiveGame(self.run_dir)
                else:
                    self.root.after(500, self.game_tick)
                    return
            if not self.paused.get() and not (self.run_dir / 'STOP_DEMOS').exists():
                if not self.game.active:
                    if time.monotonic() >= self.next_game_at:
                        if not self.game.new_game():
                            delay = 1000
                else:
                    self.game.step()
                    if not self.game.active:
                        self.next_game_at = time.monotonic() + .35
                self.draw_board()
            self.update_game_labels()
        except Exception as error:
            self.game_text.set(f'Гра: {type(error).__name__}: {error}')
            delay = 1500
        self.root.after(delay, self.game_tick)

    def draw_board(self):
        env = self.game.env
        self.board.delete('all')
        size = min(self.board.winfo_width(), 420)
        self.board.configure(height=size)
        cell = size / max(env.width, env.height)
        positions = {position: index for index, position in enumerate(env.snake)}
        for row in range(env.height):
            for col in range(env.width):
                position = (row, col)
                color = '#23304a' if row in (0, env.height - 1) or col in (0, env.width - 1) else '#101b2c'
                if position in positions:
                    index = positions[position]
                    color = BLUE if index == 0 else GOLD if index == len(env.snake) - 1 else GREEN
                self.board.create_rectangle(col * cell + 1, row * cell + 1, (col + 1) * cell - 1,
                                            (row + 1) * cell - 1, fill=color, outline='')
                if position == env.apple:
                    self.board.create_oval(col * cell + 7, row * cell + 7,
                                           (col + 1) * cell - 7, (row + 1) * cell - 7, fill=RED, outline='')
        if env.snake:
            row, col = env.snake[0]
            dr, dc = env.DIRECTIONS[env.direction]
            x, y = (col + .5) * cell, (row + .5) * cell
            self.board.create_line(x, y, x + dc * cell * .3, y + dr * cell * .3, fill=BG, width=3, arrow='last')

    def update_game_labels(self):
        game = self.game
        if game is None:
            return
        env = game.env
        result = game.last_result
        self.game_text.set(f'Партія № {game.games_started}  |  яблука: {env.apples} / 61\n'
                           f'Ходів: {env.episode_steps}  |  винагорода: {game.total_reward:.2f}\n'
                           f'Без яблука: {env.steps_since_apple} / {env.max_steps_without_apple}\n'
                           f'Ваги у грі: ітерація {game.loaded_iteration}\n'
                           f'Найновіші доступні: ітерація {game.latest_iteration}')
        last = 'Ще немає завершених показових ігор.' if not result else f"Остання: {REASON_UA[result['reason']]} • {result['apples']} яблук\nВинагорода: {result['reward']:.2f} • {result['steps']} ходів"
        self.game_footer.set(f'Показових ігор: {game.games_finished}  |  перемог: {game.wins}\n{last}' +
                             (f'\nОчікування ваг: {game.error}' if game.error else ''))

    def stop_training(self):
        (self.run_dir / 'STOP').write_text('requested from live dashboard\n', encoding='utf-8')
        self.stop_button.configure(text='Збереження після поточної ітерації…', state='disabled')

    def close(self):
        self.closed = True
        if self.game:
            self.game.env.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = tk.Tk()
    Dashboard(root, args.run_dir)
    root.mainloop()


if __name__ == '__main__':
    main()
