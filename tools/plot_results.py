"""Regenerate repository figures exclusively from public game/training data."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'results/latest'
    rows = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    output = ROOT / 'docs/assets'
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .2})
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    x = [r['iteration'] for r in rows]
    axes[0].plot(x, [r['apples_mean'] for r in rows], color='#2563eb', linewidth=.8)
    axes[0].set(title='Яблука за гру', xlabel='Ітерація', ylabel='Середнє за останні 100 ігор', ylim=(0, 61), xlim=(1, 2000))
    axes[1].plot(x, [100 * r['win_rate'] for r in rows], color='#079c8e', linewidth=.8)
    axes[1].set(title='Перемоги в навчальних іграх', xlabel='Ітерація', ylabel='Останні 100 ігор, %', ylim=(0, 100), xlim=(1, 2000))
    fig.savefig(output / 'training_history.png', dpi=160)
    plt.close(fig)
    values = []
    for step in (950, 2000):
        data = json.loads((folder / f'evaluation_{step:06d}.json').read_text(encoding='utf-8'))
        values.append(100 * data['win_rate'])
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    bars = ax.bar(['Ітерація 950', 'Ітерація 2000'], values, color=['#2563eb', '#079c8e'], width=.55)
    ax.bar_label(bars, labels=['77%', '89%'], padding=5, fontsize=14)
    ax.set(title='Однакові 200 нових тестових ігор', ylabel='Перемоги, %', ylim=(0, 100))
    fig.savefig(output / 'model_comparison.png', dpi=160)
    plt.close(fig)
    print('Updated training_history.png and model_comparison.png')


if __name__ == '__main__':
    main()
