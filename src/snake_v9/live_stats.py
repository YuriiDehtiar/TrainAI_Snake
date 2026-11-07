"""Live episode statistics and small, atomically published inference snapshots."""

from collections import Counter, deque
from datetime import datetime
import json
import csv
from pathlib import Path
from statistics import mean
import os
import math
from zoneinfo import ZoneInfo

import torch

from .io import write_json


REASONS = ('running', 'wall', 'body', 'starvation', 'win', 'time_limit')


class EpisodeTracker:
    def __init__(self, window=100):
        self.recent = deque(maxlen=window)
        self.total = 0
        self.reasons = Counter()
        self.records = {}

    def add(self, rows, iteration):
        episodes = []
        for row in rows:
            episode = {'iteration': int(iteration), 'apples': int(row[1]),
                       'steps': int(row[2]), 'reward': float(row[3]),
                       'reason': REASONS[int(row[4])]}
            self.total += 1
            self.reasons[episode['reason']] += 1
            self.recent.append(episode)
            episodes.append(episode)
            for metric in ('apples', 'reward', 'steps'):
                for prefix, better in [('best', lambda a, b: a > b),
                                       ('worst', lambda a, b: a < b)]:
                    key = f'{prefix}_{metric}'
                    if key not in self.records or better(episode[metric], self.records[key][metric]):
                        self.records[key] = episode.copy()
        return episodes

    def summary(self):
        result = {'episodes_total': self.total, 'wins_total': self.reasons['win'],
                  'win_rate_total': self.reasons['win'] / self.total if self.total else 0,
                  'end_reasons_total': dict(self.reasons), 'records': self.records.copy()}
        if not self.recent:
            return result
        rows = list(self.recent)
        result.update(apples_mean=mean(r['apples'] for r in rows),
                      episode_return_mean=mean(r['reward'] for r in rows),
                      episode_len_mean=mean(r['steps'] for r in rows),
                      win_rate=mean(r['reason'] == 'win' for r in rows),
                      zero_apple_rate=mean(r['apples'] == 0 for r in rows),
                      early_death_rate=mean(r['steps'] <= 20 and r['reason'] != 'win' for r in rows),
                      starvation_rate=mean(r['reason'] == 'starvation' for r in rows),
                      recent_episodes=len(rows))
        for metric in ('apples', 'reward', 'steps'):
            result[f'{metric}_min'] = min(r[metric] for r in rows)
            result[f'{metric}_max'] = max(r[metric] for r in rows)
            result[f'{metric}_best_all'] = self.records[f'best_{metric}'][metric]
            result[f'{metric}_worst_all'] = self.records[f'worst_{metric}'][metric]
        return result


class LivePublisher:
    def __init__(self, run_dir, bootstrap_checkpoint, keep=3):
        self.directory = Path(run_dir) / 'live'
        self.directory.mkdir(exist_ok=True)
        self.bootstrap = str(bootstrap_checkpoint)
        self.keep = max(2, keep)
        self.last_iteration = None

    def publish(self, module, iteration, environment_steps):
        if iteration == self.last_iteration:
            return
        destination = self.directory / f'weights_{iteration:06d}.pt'
        temporary = destination.with_suffix(f'.{os.getpid()}.tmp')
        state = {key: value.detach().cpu() for key, value in module.state_dict().items()}
        torch.save({'iteration': iteration, 'environment_steps': environment_steps,
                    'state_dict': state}, temporary)
        temporary.replace(destination)
        write_json(self.directory / 'latest.json', {
            'iteration': iteration, 'environment_steps': environment_steps,
            'snapshot': str(destination), 'bootstrap_checkpoint': self.bootstrap,
            'published_at': datetime.now(ZoneInfo('Europe/Kyiv')).isoformat(),
        })
        self.last_iteration = iteration
        for old in sorted(self.directory.glob('weights_*.pt'))[:-self.keep]:
            try:
                old.unlink()
            except (PermissionError, FileNotFoundError):
                pass  # A reader may still be opening this generation on Windows.


class JsonlTail:
    def __init__(self, path):
        self.path = Path(path)
        self.position = 0
        self.pending = b''

    def read(self):
        try:
            with self.path.open('rb') as stream:
                if self.path.stat().st_size < self.position:
                    self.position = 0
                    self.pending = b''
                stream.seek(self.position)
                data = self.pending + stream.read()
                self.position = stream.tell()
        except FileNotFoundError:
            return []
        lines = data.split(b'\n')
        self.pending = lines.pop()
        return [json.loads(line.decode('utf-8')) for line in lines if line.strip()]


def scalar_metrics(result):
    """Read RLlib scalar metrics without serializing model tensors or Stats objects."""
    found = {}
    wanted = {'policy_loss', 'vf_loss', 'total_loss', 'entropy', 'mean_kl_loss',
              'curr_kl_coeff', 'vf_explained_var', 'gradients_default_optimizer_global_norm'}

    def visit(item, key=''):
        if isinstance(item, dict):
            for name, value in item.items():
                visit(value, str(name))
        elif key in wanted:
            try:
                if hasattr(item, 'peek'):
                    item = item.peek()
                if hasattr(item, 'item'):
                    item = item.item()
                found[key] = float(item)
            except (ValueError, TypeError, RuntimeError):
                pass
    visit(result)
    return found


def copy_history_until(source, destination, iteration, window=100, stdout_path=None):
    """Copy only history represented by a full checkpoint; keep source untouched."""
    source, destination = Path(source), Path(destination)
    tracker, last_metrics = EpisodeTracker(window), {}
    history, recovered, invalid_metrics, invalid_episodes = {}, [], 0, 0
    # A power loss may leave a zero-filled or incomplete JSONL tail. The flushed
    # console log is an independent copy of the iteration's scalar metrics.
    with (source / 'metrics.jsonl').open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                index = int(row['iteration'])
                if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in row.values()):
                    raise ValueError('Nonfinite metric')
                if 1 <= index <= iteration:
                    history[index] = row
            except (ValueError, TypeError, KeyError):
                invalid_metrics += 1
    stdout_path = Path(stdout_path) if stdout_path else source.parent.parent / 'sessions' / (source.name + '.stdout.log')
    if stdout_path.exists():
        with stdout_path.open(encoding='utf-8', errors='replace') as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    index = int(row['iteration'])
                    if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in row.values()):
                        continue
                    if 1 <= index <= iteration and index not in history:
                        row['recovered_from_stdout'] = True
                        history[index] = row
                        recovered.append(index)
                except (ValueError, TypeError, KeyError):
                    pass

    previous_iteration = None

    def finish_group(index):
        if index in recovered:
            history[index].update(tracker.summary())

    with (source / 'episodes.csv').open(newline='', encoding='utf-8', errors='replace') as stream:
        with (destination / 'episodes.csv').open('w', newline='', encoding='utf-8') as output:
            reader = csv.DictReader(stream)
            writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                try:
                    index = int(row['iteration'])
                    event = [1, int(row['apples']), int(row['steps']), float(row['reward']), REASONS.index(row['reason'])]
                    if not math.isfinite(event[3]):
                        raise ValueError('Nonfinite reward')
                except (ValueError, TypeError, KeyError):
                    invalid_episodes += 1
                    continue
                if index > iteration:
                    continue
                if previous_iteration is not None and index != previous_iteration:
                    finish_group(previous_iteration)
                tracker.add([event], index)
                writer.writerow(row)
                previous_iteration = index
            if previous_iteration is not None:
                finish_group(previous_iteration)
            output.flush()
            os.fsync(output.fileno())
    with (destination / 'metrics.jsonl').open('w', encoding='utf-8') as output:
        for index in sorted(history):
            row = history[index]
            output.write(json.dumps(row, allow_nan=False) + '\n')
            last_metrics = row
        output.flush()
        os.fsync(output.fileno())
    write_json(destination / 'history_recovery.json', {
        'source': str(source), 'checkpoint_iteration': iteration,
        'invalid_metric_lines_skipped': invalid_metrics, 'invalid_episode_rows_skipped': invalid_episodes,
        'iterations_recovered_from_stdout': sorted(recovered),
        'missing_iterations': [i for i in range(1, iteration + 1) if i not in history],
        'source_files_modified': False,
    })
    return tracker, last_metrics
