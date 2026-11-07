import json
from pathlib import Path
import tempfile
import unittest

import torch

from snake_v9.live_stats import EpisodeTracker, JsonlTail, LivePublisher, copy_history_until
from snake_v9.live_demo import LiveGame


class LiveTests(unittest.TestCase):
    def test_recovery_uses_independent_console_log_after_zero_filled_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / 'source', Path(directory) / 'target'
            source.mkdir()
            target.mkdir()
            (source / 'episodes.csv').write_text('iteration,apples,steps,reward,reason\n1,1,5,0.9,wall\n2,61,400,60,win\n3,0,1,-1,body\n')
            damaged = b'{"iteration": 1, "apples_mean": 1}\n' + b'\0' * 256
            (source / 'metrics.jsonl').write_bytes(damaged)
            console = Path(directory) / 'stdout.log'
            console.write_text('startup\n{"iteration": 2, "total_loss": 0.4}\n{"iteration": 3}\n')
            tracker, metrics = copy_history_until(source, target, 2, stdout_path=console)
            rows = [json.loads(line) for line in (target / 'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([row['iteration'] for row in rows], [1, 2])
            self.assertEqual(metrics['episodes_total'], 2)
            self.assertEqual(metrics['records']['best_apples']['apples'], 61)
            self.assertEqual(json.loads((target / 'history_recovery.json').read_text())['iterations_recovered_from_stdout'], [2])
            self.assertEqual((source / 'metrics.jsonl').read_bytes(), damaged)

    def test_recovery_history_excludes_updates_after_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / 'source', Path(directory) / 'target'
            source.mkdir()
            target.mkdir()
            (source / 'episodes.csv').write_text('iteration,apples,steps,reward,reason\n1,1,5,0.9,wall\n2,61,400,60,win\n3,0,1,-1,body\n')
            original = ''.join(json.dumps({'iteration': i, 'apples_mean': i}) + '\n' for i in (1, 2, 3))
            (source / 'metrics.jsonl').write_text(original)
            tracker, metrics = copy_history_until(source, target, 2)
            self.assertEqual(tracker.total, 2)
            self.assertEqual(tracker.summary()['wins_total'], 1)
            self.assertEqual(metrics['iteration'], 2)
            self.assertEqual(len((target / 'metrics.jsonl').read_text().splitlines()), 2)
            self.assertEqual((source / 'metrics.jsonl').read_text(), original)

    def test_game_refreshes_weights_between_games_and_reuses_without_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'run.json').write_text(json.dumps({'env_config': {}}))
            source = torch.nn.Linear(2, 1)
            publisher = LivePublisher(root, root / 'bootstrap')
            with torch.no_grad():
                source.weight.fill_(1)
            publisher.publish(source, 0, 0)
            game = LiveGame(root, loader=lambda _: torch.nn.Linear(2, 1), action_fn=lambda *_: 0)
            self.assertTrue(game.new_game())
            with torch.no_grad():
                source.weight.fill_(2)
            publisher.publish(source, 1, 64)
            while game.active:
                game.step()
                self.assertEqual(game.loaded_iteration, 0)
                self.assertTrue(torch.all(game.module.weight == 1))
            self.assertEqual(game.last_result['weights_iteration'], 0)
            self.assertTrue(game.new_game())
            self.assertEqual(game.loaded_iteration, 1)
            self.assertTrue(torch.all(game.module.weight == 2))
            while game.active:
                game.step()
            self.assertTrue(game.new_game())
            self.assertEqual(game.loaded_iteration, 1)
            rows = [json.loads(line) for line in (root / 'demo_games.jsonl').read_text().splitlines()]
            self.assertEqual([r['weights_iteration'] for r in rows], [0, 1])

    def test_window_and_independent_lifetime_records(self):
        tracker = EpisodeTracker(window=2)
        tracker.add([[1, 8, 90, -2, 2], [1, 2, 40, 1.5, 1]], 1)
        tracker.add([[1, 61, 500, 50, 4]], 2)
        result = tracker.summary()
        self.assertEqual(result['episodes_total'], 3)
        self.assertEqual(result['win_rate'], 0.5)
        self.assertEqual(result['apples_min'], 2)
        self.assertEqual(result['reward_worst_all'], -2)
        self.assertEqual(result['records']['best_reward']['iteration'], 2)
        self.assertEqual(result['end_reasons_total'], {'body': 1, 'wall': 1, 'win': 1})

    def test_tailer_waits_for_complete_line_and_handles_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.jsonl'
            tail = JsonlTail(path)
            self.assertEqual(tail.read(), [])
            path.write_bytes(b'{"value": 1}\n{"value":')
            self.assertEqual(tail.read(), [{'value': 1}])
            with path.open('ab') as stream:
                stream.write(b' 2}\n')
            self.assertEqual(tail.read(), [{'value': 2}])
            self.assertEqual(tail.read(), [])
            path.write_bytes(b'{"new": 3}\n')
            self.assertEqual(tail.read(), [{'new': 3}])

    def test_published_weights_match_pointer_and_keep_recent_generations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module = torch.nn.Linear(2, 1)
            publisher = LivePublisher(root, root / 'bootstrap', keep=2)
            for iteration in range(4):
                with torch.no_grad():
                    module.weight.fill_(iteration)
                publisher.publish(module, iteration, iteration * 10)
            pointer = json.loads((root / 'live/latest.json').read_text())
            payload = torch.load(pointer['snapshot'], weights_only=False)
            self.assertEqual(payload['iteration'], pointer['iteration'])
            self.assertEqual(payload['environment_steps'], 30)
            self.assertTrue(torch.equal(payload['state_dict']['weight'], module.weight))
            self.assertEqual(len(list((root / 'live').glob('weights_*.pt'))), 2)


if __name__ == '__main__':
    unittest.main()
