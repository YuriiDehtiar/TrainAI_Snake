"""Fresh CUDA PPO training with episode records and current inference weights."""

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import torch

from snake_v9.config import make_config
from snake_v9.checkpoints import checkpoint_settings
from snake_v9.env import SnakeEnv
from snake_v9.gpu_env import GPUSnakeBatch
from snake_v9.gpu_rollout import GPURollout
from snake_v9.io import write_json
from snake_v9.live_stats import EpisodeTracker, LivePublisher, scalar_metrics, copy_history_until
from snake_v9.numerics import LogSpaceCategorical, require_finite_weights
from snake_v9.performance import configure_gpu

from snake_v9.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--settings-from', type=Path)
    parser.add_argument('--resume', type=Path, help='Restore a full checkpoint into a new run; iterations is the total target.')
    parser.add_argument('--weights', type=Path, help='Start a new run from portable weights, with a fresh optimizer and iteration counter.')
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--num-envs', type=int, default=1024)
    parser.add_argument('--horizon', type=int, default=64)
    parser.add_argument('--minibatch-size', type=int, default=16384)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-every', type=int, default=20)
    args = parser.parse_args()
    if args.resume and args.weights:
        parser.error('Use either --resume or --weights.')
    if not torch.cuda.is_available():
        parser.error('CUDA is required.')
    if min(args.iterations, args.num_envs, args.horizon, args.minibatch_size,
           args.epochs, args.save_every) < 1:
        parser.error('Counts must be positive.')
    batch_size = args.num_envs * args.horizon
    if batch_size % args.minibatch_size:
        parser.error('The batch must contain a whole number of minibatches.')
    if Path(args.run_name).name != args.run_name or any(c in args.run_name for c in '\\/:*?"<>|') or args.run_name in ('.', '..'):
        parser.error('run-name must be one directory name.')
    resume = args.resume.resolve() if args.resume else None
    if resume:
        if not (resume / 'learner_group/learner').is_dir() or not (resume / 'environment.pt').is_file():
            parser.error('Resume requires a full CUDA checkpoint with learner and environment state.')
        resumed_state = json.loads((resume / 'trainer_state.json').read_text(encoding='utf-8-sig'))
        if args.iterations <= resumed_state['iteration']:
            parser.error('iterations must exceed the checkpoint iteration (total target).')
        source_run = resume.parent.parent
        settings = json.loads((source_run / 'run.json').read_text(encoding='utf-8-sig'))['env_config']
    elif args.weights:
        settings = checkpoint_settings(args.weights)
    elif args.settings_from:
        settings = json.loads(args.settings_from.read_text(encoding='utf-8-sig'))['env_config']
    else:
        reference = SnakeEnv()
        settings = reference.settings()
        reference.close()
    run_dir = ROOT / 'runs' / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'checkpoints').mkdir()
    write_json(run_dir / 'run.json', {
        'version': '9.0.0', 'backend': 'cuda_env_rllib_learner_live',
        'created': datetime.now(ZoneInfo('Europe/Kyiv')).isoformat(),
        'purpose': 'training', 'initialization': 'checkpoint' if resume else 'weights' if args.weights else 'random',
        'resumed_from': str(resume) if resume else None,
        'numerical_fix': 'categorical KL from normalized log probabilities; no action masks or probability floors',
        'env_config': settings,
        'arguments': {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
    }, durable=True)
    write_json(ROOT / 'runs' / 'latest_live_run.json', {'run_dir': str(run_dir)})
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    phase, iteration, environment_steps = 'starting', 0, 0
    metrics, last_saved, learner, publisher = {}, None, None, None
    last_error = None
    tracker = EpisodeTracker(100)
    if resume:
        iteration = int(resumed_state['iteration'])
        environment_steps = int(resumed_state['environment_steps'])
        tracker, metrics = copy_history_until(source_run, run_dir, iteration)

    def status():
        write_json(run_dir / 'status.json', {
            'phase': phase, 'pid': os.getpid(), 'iteration': iteration,
            'total_iterations': args.iterations, 'initialization': 'checkpoint' if resume else 'weights' if args.weights else 'random',
            'device': 'CUDA', 'backend': 'CUDA environments + RLlib PPO learner',
            'elapsed_seconds': round(time.monotonic() - started, 1),
            'environment_count': args.num_envs, 'environment_steps': environment_steps,
            'batch_size': batch_size, 'minibatch_size': args.minibatch_size,
            'epochs': args.epochs, 'metrics': metrics, 'error': last_error,
        }, durable=True)

    def save():
        nonlocal last_saved
        if last_saved == iteration:
            return run_dir / 'checkpoints' / f'iteration_{iteration:06d}'
        destination = run_dir / 'checkpoints' / f'iteration_{iteration:06d}'
        destination.mkdir(exist_ok=False)
        learner.save_to_path(str(destination / 'learner_group' / 'learner'))
        torch.save({'environment': env.state_dict(), 'torch_cuda_rng': torch.cuda.get_rng_state(),
                    'torch_cpu_rng': torch.get_rng_state()},
                   destination / 'environment.pt')
        write_json(destination / 'trainer_state.json', {
            'iteration': iteration, 'environment_steps': environment_steps})
        write_json(destination / 'rllib_checkpoint.json', {
            'checkpoint_version': '2.0', 'backend': 'cuda_env_rllib_learner', 'component': 'learner'})
        # Commit the full learner, optimizer, and environment before publishing
        # a pointer which claims this checkpoint is ready for recovery.
        for checkpoint_file in destination.rglob('*'):
            if checkpoint_file.is_file():
                with checkpoint_file.open('r+b') as stream:
                    os.fsync(stream.fileno())
        write_json(run_dir / 'latest.json', {'checkpoint': str(destination), 'iteration': iteration}, durable=True)
        write_json(ROOT / 'runs' / 'latest_run.json', {'run_dir': str(run_dir)})
        last_saved = iteration
        return destination

    try:
        status()
        env = GPUSnakeBatch(args.num_envs, settings, device='cuda', seed=args.seed)
        cfg = make_config(env_config=settings, workers=0, gpu=1, batch_size=batch_size,
                          minibatch_size=args.minibatch_size, epochs=args.epochs, seed=args.seed)
        learner = cfg.build_learner(spaces={'default_policy': (env.observation_space, env.action_space)})
        if resume:
            learner.restore_from_path(str(resume / 'learner_group/learner'))
        configure_gpu(True)
        if resume:
            saved = torch.load(resume / 'environment.pt', map_location='cpu', weights_only=False)
            if not env.load_state_dict(saved['environment']):
                raise RuntimeError('Saved environment does not match the current GPU batch.')
            torch.cuda.set_rng_state(saved['torch_cuda_rng'])
            if 'torch_cpu_rng' in saved:
                torch.set_rng_state(saved['torch_cpu_rng'])
        module = learner.module['default_policy']
        if args.weights:
            from snake_v9.evaluation import load_policy
            initial_policy = load_policy(args.weights)
            module.load_state_dict(initial_policy.state_dict(), strict=True)
            del initial_policy
            configure_gpu(True)
        module.action_dist_cls = LogSpaceCategorical
        if next(module.parameters()).device.type != 'cuda':
            raise RuntimeError('The learner is not on CUDA.')
        (run_dir / 'inference_architecture.txt').write_text(str(module), encoding='utf-8')
        require_finite_weights(module)
        bootstrap = save()
        publisher = LivePublisher(run_dir, bootstrap)
        publisher.publish(module, iteration, environment_steps)
        rollout = GPURollout(module, env, args.horizon, gamma=cfg.gamma, gae_lambda=cfg.lambda_)
        phase = 'training'
        status()
        print(f'{"RESTORED" if resume else "FRESH"} TRAINING: {args.num_envs} games, {batch_size} new steps, '
              f'{args.minibatch_size} minibatch, {args.epochs} epochs; start iteration {iteration}.', flush=True)
        episode_path = run_dir / 'episodes.csv'
        write_header = not episode_path.exists() or episode_path.stat().st_size == 0
        with episode_path.open('a', newline='', encoding='utf-8') as episode_file:
            episode_csv = csv.DictWriter(episode_file, fieldnames=['iteration', 'apples', 'steps', 'reward', 'reason'])
            if write_header:
                episode_csv.writeheader()
            while iteration < args.iterations:
                if (run_dir / 'STOP').exists():
                    phase = 'stopped'
                    break
                torch.cuda.synchronize()
                tick = time.perf_counter()
                batch, completed = rollout.sample()
                torch.cuda.synchronize()
                sample_seconds = time.perf_counter() - tick
                update_tick = time.perf_counter()
                result = learner.update_from_batch(
                    batch, timesteps={'num_env_steps_sampled_lifetime': environment_steps + batch_size},
                    num_epochs=args.epochs, minibatch_size=args.minibatch_size,
                    shuffle_batch_per_epoch=True)
                torch.cuda.synchronize()
                update_seconds = time.perf_counter() - update_tick
                iteration += 1
                environment_steps += batch_size
                episodes = tracker.add(completed.tolist(), iteration)
                episode_csv.writerows(episodes)
                episode_file.flush()
                os.fsync(episode_file.fileno())
                seconds = time.perf_counter() - tick
                metrics = {
                    'iteration': iteration, 'environment_steps': environment_steps,
                    'iteration_seconds': seconds, 'sample_seconds': sample_seconds,
                    'update_seconds': update_seconds,
                    'environment_steps_per_second': batch_size / seconds,
                    'completed_episodes': len(completed),
                    'peak_cuda_memory_mib': torch.cuda.max_memory_allocated() / 1024**2,
                    **tracker.summary(), **scalar_metrics(result),
                }
                invalid = [key for key, value in metrics.items()
                           if isinstance(value, (int, float)) and not math.isfinite(value)]
                if invalid:
                    raise RuntimeError('Nonfinite training metric: ' + ', '.join(invalid))
                require_finite_weights(module)
                publish_tick = time.perf_counter()
                publisher.publish(module, iteration, environment_steps)
                metrics['publish_seconds'] = time.perf_counter() - publish_tick
                metrics['iteration_seconds'] = time.perf_counter() - tick
                metrics['environment_steps_per_second'] = batch_size / metrics['iteration_seconds']
                with (run_dir / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(metrics) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                print(json.dumps({k: v for k, v in metrics.items() if k not in ('records', 'end_reasons_total')}), flush=True)
                status()
                if iteration % args.save_every == 0:
                    save()
        save()
        if phase != 'stopped':
            phase = 'completed'
    except KeyboardInterrupt:
        phase = 'stopped'
        if learner is not None and publisher is not None:
            save()
    except Exception as error:
        phase = 'failed'
        last_error = f'{type(error).__name__}: {error}'
        traceback.print_exc()
        status()
        raise
    finally:
        status()


if __name__ == '__main__':
    main()
