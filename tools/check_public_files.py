"""Inspect staged file bytes, model metadata, and the public data allowlist."""

import io
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'models/iteration_000950/policy.pt', 'models/iteration_002000/policy.pt'}
PRIVATE_PARTS = {'runs', 'sessions', 'diagnostics', 'private', '.venv', 'site-packages', '__pycache__', '.vs'}
MACHINE_KEYS = {'hostname', 'computername', 'username', 'user_profile', 'gpu_name', 'gpu_uuid', 'gpu_driver',
                'gpu_core_c', 'cpu_max_c', 'gpu_power_w', 'gpu_power_limit_w', 'thermal', 'pid',
                'peak_cuda_memory_mib', 'environment_steps_per_second', 'iteration_seconds',
                'sample_seconds', 'update_seconds', 'publish_seconds'}
PATTERNS = [
    ('Windows absolute path', re.compile(r'(?i)\b[A-Z]:[\\/]+[^\s\"\x27]+')),
    ('User home path', re.compile(r'(?i)(?:/Users/|/home/|\\Users\\)[a-z0-9_.-]+')),
    ('GitHub credential', re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})')),
    ('Private key', re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')),
    ('Cloud credential', re.compile(r'\bAKIA[0-9A-Z]{16}\b')),
]


def check_tree(value, location, failures):
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in MACHINE_KEYS:
                failures.append(f'{location}: private machine field {key}')
            check_tree(child, location, failures)
    elif isinstance(value, list):
        for child in value:
            check_tree(child, location, failures)


def main():
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    failures, found_models, count = [], set(), 0
    for name in filter(None, paths):
        count += 1
        parts = Path(name).parts
        if PRIVATE_PARTS.intersection(parts):
            failures.append(f'{name}: private execution directory')
        data = subprocess.check_output(['git', 'show', ':' + name], cwd=ROOT)
        if len(data) > 50 * 1024 * 1024:
            failures.append(f'{name}: unexpected large file')
        if name.endswith('.pt'):
            found_models.add(name)
            if name not in MODELS:
                failures.append(f'{name}: model outside the two approved paths')
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                data = b'\n'.join(archive.read(p) for p in archive.namelist() if p.endswith('.pkl'))
        elif b'\0' in data[:4096]:
            continue
        text = data.decode('utf-8', errors='replace')
        for label, pattern in PATTERNS:
            if pattern.search(text):
                failures.append(f'{name}: {label}')
        if name.startswith(('results/', 'models/', 'configs/')):
            if name.endswith('.json'):
                check_tree(json.loads(text), name, failures)
            elif name.endswith('.jsonl'):
                rows = [json.loads(line) for line in text.splitlines() if line.strip()]
                if [r['iteration'] for r in rows] != list(range(1, 2001)):
                    failures.append(f'{name}: missing or duplicated iterations')
                for row in rows:
                    check_tree(row, name, failures)
    if found_models != MODELS:
        failures.append('Expected exactly the models from iterations 950 and 2000.')
    if failures:
        print('\n'.join(failures))
        return 1
    print(f'Passed: {count} staged files, exactly 2 models, no detected credentials or machine data.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
