"""Locate v9 checkpoints without importing any graphical interface."""

import json
from pathlib import Path

from .paths import ROOT


def resolve_checkpoint(value=None):
    if value is not None:
        return Path(value).resolve()
    pointer = ROOT / "runs" / "latest_run.json"
    if not pointer.is_file():
        bundled = ROOT / 'models' / 'iteration_002000'
        if (bundled / 'model.json').is_file():
            return bundled
        raise FileNotFoundError('No trained model found. Select --checkpoint or start training.')
    run = Path(json.loads(pointer.read_text(encoding="utf-8"))["run_dir"])
    pointer = run / "best.json"
    if not pointer.is_file():
        pointer = run / "latest.json"
    return Path(json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"])


def checkpoint_settings(checkpoint):
    checkpoint = Path(checkpoint)
    if checkpoint.is_file():
        checkpoint = checkpoint.parent
    portable = checkpoint / 'model.json'
    if portable.is_file():
        return json.loads(portable.read_text(encoding='utf-8'))['env_config']
    for parent in (checkpoint, *checkpoint.parents):
        metadata = parent / "run.json"
        if metadata.is_file():
            data = json.loads(metadata.read_text(encoding="utf-8"))
            if data.get("version") != "9.0.0":
                raise ValueError("The selected model is not v9.")
            return data["env_config"]
    raise FileNotFoundError("Missing run.json for the selected v9 checkpoint.")
