"""Repository resources and local run storage, without machine-specific paths."""

import os
from pathlib import Path

_source_root = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ['SNAKEAI_HOME']).expanduser().resolve() if os.environ.get('SNAKEAI_HOME') else (
    _source_root if (_source_root / 'pyproject.toml').is_file() else Path.cwd()
)
