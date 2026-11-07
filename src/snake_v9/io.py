"""Publish JSON pointers atomically for concurrent training and viewers."""

import json
import os
import time
from pathlib import Path


def write_json(path, data, *, durable=False):
    path = Path(path)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        if durable:
            stream.flush()
            os.fsync(stream.fileno())
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05)
