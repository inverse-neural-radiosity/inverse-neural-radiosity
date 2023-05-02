import json
from os.path import isfile
from pathlib import Path


def read_json(file, default_factory=None):
    if not isfile(file):
        return default_factory()

    with open(file, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def write_json(file, data):
    if file is None:
        return

    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2, sort_keys=True))
