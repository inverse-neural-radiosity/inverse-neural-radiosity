from typing import Any


def inject_dict(data: dict[str, Any], injection: dict[str, Any]) -> dict[str, Any]:
    for k, v in injection.items():
        if k in data and data[k] is None:
            data[k] = v
    return data
