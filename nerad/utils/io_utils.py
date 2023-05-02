import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def glob_sorted(root: str | Path, patterns: str | list[str]):
    """Load files from glob patterns and sort them.

    Sorting supports file names like 1.png, 2.png, ..., 10.png.
    """
    root = Path(root)
    if not isinstance(patterns, list):
        patterns = [patterns]

    files: list[Path] = []
    for pattern in patterns:
        files += list(root.glob(pattern))

    if len(files) == 0:
        return files

    # if all files have same length, assume format 000.png, 001.png, ...
    # so just do string sort
    if all((len(f.name) == len(files[0].name) for f in files)):
        return sorted(files)

    try:
        return sorted(files, key=lambda f: int(f.stem))
    except ValueError:
        logger.warning(
            f"Not all files in {root} ({patterns}) have numbered names. "
            "Fallback to string sort, results may be unexpected."
        )
        return sorted(files)
