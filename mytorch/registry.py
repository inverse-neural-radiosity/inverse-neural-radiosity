import importlib
import logging
import os
from pathlib import Path
from typing import Any, Callable, Generic, Type, TypeVar

T = TypeVar("T")
TImpl = TypeVar("TImpl")
logger = logging.getLogger(__name__)


class Registry(Generic[T]):
    def __init__(self, name: str, base: Type[T]) -> None:
        self.name = name
        self.constructors: dict[str, Type[T]] = {}

    def add(self, name: str, constructor: Type[T]) -> None:
        logger.info(f"Register {self.name}: {name}")
        self.constructors[name] = constructor

    def register(self, name: str) -> Callable[[Type[TImpl]], Type[TImpl]]:
        def func(constructor: Type[TImpl]):
            self.add(name, constructor)
            return constructor
        return func

    def get(self, name: str) -> Type[T]:
        return self.constructors[name]

    def build(self, name: str, kwargs: dict[str, Any]) -> T:
        return self.get(name)(**kwargs)


def import_children(path: str, module: str):
    """Import all py files in the same folder"""
    assert path.endswith("__init__.py"), "import_children should only be called from __init__.py"
    folder = Path(path).parent
    files = os.listdir(folder)
    for file in files:
        if file == "__init__.py" or not file.endswith(".py"):
            continue
        name = module + "." + os.path.splitext(file)[0]
        importlib.import_module(name)
