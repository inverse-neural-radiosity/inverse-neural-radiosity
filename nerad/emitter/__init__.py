from typing import Callable, TypeVar

import mitsuba as mi

from mytorch.registry import import_children

T = TypeVar("T")
registered_emitters = []


def register_emitter(name: str) -> Callable[[T], T]:
    def register(cls: T):
        registered_emitters.append(name)
        mi.register_emitter(name, cls)
        return cls
    return register


import_children(__file__, __name__)
