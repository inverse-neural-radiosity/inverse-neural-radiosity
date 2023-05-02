from typing import Callable, TypeVar

import mitsuba as mi

from mytorch.registry import import_children

T = TypeVar("T")
registered_bsdfs = []


def register_bsdf(name: str) -> Callable[[T], T]:
    def register(cls: T):
        registered_bsdfs.append(name)
        mi.register_bsdf(name, cls)
        return cls
    return register


import_children(__file__, __name__)
