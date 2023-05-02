from typing import Callable, TypeVar

import mitsuba as mi

from mytorch.registry import import_children

T = TypeVar("T")
registered_integrators = []


def register_integrator(name: str) -> Callable[[T], T]:
    def register(cls: T):
        registered_integrators.append(name)
        mi.register_integrator(name, cls)
        return cls
    return register


import_children(__file__, __name__)
