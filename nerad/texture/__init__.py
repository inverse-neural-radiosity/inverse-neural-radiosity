from typing import Callable, TypeVar

import mitsuba as mi

from mytorch.registry import import_children

T = TypeVar("T")
registered_textures = []


def register_texture(name: str) -> Callable[[T], T]:
    def register(cls: T):
        registered_textures.append(name)
        mi.register_texture(name, cls)
        return cls
    return register


import_children(__file__, __name__)
