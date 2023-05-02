import mitsuba as mi

from nerad.texture import register_texture


@register_texture("dict")
class MiDictionary(mi.Texture):
    """This allows us to pass a flat dictionary when constructing a Mitsuba object from dict.

    It enables us to define typed parameters in self._init() of Integrators,
    which makes it easier to track what should / can be in the configs (i.e. arguments of the _init function)
    """

    def __init__(self, props: mi.Properties) -> None:
        super().__init__(props)
        kwargs = {}
        for key in props.property_names():
            kwargs[key] = props.get(key)
        self.dict = kwargs
