from warnings import warn

class Registry(object):
    def __init__(self, name):
        self._dict = {}
        self._name = name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable!\nValue: {value}")
        if key in self._dict:
            warn("{} already registered".format(key))
        self._dict[key] = value

    def register(self, obj):
        """Decorator to register a function or class

            Usags:
                @register.register
                class or func

                @register.register("name")
                class or func
        """

        def add(key, value):
            self[key] = value
            return value

        if callable(obj):
            # @register.register
            return add(obj.__name__, obj)
        # @register.register("name")
        return lambda x: add(obj, x)

    def __getitem__(self, key):
        if key not in self._dict:
            raise Exception("{} not registered".format(key))
        return self._dict[key]

    def get(self, key):
        if key not in self._dict:
            raise Exception("{} not registered".format(key))
        return self._dict[key]

    def keys(self):
        """key"""
        return self._dict.keys()