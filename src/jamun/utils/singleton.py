import threading

def singleton(cls):
    """
    Decorator that implements the singleton pattern by modifying __init__.
    """
    _instances = {}
    _lock = threading.Lock()

    original_init = cls.__init__

    def __init__(self, *args, **kwargs):
        # Convert args and kwargs to hashable types
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, list):
                args[i] = tuple(arg)
            if isinstance(arg, dict):
                args[i] = frozenset(arg.items())
        for key, value in kwargs.items():
            if isinstance(value, list):
                kwargs[key] = tuple(value)
            if isinstance(value, dict):
                kwargs[key] = frozenset(value.items())

        obj_key = (tuple(args), frozenset(kwargs.items()))

        if obj_key not in _instances:
            with _lock:
                if obj_key not in _instances:
                    _instances[obj_key] = self
                    original_init(self, *args, **kwargs)
                    return

        # Copy state from singleton instance
        self.__dict__.update(_instances[obj_key].__dict__)

    cls.__init__ = __init__
    return cls
