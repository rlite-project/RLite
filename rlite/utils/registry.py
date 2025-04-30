from __future__ import annotations


class Registry:
    def __init__(self, name: str):
        self._name = name
        self._registry = {}

    def register(self, names: list[str] | str | None = None):
        if names is None:
            names = []
        elif isinstance(names, str):
            names = [names]

        def decorator(cls):
            for name in names:
                self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str, default=None) -> type:
        return self._registry.get(name, default)

    def __getitem__(self, name):
        try:
            return self._registry[name]
        except KeyError:
            raise KeyError(f"{name} is not registered in Registry {self._name}")
