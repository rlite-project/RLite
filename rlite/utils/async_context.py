class AsyncRayContextManager:
    def __init__(self):
        self._is_in_context = False

    def __enter__(self):
        self._is_in_context = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._is_in_context = False

    def is_active(self):
        return self._is_in_context

    def __call__(self):
        return self
