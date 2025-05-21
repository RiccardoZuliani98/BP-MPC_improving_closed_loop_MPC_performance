class CallableWrapper:
    """
    Given a python function f(x), the object g = CallableWrapper(f)
    produces the same output as f(x) but can be called using g.call(x).
    """
    def __init__(self, func):
        self.func = func

    def call(self, *args, **kwargs):
        return self.func(*args, **kwargs)