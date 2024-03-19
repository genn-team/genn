from functools import wraps
from inspect import ismethod
from warnings import warn

def deprecated(message):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if ismethod(func):
                warn(f"Call to deprecated method {func.__name__}."
                     f" ({message}) -- Deprecated since version 4.9.0",
                     category=FutureWarning)
            else:
                warn(f"Call to deprecated function {func.__name__}."
                     f" ({message}) -- Deprecated since version 4.9.0",
                     category=FutureWarning)
                
            return func(*args, **kwargs)
        
        wrapper.is_deprecated = True
        return wrapper

    return decorator