import functools
import warnings

from nexusml.enums import ElementValueType


def deprecated(description='This function is deprecated and will be removed in future versions.'):
    """ Decorator to mark functions as deprecated with an optional description. """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(f'{func.__name__} is deprecated: {description}', category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


FILE_TYPES = {
    ElementValueType.DOCUMENT_FILE, ElementValueType.IMAGE_FILE, ElementValueType.VIDEO_FILE,
    ElementValueType.AUDIO_FILE
}
