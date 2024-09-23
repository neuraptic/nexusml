import functools
import warnings

import boto3

from nexusml.api import config
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

#############
# Amazon S3 #
#############

_s3_client = None


def s3_client():
    """
    Retrieve a boto3 S3 client instance, initializing it if necessary.

    Returns:
        boto3.S3.Client: The S3 client instance.
    """
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client('s3')
    return _s3_client


def get_s3_config() -> dict:
    """
    Returns S3 configuration.

    Returns:
        dict: The S3 configuration.

    Raises:
        KeyError: If the S3 configuration is not found.
    """
    return config.get('storage')['files']['s3']
