import logging
from functools import wraps
from datetime import datetime

"""Decorator for logging the duration of a method call"""


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        duration = end - start
        # TODO(df): Reuse logger / level from calling module
        logger = logging.getLogger('pdf_extractor')
        logger.setLevel(logging.INFO)
        logger.info(f'Duration for {func.__name__}: {duration}')
        return result
    return wrapper
