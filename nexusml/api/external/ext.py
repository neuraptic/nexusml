import functools
import os

from celery import Celery
from flask_apispec import FlaskApiSpec
from flask_caching import Cache as FlaskCache
from flask_cors import CORS
from flask_mail import Mail
from redis import StrictRedis

from nexusml.api.utils import config
from nexusml.constants import DEFAULT_CELERY_BROKER_URL
from nexusml.env import ENV_CELERY_BROKER_URL

###########################
# Custom Flask extensions #
###########################


class Cache(FlaskCache):
    """
    Class that enables `@memoize` decorator to set a dynamic timeout value (loaded at runtime).
    """

    def memoize(self, *args, **kwargs):
        """
        Decorator to memoize function results with a dynamic timeout value loaded at runtime.

        Args:
            *args: Positional arguments passed to the original memoize decorator.
            **kwargs: Keyword arguments passed to the original memoize decorator.

        Returns:
            function: A decorated function with memoization and dynamic timeout capabilities.
        """

        def dynamic_timeout(f):
            """
            Inner decorator to set the cache timeout dynamically from the configuration.

            Args:
                f (function): The function to be decorated.

            Returns:
                function: The function wrapped with dynamic timeout logic.
            """

            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                f.cache_timeout = config.get('limits')['requests']['cache_timeout']
                return f(*args, **kwargs)

            return wrapper

        def decorator(f):
            """
            Combine dynamic timeout behavior with the original memoization behavior.

            Args:
                f (function): The function to be decorated.

            Returns:
                function: The function wrapped with both dynamic timeout and memoization.
            """
            # First apply the dynamic timeout behavior
            with_dynamic_timeout = dynamic_timeout(f)
            # Then apply the original memoization behavior
            memoized_with_dynamic_timeout = FlaskCache.memoize(self, *args, **kwargs)(with_dynamic_timeout)
            return memoized_with_dynamic_timeout

        return decorator


######################################
# Initialize Flask extension objects #
######################################

cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})  # WARNING: `SimpleCache` is not thread safe
celery = Celery(include=['nexusml.api.jobs.periodic_jobs', 'nexusml.api.jobs.event_jobs'])
cors = CORS()  # TODO: add allowed origin URIs to be more restrictive
docs = FlaskApiSpec()
mail = Mail()

#########
# Redis #
#########

redis_buffer = StrictRedis.from_url(os.environ.get(ENV_CELERY_BROKER_URL, DEFAULT_CELERY_BROKER_URL))

####################
# Helper functions #
####################


def init_celery(app, celery):
    """
    Initialize the Celery app with Flask application context and configuration.

    Args:
        app (Flask): The Flask application instance.
        celery (Celery): The Celery application instance.
    """
    celery.conf.update(app.config)
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        """
        Custom Celery Task class that ensures tasks run within Flask application context.
        """
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    celery.Task = ContextTask
    celery.config_from_object(app.config['CELERY'])
