from nexusml.api import create_app
from nexusml.api.external.ext import celery

flask_app = create_app()
celery_app = celery
