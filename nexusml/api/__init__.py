from datetime import timedelta
import os
import sys

from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from flask import Flask
from flask import redirect
from flask_apispec import FlaskApiSpec
from webargs.flaskparser import abort
from webargs.flaskparser import parser
import yaml

from nexusml.api import routes
from nexusml.api.ext import cache
from nexusml.api.ext import celery
from nexusml.api.ext import cors
from nexusml.api.ext import docs
from nexusml.api.ext import init_celery
from nexusml.api.ext import mail
from nexusml.api.utils import config
from nexusml.api.utils import DEFAULT_CONFIG
from nexusml.api.views import ai
from nexusml.api.views import services
from nexusml.api.views import tasks
from nexusml.api.views.core import limiter
from nexusml.api.views.core import register_all_endpoints_docs
from nexusml.constants import CONFIG_FILE
from nexusml.constants import SWAGGER_UI_URL
from nexusml.constants import SWAGGER_URL
from nexusml.database.core import create_tables
from nexusml.database.core import db
from nexusml.database.organizations import create_known_clients_and_reserved_clients
from nexusml.database.organizations import create_main_admin_and_maintainer
from nexusml.database.organizations import create_main_organization
from nexusml.database.subscriptions import create_default_plans
from nexusml.env import AWS_S3_BUCKET
from nexusml.env import ENV_CELERY_BROKER_URL
from nexusml.env import ENV_CELERY_RESULT_BACKEND
from nexusml.env import ENV_DB_NAME
from nexusml.env import ENV_DB_PASSWORD
from nexusml.env import ENV_DB_USER
from nexusml.env import ENV_MAIL_PASSWORD
from nexusml.env import ENV_MAIL_SERVER
from nexusml.env import ENV_MAIL_USERNAME
from nexusml.env import ENV_NOTIFICATION_EMAIL

__all__ = ['create_app']
"""
Flask Application Factory Pattern
"""


def create_app(setup_database: bool = True):
    app = Flask(__name__)

    # Set app config
    _set_app_config(app)

    # Set config for Swagger documentation
    _set_swagger_config(app)

    # Set config for Amazon S3
    _config = config.get()
    s3_config = _config['storage']['files']['s3']
    s3_config['bucket'] = os.environ[AWS_S3_BUCKET]
    config.set(_config)

    # Set config for Celery
    _set_celery_config(app)

    # Set routes
    _set_routes(app=app)

    # Add endpoint documentation with Swagger
    _add_docs(app=app, docs=docs)

    # Initialize Flask extensions
    cache.init_app(app)
    cors.init_app(app)
    docs.init_app(app)
    limiter.init_app(app)
    mail.init_app(app)

    init_celery(app, celery)

    # Initialize database
    db.init_app(app)

    if setup_database:
        _setup_database(app)

    return app


def _set_app_config(app):
    # Initialize app config
    config.init_app(app)

    # Load specific config from file
    if os.path.isfile(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            _config = yaml.load(f, yaml.SafeLoader)
    else:
        _config = DEFAULT_CONFIG
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(_config, f)

    config.set(_config)

    # Set database config
    db_uri = _config['storage']['database']['uri']

    _db_connection = {
        '<database>': ENV_DB_NAME,
        '<user>': ENV_DB_USER,
        '<password>': ENV_DB_PASSWORD,
    }

    for param, value in _db_connection.items():
        if value not in os.environ:
            print(f'FATAL: environment variable "{value}" has not been set. Exiting')
            sys.exit(1)
        else:
            db_uri = db_uri.replace(param, os.environ[value])

    app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    # app.config['SQLALCHEMY_POOL_SIZE'] = 20  # disable max concurrent connections
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_pre_ping': True, 'pool_recycle': 7200}

    # Set security config
    app.config['RSA_PUB_KEY'] = config.rsa_public_key()
    app.config['SECRET_KEY'] = os.urandom(24)
    app.config['SESSION_COOKIE_SECURE'] = True

    # Set email config
    app.config['MAIL_SERVER'] = os.environ[ENV_MAIL_SERVER]
    app.config['MAIL_PORT'] = _config['notifications']['mail_port']
    app.config['MAIL_USE_TLS'] = _config['notifications']['use_tls']
    app.config['MAIL_USE_SSL'] = _config['notifications']['use_ssl']
    app.config['MAIL_USERNAME'] = os.environ[ENV_MAIL_USERNAME]
    app.config['MAIL_PASSWORD'] = os.environ[ENV_MAIL_PASSWORD]
    app.config['MAIL_DEFAULT_SENDER'] = os.environ[ENV_NOTIFICATION_EMAIL]

    # Set payload size limit
    app.config['MAX_CONTENT_LENGTH'] = _config['limits']['requests']['max_payload']


def _set_swagger_config(app):
    api_url = config.get('server')['api_url']

    app.config.update({
        'APISPEC_SPEC':
            APISpec(
                title='NexusML API',
                version='v0',
                # Apparently, `flask-apispec` doesn't fully support OpenAPI 3 yet:
                #   - https://github.com/jmcarp/flask-apispec/issues/88
                #   - https://github.com/jmcarp/flask-apispec/issues/170
                openapi_version='2.0',
                plugins=[MarshmallowPlugin()],
            ),
        'APISPEC_SWAGGER_URL':
            api_url + SWAGGER_URL,
        'APISPEC_SWAGGER_UI_URL':
            api_url + SWAGGER_UI_URL,
    })


def _set_celery_config(app):
    _celery_beat_schedule = {
        'cl_service': {
            'task': 'nexusml.api.jobs.periodic_jobs.run_cl_service',
            'schedule': timedelta(days=1),
        },
        'al_service': {
            'task': 'nexusml.api.jobs.periodic_jobs.run_al_service',
            'schedule': timedelta(days=1),
        },
        'notifier': {
            'task': 'nexusml.api.jobs.periodic_jobs.notify',
            'schedule': config.get('notifications')['interval'],
        },
        'biller': {
            'task': 'nexusml.api.jobs.periodic_jobs.bill',
            'schedule': timedelta(days=1),
        },
        'upload_cleaner': {
            'task': 'nexusml.api.jobs.periodic_jobs.abort_incomplete_uploads',
            'schedule': timedelta(days=1),
        },
        'waitlist_sender': {
            'task': 'nexusml.api.jobs.periodic_jobs.send_waitlist',
            'schedule': timedelta(days=1),
        },
        'prediction_logger': {
            'task': 'nexusml.api.views.ai.save_buffered_prediction_logs',
            'schedule': timedelta(minutes=1),
        },
        'pending_test_prediction_trigger': {
            'task': 'nexusml.api.jobs.periodic_jobs.trigger_all_pending_test_predictions',
            'schedule': timedelta(minutes=1),
        },
        'expired_invitations_cleaner': {
            'task': 'nexusml.api.jobs.periodic_jobs.remove_expired_invitations',
            'schedule': timedelta(days=1),
        },
    }

    app.config.from_mapping(CELERY=dict(broker_url=os.environ[ENV_CELERY_BROKER_URL],
                                        result_backend=os.environ[ENV_CELERY_RESULT_BACKEND],
                                        task_ignore_result=True,
                                        beat_schedule=_celery_beat_schedule),)


def _set_routes(app: Flask):
    api_url = config.get('server')['api_url']

    @app.route('/')
    def index():
        return redirect(api_url + SWAGGER_UI_URL)

    routes.register_myaccount_endpoints(app=app)
    routes.register_organizations_endpoints(app=app)
    routes.register_files_endpoints(app=app)
    routes.register_tasks_endpoints(app=app)
    routes.register_ai_endpoints(app=app)
    routes.register_services_endpoints(app=app)
    routes.register_examples_endpoints(app=app)
    routes.register_tags_endpoints(app=app)

    # This error handler is necessary when using Flask-RESTful
    @parser.error_handler
    def handle_request_parsing_error(err, req, schema, *, error_status_code, error_headers):
        """
        webargs error handler that uses Flask-RESTful's abort function to return a JSON error response to the client.
        """
        if error_status_code is None:
            error_status_code = 400
        abort(error_status_code, errors=err.messages)


def _add_docs(app: Flask, docs: FlaskApiSpec):
    exclude = [
        # Tasks
        tasks.TaskStatusView,
        tasks.TaskQuotaUsageView,
        # AI
        ai.PredictionLoggingView,
        # Services
        services.InferenceServiceStatusView,
        services.CLServiceStatusView,
        services.ALServiceStatusView,
        services.MonitoringServiceStatusView,
        services.TestingServiceStatusView,
        # Notifications
        services.InferenceServiceNotificationsView,
        services.CLServiceNotificationsView,
        services.ALServiceNotificationsView,
        services.MonitoringServiceNotificationsView,
        services.TestingServiceNotificationsView,
        # Templates
        services.MonitoringServiceTemplatesView,
    ]

    register_all_endpoints_docs(app=app, docs=docs, exclude=exclude)


def _setup_database(app):
    with app.app_context():
        # Create tables
        create_tables()

        # Create reserved organization and clients
        create_main_organization()
        create_main_admin_and_maintainer()
        create_known_clients_and_reserved_clients()

        # Create default subscription plans
        create_default_plans()
