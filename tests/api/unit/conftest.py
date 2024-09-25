from flask import Flask
import pytest

from nexusml.api import create_app
from nexusml.api.utils import config
from tests.api.conftest import empty_db
from tests.api.conftest import restore_db
from tests.api.conftest import set_app_config

############
# Mock App #
############


@pytest.fixture(scope='session', autouse=True)
def app() -> Flask:
    # Create app
    try:
        app_ = create_app()
    except RuntimeError:
        # TODO: Resolve the `RuntimeError: NexusML config already initialized` issue.
        #       This error emerges when an app has already been created by integration tests,
        #       due to the API's reliance on a centralized config object.
        #       A temporary workaround involves overriding that object.
        #       Decentralizing config would be a more permanent solution.
        config._app = None
        app_ = create_app()

    # Set app config
    set_app_config(app=app_)

    return app_


@pytest.fixture(scope='function', autouse=True)
def app_context(app):
    with app.app_context():
        yield


@pytest.fixture
def client(app):
    return app.test_client()


############
# Database #
############


@pytest.fixture(scope='function', autouse=True)
def _restore_db(app_context, mock_client_id, session_user_id, session_user_auth0_id):
    restore_db(mock_client_id=mock_client_id,
               session_user_id=session_user_id,
               session_user_auth0_id=session_user_auth0_id)


@pytest.fixture(scope='session', autouse=True)
def _empty_db(app):
    """
    Deletes all database tables after each test session execution.
    This fixture is used because `nexusml.api.create_app()` requires the existing database to meet certain conditions.
    If the database is empty, it automatically inserts the entries required to fulfill these conditions.
    """
    # Run test session
    yield
    # Delete all tables
    empty_db(app=app)
