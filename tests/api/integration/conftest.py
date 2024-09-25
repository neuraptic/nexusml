from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from datetime import datetime
from datetime import timedelta
import json
import os
import random
import re
import string
import sys
from tempfile import NamedTemporaryFile
from threading import Thread
import time
from typing import Iterable, List, Optional, Tuple, Union
from unittest.mock import MagicMock
import urllib.parse
import uuid

import boto3
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from flask import Flask
from flask import Response
from flask.testing import FlaskClient as _FlaskClient
import flask_mail
from jwcrypto import jwk
import jwt
from moto import mock_s3 as moto_mock_s3
import pytest
import requests
from werkzeug.test import TestResponse

from nexusml import constants  # Import the whole module for monkey patches
from nexusml.api import create_app
from nexusml.api.ext import cache
from nexusml.api.resources.organizations import Organization
from nexusml.api.resources.tasks import Task
from nexusml.api.utils import API_DOMAIN
from nexusml.api.utils import config
from nexusml.api.views import organizations as organizations_views
from nexusml.constants import CONFIG_FILE
from nexusml.constants import DEFAULT_API_KEY_FILE
from nexusml.database.organizations import client_scopes
from nexusml.enums import FileStorageBackend
from nexusml.env import ENV_AUTH0_JWKS
from nexusml.env import ENV_AUTH0_TOKEN_AUDIENCE
from nexusml.env import ENV_AUTH0_TOKEN_ISSUER
from nexusml.env import ENV_AWS_ACCESS_KEY_ID
from nexusml.env import ENV_AWS_SECRET_ACCESS_KEY
from nexusml.env import ENV_RSA_KEY_FILE
from nexusml.utils import deprecated
from nexusml.utils import get_s3_config
from tests.api.conftest import empty_db
from tests.api.conftest import populate_db
from tests.api.conftest import restore_db
from tests.api.conftest import set_app_config
from tests.api.constants import CLIENT_MAX_THREADS
from tests.api.constants import CLIENT_SCOPES
from tests.conftest import Singleton

###########
# Backend #
###########


class Backend(metaclass=Singleton):
    # TODO: This class shouldn't be necessary if the Flask test client is used.
    #       Refactor tests to use the Flask test client.

    def __init__(self):
        self._running = False
        self._app = create_app()

    @property
    def app(self) -> Flask:
        return self._app

    def run(self):
        if self._running:
            return
        Thread(target=self._run_backend_thread, daemon=True).start()
        time.sleep(2)  # Give the backend some time to start
        self._running = True

    def _run_backend_thread(self):
        self.set_app_config()
        # Note: when setting `debug=True`, a `ValueError` is raised with "signal only works in main thread"
        self.app.run(debug=False)

    def set_app_config(self):
        set_app_config(app=self.app)

    @staticmethod
    def update_app_config(param_path: List[str], value: object):
        assert len(param_path) > 1  # include at least one section

        # Get a copy of the current config
        new_config = config.get()

        # Update config
        param_name = param_path[-1]
        param_parent = new_config[param_path[0]]

        for subsec in param_path[1:-1]:
            param_parent = param_parent[subsec]

        param_parent[param_name] = value

        # Set the new config
        config.set(new_config)


@pytest.fixture(scope='session', autouse=True)
def backend() -> Backend:
    # Create and run the backend
    backend_ = Backend()
    backend_.run()

    # Run test session
    yield backend_

    # Delete the files related to the backend
    try:
        os.remove(CONFIG_FILE)
        os.remove(DEFAULT_API_KEY_FILE)
    except FileNotFoundError:
        pass


@pytest.fixture(scope='function', autouse=True)
def _backend_app_context(backend):
    with backend.app.app_context():
        yield


@pytest.fixture(scope='session', autouse=True)
def _setup_db(backend, mock_client_id, session_user_id, session_user_auth0_id):
    # TODO: Is this necessary? `_restore_environment` already calls `restore_db()`.
    with backend.app.app_context():
        populate_db(mock_client_id=mock_client_id,
                    session_user_id=session_user_id,
                    session_user_auth0_id=session_user_auth0_id)


@pytest.fixture(scope='session', autouse=True)
def _empty_db(backend):
    """
    Deletes all database tables after each test session execution.
    This fixture is used because `nexusml.api.create_app()` requires the existing database to meet certain conditions.
    If the database is empty, it automatically inserts the entries required to fulfill these conditions.
    """
    # Run test session
    yield
    # Delete all tables
    empty_db(app=backend.app)


###############
# Mock client #
###############


class MockClient:

    def __init__(self, app: Flask, client_id: str, session_user_id: str, session_user_auth0_id: str):
        # Set Flask app
        self._app = app
        self._app.test_client_class = FlaskClient
        self._always_use_test_client = False  # TODO: Remove this attribute when `send_request()` is removed

        # Set client data
        self.client_id = client_id

        # Set user data
        self._user_data = {
            'uuid': session_user_id,
            'auth0_id': session_user_auth0_id,
            'email': 'test@testorg.com',
            'password': '',
            'password_salt': '',
            'first_name': 'Test',
            'last_name': 'User',
            'roles': 'customer',
            'company': 'Test Co.',
            'email_verified': True,
        }

        # Set Auth0 token
        self.token_type = 'auth0'
        self.token_scopes = CLIENT_SCOPES
        self.token = self._generate_mock_token()

        # Store initial values
        self._init_client_id = self.client_id
        self._init_token_type = self.token_type
        self._init_token_scopes = self.token_scopes
        self._init_token = self.token
        self._init_user_data = dict(self._user_data)

    def restore(self):
        self.client_id = self._init_client_id
        self.token_type = self._init_token_type
        self.token_scopes = self._init_token_scopes
        self.token = self._init_token
        self._user_data = dict(self._init_user_data)

    def _generate_mock_token(self):
        return self._generate_mock_auth0_token() if self.token_type == 'auth0' else self._generate_mock_api_key()

    def _generate_mock_auth0_token(self) -> str:
        assert self.client_id
        assert self.token_scopes
        now = datetime.utcnow()
        claims = {
            'iss': os.environ[ENV_AUTH0_TOKEN_ISSUER],
            'aud': os.environ[ENV_AUTH0_TOKEN_AUDIENCE],
            'iat': now,
            'exp': now + timedelta(minutes=120),
            'scope': self.token_scopes,
            'azp': self.client_id,
            'sub': self._user_data['auth0_id'],
            'name': self._user_data['first_name'] + ' ' + self._user_data['last_name'],
            'given_name': self._user_data['first_name'],
            'family_name': self._user_data['last_name'],
            'email': self._user_data['email'],
            'email_verified': self._user_data['email_verified'],
            'company': self._user_data['company']
        }

        # Create tmp JWKS for token validation
        try:
            with open(os.environ[ENV_RSA_KEY_FILE], 'rb') as fd:
                key_from_pem = jwk.JWK.from_pem(fd.read())

                kid = 'jpkx3y6aihtbjw9cw6j3j0257yj'
                jwks = {
                    'keys': [{
                        'kty': 'RSA',
                        'n': key_from_pem['n'],
                        'e': key_from_pem['e'],
                        'kid': kid,
                        'alg': 'RS256',
                        'use': 'sig'
                    }]
                }

                jwks_file = NamedTemporaryFile(delete=False, suffix='.json')

                with open(jwks_file.name, 'w', newline='') as f:
                    json.dump(jwks, f, indent=4)

                os.environ[ENV_AUTH0_JWKS] = f'file:///{jwks_file.name}'
                jwks_file.close()
        except Exception:
            print(f'ERROR: Failed to load RSA key from "{os.environ[ENV_RSA_KEY_FILE]}"')
            print('Exiting')
            sys.exit(1)

        with open(os.environ[ENV_RSA_KEY_FILE], 'rb') as fd:
            private_key = serialization.load_pem_private_key(fd.read(), password=None, backend=default_backend())
        return jwt.encode(claims, private_key, 'RS256', {'kid': kid})

    def _generate_mock_api_key(self) -> str:
        assert self.client_id
        assert self.token_scopes
        assert all(scope in client_scopes for scope in self.token_scopes)
        api_key_exp = config.get('security')['api_keys']['expiration']
        now = datetime.utcnow()
        expire_at = now + timedelta(seconds=api_key_exp)
        token_claims = {
            'iss': constants.API_NAME,
            'aud': self.client_id,
            'exp': expire_at,
            'iat': now,
            'jti': str(uuid.uuid4()),
            'scope': (' '.join(sorted(self.token_scopes))),
            'api_version': constants.API_VERSION,
        }
        return jwt.encode(token_claims, config.rsa_private_key(), 'RS256')

    def update_client_id(self, client_id: str, token_type: str = 'auth0', token_scopes: List[str] = None):
        assert token_type in ['auth0', 'api_key']
        self.client_id = client_id
        self.token_type = token_type
        if token_scopes:
            self.token_scopes = token_scopes
        else:
            self.token_scopes = client_scopes if token_type == 'api_key' else CLIENT_SCOPES
        self.token = self._generate_mock_token()

    @deprecated('`send_request()` function is deprecated and will be removed in a future version. '
                'Please use `get()`, `post()`, `put()`, and `delete()` functions instead.')
    def send_request(self,
                     method: str,
                     url: str,
                     data: Optional[dict] = None,
                     json: Optional[Union[dict, Iterable[dict]]] = None,
                     files: Optional[dict] = None,
                     authenticate: bool = True,
                     use_test_client: bool = False) -> Union[TestResponse, requests.Response]:
        """
        Sends a request to the Flask app using the specified method, URL, and payload.

        Args:
            method (str): HTTP method (e.g., 'GET', 'POST', 'PUT', 'PATCH', 'DELETE')
            url (str): URL to send the request to
            data (dict): Form data to send with the request
            json (dict): JSON data to send with the request
            files (dict): Files to upload
            authenticate (bool): Whether to include the Authorization header in the request
            use_test_client (bool): Whether to use the Flask test client or the `requests` library.
                                    If `self._always_use_test_client` is `True`, this argument is ignored.

        Returns:
            Union[TestResponse, requests.Response]: Response object
        """

        assert self.token is not None

        use_test_client = self._always_use_test_client or use_test_client

        kwargs = dict()

        if authenticate:
            kwargs['headers'] = {'Authorization': f'Bearer {self.token}'}
        if data:
            kwargs['data'] = data
        if json:
            kwargs['json'] = json
        if files:
            kwargs['files'] = files

        method_calls = {
            'GET': self.get if use_test_client else requests.get,
            'POST': self.post if use_test_client else requests.post,
            'PUT': self.put if use_test_client else requests.put,
            'PATCH': self.patch if use_test_client else requests.patch,
            'DELETE': self.delete if use_test_client else requests.delete
        }
        method_call = method_calls[method.strip().upper()]

        return method_call(url, **kwargs)

    def get(self, *args, **kwargs) -> TestResponse:
        with self._app.app_context():
            return self._app.test_client().get(*args, **kwargs)

    def post(self, *args, **kwargs) -> TestResponse:
        with self._app.app_context():
            kwargs = self._set_file_data_for_test_client(**kwargs)
            return self._app.test_client().post(*args, **kwargs)

    def put(self, *args, **kwargs) -> TestResponse:
        with self._app.app_context():
            kwargs = self._set_file_data_for_test_client(**kwargs)
            return self._app.test_client().put(*args, **kwargs)

    def patch(self, *args, **kwargs) -> TestResponse:
        with self._app.app_context():
            return self._app.test_client().patch(*args, **kwargs)

    def delete(self, *args, **kwargs) -> TestResponse:
        with self._app.app_context():
            return self._app.test_client().delete(*args, **kwargs)

    def upload_files(self, files: Iterable[dict], parent: Union[Organization, Task],
                     storage_backend: FileStorageBackend) -> dict:
        """ Uploads files' metadata and (random) content to the API and the file storage backend, respectively.

        Args:
            files (list): metadata of the files to be uploaded
            parent (Resource): parent organization or task
            storage_backend (FileStorageBackend): file storage backend to upload the files to

        Returns:
            dict: a dictionary containing the following keys:
                  - `uploaded_metadata`: JSON with the metadata of the files uploaded to the API
                  - `uploaded_data`: IDs of the files uploaded to the file storage backend
        """

        def _upload_file(file: dict) -> Tuple[Response, Optional[Response]]:
            file_data = os.urandom(file['size'])

            # Upload file metadata to the API
            api_url = config.get('server')['api_url']
            if isinstance(parent, Task):
                endpoint = constants.ENDPOINT_TASK_FILES.replace('<task_id>', parent.public_id())
            else:
                endpoint = constants.ENDPOINT_ORG_FILES.replace('<organization_id>', parent.public_id())
            endpoint_url = API_DOMAIN + api_url + endpoint
            api_response = self.send_request(method='POST', url=endpoint_url, json=file)
            if api_response.status_code != constants.HTTP_POST_STATUS_CODE:
                return api_response, None
            api_res_json = api_response.json if isinstance(api_response, TestResponse) else api_response.json()
            upload_url = api_res_json['upload_url']

            # Upload file to the file storage backend
            if storage_backend == FileStorageBackend.S3:
                store_response = requests.post(url=upload_url['url'],
                                               data=upload_url['fields'],
                                               files={'file': file_data})
            else:
                store_response = self.send_request(method='POST',
                                                   url=upload_url['url'],
                                                   data=upload_url['fields'],
                                                   files={'file': file_data},
                                                   authenticate=False)

            # Mock thumbnail creation process
            if storage_backend == FileStorageBackend.S3:
                # Note: thumbnails are automatically created by AWS Lambda when a new image is uploaded to S3.
                #       See `scripts.aws.lambda.thumbnail_creator.lambda_handler()`)
                s3_client = boto3.client('s3')
                s3_config = get_s3_config()
                obj_prefixes = upload_url['fields']['key'].split('/')
                thumb_key = '/'.join(obj_prefixes[:-2]) + '/' + constants.PREFIX_THUMBNAILS + obj_prefixes[-1]
                thumb_response = s3_client.put_object(Body=os.urandom(4), Bucket=s3_config['bucket'], Key=thumb_key)
                assert thumb_response['ResponseMetadata']['HTTPStatusCode'] == 200
                assert thumb_response['ETag']
            else:
                pass  # TODO

            return api_response, store_response

        # Upload files in parallel
        # Note: using multiple workers results in a `ConnectionAbortedError`
        futures = []
        with ThreadPoolExecutor(max_workers=CLIENT_MAX_THREADS) as executor:
            for file in files:
                futures.append(executor.submit(_upload_file, file))
            wait(futures)

        # Check uploads
        uploaded_metadata = []
        uploaded_data = []

        for future in futures:
            api_response, store_response = future.result()
            api_response_json = api_response.json if isinstance(api_response, TestResponse) else api_response.json()
            # Check API response
            if api_response.status_code == constants.HTTP_POST_STATUS_CODE:
                uploaded_metadata.append(api_response_json)
            # Check storage backend response
            if store_response is not None and store_response.status_code == 204:
                uploaded_data.append(api_response_json['uuid'])

        return {'uploaded_metadata': uploaded_metadata, 'uploaded_data': uploaded_data}

    @staticmethod
    def _set_file_data_for_test_client(**kwargs) -> dict:
        """
        Prepares the file upload data for the Flask test client. Not calling this function results in the following
        error: "TypeError: EnvironBuilder.__init__() got an unexpected keyword argument 'files'".

        Args:
            **kwargs: the keyword arguments passed to the request method

        Returns:
            dict: the updated keyword arguments with the file upload data formatted correctly
        """
        # TODO: We haven't been able to make this work with the Flask test client.
        #       The `files` field is not recognized in the view function.
        # TODO: Uncomment the code below when the issue is resolved.
        return kwargs

        # # Check if the "files" field is present and contains a "file" field
        # if 'files' not in kwargs or 'file' not in kwargs['files']:
        #     return kwargs
        #
        # # Prepare a copy of `kwargs`
        # kwargs = dict(**kwargs)
        # if 'data' not in kwargs:
        #     kwargs['data'] = {}
        #
        # # Add the file to the "data" field.
        # # Note: The filename is ignored by the backend.
        # # TODO: The validation made by `@use_kwargs(_local_store_upload_form_fields, location='form')` returns:
        # #       {"errors": {"form": {"file": ["Unknown field."]}}}
        # kwargs['data']['file'] = (BytesIO(kwargs['files']['file']), 'filename.txt'),
        #
        # # Remove the "files" field as it has been processed
        # kwargs.pop('files')
        #
        # # Set the content type
        # kwargs['content_type'] = 'multipart/form-data'
        #
        # return kwargs


class FlaskClient(_FlaskClient):
    """ Custom Flask test client that automatically includes the Authorization header. """

    def __init__(self, *args, token: Optional[str] = None, **kwargs):
        self._token = token
        super().__init__(*args, **kwargs)

    def open(self, *args, **kwargs):
        # Convert absolute URL into relative URL
        if 'path' in kwargs:
            if kwargs['path'].startswith(API_DOMAIN):
                kwargs['path'] = kwargs['path'][len(API_DOMAIN):]
        elif args and isinstance(args[0], str) and args[0].startswith(API_DOMAIN):
            args = list(args)
            args[0] = args[0][len(API_DOMAIN):]

        # Add authorization header
        if self._token:
            headers = kwargs.pop('headers', dict())
            headers['Authorization'] = f'Bearer {self._token}'
            kwargs['headers'] = headers

        # Make request
        return super().open(*args, **kwargs)


@pytest.fixture(scope='function')
def client(backend, mock_client_id, session_user_id, session_user_auth0_id) -> MockClient:
    return MockClient(app=backend.app,
                      client_id=mock_client_id,
                      session_user_id=session_user_id,
                      session_user_auth0_id=session_user_auth0_id)


@pytest.fixture(scope='function')
def custom_client(request, client) -> MockClient:
    """
    Accepts the following arguments (captured by `request`):
        - Client UUID
        - Token type
        - Token scopes
    """

    init_client_id = client.client_id

    # Get fixture params
    kwargs = dict()
    assert hasattr(request, 'param') and isinstance(request.param, (list, tuple)) and len(request.param) > 0

    if len(request.param) >= 1:
        kwargs['client_id'] = request.param[0]
    if len(request.param) >= 2:
        token_type = request.param[1]
        if token_type is not None:
            kwargs['token_type'] = token_type
    if len(request.param) >= 3:
        token_scopes = request.param[2]
        if token_scopes is not None:
            kwargs['token_scopes'] = token_scopes

    # Update client ID and token
    client.update_client_id(**kwargs)

    # Yield client
    yield client

    # Restore client ID and token
    client.update_client_id(client_id=init_client_id)


###############
# Environment #
###############


@pytest.fixture(scope='session', autouse=True)
def aws_credentials():
    os.environ[ENV_AWS_ACCESS_KEY_ID] = 'testing'
    os.environ[ENV_AWS_SECRET_ACCESS_KEY] = 'testing'


@pytest.fixture(scope='function')
def mock_s3():
    with moto_mock_s3():
        conn = boto3.resource('s3')
        conn.create_bucket(Bucket=get_s3_config()['bucket'])
        yield


@pytest.fixture(scope='function', autouse=True)
def _restore_environment(backend, client, mock_client_id, session_user_id, session_user_auth0_id):
    # Restore backend
    backend.set_app_config()
    restore_db(mock_client_id=mock_client_id,
               session_user_id=session_user_id,
               session_user_auth0_id=session_user_auth0_id)
    cache.clear()
    # Restore client
    client.restore()


##################
# Monkey patches #
##################


def mock_celery_task(*args, **kwargs):
    pass  # Do nothing


def _generate_auth0_string():
    prefix = 'auth0|'
    suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=24))
    return prefix + suffix


mock_celery_task.delay = (lambda *args, **kwargs: None)  # Do nothing when calling `delay()`

mock_auth0_users = [
    MagicMock(json=lambda: {
        'email': 'test@testorg.com',
        'given_name': 'Test',
        'family_name': 'User',
        'user_id': 'auth0|111111111111111111111111',
        'email_verified': True
    },
              status_code=200),
    MagicMock(json=lambda: {
        'email': 'user_2@org2.com',
        'given_name': 'User',
        'family_name': '2',
        'user_id': 'auth0|222222222222222222222222',
        'email_verified': True
    },
              status_code=200),
    MagicMock(json=lambda: {
        'email': 'user_3@org2.com',
        'given_name': 'User',
        'family_name': '3',
        'user_id': 'auth0|333333333333333333333333',
        'email_verified': True
    },
              status_code=200),
    MagicMock(json=lambda: {
        'email': 'user_4@org2.com',
        'given_name': 'User',
        'family_name': '4',
        'user_id': 'auth0|444444444444444444444444',
        'email_verified': True
    },
              status_code=200),
    MagicMock(json=lambda: {
        'email': 'user_5@org3.com',
        'given_name': 'User',
        'family_name': '5',
        'user_id': 'auth0|555555555555555555555555',
        'email_verified': True
    },
              status_code=200),
    MagicMock(json=lambda: {
        'email': 'user_6@org3.com',
        'given_name': 'User',
        'family_name': '6',
        'user_id': 'auth0|666666666666666666666666',
        'email_verified': True
    },
              status_code=200),
    MagicMock(json=lambda: {
        'email': 'user_7@org3.com',
        'given_name': 'User',
        'family_name': '7',
        'user_id': 'auth0|777777777777777777777777',
        'email_verified': True
    },
              status_code=200),
    MagicMock(json=lambda: {
        'email': 'user_8@org3.com',
        'given_name': 'User',
        'family_name': '8',
        'user_id': 'auth0|888888888888888888888888',
        'email_verified': True
    },
              status_code=200),
    MagicMock(json=lambda: {
        'email': 'test@org2.com',
        'given_name': 'Test',
        'family_name': 'User 1',
        'user_id': 'auth0|789078907890789078907890',
        'email_verified': True
    },
              status_code=200),
    MagicMock(json=lambda: {
        'email': 'test2@org2.com',
        'given_name': 'Test',
        'family_name': 'User 2',
        'user_id': 'auth0|222222222222221111111111',
        'email_verified': True
    },
              status_code=200)
]
mock_get_user: MagicMock = MagicMock(json=lambda: {
    'email': 'test@testorg.com',
    'given_name': 'Test',
    'family_name': 'User',
    'user_id': 'auth0|111111111111111111111111',
    'email_verified': True
},
                                     status_code=200)

mock_delete_responses = {'https://[^/]+/api/v2/users/[^/]+': Response(status=204)}
mock_get_responses = {
    'https://[^/]+/oauth/token': MagicMock(json=lambda: {'access_token': 'test_token'}, status_code=200),
    r'https://[^/]+/api/v2/users\?q=email:[^&]+&search_engine=v3': mock_get_user,
    'https://[^/]+/api/v2/users/[^/]+': mock_get_user
}
mock_post_responses = {
    'https://[^/]+/oauth/token':
        MagicMock(json=lambda: {'access_token': 'test_token'}),
    'https://[^/]+/api/v2/tickets/password-change':
        MagicMock(
            json=lambda: {'ticket': 'https://testorg.eu.auth0.com/lo/reset?ticket=JeF44XnsKOp9IMaypUfTtjqFkisILQnS#'},
            status_code=201),
    'https://[^/]+/api/v2/users': {
        'test1@org2.com':
            MagicMock(json=lambda: {
                'email': 'test1@org2.com',
                'user_id': 'auth0|111111111112222222222222'
            },
                      status_code=201),
        'test2@org2.com':
            MagicMock(json=lambda: {
                'email': 'test2@org2.com',
                'user_id': 'auth0|222222222222221111111111'
            },
                      status_code=201),
        'user_2@org2.com':
            MagicMock(json=lambda: {
                'email': 'user_2@org2.com',
                'user_id': 'auth0|222222222222222222222222'
            },
                      status_code=201),
        'user_5@org2.com':
            MagicMock(json=lambda: {
                'email': 'user_5@org2.com',
                'user_id': 'auth0|555555555558888888888888'
            },
                      status_code=201),
        'user_8@org3.com':
            MagicMock(json=lambda: {
                'email': 'user_8@org3.com',
                'user_id': 'auth0|877777777788888888888888'
            },
                      status_code=201),
        'test@testorg.com':
            MagicMock(json=lambda: {
                'email': 'test@testorg.com',
                'user_id': 'auth0|123412341234123412341234'
            },
                      status_code=201),
        'invitation_test@testorg.com':
            MagicMock(json=lambda: {
                'email': 'invitation_test@testorg.com',
                'user_id': 'auth0|123412349234123412310234'
            },
                      status_code=201),
    },
    'https://[^/]+/api/v2/users?q=email:[^&]+&search_engine=v3':
        MagicMock(json=lambda: [{
            'first_name': 'Test',
            'last_name': 'User',
            'email': 'test@test.com'
        }], status_code=201),
    'https://[^/]+/api/v2/users/[^/]+':
        MagicMock(json=lambda: [{
            'first_name': 'Test',
            'last_name': 'User',
            'email': 'test@test.com'
        }], status_code=201),
}
mock_put_responses = dict()  # TODO: key = URL regex; value = mock JSON
mock_patch_responses = {'https://[^/]+/api/v2/users/[^/]+': Response(status=200)}

original_requests_methods = {
    'delete': requests.delete,
    'get': requests.get,
    'post': requests.post,
    'put': requests.put,
    'patch': requests.patch
}


def _extract_email(kwargs, url_):
    email_regex = re.compile(r'[^@]+@[^@]+\.[^@]+')
    email_url_regex = re.compile(r'https://[^/]+/api/v2/users\?q=email:([^&]+)&search_engine=v3')

    for key, value in kwargs.items():
        if email_regex.match(str(value)):
            email = value.get('email')
            if email is not None:
                return email

    match = re.search(email_url_regex, url_)
    if match:
        return match.group(1)

    return None


def _extract_auth0_user_id(url_):
    auth0_user_id_regex = re.compile(r'auth0\|[a-zA-Z0-9]{24}')
    match = auth0_user_id_regex.findall(url_)
    if match and len(match) == 1:
        return match[0]
    return None


def _find_matching_user(email=None, auth0_id=None):
    if email:
        return next((user for user in mock_auth0_users if user.json()['email'] == email), None)
    if auth0_id:
        return next((user for user in mock_auth0_users if user.json()['user_id'] == auth0_id), None)
    return None


def _mock_request_response(http_method: str, *args, **kwargs):
    http_method_mock_responses = {
        'delete': mock_delete_responses,
        'get': mock_get_responses,
        'post': mock_post_responses,
        'put': mock_put_responses,
        'patch': mock_patch_responses
    }
    mock_responses: dict = http_method_mock_responses[http_method]

    url_: str = kwargs.get('url', args[0])

    url_ = urllib.parse.unquote(url_)

    email = _extract_email(kwargs, url_)
    auth0_id = _extract_auth0_user_id(url_)

    for url_regex, response in mock_responses.items():
        if re.match(url_regex, url_):
            if email or auth0_id:
                # Return a specific mock response if email matches
                if email and http_method == 'post' and email in response:
                    return response[email]

                matching_user = _find_matching_user(email, auth0_id)
                if matching_user:
                    if http_method == 'delete':
                        matching_user.status_code = 204
                    elif http_method == 'get':
                        matching_user.status_code = 200

                    return matching_user
                if matching_user == {}:
                    return MagicMock(json=lambda: [{}], status_code=200)
            if http_method == 'get':
                matching_user = _find_matching_user(email, auth0_id)
                if matching_user:
                    return matching_user

            return response

    # Fallback to real request if no mock matches
    return original_requests_methods[http_method](*args, **kwargs)


@pytest.fixture(scope='function')
def mock_request_responses(monkeypatch):
    monkeypatch.setattr(requests, 'delete', (lambda *a, **k: _mock_request_response('delete', *a, **k)))
    monkeypatch.setattr(requests, 'get', (lambda *a, **k: _mock_request_response('get', *a, **k)))
    monkeypatch.setattr(requests, 'post', (lambda *a, **k: _mock_request_response('post', *a, **k)))
    monkeypatch.setattr(requests, 'put', (lambda *a, **k: _mock_request_response('put', *a, **k)))
    monkeypatch.setattr(requests, 'patch', (lambda *a, **k: _mock_request_response('patch', *a, **k)))


@pytest.fixture(scope='function', autouse=True)
def monkey_patches(monkeypatch, backend):
    """
    Fixture for monkey patching modules and objects in tests. Any change made using this fixture is temporary and
    automatically reverted to its original state after each test function completes. This ensures that modifications
    do not leak into other tests, maintaining test isolation and consistency.

    Args:
        monkeypatch: built-in Pytest fixture that allows modifying or replacing attributes, methods, or functions in
                     the code during the execution of an individual test function. This is particularly useful for
                     altering the behavior of external dependencies or changing global settings without affecting the
                     actual application logic.
        backend: fixture running the API.
    """
    ##############################################################
    # Constants.                                                 #

    # Use `monkeypatch.setattr(constants, <constant>, <value>)`. #
    ##############################################################
    pass
    #############################################################
    # Functions.                                                #

    # Use `monkeypatch.setattr(<module>, <function>, <value>)`. #
    #############################################################
    monkeypatch.setattr(flask_mail.Mail, 'send', (lambda self, message: None))
    ########################################################################################
    # Celery tasks (required to bypass `@shared_task`, which injects `.delay()` function). #

    # Use `monkeypatch.setattr(<module>, <function>, mock_celery_task)`.                   #
    ########################################################################################
    # Warning: don't move these imports as they require `backend` (i.e., an app context).

    monkeypatch.setattr(organizations_views, '_populate_demo_tasks', mock_celery_task)
