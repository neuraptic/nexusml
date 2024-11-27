from datetime import datetime
from datetime import timedelta
import functools
from io import BytesIO
import os
import re
import sys
import tempfile
from typing import Any, IO, Optional, Union
import urllib.parse
from urllib.parse import unquote_plus
import uuid

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.dh import DHPublicKey
from cryptography.hazmat.primitives.asymmetric.types import PrivateKeyTypes
from flask import Flask
import jwt
from PIL import Image
from platformdirs import user_data_dir
import requests
from requests import Response
from werkzeug.exceptions import BadRequest

from nexusml.constants import API_NAME
from nexusml.constants import API_VERSION
from nexusml.constants import PREFIX_THUMBNAILS
from nexusml.constants import THUMBNAIL_SIZE
from nexusml.enums import EngineType
from nexusml.enums import FileStorageBackend
from nexusml.env import ENV_API_DOMAIN
from nexusml.env import ENV_AUTH0_CLIENT_ID
from nexusml.env import ENV_AUTH0_CLIENT_SECRET
from nexusml.env import ENV_AUTH0_DOMAIN
from nexusml.env import ENV_AUTH0_JWKS
from nexusml.env import ENV_AUTH0_TOKEN_AUDIENCE
from nexusml.env import ENV_AUTH0_TOKEN_ISSUER
from nexusml.env import ENV_RSA_KEY_FILE

##########
# CONFIG #
##########

API_DOMAIN = 'https://' + os.environ[ENV_API_DOMAIN]

DEFAULT_CONFIG = {
    'engine': {
        'worker': {
            'type': 'local'
        },
        'services': {
            'inference': {
                'enabled': True,
            },
            'continual_learning': {
                'enabled': True,
                'min_days': 7.0,  # Maximum frequency at which the AI can be retrained.
                'max_days': 28.0,  # Minimum frequency at which the AI should be retrained.
                'min_sample': 0.2,  # Minimum sample size to trigger retraining, relative to current number of examples.
                # Value between 0 and 1, representing the percentage of current number of examples.
                'min_cpu_quota': 600.0,  # In hours.
                'max_cpu_quota': 900.0,  # In hours.
                'cpu_hard_limit': 2000.0,  # CPU hard limit to guarantee quality of service (in hours).
                'min_gpu_quota': 300.0,  # In hours.
                'max_gpu_quota': 450.0,  # In hours.
                'gpu_hard_limit': 1000.0,  # GPU hard limit to guarantee quality of service (in hours).
            },
            'active_learning': {
                'enabled': True,
                'query_interval': 7,  # Interval in days between active learning queries.
                'max_examples_per_query': 50,  # Maximum number of examples to be queried per query.
            },
            'monitoring': {
                'enabled': True,
                'refresh_interval': 100,
                # Interval in which metrics are refreshed (in number of predictions). Setting it
                # to 1 forces metrics to be refreshed each time a new prediction is made.
                'ood_predictions': {  # Detection of out-of-distribution (OOD) predictions.
                    'min_sample': 100,  # Minimum number of predictions required for running detection.
                    'sensitivity': 0.5,  # Sensitivity to anomalies (value between 0 and 1).
                    'smoothing': 0.8,  # Smoothing factor (value between 0 and 1). Low values result in less smoothing
                    # and thus a high responsiveness to variations in predictions.
                }
            },
            'testing': {
                'enabled': True,
            },
        },
    },
    'general': {
        'auth_enabled': False  # WARNING: Authentication should be enabled in production.
    },
    'jobs': {
        'abort_upload_after': 7,  # Uploads that don't complete within the specified number of days will be aborted.
        'billing_time': '02:00',  # UTC at which billing jobs run every day (HH:MM format string).
        'log_buffer_size': 100,  # Buffer size for prediction logging (in number of predictions).
        'max_workers': 5,
    },
    'limits': {
        'organizations': {
            'num_organizations': 100,  # Maximum number of organizations.
            'picture_size': 1 * 1024**2,  # Maximum picture size (in bytes). Default: 1 MB.
            'waitlist': 10**4,  # Maximum number of entries in the wait list.
        },
        'quotas': {
            'free_plan': {
                'max_apps': 3,
                'max_collaborators': 3,
                'max_cpu_hours': 0,  # Per billing cycle.
                'max_deployments': 0,  # Maximum number of AI model deployments.
                'max_gpu_hours': 0,  # Per billing cycle.
                'max_predictions': 0,  # Per billing cycle.
                'max_roles': 5,  # "admin" and "maintainer" roles don't count.
                'max_tasks': 1,
                'max_users': 10,
                'max_examples': 10**4,
                'space_limit': 50 * 1024**2,  # In bytes. Default: 50 MB.
            },
        },
        'requests': {
            'cache_timeout': 60,  # In seconds.
            'max_payload': 20 * 1024**2,  # In bytes. Default: 20 MB.
            'requests_per_day': 10**7,
            'requests_per_hour': 10**6,
            'requests_per_minute': 10**4,
            'requests_per_second': 10**3,
        },
        'tasks': {
            'picture_size': 1 * 1024**2,  # In bytes. Default: 1 MB.
            'max_preloaded_categories': 100,  # TODO: this parameter should be removed when the cache works correctly.
        },
    },
    'notifications': {
        'interval': 600,  # Interval in which notifications will be sent (in seconds).
        'max_source_events': 50,  # Maximum number of individual notifications to be stored in the database
        # for each pair (source_type, event).
        'mail_port': 587,
        'use_tls': True,
        'use_ssl': False
    },
    'security': {
        'api_keys': {
            'expiration': 60 * 60 * 24 * 30,  # In seconds. Default: 30 days.
        },
        'public_id': {
            'min_length': 8,
        },
    },
    'server': {
        'api_url': f'/v{API_VERSION}'
    },
    'storage': {
        'database': {
            'type': 'mysql',  # For the moment, only MySQL is supported.
            'uri': 'mysql+pymysql://<user>:<password>@localhost:3306/<database>',
        },
        'files': {
            'backend': 's3',  # "s3" or "local".
            'local': {
                'max_upload_size': 100 * 1024**2,  # In bytes. Default: 100 MB.
                'root_path': user_data_dir(API_NAME),
                'url_expiration': 600,  # Presigned URLs' expiration (in seconds). Default: 10 minutes.
            },
            's3': {
                'max_upload_size': 100 * 1024**2,  # In bytes. Default: 100 MB.
                'url_expiration': 600,  # Presigned URLs' expiration (in seconds). Default: 10 minutes.
            },
        }
    },
    'views': {
        'default_items_per_page': 25,
        'max_items_per_page': 100,
    }
}


class NexusMLConfig:
    """
    Configuration class for handling the NexusML app's config and RSA key management.

    This class is responsible for initializing and storing the Flask app configuration,
    handling the private RSA key used for signing tokens, and managing access to configuration
    values by verifying if the app has been initialized. It also provides methods to retrieve the
    public RSA key and set/reset configuration values.
    """

    def __init__(self):
        """
        Initializes the NexusMLConfig instance with default values.

        Attributes:
            _app (Flask): Flask app instance.
            _rsa_private_key (PrivateKeyTypes): RSA private key instance.
        """
        self._app: Flask = None
        self._rsa_private_key: PrivateKeyTypes = None

    def init_app(self, app: Flask):
        """
        Initializes the NexusMLConfig with a Flask app instance.

        Args:
            app (Flask): The Flask app instance to initialize with.

        Raises:
            RuntimeError: If the app has already been initialized.
        """
        if self._app is not None:
            raise RuntimeError('NexusML config already initialized')
        self._app = app

    @staticmethod
    def verify_initialized_app(func):
        """
        A decorator that verifies if the NexusMLConfig has been initialized with a Flask app.

        Args:
            func: The function to wrap with initialization check.

        Returns:
            Callable: The wrapped function that checks initialization before execution.

        Raises:
            RuntimeError: If the app has not been initialized.
        """

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self._app is None:
                raise RuntimeError('NexusML config not initialized. Call `init_app()` first')
            return func(self, *args, **kwargs)

        return wrapper

    @property
    def app_name(self) -> str:
        return API_NAME.upper().replace(' ', '_')

    def _get(self) -> dict:
        prefix_len = len(self.app_name) + 1
        _config = {k[prefix_len:]: v for k, v in self._app.config.items() if k.startswith(self.app_name + '_')}
        return self._to_lowercase_keys(_config)

    @verify_initialized_app
    def get(self, section: Optional[str] = None) -> Any:
        """
        Returns a copy of the configuration of the app.

        Args:
            section (Optional[str]): The section of the configuration to return.
                                     If `None`, the entire configuration will be returned.

        Returns:
            Any: A copy of the configuration of the app.
        """
        if section:
            return self._get()[section]
        else:
            return self._get()

    @verify_initialized_app
    def set(self, config: dict):
        """
        Sets configuration values for the NexusML app by clearing previous config and setting new ones.

        Args:
            config (dict): The dictionary containing new configuration values.
        """
        # TODO: Should we flatten the config dict to a single level?
        #       Typically, config values in Flask are stored as single-level key-value pairs
        #       (e.g., strings, integers, booleans)
        # Clear previous config
        previous_config = self._get()
        for section in previous_config:
            self._app.config.pop((self.app_name + '_' + section).upper())
        # Convert all keys to uppercase
        new_config = self._to_uppercase_keys(config)
        # Set new config
        for section, params in new_config.items():
            self._app.config[self.app_name + '_' + section] = params

    @staticmethod
    def _to_uppercase_keys(data: Any) -> Any:
        if isinstance(data, dict):
            new_dict = dict()
            for key, value in data.items():
                # Recursively call this function if the value is also a dict
                new_dict[key.upper()] = NexusMLConfig._to_uppercase_keys(value)
            return new_dict
        elif isinstance(data, list):
            # If the value is a list, recursively process each item
            return [NexusMLConfig._to_uppercase_keys(item) for item in data]
        else:
            # If it's neither a dict nor a list, just return the value
            return data

    @staticmethod
    def _to_lowercase_keys(data: Any) -> Any:
        if isinstance(data, dict):
            new_dict = dict()
            for key, value in data.items():
                # Recursively call this function if the value is also a dict
                new_dict[key.lower()] = NexusMLConfig._to_lowercase_keys(value)
            return new_dict
        elif isinstance(data, list):
            # If the value is a list, recursively process each item
            return [NexusMLConfig._to_lowercase_keys(item) for item in data]
        else:
            # If it's neither a dict nor a list, just return the value
            return data

    def rsa_private_key(self) -> PrivateKeyTypes:
        """
        Retrieves the private RSA key used for signing data.

        If the private key is already loaded, it returns the cached key.
        Otherwise, it loads the key from the file specified in the environment variable.

        Returns:
            PrivateKeyTypes: The RSA private key used for signing.

        Raises:
            SystemExit: If the key file cannot be loaded.
        """
        if self._rsa_private_key is not None:
            return self._rsa_private_key

        try:
            with open(os.environ[ENV_RSA_KEY_FILE], 'rb') as fd:
                self._rsa_private_key = serialization.load_pem_private_key(fd.read(),
                                                                           password=None,
                                                                           backend=default_backend())
        except Exception:
            print(f'ERROR: Failed to load RSA key from "{os.environ[ENV_RSA_KEY_FILE]}"')
            print('Exiting')
            sys.exit(1)

        return self._rsa_private_key

    def rsa_public_key(self) -> DHPublicKey:
        """
        Retrieves the public RSA key derived from the private RSA key.

        This method calls the `rsa_private_key` method to retrieve the private key,
        and from that key it derives the public key.

        Returns:
            DHPublicKey: The RSA public key.
        """
        return self.rsa_private_key().public_key()


# TODO: The current centralized config approach with a global `NexusMLConfig` instance
#       doesn't support multiple Flask apps running in the same process, as it creates a
#       shared configuration object for all apps. Consider refactoring to instantiate `NexusMLConfig`
#       for each Flask app separately or use `current_app.config` directly within each app's context.
config = NexusMLConfig()

##########
# ENGINE #
##########


def get_engine_type() -> EngineType:
    return EngineType[config.get('engine')['worker']['type'].upper()]


##########
# TOKENS #
##########


def generate_tmp_token(agent_uuid: str, expires_in: int, custom_claims: dict, custom_claims_key: str) -> str:
    """
    Generates a temporary token with the provided claims.

    Args:
        agent_uuid (str): UUID of the agent that requested the token
        expires_in (int): token expiration time in seconds
        custom_claims (dict): custom token claims
        custom_claims_key (str): key for the custom claims

    Returns:
        str: the generated temporary token
    """
    utc_now = datetime.utcnow()
    token_claims = {
        'iss': API_NAME,  # Issuer (current API)
        'aud': agent_uuid,  # Audience (who is the token for)
        'jti': str(uuid.uuid4()),  # Unique identifier for the token
        'iat': utc_now,  # Issued at (current UTC time)
        'exp': utc_now + timedelta(seconds=expires_in),  # Expiration time
        'api_version': API_VERSION,  # API version that issued the token
        custom_claims_key: custom_claims  # Provided custom claims
    }
    return jwt.encode(payload=token_claims, key=config.rsa_private_key(), algorithm='RS256')


def _decode_jwt(token: str, public_key, issuer: str = None, audience: str = None, verify: bool = True) -> dict:
    """
    Decodes a JWT token and optionally verifies its claims.

    This function decodes a JWT token using the provided public key, and can optionally verify the token's
    issuer and audience. It also disables certain date verifications to avoid issues with local timezone usage.
    After decoding, the token's claims are validated, such as issued-at, not-before, and expiration time.

    Args:
        token (str): The JWT token to decode.
        public_key: The public key used for decoding and verifying the token.
        issuer (str, optional): The expected issuer of the token. Defaults to None.
        audience (str, optional): The expected audience of the token. Defaults to None.
        verify (bool, optional): Whether to verify the token claims. Defaults to True.

    Returns:
        dict: The decoded token claims.

    Raises:
        ValueError: If the token's issued-at, not-before, or expiration claims are invalid.
    """
    algorithms = ['RS256']

    if not verify:
        return jwt.decode(token, public_key, algorithms=algorithms, options={'verify_signature': False})

    # Note: we disable datetime verifications because PyJWT uses the local timezone instead of UTC
    options = {'verify_iat': False, 'verify_nbf': False, 'verify_exp': False}

    if not issuer:
        options['verify_iss'] = False
    if not audience:
        options['verify_aud'] = False

    dec_token = jwt.decode(jwt=token,
                           key=public_key,
                           algorithms=algorithms,
                           issuer=issuer,
                           audience=audience,
                           options=options)

    now = datetime.utcnow().timestamp()

    try:
        int(dec_token.get('iat', 0))
    except ValueError:
        raise ValueError('issued-at claim (iat) must be an integer')
    if dec_token.get('nbf', 0) > now:
        raise ValueError('the token is not yet valid')
    if dec_token.get('exp', sys.maxsize) < now:
        raise ValueError('the token has expired')

    return dec_token


def decode_auth0_token(auth0_token: str) -> dict:
    """
    Decodes an Auth0 access token using the JSON Web Key Set (JWKS) from the Auth0 domain.

    The function retrieves the JWKS from the Auth0 domain, extracts the signing key from it,
    and uses the key to decode the token.

    Args:
        auth0_token (str): The Auth0 access token to decode.

    Returns:
        dict: The decoded token claims.
    """
    jwks_uri = os.environ[ENV_AUTH0_JWKS]
    jwks_client = jwt.PyJWKClient(jwks_uri)
    key = jwks_client.get_signing_key_from_jwt(auth0_token).key

    return _decode_jwt(token=auth0_token,
                       public_key=key,
                       verify=True,
                       issuer=os.environ[ENV_AUTH0_TOKEN_ISSUER],
                       audience=os.environ[ENV_AUTH0_TOKEN_AUDIENCE])


def decode_api_key(api_key: str, verify: bool = True) -> dict:
    """
    Decodes a NexusML API key (JWT token) using the public RSA key.

    This function decodes the provided API key and optionally verifies its claims, such as issuer
    and audience, against the expected values.

    Args:
        api_key (str): The encoded API key (JWT token).
        verify (bool, optional): Whether to verify the token claims. Defaults to True.

    Returns:
        dict: The decoded API key claims.
    """
    return _decode_jwt(token=api_key, public_key=config.rsa_public_key(), verify=verify, issuer=API_NAME)


#############
# Auth0 API #
#############


def get_auth0_management_api_token() -> str:
    """
    Retrieves the Auth0 Management API token required for accessing the Auth0 database.

    This function sends a POST request to the Auth0 token endpoint with the necessary credentials,
    including client ID, client secret, and audience. The returned access token is used for making
    further Auth0 Management API calls.

    Returns:
        str: The Auth0 Management API access token.
    """
    access_token: str
    payload: dict = {
        'grant_type': 'client_credentials',
        'client_id': os.environ[ENV_AUTH0_CLIENT_ID],
        'client_secret': os.environ[ENV_AUTH0_CLIENT_SECRET],
        'audience': f'https://{os.environ[ENV_AUTH0_DOMAIN]}/api/v2/'
    }

    headers: dict = {'Content-Type': 'application/json'}

    response_data: Response = requests.post(f'https://{os.environ[ENV_AUTH0_DOMAIN]}/oauth/token',
                                            json=payload,
                                            headers=headers)
    json_data: dict = response_data.json()
    access_token = json_data['access_token']

    return access_token


def get_auth0_user_data(access_token: str, auth0_id_or_email: str) -> dict:
    """
    Matches an Auth0 ID or email to retrieve the associated user data.

    This function checks if the provided identifier is an email or an Auth0 ID, constructs the appropriate URL,
    and sends a GET request to retrieve the user account data. If the identifier is an email, it searches by email;
    otherwise, it searches by Auth0 ID.

    Args:
        access_token (str): The Auth0 access token for authorization.
        auth0_id_or_email (str): The Auth0 ID or email to match.

    Returns:
        dict: The matched user data.

    Raises:
        BadRequest: If no Auth0 user is associated with the provided identifier.
    """
    auth0_user_data: dict

    headers: dict = {'Authorization': 'Bearer ' + access_token}
    regex_email = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'

    encoded_email_or_auth0_id = urllib.parse.quote(auth0_id_or_email)

    auth0_domain = os.environ[ENV_AUTH0_DOMAIN]
    if re.fullmatch(regex_email, auth0_id_or_email):
        url = f'https://{auth0_domain}/api/v2/users?q=email:{encoded_email_or_auth0_id}&search_engine=v3'
    else:
        url = f'https://{auth0_domain}/api/v2/users/{encoded_email_or_auth0_id}'

    response: Response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise BadRequest(description=f'No Auth0 user associated with "{auth0_id_or_email}"')

    auth0_user_data = response.json()
    if auth0_user_data and isinstance(auth0_user_data, list):
        auth0_user_data: dict = auth0_user_data[0]

    return auth0_user_data


def delete_auth0_user(auth0_id: str) -> None:
    """
    Deletes an Auth0 user account based on the provided Auth0 ID.

    This function retrieves an Auth0 management API token, constructs the URL for the user deletion endpoint,
    and sends a DELETE request to remove the user.

    Args:
        auth0_id (str): The Auth0 ID of the user to delete.

    Raises:
        AssertionError: If the DELETE request does not return a status code of 204.
    """
    auth0_token = get_auth0_management_api_token()
    url = f'https://{os.environ[ENV_AUTH0_DOMAIN]}/api/v2/users/{urllib.parse.quote(auth0_id)}'
    headers = {'Authorization': 'Bearer ' + auth0_token}
    res: Response = requests.delete(url, headers=headers)
    assert res.status_code == 204


################
# FILE STORAGE #
################


def get_file_storage_backend() -> FileStorageBackend:
    """
    Retrieves the current file storage backend in use.

    The function checks the configuration for the file storage backend, which can either be local or S3,
    and returns the corresponding enum value.

    Returns:
        FileStorageBackend: The file storage backend in use (LOCAL or S3).
    """
    return FileStorageBackend[config.get('storage')['files']['backend'].upper()]


def generate_thumbnail(fp: Union[str, IO[bytes]]):
    """
    Generates a thumbnail from an image.

    Args:
        fp (Union[str, IO[bytes]]): The file path or file object of the image.

    Returns:
        tempfile.NamedTemporaryFile: The temporary file containing the generated thumbnail.
    """
    # Create the thumbnail
    img = Image.open(fp).convert('RGB')
    img.thumbnail(THUMBNAIL_SIZE)
    # Save the thumbnail to a temporary file
    tmp_file = tempfile.NamedTemporaryFile(delete=False)  # WARNING: `delete=True` fails on Windows
    img.save(tmp_file.name, 'JPEG')
    # Return the temporary file
    return tmp_file


#######################
# FILE STORAGE: LOCAL #
#######################


def get_local_file_storage_config() -> dict:
    """
    Returns the local file store configuration.

    Returns:
        dict: The local file store configuration.

    Raise:
        KeyError: If the local file store configuration is not found.
    """
    return config.get('storage')['files']['local']


def save_thumbnail_to_local_file_store(thumbnail_path: str, file_content: bytes):
    """
    Saves a thumbnail image to the local file store.

    This function generates a thumbnail from the provided image content and writes it to
    the specified file path in the local file store.

    Args:
        thumbnail_path (str): The path where the thumbnail will be saved.
        file_content (bytes): The binary content of the original image file.
    """
    # Create the thumbnail
    tmp_file = generate_thumbnail(fp=BytesIO(file_content))
    # Save the thumbnail to the local file store
    with open(thumbnail_path, 'wb') as f:
        f.write(tmp_file.read())


########################
# FILE STORAGE: AWS S3 #
########################


def save_thumbnail_to_s3(s3_client, bucket: str, object_key: str):
    """
    Saves a generated thumbnail to an AWS S3 bucket.

    This function retrieves the original image from S3, generates a thumbnail, and then uploads
    the thumbnail to the specified S3 bucket under the appropriate key.

    Args:
        s3_client: The S3 client used for interacting with AWS S3.
        bucket (str): The name of the S3 bucket where the thumbnail will be saved.
        object_key (str): The S3 object key of the original image, which will be used to generate the thumbnail.
    """
    # Note: pass `s3_client` as argument to allow tests to use mock S3 client

    # Get the original image from S3
    src_key = unquote_plus(object_key)
    src_prefixes = src_key.split('/')
    src_uuid = src_prefixes[-1]

    s3_object = s3_client.get_object(Bucket=bucket, Key=src_key)

    # Create the thumbnail
    tmp_file = generate_thumbnail(fp=s3_object['Body'])

    # Save the thumbnail to S3
    dst_key = '/'.join(src_prefixes[:-2]) + '/' + PREFIX_THUMBNAILS + src_uuid
    s3_client.upload_file(tmp_file.name, bucket, dst_key)

    # Close and remove the temporary file
    tmp_file.close()
    os.remove(tmp_file.name)


########
# MISC #
########


def encode_url(url: str):
    """
    Encodes a URL string by replacing certain characters with their percent-encoded equivalents.

    Args:
        url (str): The URL to encode.

    Returns:
        str: The encoded URL string.
    """
    return url.replace('-', '%2D').replace('_', '%5F').replace('.', '%2E').replace(' ', '%20').replace('/', '%2F')


def split_str(str_to_split: str, separators: list) -> list:
    """
    Splits a string using multiple separators and strips any extra whitespace.

    Args:
        str_to_split (str): The string to split.
        separators (list): A list of separator characters or patterns.

    Returns:
        list: A list of the split and stripped substrings.
    """
    sep_re = '|'.join(separators)
    return list(map(lambda x: x.strip(), re.split(sep_re, str_to_split)))


def camel_to_snake(string: str) -> str:
    """
    Converts a string from CamelCase to snake_case.

    Args:
        string (str): The CamelCase string to convert.

    Returns:
        str: The converted snake_case string.
    """
    return ''.join(['_' + c.lower() if c.isupper() else c for c in string]).lstrip('_')
