import functools
import inspect
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple, Type, Union

from flask import Flask
from flask import g
from flask import jsonify
from flask import request
from flask import Response
from flask_apispec import FlaskApiSpec
from flask_apispec import MethodResource
from flask_limiter import Limiter
from flask_restful import Api
import jwt
from marshmallow import Schema
from sqlalchemy.orm import Query
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql.elements import UnaryExpression
from werkzeug.exceptions import BadRequest

from nexusml.api.ext import cache
from nexusml.api.resources import DuplicateResourceError
from nexusml.api.resources import ImmutableResourceError
from nexusml.api.resources import InvalidDataError
from nexusml.api.resources import QuotaError
from nexusml.api.resources import ResourceError
from nexusml.api.resources import ResourceOutOfSyncError
from nexusml.api.resources import UnprocessableRequestError
from nexusml.api.resources.base import PermissionDeniedError
from nexusml.api.resources.base import Resource
from nexusml.api.resources.base import ResourceNotFoundError
from nexusml.api.resources.organizations import get_resource_organization
from nexusml.api.resources.organizations import get_user_roles
from nexusml.api.resources.organizations import Organization
from nexusml.api.resources.tasks import Task
from nexusml.api.schemas.base import PageSchema
from nexusml.api.utils import config
from nexusml.api.utils import decode_api_key
from nexusml.api.utils import decode_auth0_token
from nexusml.constants import API_NAME
from nexusml.constants import HTTP_BAD_REQUEST_STATUS_CODE
from nexusml.constants import HTTP_CONFLICT_STATUS_CODE
from nexusml.constants import HTTP_DELETE_STATUS_CODE
from nexusml.constants import HTTP_FORBIDDEN_STATUS_CODE
from nexusml.constants import HTTP_NOT_FOUND_STATUS_CODE
from nexusml.constants import HTTP_PAYLOAD_TOO_LARGE_STATUS_CODE
from nexusml.constants import HTTP_POST_STATUS_CODE
from nexusml.constants import HTTP_PUT_STATUS_CODE
from nexusml.constants import HTTP_UNAUTHORIZED_STATUS_CODE
from nexusml.constants import HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
from nexusml.database.base import DBModel
from nexusml.database.organizations import Agent
from nexusml.database.organizations import ClientDB
from nexusml.database.organizations import KNOWN_CLIENT_IDS
from nexusml.database.organizations import UserDB
from nexusml.database.services import Service
from nexusml.database.subscriptions import ClientRateLimits
from nexusml.database.subscriptions import get_user_roles_rate_limits
from nexusml.database.subscriptions import UserRateLimits
from nexusml.enums import ResourceAction
from nexusml.enums import ResourceType
from nexusml.enums import ServiceType

__all__ = [
    'register_endpoint',
    'register_endpoints',
    'register_all_endpoints_docs',
    'AUTH0_REQUIRED_ERR_MSG',
    'roles_required',
    'allowed_clients',
    'agent_from_token',
    'client_from_token',
    'error_response',
    'process_get_request',
    'process_post_or_put_request',
    'process_delete_request',
    'get_page_db_objects',
    'get_page_resources',
    'page_url',
    'limiter',
    'rate_limits',
]

#####################
# View registration #
#####################


def register_endpoint(api: Api,
                      method_resource: Type[MethodResource],
                      api_url: str,
                      endpoint_url: str,
                      endpoint_name: Optional[str] = None):
    """
    Registers a given endpoint within a Flask-RESTful API instance. It constructs the endpoint URL
    by concatenating the parent API URL and the endpoint URL. It assigns a default view name based on the method
    resource class name, ensuring it ends with 'view'. Additionally, if the endpoint represents a resource,
    it attempts to register an associated permissions endpoint.

    Args:
        api (Api): API (instance of `flask_restful.Api`) in which the endpoint will be registered.
        method_resource(Type[MethodResource]): The class of the view.
        api_url (str): Parent URL.
        endpoint_url (str): URL of the endpoint.
        endpoint_name (str): Name of the endpoint, optional.

    """
    view_name = method_resource.__name__.lower()
    if not view_name.endswith('view'):
        view_name += 'view'

    endpoint_url = api_url + endpoint_url
    endpoint_name = endpoint_name or view_name

    # Register endpoint
    api.add_resource(method_resource, endpoint_url, endpoint=endpoint_name)

    # Register permissions endpoint
    is_resource = endpoint_url.endswith('>')
    if is_resource:
        try:
            permissions_view = getattr(method_resource, 'permissions_view', None)
            if permissions_view is not None:
                permissions_url = endpoint_url + '/permissions'
                permissions_endpoint_name = endpoint_name + '_permissions'
                api.add_resource(permissions_view, permissions_url, endpoint=permissions_endpoint_name)
        except Exception as e:
            print(f'ERROR: Cannot register "{endpoint_url}". Raised error: {str(e)}')
            print('Exiting')
            sys.exit(1)


def register_endpoints(api_url: str,
                       endpoint_urls: Dict[Type[MethodResource], str],
                       endpoint_names: Dict[Type[MethodResource], str] = None):
    """
    Registers multiple endpoints in a Flask-RESTful API instance. It iterates over the provided endpoint
    URLs and names, creates or reuses the necessary API objects for each blueprint, and registers the endpoints.

    Args:
        api_url (str): Parent URL.
        endpoint_urls (dict): Mapping where the key is the view's class and the value is the endpoint URL.
        endpoint_names (dict): Mapping where the key is the view's class and the value is the endpoint name.

    Raises:
        ValueError: If keys given by `endpoint_names` are not a subset of those given by `endpoint_urls`.

    """

    endpoint_names = endpoint_names or dict()

    # Check arguments
    if not set(endpoint_names.keys()).issubset(set(endpoint_urls.keys())):
        raise ValueError('Keys given by `endpoint_names` must be a subset of those given by `endpoint_urls`')

    # Register endpoints
    apis = dict()  # Contains the `flask_restful.Api` object associated with each blueprint

    for view_class, endpoint_url in endpoint_urls.items():
        # Get view class' blueprint
        module = sys.modules[view_class.__module__]
        blueprint = module.blueprint

        # Create the `flask_restful.Api` object for the blueprint (if necessary)
        if blueprint not in apis:
            apis[blueprint] = Api(blueprint)

        # Register endpoint
        register_endpoint(api=apis[blueprint],
                          method_resource=view_class,
                          api_url=api_url,
                          endpoint_url=endpoint_url,
                          endpoint_name=endpoint_names.get(view_class))


######################
# View documentation #
######################


def register_all_endpoints_docs(app: Flask, docs: FlaskApiSpec, exclude: List[Type[MethodResource]] = None):
    """
    Registers the documentation of all subclasses of `MethodResource` defined in all blueprints registered in the app.

    Args:
        app (Flask): App in which the endpoints were registered.
        docs (FlaskApiSpec): Instance of `FlaskApiSpec` used for documenting the endpoint.
        exclude (list): List of `MethodResource` subclasses to be excluded from the documentation.

    """
    for endpoint, view_class in app.view_functions.items():
        # Check view class
        if not hasattr(view_class, 'view_class'):
            continue
        view_class = view_class.view_class
        if not issubclass(view_class, MethodResource) or view_class in exclude:
            continue
        # Register docs
        docs.register(target=view_class, endpoint=endpoint)


###################
# View protection #
###################

AUTH0_REQUIRED_ERR_MSG = 'Auth0 access token required. API keys not supported in this endpoint'


def validate_payload_size(func):
    """
    Decorator for validating request payload size. Requires an HTTP request context.

    WARNING: this decorator is useful because Flask (Werkzeug actually) doesn't check the payload until the
             corresponding request attribute (`data`, `form`, `files`, `json`, etc.) is accessed.
             See https://github.com/pallets/werkzeug/issues/1513 for more info.

    WARNING: according to https://github.com/pallets/werkzeug/issues/1513, Werkzeug expects the WSGI or the HTTP
             server to be responsible for rejecting too large payloads before loading the data stream sent by the
             client in memory. Werkzeug is expected to act only on safe requests, so the WSGI or the HTTP server
             would have to make a decision about how much to read.
             See https://github.com/pallets/werkzeug/pull/2051#issuecomment-787103215) and
             https://github.com/pallets/werkzeug/issues/1513#issuecomment-764901165
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        max_payload = config.get('limits')['requests']['max_payload']

        # Validate payload
        if request.content_length:
            if request.content_length > max_payload:
                max_payload_mb = round(max_payload / 1024 / 1024)
                return error_response(code=HTTP_PAYLOAD_TOO_LARGE_STATUS_CODE,
                                      message=f'Payload too large (limit: {max_payload} bytes ({max_payload_mb} MB))')
        else:
            if request.method in ['post', 'put']:
                return error_response(code=HTTP_BAD_REQUEST_STATUS_CODE, message='No content provided')

        # Call wrapped function
        return func(*args, **kwargs)

    return wrapper


def validate_token(resource_types: List[Type[Resource]] = None, reject_api_keys: Iterable[str] = None):
    """
    Decorator for validating request bearer token before accessing the endpoint.
    Supports OAuth2 access tokens (for user requests) and API Keys (for client requests).
    Requires an HTTP request context.

    Note: this function injects the following attributes in `Flask.g`:
              - `token` (dict): Contains the decoded token.
              - `token_type` (str): "auth0_token" or "api_key".
              - `agent_uuid` (str): UUID of the user or client identified by `token`.
              - `client_uuid` (str): UUID of the client making the request

    Args:
        resource_types (list): resource types involved in the endpoint. The number and the order of provided types
                               must match that of the resource IDs specified in the URLs.
        reject_api_keys (Iterable): HTTP methods that must reject API keys (i.e. support only Auth0 access tokens)
    """

    def decorator(func):

        def _valid_scopes(token: dict, token_type: str) -> bool:
            assert token_type in ['auth0_token', 'api_key']

            # Map scopes to resource types.
            if not resource_types:
                return True

            resource_type = resource_types[-1]
            parent_type = resource_types[-2] if len(resource_types) > 1 else None

            if (resource_type.permission_resource_type() is None and
                (parent_type is None or parent_type.permission_resource_type() is None)):
                return True

            scope_resource_types = {
                'organizations': ResourceType.ORGANIZATION,
                'tasks': ResourceType.TASK,
                'files': ResourceType.FILE,
                'models': ResourceType.AI_MODEL,
                'examples': ResourceType.EXAMPLE,
                'predictions': ResourceType.PREDICTION
            }

            assert all(rt in scope_resource_types.values() for rt in ResourceType)

            # Map HTTP methods to actions
            method_actions = {
                'POST': ResourceAction.CREATE,
                'GET': ResourceAction.READ,
                'PUT': ResourceAction.UPDATE,
                'DELETE': ResourceAction.DELETE
            }

            assert all(a in method_actions.values() for a in ResourceAction)

            action = method_actions[request.method]

            # Get the actions allowed by the token for the given resource type and the direct parent
            token_scopes = token.get('scope', '').split()

            token_actions = dict()

            if parent_type is not None:
                involved_resource_types = (resource_type.permission_resource_type(),
                                           parent_type.permission_resource_type())
            else:
                involved_resource_types = (resource_type.permission_resource_type(),)

            for rsrc_type in involved_resource_types:
                token_actions[rsrc_type] = []

            for scope in token_scopes:
                try:
                    scope_parts = scope.split('.')  # scope = resource_type.action
                    scope_resource_type = scope_resource_types.get(scope_parts[0])
                    scope_action = scope_parts[1]
                    if scope_resource_type not in involved_resource_types:
                        continue
                    if scope_resource_type not in token_actions:
                        token_actions[scope_resource_type] = []
                    token_actions[scope_resource_type].append(ResourceAction[scope_action.upper()])
                except Exception:
                    continue

            # Check if the token doesn't allow any action on the resource
            if not token_actions:
                return False

            # Permissions on parent resource
            parent_required_permission = parent_type.permission_resource_type() if parent_type is not None else None

            if parent_required_permission is None:
                allowed_parent = True
            else:
                if action == ResourceAction.READ or not resource_type.touch_parent():
                    allowed_parent = ResourceAction.READ in token_actions[parent_required_permission]
                else:
                    allowed_parent = ResourceAction.UPDATE in token_actions[parent_required_permission]

            # Permissions on parent collection, current resource, or child collection
            current_required_permission = resource_type.permission_resource_type()

            if current_required_permission is None:
                allowed_current = True
            else:
                url_ = str(request.url_rule).split('/')
                last_url_item = url_[-1]
                is_resource = last_url_item.startswith('<') and last_url_item.endswith('>')
                num_resources = len([x for x in url_ if x.startswith('<') and x.endswith('>')])
                num_resource_types = len(resource_types)
                assert num_resources in [num_resource_types - 1, num_resource_types]
                if num_resources < num_resource_types:
                    # Parent collection (e.g., `POST /tasks`)
                    allowed_current = action in token_actions[current_required_permission]
                elif is_resource:
                    # Current resource (e.g., `PUT /tasks/<task_id>`)
                    allowed_current = action in token_actions[current_required_permission]
                else:
                    # Child collection (e.g., `POST /tasks/<task_id>/usage`)
                    allowed_current = ResourceAction.READ in token_actions[current_required_permission]

            return allowed_current and allowed_parent

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if a token is provided
            token_provided = 'Authorization' in request.headers and 'Bearer ' in request.headers['Authorization']
            if token_provided:
                # Get the token
                try:
                    enc_token = request.headers['Authorization'].replace('Bearer ', '')
                    # Try to decode an API key
                    try:
                        token = decode_api_key(api_key=enc_token)
                        token_type = 'api_key'
                    except (jwt.InvalidSignatureError, jwt.InvalidTokenError, ValueError):
                        # If it's not an API key, try to decode an Auth0 token
                        token = decode_auth0_token(auth0_token=enc_token)
                        token_type = 'auth0_token'
                except (jwt.InvalidSignatureError, jwt.PyJWKClientError, KeyError):
                    # Note: `KeyError` may be raised if Auth0 env variables are not set
                    return error_response(code=HTTP_BAD_REQUEST_STATUS_CODE, message='Invalid token')
                except Exception:
                    return error_response(code=HTTP_BAD_REQUEST_STATUS_CODE, message='Invalid token')
            else:
                # If no token is provided, check if authentication is enabled
                if config.get('general')['auth_enabled']:
                    return error_response(code=HTTP_UNAUTHORIZED_STATUS_CODE, message='Unauthorized')
                else:
                    # If authentication is disabled, use the default client's API key
                    default_client = ClientDB.get(client_id=KNOWN_CLIENT_IDS['default'])
                    token = decode_api_key(api_key=default_client.api_key)
                    token_type = 'api_key'

            # Check whether API keys are supported in this endpoint
            reject_api_keys_ = [x.upper() for x in reject_api_keys] if reject_api_keys else []
            if token_type == 'api_key' and request.method in reject_api_keys_:
                return error_response(code=HTTP_FORBIDDEN_STATUS_CODE, message=AUTH0_REQUIRED_ERR_MSG)

            # Check required token scopes
            if not _valid_scopes(token=token, token_type=token_type):
                return error_response(code=HTTP_FORBIDDEN_STATUS_CODE, message='Invalid scopes')

            # Inject token and agent
            g.token = token
            g.token_type = token_type

            if token_type == 'auth0_token':
                g.user_auth0_id = g.token['sub']
                g.client_auth0_id = g.token['azp']
                g.client_uuid = None
                # Set agent
                user_db_obj = UserDB.query().filter_by(auth0_id=g.token['sub']).first()
                if user_db_obj is None:
                    g.agent_uuid = None
                else:
                    g.agent_uuid = user_db_obj.uuid
            else:
                g.client_uuid = g.token['aud']
                g.client_auth0_id = None
                g.agent_uuid = g.token['aud']

            # Call wrapped function
            return func(*args, **kwargs)

        return wrapper

    return decorator


def roles_required(roles: Iterable[str], require_all: bool = True):
    """
    Decorator for requiring the specified roles for accessing the endpoint. Requires an HTTP request context.

    Args:
        roles (list): Names of the roles required for accessing the endpoint.
        require_all (bool): Require all the specified roles.

    Raises:
        PermissionDeniedError: If the user lacks the necessary roles.

    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            session_user = agent_from_token()
            # If provided token is an API key, bypass role check (always allow access)
            if not isinstance(session_user, UserDB):
                return func(*args, **kwargs)
            # Check whether the user belongs to the same organization as that of the resource being accessed.
            # Note: resources not belonging to an organization cannot be accessed through the API.
            organization = None
            resources = kwargs.get('resources')
            if resources:
                organization = get_resource_organization(resource=resources[0])
            elif 'organization_id' in kwargs:
                organization = Organization.get(agent=session_user, db_object_or_id=kwargs['organization_id'])
            if organization is None or session_user.organization_id != organization.db_object().organization_id:
                raise PermissionDeniedError()
            # Check user roles
            user_roles = get_user_roles(user=session_user)
            if require_all:
                if not set(roles).issubset(set(user_roles)):
                    raise PermissionDeniedError()
            else:
                if not any(required_role in user_roles for required_role in roles):
                    raise PermissionDeniedError()
            # Call wrapped function
            return func(*args, **kwargs)

        return wrapper

    return decorator


def allowed_clients(client_ids: Iterable[str] = None,
                    service_types: Iterable[ServiceType] = None,
                    error_code: int = HTTP_FORBIDDEN_STATUS_CODE):
    """
    Decorator for rejecting requests made by clients different from the specified ones.
    Requires an HTTP request context. If the client is not allowed, an error response is returned.

    Args:
        client_ids (Iterable): Auth0 ID or UUID of allowed clients
        service_types (Iterable): Allowed service types (only for task-related endpoints).
                                  If specified, a `resources` argument is expected to be passed to the wrapped function
        error_code (bool): HTTP status code returned when rejecting a request
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            resources = kwargs.get('resources')
            if not resources:
                return error_response(code=error_code, message='')

            allowed_uuid = False
            allowed_service = False

            # Check provided IDs
            if client_ids:
                allowed_uuid = (g.client_uuid in client_ids) or (g.client_auth0_id in client_ids)

            # Check service clients
            if service_types:
                task = resources[-1]
                if isinstance(task, Task):
                    task_services = Service.filter_by_task(task_id=task.db_object().task_id)
                    service_client_uuids = [x.client.uuid for x in task_services]
                    allowed_service = g.client_uuid in service_client_uuids

            # Call wrapped function if the client is allowed
            if not (allowed_uuid or allowed_service):
                return error_response(code=error_code, message='')
            return func(*args, **kwargs)

        return wrapper

    return decorator


###########
# Session #
###########


@cache.memoize()
def _user_from_token(user_uuid: str) -> Optional[UserDB]:
    """
    Retrieves a user from the database using their UUID and ensures that all necessary
    relationships are loaded by forcing relationship loading. If no user is found, the
    function will return None.

    Args:
        user_uuid (str): The UUID of the user to retrieve from the database.

    Returns:
        Optional[UserDB]: The user object if found, otherwise None.
    """
    user = UserDB.get_from_uuid(user_uuid)
    if user is None:
        return
    user.force_relationship_loading()
    return user


@cache.memoize()
def _client_from_token(client_uuid: str) -> Optional[ClientDB]:
    """
    Retrieves a client from the database using their UUID and ensures that all necessary
    relationships are loaded by forcing relationship loading. If no client is found, the
    function will return None.

    Args:
        client_uuid (str): The UUID of the client to retrieve from the database.

    Returns:
        Optional[ClientDB]: The client object if found, otherwise None.
    """
    client = ClientDB.get_from_uuid(client_uuid)
    if client is None:
        return
    client.force_relationship_loading()
    return client


def agent_from_token() -> Agent:
    """
    Returns the user identified by the Auth0 token or the client identified by the API key.

    Returns:
        Agent: User (if the token is an Auth0 token) or Client (if the token is an API key)
    
    Raises:
        jwt.InvalidTokenError: If the agent is not registered.
    """

    assert g.token_type in ['auth0_token', 'api_key']

    if g.token_type == 'auth0_token':
        user = _user_from_token(user_uuid=g.agent_uuid)
        if user is None:
            raise jwt.InvalidTokenError(f'User not registered in {API_NAME}')
        return user
    else:
        client = _client_from_token(client_uuid=g.agent_uuid)
        if client is None:
            raise jwt.InvalidTokenError(f'Client not registered in {API_NAME}')
        return client


def client_from_token() -> ClientDB:
    """
    Returns the client identified by the API key.

    Returns:
        ClientDB: Client associated with the API key.
    """
    return _client_from_token(client_uuid=g.client_uuid)


##########################
# Requests and Responses #
##########################


def error_response(code: int, message: str) -> Response:
    """
    Constructs an error response with the given code and message.

    Args:
        code (int): HTTP status code.
        message (str): Error message.

    Returns:
        Response: The response with the error code and message.
    """
    response_dict = {'error': {'code': code, 'message': message}}
    response = jsonify(response_dict)
    response.status_code = code
    return response


def capture_request_errors(func):
    """
    Decorator that wraps a function to capture and handle common request-related errors.
    It returns appropriate HTTP error responses based on the type of the exception raised.

    Args:
        func (Callable): The function to wrap.

    Returns:
        Callable: The wrapped function which returns an error response if an exception is raised.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except jwt.InvalidTokenError as e:
            return error_response(code=HTTP_UNAUTHORIZED_STATUS_CODE, message=str(e))
        except PermissionDeniedError as e:
            return error_response(code=HTTP_FORBIDDEN_STATUS_CODE, message=str(e))
        except InvalidDataError as e:
            return error_response(code=HTTP_BAD_REQUEST_STATUS_CODE, message=str(e))
        except UnprocessableRequestError as e:
            return error_response(code=HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE, message=str(e))
        except ResourceNotFoundError as e:
            # TODO: Returning a "422 Unprocessable Entity" may be more appropriate if the resource is referenced
            #       within the request payload. That is, the URL is found on the server, but the referenced
            #       resource does not exist.
            return error_response(code=HTTP_NOT_FOUND_STATUS_CODE, message=str(e))
        except DuplicateResourceError as e:
            return error_response(code=HTTP_CONFLICT_STATUS_CODE, message=str(e))
        except ImmutableResourceError as e:
            return error_response(code=HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE, message=str(e))
        except ResourceOutOfSyncError as e:
            return error_response(code=HTTP_CONFLICT_STATUS_CODE, message=str(e))
        except ResourceError as e:
            return error_response(code=HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE, message=str(e))
        except QuotaError as e:
            # TODO: Consider using "403 Forbidden" rather than "422 Unprocessable Entity"
            return error_response(code=HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE, message=str(e))

    return wrapper


def capture_schema_errors(func):
    """
    Decorator that captures and handles schema-related errors, specifically those
    arising from schema validation frameworks like Marshmallow. It raises
    `InvalidDataError` with detailed schema validation error messages if applicable.

    Args:
        func (Callable): The function to wrap.

    Returns:
        Callable: The wrapped function which raises schema-specific exceptions if validation fails.

    Raises:
        BadRequest: If the incoming request does not conform to the expected schema.
        InvalidDataError: If schema validation errors are found.
        Exception: If a non-schema-related error occurs, the original exception is re-raised.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BadRequest as e:
            # Get errors in Marshmallow schemas
            try:
                errors = e.data['errors']['json']['_schema']
            except (AttributeError, KeyError):
                raise e
            assert isinstance(errors, list)
            raise InvalidDataError('\n'.join(errors))
        except Exception as e:
            # If the origin of the exception is a subclass of the base Marshmallow schema, return an error response
            frame = inspect.trace()[-1].frame
            if isinstance(frame.f_locals.get('self'), Schema):
                raise InvalidDataError()
            # Otherwise, just re-raise the exception
            raise e

    return wrapper


def load_url_resources(agent: Agent, resource_ids: List[str], resource_types: List[Type[Resource]]) -> List[Resource]:
    """
    Loads the resources specified in the URL.

    Args:
        agent (Agent): user or client making the request
        resource_ids (list): IDs of the resources specified in the URL
        resource_types (list): types of the resources specified in the URL

    Returns:
        list: resources specified in the URL
    """
    parents = [(model, id_) for model, id_ in zip(resource_types[:-1], resource_ids[:-1])]
    resource = resource_types[-1].get(agent=agent, db_object_or_id=resource_ids[-1], parents=parents)
    return resource.parents() + [resource]


def process_get_request(resource: Resource, dump_args: dict = None) -> Response:
    """
    Handles a GET request for a specified resource, converting it to JSON and setting the appropriate headers.

    Args:
        resource (Resource): Requested resource.
        dump_args (dict): Arguments to be passed to `Resource.dump()`.

    Returns:
        Response: The response with the resource data in JSON format.
    """
    response = jsonify(resource.dump(**(dump_args or dict())))
    response.headers['Location'] = resource.url()
    return response


def process_post_or_put_request(agent: Agent,
                                resource_or_model: Union[Resource, Type[Resource]],
                                json: dict,
                                parents: List[Resource] = None,
                                dump_resource=True) -> Response:
    """
    Handles a POST or PUT request.

    Args:
        agent (Agent): User or client making the request.
        resource_or_model (Union[Resource, type]): Resource (for PUT) or resource model (for POST).
        json (dict): JSON payload.
        parents (list): Parent resources (in order of appearance in the URL). Only for POST.
        dump_resource (bool): Include resource in the response.

    Returns:
        Response: The response with the resource data if specified, or a status code.

    Raises:
        ResourceNotFoundError: If the resource or model is not found.
    """
    if resource_or_model is None:
        raise ResourceNotFoundError()

    resource = resource_or_model if isinstance(resource_or_model, Resource) else None
    resource_type = resource_or_model if resource is None else type(resource)
    parents = parents if resource is None else resource.parents()

    creation_mode = resource is None

    # Get the users that must be notified (currently every user having access to the task)
    # TODO: move notifications' business logic to models
    # TODO: get user preferences to see if the user has notifications enabled
    notify_to = None
    if isinstance(resource, Task):
        notify_to = resource.users_with_access()
    elif parents:
        for parent in parents:
            if isinstance(parent, Task):
                notify_to = parent.users_with_access()
                break

    # Create or update the resource
    if creation_mode:
        resource = resource_type.post(agent=agent, data=json, parents=parents, notify_to=notify_to)
    else:
        resource.put(data=json, notify_to=notify_to)

    # Include the resource in the response (if specified)
    if dump_resource:
        response = jsonify(resource.dump())
        response.status_code = HTTP_POST_STATUS_CODE if creation_mode else HTTP_PUT_STATUS_CODE

    else:
        response = Response(status=204)

    # Include the location of the resource
    response.headers['Location'] = resource.url()

    return response


def process_delete_request(resource: Resource) -> Response:
    """
    Handles a DELETE request.

    Args:
        resource (Resource): Requested resource.

    Returns:
        Response: The response with the status code indicating the result of the deletion.
    """
    # Get the users that must be notified (currently every user having access to the task)
    # TODO: move notifications' business logic to models
    # TODO: get user preferences to see if the user has notifications enabled
    notify_to = None
    for parent in resource.parents():
        if isinstance(parent, Task):
            notify_to = parent.users_with_access()
            break
    # Process the request
    resource.delete(notify_to=notify_to)
    return Response(status=HTTP_DELETE_STATUS_CODE)


def get_page_db_objects(query: Query,
                        page_number: int,
                        per_page: int,
                        order_by: Union[InstrumentedAttribute, UnaryExpression] = None,
                        total_count: bool = False) -> dict:
    """
    Returns the specified page of the collection.

    WARNING: This function requires an HTTP request context.

    Args:
        query (Query): Base query to be paginated. It must return all the items of the collection.
        page_number (int): Page number.
        per_page (int): Number of items per page.
        order_by: Column or expression generated by `desc()` or `asc()` in `sqlalchemy.sql.operators.ColumnOperators`.
        total_count (bool): Get the total number of items in the collection (including all pages).

    Returns:
        dict: Dictionary with the following keys:
            - "data": Items (database objects) in the requested page.
            - "links":
                - "previous": Previous page URL.
                - "current": Current page URL.
                - "next": Next page URL.
            - "total_count": Total number of items in the collection (including all pages). This field will be present
               only if `total_count=True`.

    Raises:
        ResourceNotFoundError: If the requested page doesn't exist.
    """
    # Get page
    if order_by is not None:
        query = query.order_by(order_by)

    page = query.paginate(page=page_number, per_page=per_page)
    if isinstance(page, Response):
        raise ResourceNotFoundError(page.text)  # 404 error response: requested page doesn't exist

    # Get previous/current/next pages' URL
    previous_page_url = page_url(page_number=page.prev_num, current_page_number=page_number) if page.has_prev else None
    current_page_url = page_url(page_number=page_number, current_page_number=page_number)
    next_page_url = page_url(page_number=page.next_num, current_page_number=page_number) if page.has_next else None

    return {
        'data': page,
        'links': {
            'previous': previous_page_url,
            'current': current_page_url,
            'next': next_page_url
        },
        **({
            'total_count': page.total
        } if total_count else {})
    }


def get_page_resources(query: Query,
                       page_number: int,
                       per_page: int,
                       order_by: Union[InstrumentedAttribute, UnaryExpression] = None,
                       total_count: bool = False,
                       resource_type: Type[Resource] = None,
                       parents: List[Resource] = None,
                       ignore_forbidden: bool = True,
                       dump_args: dict = None) -> dict:
    """
    Returns the specified page of the collection.

    WARNING: This function requires an HTTP request context.

    Args:
        query (Query): Base query to be paginated. It must return instances of `DBModel`.
        page_number (int): Page number.
        per_page (int): Number of items per page.
        order_by: Column or expression generated by `desc()` or `asc()` in `sqlalchemy.sql.operators.ColumnOperators`.
        total_count (bool): Get the total number of items in the collection (including all pages).
        resource_type (type): `Resource` subclass. All page objects will be converted into instances of this class.
                              If provided, `query` must return instances of `resource_type.db_model()`.
        parents (list): `Resource` objects representing parent resources (only if `resource_type` is given).
        ignore_forbidden (bool): Ignore forbidden resources (only if `resource_type` is given). If `True`, resources
                                 the user has no permissions to access will be omitted in the response. Otherwise, a
                                 `PermissionDeniedError` will be raised if the user has no permissions to access any
                                 of the resources.
        dump_args (dict): Arguments to be passed to `Resource.dump()` (only used if `resource_type` is provided).

    Returns:
        dict: `PageSchema` ready to be included in the response JSON.
            - "data": Items (resources) in the requested page.
            - "links":
                - "previous": Previous page URL.
                - "current": Current page URL.
                - "next": Next page URL.
            - "total_count": Total number of items in the collection (including all pages). This field will be present
               only if `total_count=True`.

    Raises:
        PermissionDeniedError: If the user has no permissions to access any of the resources
                               and `ignore_forbidden` is False.
    """
    # Get the database objects in the requested page
    page_db_objects = get_page_db_objects(query=query,
                                          page_number=page_number,
                                          per_page=per_page,
                                          order_by=order_by,
                                          total_count=total_count)

    # Convert database objects into resources
    page_resources = dict(page_db_objects)
    page_resources['data'] = []
    if resource_type is not None:
        for db_object in page_db_objects['data'].items:
            try:
                resource = resource_type.get(agent=agent_from_token(),
                                             db_object_or_id=db_object,
                                             parents=parents,
                                             check_parents=False)
                page_resources['data'].append(resource.dump(**(dump_args or dict())))
            except PermissionDeniedError as e:
                if not ignore_forbidden:
                    raise e
    else:
        for db_object in page_db_objects['data'].items:
            assert isinstance(db_object, DBModel)
            page_resources['data'].append(db_object.to_dict())

    # Dump paginated result schema
    return PageSchema().dump(page_resources)


def page_url(page_number: int, current_page_number: int) -> str:
    """
    Constructs the URL for the specified page.

    WARNING: This function requires an HTTP request context.

    Args:
        page_number (int): Target page number.
        current_page_number (int): Current page number.

    Returns:
        str: URL for the target page.
    """
    if '?page=' in request.url:
        page_url_ = request.url.replace(f'?page={current_page_number}', f'?page={page_number}')
    elif '&page=' in request.url:
        page_url_ = request.url.replace(f'&page={current_page_number}', f'&page={page_number}')
    else:
        if '?' in request.url:
            base_url = request.url[:request.url.index('?')]
            query_str = request.url[request.url.index('?') + 1:]
            page_url_ = base_url + f'?page={page_number}' + '&' + query_str
        else:
            page_url_ = request.url + f'?page={page_number}'

    page_url_ = re.sub('total_count=(T|t)rue(&)?', '', page_url_)

    return page_url_ if not page_url_.endswith(('?', '&')) else page_url_[:-1]


#############
# API rates #
#############

limiter = Limiter(key_func=lambda: getattr(g, 'agent_uuid', None))


def rate_limits() -> str:
    """
    Retrieves the rate limits for the current agent.

    Returns:
        str: Rate limits in the format 'day_limit/hour_limit/minute_limit/second_limit'.
    """

    @cache.memoize()
    def get_agent_limits(agent_uuid: str) -> Tuple[int, int, int, int]:
        if g.token_type == 'auth0_token':
            user = UserDB.get_from_uuid(agent_uuid)
            try:
                agent_limits = UserRateLimits.get(user_id=user.user_id)
            except Exception:
                agent_limits = get_user_roles_rate_limits(user_id=user.user_id)
        else:
            client = ClientDB.get_from_uuid(agent_uuid)
            agent_limits = ClientRateLimits.get(client_id=client.client_id)
        day_limit = agent_limits.requests_per_day
        hour_limit = agent_limits.requests_per_hour
        minute_limit = agent_limits.requests_per_minute
        second_limit = agent_limits.requests_per_second
        return day_limit, hour_limit, minute_limit, second_limit

    # WARNING: don't remove `get_agent_limits()` function as the cache takes into account passed arguments
    try:
        agent_limits = get_agent_limits(agent_uuid=g.agent_uuid)
    except Exception:
        _default_limits = config.get('limits')['requests']

        day_limit = _default_limits['requests_per_day']
        hour_limit = _default_limits['requests_per_hour']
        minute_limit = _default_limits['requests_per_minute']
        second_limit = _default_limits['requests_per_second']

        agent_limits = day_limit, hour_limit, minute_limit, second_limit

    return f'{agent_limits[0]}/day;{agent_limits[1]}/hour;{agent_limits[2]}/minute;{agent_limits[3]}/second'
