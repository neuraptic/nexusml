from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
import contextlib
import copy
from datetime import datetime
from enum import Enum
import functools
import random
import time
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
import uuid

from flask import Response
from flask_apispec import MethodResource
import requests
from sqlalchemy import and_ as sql_and
from sqlalchemy import bindparam

from nexusml.api.endpoints import ENDPOINT_AI_PREDICTION_LOG
from nexusml.api.endpoints import ENDPOINT_EXAMPLE
from nexusml.api.ext import cache
from nexusml.api.resources.ai import PredictionLog
from nexusml.api.resources.base import Permission
from nexusml.api.resources.base import Resource
from nexusml.api.resources.examples import Example
from nexusml.api.resources.tasks import Task
from nexusml.api.schemas.ai import PredictionLoggingRequest
from nexusml.api.utils import API_DOMAIN
from nexusml.api.utils import config
from nexusml.api.utils import decode_auth0_token
from nexusml.constants import DATETIME_FORMAT
from nexusml.constants import HTTP_BAD_REQUEST_STATUS_CODE
from nexusml.constants import HTTP_CONFLICT_STATUS_CODE
from nexusml.constants import HTTP_DELETE_STATUS_CODE
from nexusml.constants import HTTP_FORBIDDEN_STATUS_CODE
from nexusml.constants import HTTP_GET_STATUS_CODE
from nexusml.constants import HTTP_METHOD_NOT_ALLOWED_STATUS_CODE
from nexusml.constants import HTTP_POST_STATUS_CODE
from nexusml.constants import HTTP_PUT_STATUS_CODE
from nexusml.constants import HTTP_UNAUTHORIZED_STATUS_CODE
from nexusml.constants import HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
from nexusml.constants import NULL_UUID
from nexusml.constants import UUID_VERSION
from nexusml.database.ai import AIModelDB
from nexusml.database.ai import PredCategory
from nexusml.database.ai import PredFile
from nexusml.database.ai import PredictionDB
from nexusml.database.ai import PredValue
from nexusml.database.core import db_commit
from nexusml.database.core import db_execute
from nexusml.database.core import db_rollback
from nexusml.database.core import empty_table
from nexusml.database.core import save_to_db
from nexusml.database.examples import ExampleDB
from nexusml.database.examples import ExCategory
from nexusml.database.examples import ExFile
from nexusml.database.examples import ExValue
from nexusml.database.files import OrgFileDB
from nexusml.database.files import TaskFileDB
from nexusml.database.organizations import MutableEntity
from nexusml.database.organizations import OrganizationDB
from nexusml.database.organizations import RoleDB
from nexusml.database.organizations import user_roles as user_roles_table
from nexusml.database.organizations import UserDB
from nexusml.database.permissions import RolePermission
from nexusml.database.permissions import UserPermission
from nexusml.database.services import Service
from nexusml.database.tasks import CategoryDB
from nexusml.database.tasks import ElementDB
from nexusml.database.tasks import TaskDB
from nexusml.database.utils import get_children
from nexusml.enums import AIEnvironment
from nexusml.enums import ElementMultiValue
from nexusml.enums import ElementType
from nexusml.enums import ElementValueType
from nexusml.enums import FileStorageBackend
from nexusml.enums import OrgFileUse
from nexusml.enums import PredictionState
from nexusml.enums import ResourceAction
from nexusml.enums import ResourceType
from nexusml.enums import ServiceType
from nexusml.enums import TaskFileUse
from tests.api.conftest import restore_db
from tests.api.constants import BACKEND_JSON_FIELDS
from tests.api.constants import CLIENT_MAX_THREADS
from tests.api.integration.conftest import MockClient
from tests.api.utils import assert_same_element_values
from tests.api.utils import db_commit_and_expire
from tests.api.utils import get_files_in_s3
from tests.api.utils import get_json_from_db_object
from tests.api.utils import get_shape_or_slice_json_from_db_object
from tests.api.utils import load_default_resource
from tests.api.utils import set_quota_usage
from tests.api.utils import verify_quota_usage


def send_request(method: str, url: str, api_key: str, json: Union[dict, List[dict]] = None) -> requests.Response:
    kwargs = {'headers': {'Authorization': f'Bearer {api_key}'}}
    if json:
        kwargs['json'] = json
    method_call = getattr(requests, method.strip().lower())
    return method_call(url, **kwargs)


def get_endpoint(parameterized_endpoint: str,
                 resource: Resource = None,
                 use_uuids: bool = False,
                 absolute_url: bool = True) -> str:
    api_url = config.get('server')['api_url']

    endpoint = parameterized_endpoint

    if resource is not None:
        params = [x for x in endpoint.split('/') if x.startswith('<') and x.endswith('>')]
        assert len(params) == len(resource.parents()) + 1
        for param, rsrc in zip(params, resource.parents() + [resource]):
            endpoint = endpoint.replace(param, rsrc.uuid() if use_uuids else rsrc.public_id())

    return (API_DOMAIN + api_url + endpoint) if absolute_url else endpoint


def verify_endpoints_protection(client: MockClient,
                                endpoint: str,
                                view: MethodResource,
                                mock_client_id: str,
                                session_user_id: str,
                                session_user_auth0_id: str):
    """ Class decorator for verifying all endpoints' access protection before making the corresponding requests. """

    # TODO: this decorator makes tests extremely slow and doesn't scale at all

    DECORATED_METHODS = ['test_get', 'test_post', 'test_put', 'test_delete']

    def _clear_permission_and_role_assignments():
        empty_table(UserPermission)
        empty_table(RolePermission)
        db_execute(user_roles_table.delete())
        db_commit()

    def _verify_user_permission(user_id: int,
                                method: str,
                                endpoint_url: str,
                                permission: Permission = None,
                                inherited_from_role: bool = False,
                                payload: dict = None):
        # Clear all previous permission/role assignments
        _clear_permission_and_role_assignments()

        # Assign the permission to the user directly
        # or assign "dasci" role to the user and assign the permission to the role
        # (the user inherits the role's permissions)
        user = UserDB.get(user_id=user_id)
        organization = OrganizationDB.get(organization_id=user.organization_id)
        if not (permission is None or inherited_from_role):
            user_perm = UserPermission(user_id=user_id,
                                       organization_id=organization.organization_id,
                                       **permission._asdict())
            save_to_db(user_perm)
        elif permission is not None:
            role = RoleDB.get_from_id(id_value='dasci', parent=organization)
            # Assign permission to role
            role_perm = RolePermission(role_id=role.role_id, **permission._asdict())
            save_to_db(role_perm)
            # Assign role to user
            stmt = user_roles_table.insert().values(user_id=bindparam('user_id'), role_id=bindparam('role_id'))
            db_execute(stmt, {'user_id': user_id, 'role_id': role.role_id})
            db_commit()

        # Make request
        if payload or method not in ['post', 'put']:  # ignore `POST` and `PUT` if no payload was given
            response = client.send_request(method, endpoint_url, json=payload)
            # Note: permissions must explicitly be allowed to grant access
            if permission is not None and permission.allow:
                assert str(response.status_code).startswith('2')
            else:
                no_content = not (response.content and response.json())
                assert response.status_code == HTTP_FORBIDDEN_STATUS_CODE or no_content
        # TODO: don't ignore `POST` and `PUT` even if no payload is given. We are now ignoring them because
        #       requests' payload verification is being performed before token/permissions verification, returning bad
        #       request errors (code 400) instead of unauthorized/forbidden errors (codes 401 and 403). To fix this,
        #       we must ensure that `views.core.View.decorators` run before
        #       `flask_apispec.annotations.marshal_with` decorator in all endpoints.

        # Clear permission/role assignments again
        _clear_permission_and_role_assignments()

    def _verify_permissions(method: str, endpoint_url: str, payload: Optional[dict], resource_type: ResourceType,
                            action: ResourceAction, target_resource: Optional[Resource]):

        empty_table(UserPermission)
        empty_table(RolePermission)

        resource_uuid = target_resource.uuid() if target_resource is not None else NULL_UUID

        allow_perm = Permission(resource_uuid=resource_uuid, resource_type=resource_type, action=action, allow=True)
        deny_perm = Permission(resource_uuid=resource_uuid, resource_type=resource_type, action=action, allow=False)

        # User permissions
        _verify_user_permission(user_id=1,
                                permission=allow_perm,
                                method=method,
                                endpoint_url=endpoint_url,
                                payload=payload)
        _verify_user_permission(user_id=1,
                                permission=deny_perm,
                                method=method,
                                endpoint_url=endpoint_url,
                                payload=payload)

        # Role permissions
        _verify_user_permission(user_id=1,
                                permission=allow_perm,
                                inherited_from_role=True,
                                method=method,
                                endpoint_url=endpoint_url,
                                payload=payload)
        _verify_user_permission(user_id=1,
                                permission=deny_perm,
                                inherited_from_role=True,
                                method=method,
                                endpoint_url=endpoint_url,
                                payload=payload)

        # No permissions
        _verify_user_permission(user_id=1, permission=None, method=method, endpoint_url=endpoint_url, payload=payload)

        # TODO: actually, testing an explicitly denied permission requires that the same action has previously been
        #       allowed for the same resource type (or a resource of the same type), either for the user or
        #       the user's role, since permissions must explicitly be allowed to grant access.

    def _verify_endpoint_protection(method: str, url: str, payload: Optional[dict], resource: Optional[Resource]):
        method = method.strip().lower()

        method_requests = {'get': requests.get, 'post': requests.post, 'put': requests.put, 'delete': requests.delete}
        method_request = method_requests[method]

        method_actions = {
            'get': ResourceAction.READ,
            'post': ResourceAction.CREATE,
            'put': ResourceAction.UPDATE,
            'delete': ResourceAction.DELETE
        }
        method_action = method_actions[method]
        #####################
        # Test access token #
        #####################

        # Don't provide any token
        response = method_request(url)
        if response.status_code == HTTP_METHOD_NOT_ALLOWED_STATUS_CODE:
            return
        assert response.status_code == HTTP_UNAUTHORIZED_STATUS_CODE

        # Provide invalid tokens
        response = method_request(url, headers={'Authorization': f'Bearer {client.token[1:]}'})
        assert response.status_code in [HTTP_BAD_REQUEST_STATUS_CODE, HTTP_UNAUTHORIZED_STATUS_CODE]

        response = method_request(url, headers={'Authorization': 'Bearer '})
        assert response.status_code in [HTTP_BAD_REQUEST_STATUS_CODE, HTTP_UNAUTHORIZED_STATUS_CODE]
        ####################
        # Test permissions #
        ####################

        permission_resource_type = view.resource_types[-1].permission_resource_type()

        for action in method_action:
            # Permissions to perform the action in the corresponding resource or resource type
            if permission_resource_type is not None:
                _verify_permissions(method=method,
                                    endpoint_url=url,
                                    payload=payload,
                                    resource_type=permission_resource_type,
                                    action=action,
                                    target_resource=resource)
            # Permissions to perform the action in any resource type (only for generic permissions)
            # elif resource is None:
            #     _verify_permissions(method=method,
            #                         endpoint_url=url,
            #                         payload=payload,
            #                         resource_type=ResourceType.ALL,  # wildcard permissions don't exist anymore
            #                         action=action,
            #                         target_resource=None)

    def _method_decorator(http_method: str):

        def decorator(func):

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Restore database
                restore_db(mock_client_id=mock_client_id,
                           session_user_id=session_user_id,
                           session_user_auth0_id=session_user_auth0_id)
                if endpoint.endswith('>'):
                    resource = load_default_resource(resource_type=view.resource_types[-1],
                                                     parents_types=view.resource_types[:-1])
                else:
                    resource = None
                # Verify endpoint protection
                _verify_endpoint_protection(method=http_method,
                                            url=get_endpoint(parameterized_endpoint=endpoint, resource=resource),
                                            payload=kwargs.get('json'),
                                            resource=resource)
                # Restore database
                restore_db(mock_client_id=mock_client_id,
                           session_user_id=session_user_id,
                           session_user_auth0_id=session_user_auth0_id)
                # Run wrapped function
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def _class_decorator(cls):
        for method_name, method in cls.__dict__.items():
            if not (method_name in DECORATED_METHODS and isinstance(method, Callable)):
                continue
            http_method = method_name.replace('test_', '')
            decorated_method = _method_decorator(http_method=http_method)(method)
            setattr(cls, method_name, decorated_method)
        return cls

    return _class_decorator


def verify_resource_request(client: MockClient,
                            method: str,
                            endpoint: str,
                            resource: Resource,
                            collection: Tuple[str, Type[Resource]] = None,
                            url_query: str = None,
                            request_json: dict = None,
                            expected_jsons: Iterable[dict] = None,
                            expected_status_code: int = None,
                            check_resource_location: bool = True) -> Response:
    """
    Performs all generic verifications at two levels:
        - Response's JSON(s):
            - Identifiers
            - Resource location
            - Fields
            - Values
            - Last activity/modification datetimes
        - Database:
            - Entry operations (creations/modifications/deletions)

    Args:
        client (MockClient): client to make the request with
        method (str): "GET", "POST", "PUT" or "DELETE"
        endpoint (str): URL of `resource` (don't include the URL of the API)
        resource (Resource): resource to which the request refers
        collection (tuple): a collection belonging to `resource`, given by a tuple of
                            (collection name, type of the resources contained in the collection).
                            Required if `method` is "POST"
        url_query (str): URL query parameters
        request_json (dict): JSON to be included in the request payload. Required if `method` is "POST" or "PUT"
        expected_jsons (list): expected list of JSONs in the response (only if `method` is not "DELETE")
        expected_status_code (int): expected HTTP status code.
                                    Useful for those cases where the request is intentionally invalid
        check_resource_location (bool): check the resource location provided in the response

    Returns:
        Response: The response
    """

    # Sanity check
    method = method.strip().lower()
    success_expected = expected_status_code is None or str(expected_status_code).startswith('2')

    if method == 'delete':
        assert expected_jsons is None
    elif success_expected:
        assert expected_jsons is not None

    if method in ['post', 'put']:
        assert request_json is not None
        if success_expected:
            assert len(expected_jsons) == 1
        if method == 'post':
            assert collection is not None

    if collection and not isinstance(resource, Task):  # `resources.tasks.Task.collections()` doesn't include elements
        assert collection[0] in resource.collections().keys() or (collection[0] + '_') in resource.collections().keys()
        assert collection[1] in resource.collections().values()

    # Set endpoint URL
    endpoint_url = get_endpoint(parameterized_endpoint=endpoint, resource=resource)
    if collection:
        endpoint_url += '/' + collection[0]
    if url_query is None:
        url_query = ''
    elif url_query and not url_query.startswith('?'):
        url_query = '?' + url_query
    endpoint_url += url_query

    # Get parent resource
    if collection is None:
        parent = resource.parents()[-1] if resource.parents() else None
    else:
        parent = resource

    # Get info about the resource for later checks
    resource_uuid = resource.uuid()

    prev_modified_at = getattr(resource.db_object(), 'modified_at', None)
    prev_activity_at = getattr(resource.db_object(), 'activity_at', None)

    if parent is not None:
        if collection is None:
            touchable_parent = resource.touch_parent() and isinstance(parent.db_object(), MutableEntity)
        else:
            touchable_parent = collection[1].touch_parent() and isinstance(parent.db_object(), MutableEntity)
        parent_prev_modified_at = parent.db_object().modified_at if touchable_parent else None
        parent_prev_activity_at = getattr(parent.db_object(), 'activity_at', None)
    else:
        touchable_parent = False
        parent_prev_modified_at = None
        parent_prev_activity_at = None

    # Send request and check response status code
    kwargs = {'json': request_json} if request_json else {}
    response = client.send_request(method, endpoint_url, **kwargs)
    if expected_status_code is not None:
        assert response.status_code == expected_status_code
        if not str(expected_status_code).startswith('2'):
            return response
    else:
        methods_status_codes = {
            'delete': HTTP_DELETE_STATUS_CODE,
            'get': HTTP_GET_STATUS_CODE,
            'post': HTTP_POST_STATUS_CODE,
            'put': HTTP_PUT_STATUS_CODE
        }
        assert response.status_code == methods_status_codes[method]

    # Commit database changes and refresh loaded resources' database objects
    db_commit_and_expire()

    # Verify database and response JSON(s)
    if method == 'delete':
        if collection is None:
            # Verify the resource was successfully deleted
            assert resource.db_model().get_from_uuid(resource_uuid) is None
        else:
            # Verify all items were removed from child collection
            assert not get_children(resource.db_object(), collection[1].db_model())
    else:
        res_json = response.json()
        res_json_schema = collection[1].dump_schema() if collection else resource.dump_schema()
        # TODO: Why don't we access `self.fields` instead?
        expected_fields = {name for name, field in res_json_schema._declared_fields.items() if field.required}
        optional_fields = {name for name, field in res_json_schema._declared_fields.items() if not field.required}
        if method in ['post', 'put']:
            # Check returned resource location
            if check_resource_location:
                if method != 'post':
                    resource_uuid = resource.uuid()
                else:
                    resource_uuid = collection[1].db_model().get_from_uuid(res_json['uuid']).uuid
                location = response.headers.get('Location') or response.headers.get('location')
                assert location and location.split('/')[-1] == resource_uuid
            # Verify response
            verify_response_json(actual_json=res_json,
                                 expected_json=expected_jsons[0],
                                 expected_fields=expected_fields,
                                 optional_fields=optional_fields)
            pass  # TODO: check database
        else:
            if 'data' in res_json and 'links' in res_json:
                # Paginated result
                res_json = res_json['data']
            verify_response_jsons(actual_jsons=(res_json if isinstance(res_json, list) else [res_json]),
                                  expected_jsons=expected_jsons,
                                  expected_fields=expected_fields,
                                  optional_fields=optional_fields)
            pass  # TODO: check database

    # Verify last modification/activity datetimes
    if method == 'put':
        if prev_modified_at is not None:
            assert resource.db_object().modified_at > prev_modified_at
        if prev_activity_at is not None:
            assert resource.db_object().activity_at > prev_activity_at

    if parent is not None:
        if touchable_parent:
            if method != 'get':
                assert parent.db_object().modified_at > parent_prev_modified_at
            else:
                assert parent.db_object().modified_at == parent_prev_modified_at
        if parent_prev_activity_at is not None:
            if method != 'get':
                assert parent.db_object().activity_at > parent_prev_activity_at
            else:
                assert parent.db_object().activity_at == parent_prev_activity_at

    return response


def verify_response_json(actual_json: dict,
                         expected_json: dict = None,
                         expected_fields: Set[str] = None,
                         optional_fields: Set[str] = None):
    """
    Note: when both `expected_json` and `expected_fields` are provided, the following logic is applied:
              - `actual_json` must contain the fields specified in `expected_fields`
              - `actual_json` cannot contain any field that has not been specified in
                `expected_fields` or `optional_fields`
              - Only the values of the fields contained in `expected_json` will be checked in `actual_json`
    """

    # Check required and optional fields
    actual_fields = set(actual_json.keys())
    expected_fields = expected_fields or set(expected_json.keys())
    optional_fields = optional_fields or set()
    assert actual_fields.issubset(expected_fields.union(optional_fields))
    assert (expected_fields - optional_fields).issubset(actual_fields)

    # Check resource UUID version (if specified)
    if 'uuid' in actual_json:
        assert uuid.UUID(actual_json['uuid']).version == UUID_VERSION

    # Check expected values
    if expected_json:
        for field, expected_value in expected_json.items():
            if field not in actual_json:
                assert field in optional_fields
                continue
            actual_value = actual_json[field]
            if isinstance(actual_value, Enum):
                actual_value = actual_value.name.lower()
            if isinstance(expected_value, Enum):
                expected_value = expected_value.name.lower()
            assert isinstance(actual_value, type(expected_value))
            # Nested resource
            if isinstance(expected_value, dict):
                verify_response_json(actual_json=actual_value,
                                     expected_json=expected_value,
                                     optional_fields=optional_fields)
            # List of IDs or nested resources
            elif isinstance(expected_value, list) and len(expected_value) > 0:
                assert len(actual_value) == len(expected_value)
                try:
                    # List of nested resources
                    for expected_nested_resource in expected_value:
                        expected_resource_found = False
                        for actual_nested_resource in actual_value:
                            try:
                                verify_response_json(actual_json=actual_nested_resource,
                                                     expected_json=expected_nested_resource,
                                                     optional_fields=optional_fields)
                                expected_resource_found = True
                                break
                            except AssertionError:
                                continue
                        assert expected_resource_found
                except Exception as e:
                    if isinstance(e, AssertionError):
                        raise e
                    # List of IDs
                    assert set(actual_value) == set(expected_value)
                    for id_ in expected_value:
                        try:
                            assert uuid.UUID(id_).version == UUID_VERSION
                            valid_id = True
                        except Exception:
                            try:
                                pass  # Decode and validate public ID
                                valid_id = True
                            except Exception:
                                valid_id = False
                        assert valid_id
            # Datetime field
            elif isinstance(expected_value, datetime):
                assert actual_value == expected_value.strftime(DATETIME_FORMAT)
            # Single-value field
            else:
                assert actual_value == expected_value


def verify_response_jsons(actual_jsons: Iterable[dict],
                          expected_jsons: Iterable[dict] = None,
                          expected_fields: Set[str] = None,
                          optional_fields: Set[str] = None):
    if expected_jsons:
        assert len(actual_jsons) == len(expected_jsons)
        for resource_json in actual_jsons:
            matching_jsons = [r for r in expected_jsons if r['uuid'] == resource_json['uuid']]
            assert len(matching_jsons) == 1
            verify_response_json(actual_json=resource_json,
                                 expected_json=matching_jsons[0],
                                 expected_fields=expected_fields,
                                 optional_fields=optional_fields)
    else:
        for resource_json in actual_jsons:
            verify_response_json(actual_json=resource_json,
                                 expected_fields=expected_fields,
                                 optional_fields=optional_fields)


def verify_response_json_with_picture(request_json: dict,
                                      response_json: dict,
                                      picture_db_object: Union[OrgFileDB, TaskFileDB],
                                      picture_field: str,
                                      optional_fields: Set[str] = None):
    """
    Verifies a response containing a reference to a picture

    Args:
        request_json: request JSON
        response_json: response JSON
        picture_db_object: expected picture's database object
        picture_field: name of the field containing the reference to the picture
        optional_fields: optional fields
    """
    assert picture_db_object.use_for in [OrgFileUse.PICTURE, TaskFileUse.PICTURE]
    # Verify response JSON (except for the picture, which is verified later)
    optional_fields = (optional_fields or set()).union(BACKEND_JSON_FIELDS).union({picture_field})
    request_json = dict(request_json)
    request_json.pop(picture_field, None)
    response_picture = response_json.pop(picture_field, None)
    verify_response_json(actual_json=response_json, expected_json=request_json, optional_fields=optional_fields)
    # Verify picture
    res_picture = dict(response_picture)
    res_picture.pop('upload_url')  # the file was not uploaded to S3
    res_picture.pop('created_at')
    res_picture.pop('created_by')
    ignore_columns = ['created_at', 'format_']
    if isinstance(picture_db_object, OrgFileDB):
        ignore_columns += ['type_', 'use_for']
    expected_picture_json = get_json_from_db_object(db_object=picture_db_object, ignore_columns=ignore_columns)
    assert res_picture == expected_picture_json


def verify_out_of_sync(client: MockClient, endpoint: str, resource: Resource, request_json: dict):
    """ Verifies an Out of Sync response to a PUT request. """
    cache.clear()

    # Set an out-of-sync state for current user
    user = UserDB.query().filter_by(auth0_id=decode_auth0_token(client.token)['sub']).first()
    if user.user_id in resource.db_object().synced_by_users:
        resource.db_object().synced_by_users.remove(user.user_id)
    save_to_db(resource.db_object())

    # Try to modify the resource
    response = verify_resource_request(client=client,
                                       method='PUT',
                                       endpoint=endpoint,
                                       resource=resource,
                                       request_json=request_json,
                                       expected_status_code=HTTP_CONFLICT_STATUS_CODE)
    assert 'out of sync' in response.json().get('error', {}).get('message', '').lower()


def verify_quota_error(client: MockClient, endpoint_url: str, request_json: dict, num_requests: int, err_msg: str):
    """ Verifies that quota limit error is returned when making concurrent requests (i.e., under race conditions). """

    def _request_thread(endpoint_url: str, request_json: dict) -> Response:
        cache.clear()  # force `resources.tasks.Task.check_quota_usage()` to use current database value
        return client.send_request(method='POST', url=endpoint_url, json=request_json)

    futures = []
    # Note: using multiple workers results in a `ConnectionAbortedError`
    with ThreadPoolExecutor(max_workers=CLIENT_MAX_THREADS) as executor:
        for i in range(num_requests):
            futures.append(executor.submit(_request_thread, endpoint_url, request_json))
        wait(futures)
        time.sleep(0.05)  # TODO: we shouldn't need this

    num_ok = 0
    for future in futures:
        response = future.result()
        if response.status_code == HTTP_POST_STATUS_CODE:
            num_ok += 1
        else:
            assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
            assert response.json()['error']['message'] == err_msg

    assert 1 < num_ok < num_requests


def verify_response_examples_or_prediction_logs(actual_jsons: Iterable[dict],
                                                expected_db_objects: List[Union[ExampleDB, PredictionDB]]):

    def _load_element_values(db_objects: Set[Union[ExValue, PredValue]]) -> Dict[str, List[dict]]:
        # Load element values from database
        elem_values_by_element = dict()
        for element_value in db_objects:
            elem_value = element_value.value
            # Handle special value types
            if isinstance(elem_value, datetime):
                elem_value = elem_value.strftime(DATETIME_FORMAT)
            # Handle special element types
            if isinstance(element_value, (ExFile, PredFile)):
                elem_value = element_value.file.public_id
            elif isinstance(element_value, (ExCategory, PredCategory)):
                elem_value = element_value.category.name
            # Add element value
            element_key = (element_value.element, getattr(element_value, 'is_target', False))
            if element_key not in elem_values_by_element:
                elem_values_by_element[element_key] = []
            elem_values_by_element[element_key].append({'value': elem_value, 'index': element_value.index})

        # Merge values of elements
        element_values = {'inputs': list(), 'outputs': list(), 'metadata': list(), 'targets': list()}
        for (element_db_object, is_target), elem_values in elem_values_by_element.items():
            # Put element value(s) in the right format
            if len(elem_values) > 1:
                elem_value = [x['value'] for x in sorted(elem_values, key=lambda x: x['index'])]
            else:
                if element_db_object.multi_value is not None:
                    elem_value = [elem_values[0]['value']]
                else:
                    elem_value = elem_values[0]['value']
            # Set destination field
            if is_target:
                assert element_db_object.element_type == ElementType.OUTPUT
                assert isinstance(expected_db_objects[0], PredictionDB)
                dst_field = 'targets'
            else:
                dst_fields = {
                    ElementType.INPUT: 'inputs',
                    ElementType.OUTPUT: 'outputs',
                    ElementType.METADATA: 'metadata',
                }
                dst_field = dst_fields[element_db_object.element_type]
            # Add element value to destination field
            element_values[dst_field].append({'element': element_db_object.name, 'value': elem_value})
        return element_values

    assert len(actual_jsons) == len(expected_db_objects)

    db_model = type(expected_db_objects[0])
    assert all(isinstance(x, db_model) for x in expected_db_objects)

    # Load database objects' own attributes
    db_commit_and_expire()
    expected_jsons = [get_json_from_db_object(db_object=x) for x in expected_db_objects]

    # Remove `size` and `trained` fields
    for expected_json in expected_jsons:
        expected_json.pop('size')
        expected_json.pop('trained', None)  # only in examples

    # Load collections ("inputs", "outputs", "metadata", "targets", "tags", "shapes")
    for db_object, json_ in zip(expected_db_objects, expected_jsons):
        # Load element values
        element_values = _load_element_values(db_objects=db_object.values())
        json_['inputs'] = element_values['inputs']
        if element_values['outputs']:
            json_['outputs'] = element_values['outputs']
        if element_values['metadata']:
            json_['metadata'] = element_values['metadata']
        if element_values['targets']:
            json_['targets'] = element_values['targets']

        # Load AI model
        if isinstance(db_object, PredictionDB):
            json_['ai_model'] = db_object.ai_model.public_id

        # Load tags
        if isinstance(db_object, ExampleDB):
            json_['tags'] = [tag.name for tag in db_object.tags]

        # Load shapes
        if isinstance(db_object, ExampleDB):
            json_['shapes'] = []
            for shape in db_object.shapes:
                shape_json = get_shape_or_slice_json_from_db_object(db_object=shape)
                json_['shapes'].append(shape_json)

        # Load slices
        if isinstance(db_object, ExampleDB):
            json_['slices'] = []
            for slice_ in db_object.slices:
                slice_json = get_shape_or_slice_json_from_db_object(db_object=slice_)
                json_['slices'].append(slice_json)

    # Check response JSONs
    if db_model == ExampleDB:
        optional_fields = {'outputs', 'metadata', 'shapes', 'slices', 'tags'}
    else:
        optional_fields = {'outputs', 'metadata', 'targets', 'removed_elements'}

    verify_response_jsons(actual_jsons=actual_jsons, expected_jsons=expected_jsons, optional_fields=optional_fields)


def mock_prediction_log_batch_request(task: Task,
                                      request_json: dict,
                                      environment: AIEnvironment = AIEnvironment.PRODUCTION) -> List[dict]:
    # Prepare request data
    request_json = PredictionLoggingRequest().load(request_json)
    # Set client making the request
    inference_service = Service.filter_by_task_and_type(task_id=task.db_object().task_id, type_=ServiceType.INFERENCE)
    task._client = inference_service.client
    task._user = None
    # Simulate a request
    prediction_logs = PredictionLog.post_batch(data=request_json['predictions'], task=task, environment=environment)
    return PredictionLog.dump_batch(predictions=prediction_logs, task=task)


def mock_element_values_json(task_id: int,
                             include_inputs: bool = True,
                             include_outputs: bool = True,
                             include_metadata: bool = True,
                             single_collection: bool = True) -> dict:

    def _input_value(input_element: ElementDB) -> Optional[object]:
        assert input_element.element_type == ElementType.INPUT
        dt_value = datetime(year=2022, month=2, day=6, hour=17).strftime(DATETIME_FORMAT)
        input_values = {
            ElementValueType.BOOLEAN: True,
            ElementValueType.INTEGER: [7, 12, 128],
            ElementValueType.FLOAT: 6.536,
            ElementValueType.TEXT: 'new_input_text',
            ElementValueType.DATETIME: dt_value
        }
        pass  # TODO: add null value (only if the element is nullable)
        pass  # TODO: add file-based value
        pass  # TODO: add shape-based value
        if input_element.value_type == ElementValueType.CATEGORY:
            category = CategoryDB.query().filter_by(element_id=input_element.element_id).first()
            return category.name
        else:
            return input_values.get(input_element.value_type)

    def _output_value(output_element: ElementDB) -> Optional[object]:
        assert output_element.element_type == ElementType.OUTPUT
        output_values = {
            ElementValueType.INTEGER: 4,
            ElementValueType.FLOAT: 0.918,
        }
        pass  # TODO: add null value (only if the element is nullable)
        pass  # TODO: add file-based value
        pass  # TODO: add shape-based value
        if output_element.value_type == ElementValueType.CATEGORY:
            categories = CategoryDB.query().filter_by(element_id=output_element.element_id).all()
            return [categories[0].name, categories[2].name]
        else:
            return output_values.get(output_element.value_type)

    def _metadata_value(metadata_element: ElementDB) -> Optional[object]:
        assert metadata_element.element_type == ElementType.METADATA
        dt_value = datetime(year=2040, month=9, day=23, hour=21).strftime(DATETIME_FORMAT)
        metadata_values = {
            ElementValueType.BOOLEAN: False,
            ElementValueType.INTEGER: 67,
            ElementValueType.FLOAT: 3.13,
            ElementValueType.TEXT: 'new_metadata_text',
            ElementValueType.DATETIME: dt_value
        }
        pass  # TODO: add null value (only if the element is nullable)
        pass  # TODO: add file-based value
        if metadata_element.value_type == ElementValueType.CATEGORY:
            category = CategoryDB.query().filter_by(element_id=metadata_element.element_id).first()
            return category.name
        else:
            return metadata_values.get(metadata_element.value_type)

    if single_collection:
        inputs_collection = outputs_collection = metadata_collection = 'values'
        json_ = {'values': []}
    else:
        inputs_collection = 'inputs'
        outputs_collection = 'outputs'
        metadata_collection = 'metadata'
        json_ = {inputs_collection: [], outputs_collection: [], metadata_collection: []}

    task = TaskDB.get(task_id=task_id)

    # Input values
    if include_inputs:
        for element in task.input_elements():
            value = _input_value(element)
            if value is None:
                continue
            json_[inputs_collection].append({'element': element.name, 'value': value})

    # Output values
    if include_outputs:
        for element in task.output_elements():
            value = _output_value(element)
            if value is None:
                continue
            json_[outputs_collection].append({'element': element.name, 'value': value})

    # Metadata values
    if include_metadata:
        for element in task.metadata_elements():
            value = _metadata_value(element)
            if value is None:
                continue
            json_[metadata_collection].append({'element': element.name, 'value': value})

    return json_


def verify_wrong_element_values(client: MockClient, resource_type: Type[Union[Example, PredictionLog]],
                                request_method: str, resource_endpoint: str, mock_example_or_prediction_jsons: dict):

    def _test_wrong_element_value(element_names: List[str],
                                  error_msg: str,
                                  missing: bool = False,
                                  wrong_value: object = None):

        db_commit_and_expire()

        db_model = resource_type.db_model()

        if request_method.strip().lower() == 'post':
            old_db_object = None
            parent_resource = load_default_resource(resource_type=Task)
            endpoint_url = get_endpoint(parameterized_endpoint=resource_endpoint, resource=parent_resource)
        else:
            resource = load_default_resource(resource_type=resource_type, parents_types=[Task])
            parent_resource = resource.parents()[0]
            resource.size(refresh=True)
            resource.db_object().force_relationship_loading()
            old_db_object = copy.deepcopy(resource.db_object())
            endpoint_url = get_endpoint(parameterized_endpoint=resource_endpoint, resource=resource)

        task: TaskDB = parent_resource.db_object()

        num_db_objects = len(db_model.filter_by_task(task_id=task.task_id))

        # Set wrong values
        request_json = copy.deepcopy(mock_example_or_prediction_jsons)
        # TODO: test with multiple examples/predictions
        batch_request = request_json.get('batch') or request_json.get('predictions')
        wrong_values = batch_request[0] if batch_request is not None else request_json

        if resource_type == PredictionLog:
            wrong_values['values'] = (wrong_values.get('inputs', []) + wrong_values.get('outputs', []) +
                                      wrong_values.get('metadata', []))

        if missing:
            if resource_type == Example:
                wrong_values['labeling_status'] = 'labeled'
            if not element_names:
                element_names = [x['element'] for x in wrong_values['values']]
            wrong_values['values'] = [x for x in wrong_values['values'] if x['element'] not in element_names]
        else:
            for wrong_input in wrong_values['values']:
                if wrong_input['element'] not in element_names:
                    continue
                wrong_input['value'] = wrong_value

        if resource_type == PredictionLog:
            inputs = [x['element'] for x in wrong_values.get('inputs', [])]
            outputs = [x['element'] for x in wrong_values.get('outputs', [])]
            metadata = [x['element'] for x in wrong_values.get('metadata', [])]

            wrong_values['inputs'] = [x for x in wrong_values['values'] if x['element'] in inputs]
            wrong_values['outputs'] = [x for x in wrong_values['values'] if x['element'] in outputs]
            wrong_values['metadata'] = [x for x in wrong_values['values'] if x['element'] in metadata]

            wrong_values.pop('values')

        # Get current quota
        prev_num_items = task.num_examples if resource_type == Example else task.num_predictions
        prev_space_usage = task.space_usage

        # Make request and verify response and database
        if resource_type == Example:
            response = client.send_request(request_method, endpoint_url, json=request_json)
            # Check response
            assert response.status_code == HTTP_BAD_REQUEST_STATUS_CODE
            expected_error = error_msg + ': ' + ', '.join(f'{x}' for x in element_names)
            assert response.json()['error']['message'] == expected_error
            # Check database
            db_commit_and_expire()
            if old_db_object is not None:
                # Verify the example was not modified and keeps the same values/objects
                new_db_object = db_model.get(example_id=old_db_object.example_id)
                assert_same_element_values(db_object_1=old_db_object, db_object_2=new_db_object)
            assert len(db_model.filter_by_task(task_id=1)) == num_db_objects
        else:
            # Note: we can't make an API request because prediction logging is an asynchronous process (Celery task).
            res_jsons = mock_prediction_log_batch_request(task=parent_resource, request_json=request_json)
            # Check response
            res_json = res_jsons[0]  # TODO: adapt it for multiple predictions
            expected_invalid_data = {k: v for k, v in wrong_values.items() if k in ['inputs', 'outputs', 'metadata']}
            assert res_json['invalid_data'] == expected_invalid_data
            assert not res_json.get('inputs')
            assert not res_json.get('outputs')
            assert not res_json.get('metadata')
            # Check database
            db_commit_and_expire()
            pass  # TODO

        # Check quota
        if resource_type == Example:
            num_items_quota = 'examples'
            expected_num_items = prev_num_items
            expected_space_usage = prev_space_usage
        else:
            num_items_quota = 'predictions'
            expected_num_items = prev_num_items + 1
            expected_space_usage = prev_space_usage + len(str(expected_invalid_data))

        verify_quota_usage(db_object=task, quota=num_items_quota, expected_usage=expected_num_items)
        verify_quota_usage(db_object=task, quota='space', expected_usage=expected_space_usage)

    @contextlib.contextmanager
    def _modify_and_restore_element(element_names, attribute_name, new_value):
        task = load_default_resource(resource_type=Task)
        elements = [ElementDB.get_from_id(id_value=x, parent=task.db_object()) for x in element_names]

        original_values = {}

        try:
            # Modify the attribute for all elements and store the original values
            for element in elements:
                original_values[element] = getattr(element, attribute_name)
                setattr(element, attribute_name, new_value)

            yield
        finally:
            # Restore the original values for all elements
            for element, original_value in original_values.items():
                setattr(element, attribute_name, original_value)

    # Try to omit required elements
    element_names = ['input_1', 'input_2', 'input_6', 'output_2', 'output_3', 'output_5', 'metadata_4']
    with _modify_and_restore_element(element_names, 'required', True):
        _test_wrong_element_value(element_names=element_names, error_msg='Missing required elements', missing=True)

    # Try to set null for a non-nullable element
    element_names = ['input_2', 'input_4', 'input_5', 'output_3', 'metadata_2', 'metadata_5']
    with _modify_and_restore_element(element_names, 'nullable', False):
        _test_wrong_element_value(element_names=element_names,
                                  error_msg='Elements not supporting null values',
                                  wrong_value=None)

    # Try to provide multiple values for an element not supporting multi-value
    element_names = ['input_3', 'output_3', 'metadata_3']
    with _modify_and_restore_element(element_names, 'multi_value', None):
        _test_wrong_element_value(element_names=element_names,
                                  error_msg='Elements not supporting multiple values',
                                  wrong_value=[3.12, 7.24, 5.03])


def mock_prediction_log_json(task_id: int,
                             include_inputs: bool = True,
                             include_outputs: bool = True,
                             include_metadata: bool = True,
                             include_targets: bool = True,
                             state: PredictionState = None) -> dict:

    def mock_category_json(categories: List[CategoryDB] = None) -> Optional[dict]:
        if not categories:
            return None
        # Get random scores
        random_scores = [random.uniform(0, 1) for _ in range(len(categories))]
        scores_sum = sum(random_scores)
        normalized_scores = [x / scores_sum for x in random_scores]
        max_score = max(normalized_scores)
        # Assign scores to categories
        cat_scores = dict()
        winner_category_name = None
        for category, score in zip(categories, normalized_scores):
            cat_scores[category.name] = score
            if score == max_score:
                winner_category_name = category.name
        return {
            'category': winner_category_name,
            'scores': cat_scores,
        }

    task = TaskDB.get(task_id=task_id)

    # Set element values
    pred_json = mock_element_values_json(task_id=task_id,
                                         include_inputs=include_inputs,
                                         include_outputs=include_outputs,
                                         include_metadata=include_metadata,
                                         single_collection=False)

    # Set category scores
    categorical_outputs = [x.name for x in task.output_elements() if x.value_type == ElementValueType.CATEGORY]
    for output_value in pred_json['outputs']:
        if output_value['element'] in categorical_outputs:
            element = ElementDB.get_from_id(id_value=output_value['element'], parent=task)
            categories = CategoryDB.query().filter_by(element_id=element.element_id).all()
            output_value['value'] = mock_category_json(categories)

    # Set target values
    if include_targets:
        pred_json['targets'] = copy.deepcopy(pred_json['outputs'])
        for output_value in pred_json['targets']:
            if isinstance(output_value['value'], dict) and set(output_value['value'].keys()) == {'category', 'scores'}:
                output_value['value'].pop('scores')

    # Set metadata
    pred_json['ai_model'] = AIModelDB.filter_by_task(task_id=task_id)[0].public_id

    state = state or (PredictionState.COMPLETE if include_outputs else PredictionState.PENDING)
    pred_json['state'] = state.name

    return pred_json


def mock_shape_or_slice_json(type_: str, task_id: int) -> dict:
    assert type_ in ['shape', 'slice']

    task = TaskDB.get(task_id=task_id)

    if type_ == 'shape':
        input_element = [x for x in task.input_elements() if x.value_type == ElementValueType.IMAGE_FILE][0]
    else:
        input_element = [
            x for x in task.input_elements()
            if (x.value_type in [ElementValueType.INTEGER, ElementValueType.FLOAT] and
                x.multi_value in [ElementMultiValue.ORDERED, ElementMultiValue.TIME_SERIES])
        ][0]

    output_elements = [
        x for x in task.output_elements() if x.value_type in [ElementValueType.CATEGORY, ElementValueType.FLOAT]
    ]

    output_values = {
        ElementValueType.CATEGORY: 'Output Category 2',
        ElementValueType.FLOAT: 0.615,
    }

    json_ = {
        'element':
            input_element.name,
        'outputs': [{
            'element': x.name,
            'value': output_values[x.value_type] if x.multi_value is None else [output_values[x.value_type]]
        } for x in output_elements]
    }

    if type_ == 'shape':
        json_['polygon'] = [{'x': 136, 'y': 14}, {'x': 201, 'y': 78}, {'x': 166, 'y': 49}]
    else:
        json_['start_index'] = 2
        json_['end_index'] = 4

    return json_


def verify_example_or_prediction_deletion(client: MockClient, resource_type: Type[Union[Example, PredictionLog]],
                                          file_storage_backend: Optional[FileStorageBackend]):
    # Get task and resources
    db_model = resource_type.db_model()

    task = load_default_resource(resource_type=Task)
    task_db_object = task.db_object()

    resources = [
        resource_type.get(agent=UserDB.get(user_id=1), db_object_or_id=x, parents=[task])
        for x in db_model.filter_by_task(task_id=task_db_object.task_id)
    ]

    # Get examples and predictions' specific attributes
    if resource_type == Example:
        pk_col = 'example_id'
        file_db_model = ExFile
        files_db_collection = 'ex_files'
        num_rsrcs_quota = 'examples'
        num_rsrcs_col = 'num_examples'
    else:
        pk_col = 'prediction_id'
        file_db_model = PredFile
        files_db_collection = 'pred_files'
        num_rsrcs_quota = 'predictions'
        num_rsrcs_col = 'num_predictions'

        # Set environment to "testing" (predictions made in production cannot be deleted)
        for prediction in resources:
            prediction.db_object().environment = AIEnvironment.TESTING
        db_commit_and_expire()

    # Delete all files
    empty_table(TaskFileDB)

    # Upload files
    NUM_FILES = 10
    use_for = {0: 'input', 1: 'output', 2: 'metadata'}
    file_jsons = [{
        'filename': f'Test File {idx + 1}',
        'size': (idx + 1) * 10,
        'use_for': use_for[idx % len(use_for)],
        'type': None
    } for idx in range(NUM_FILES)]
    res_data = client.upload_files(files=file_jsons, parent=task, storage_backend=file_storage_backend)
    assert len(res_data['uploaded_metadata']) == len(res_data['uploaded_data']) == len(file_jsons)
    db_commit_and_expire()
    files_filter = sql_and(TaskFileDB.task_id == task_db_object.task_id, TaskFileDB.use_for != TaskFileUse.AI_MODEL)
    files = [f for f in TaskFileDB.query().filter(files_filter)]

    # Associate files
    input_element = (ElementDB.query().filter_by(task_id=task_db_object.task_id,
                                                 element_type=ElementType.INPUT,
                                                 value_type=ElementValueType.IMAGE_FILE).first())
    metadata_element = (ElementDB.query().filter_by(task_id=task_db_object.task_id,
                                                    element_type=ElementType.METADATA,
                                                    value_type=ElementValueType.IMAGE_FILE).first())
    elements = [input_element, metadata_element]
    for resource in resources:
        getattr(resource.db_object(), files_db_collection).clear()
        save_to_db(resource.db_object())
    for idx, file in enumerate(files):
        pk = getattr(resources[idx % len(resources)].db_object(), pk_col)
        elem_idx = idx % len(elements)
        saved = False
        while not saved:
            assoc_file = file_db_model(**{pk_col: pk}, element_id=elements[elem_idx].element_id, value=file.file_id)
            try:
                save_to_db(assoc_file)
                saved = True
            except Exception:
                elem_idx = (elem_idx + 1) % len(elements)
                db_rollback()

    # Set current number of examples/predictions
    set_quota_usage(db_object=task_db_object, quota=num_rsrcs_quota, usage=len(resources))

    # Set actual space usage
    space_usage = sum(resource.size(refresh=True) for resource in resources) + sum(f.size for f in files)
    set_quota_usage(db_object=task_db_object, quota='space', usage=space_usage)

    # Delete each database object
    for resource in resources:
        rsrc_uuid = resource.uuid()
        rsrc_files = getattr(resource.db_object(), files_db_collection)
        rsrc_file_uuids = [x.file.uuid for x in rsrc_files]
        rsrc_size = resource.size() + sum(x.file.size for x in rsrc_files)
        prev_num_rsrcs = getattr(task_db_object, num_rsrcs_col)
        prev_space_usage = task_db_object.space_usage
        s3_files = get_files_in_s3(parent_resource=task)
        assert all(x in s3_files for x in rsrc_file_uuids)
        # Make request and verify response
        verify_resource_request(client=client,
                                method='DELETE',
                                endpoint=(ENDPOINT_EXAMPLE if resource_type == Example else ENDPOINT_AI_PREDICTION_LOG),
                                resource=resource)
        # Check database
        db_commit_and_expire()
        assert db_model.get_from_uuid(rsrc_uuid) is None
        assert all(TaskFileDB.get_from_uuid(x) is None for x in rsrc_file_uuids)
        expected_num_rsrcs = (prev_num_rsrcs - 1) if resource_type == Example else len(resources)
        verify_quota_usage(db_object=task_db_object, quota=num_rsrcs_quota, expected_usage=expected_num_rsrcs)
        verify_quota_usage(db_object=task_db_object, quota='space', expected_usage=prev_space_usage - rsrc_size)
        # Check S3
        s3_files = get_files_in_s3(parent_resource=task)
        assert all(x not in s3_files for x in rsrc_file_uuids)

    # Check database
    assert not db_model.query().filter_by(task_id=task_db_object.task_id).all()
    assert not get_files_in_s3(parent_resource=task)  # all associated files were correctly removed from S3
    expected_num_rsrcs = 0 if resource_type == Example else len(resources)
    verify_quota_usage(db_object=task_db_object, quota=num_rsrcs_quota, expected_usage=expected_num_rsrcs)
    verify_quota_usage(db_object=task_db_object, quota='space', expected_usage=0)
