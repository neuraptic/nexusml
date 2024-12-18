from typing import List, Type

import pytest

from nexusml.api.endpoints import ENDPOINT_EXAMPLE
from nexusml.api.endpoints import ENDPOINT_TAG
from nexusml.api.endpoints import ENDPOINT_TASK
from nexusml.api.external.ext import cache
from nexusml.api.resources.ai import AIModel
from nexusml.api.resources.ai import PredictionLog
from nexusml.api.resources.base import Resource
from nexusml.api.resources.examples import Example
from nexusml.api.resources.files import OrgFile
from nexusml.api.resources.files import TaskFile
from nexusml.api.resources.organizations import Organization
from nexusml.api.resources.tags import Tag
from nexusml.api.resources.tasks import Task
from nexusml.constants import ADMIN_ROLE
from nexusml.constants import HTTP_DELETE_STATUS_CODE
from nexusml.constants import HTTP_FORBIDDEN_STATUS_CODE
from nexusml.constants import HTTP_GET_STATUS_CODE
from nexusml.constants import HTTP_METHOD_NOT_ALLOWED_STATUS_CODE
from nexusml.constants import HTTP_POST_STATUS_CODE
from nexusml.constants import HTTP_PUT_STATUS_CODE
from nexusml.constants import MAINTAINER_ROLE
from nexusml.constants import NUM_RESERVED_CLIENTS
from nexusml.database.core import save_to_db
from nexusml.database.organizations import client_scopes
from nexusml.database.organizations import CollaboratorDB
from nexusml.database.organizations import OrganizationDB
from nexusml.database.organizations import RoleDB
from nexusml.database.organizations import UserDB
from nexusml.database.permissions import RolePermission
from nexusml.database.permissions import UserPermission
from nexusml.database.tasks import TaskDB
from nexusml.enums import ResourceAction
from nexusml.enums import ResourceType
from tests.api.integration.conftest import MockClient
from tests.api.integration.test_api_keys import generate_api_key
from tests.api.integration.utils import get_endpoint
from tests.api.integration.utils import mock_element_values_json
from tests.api.integration.utils import send_request
from tests.api.utils import db_commit_and_expire
from tests.api.utils import load_default_resource

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_CLIENT_ID = NUM_RESERVED_CLIENTS + 2  # First client IDs are reserved for official apps


class TestResourcePermissions:

    def test_delete(self, client: MockClient, session_user_id: str):

        def _get_permissions(task: Task) -> tuple:
            user_permissions = UserPermission.query().filter_by(resource_uuid=task.uuid()).all()
            role_permissions = (RolePermission.query().join(RoleDB).filter(
                RolePermission.resource_uuid == task.uuid(), RoleDB.name.notin_([ADMIN_ROLE, MAINTAINER_ROLE])).all())
            return user_permissions, role_permissions

        fixtures = _get_fixtures(session_user_id=session_user_id)
        _set_permissions(session_user_id=session_user_id)
        user_permissions, role_permissions = _get_permissions(task=fixtures['task'])
        assert user_permissions and role_permissions
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_TASK, resource=fixtures['task'])
        response = client.send_request(method='DELETE', url=endpoint_url + '/permissions')
        assert response.status_code == HTTP_DELETE_STATUS_CODE
        db_commit_and_expire()
        fixtures = _get_fixtures(session_user_id=session_user_id)
        user_permissions, role_permissions = _get_permissions(task=fixtures['task'])
        assert not user_permissions and not role_permissions

    def test_get(self, client: MockClient, session_user_id: str):

        def _check_permissions(actual_permissions: dict, expected_permissions: dict):

            def _check_agent_permissions(actual_perms: dict, expected_perms: dict, agent: str):
                assert len(actual_perms) == len(expected_perms)
                for expected_agent_perms in expected_perms:
                    matching_agent_perms = [x for x in actual_perms if x[agent] == expected_agent_perms[agent]]
                    assert len(matching_agent_perms) == 1
                    actual_agent_perms = matching_agent_perms[0]
                    assert len(actual_agent_perms['permissions']) == len(expected_agent_perms['permissions'])
                    assert all(x in actual_agent_perms['permissions'] for x in expected_agent_perms['permissions'])

            _check_agent_permissions(actual_perms=actual_permissions['users'],
                                     expected_perms=expected_permissions['users'],
                                     agent='user')
            _check_agent_permissions(actual_perms=actual_permissions['roles'],
                                     expected_perms=expected_permissions['roles'],
                                     agent='role')
            _check_agent_permissions(actual_perms=actual_permissions['collaborators'],
                                     expected_perms=expected_permissions['collaborators'],
                                     agent='collaborator')

        fixtures = _get_fixtures(session_user_id=session_user_id)
        _set_permissions(session_user_id=session_user_id)
        admin_role = RoleDB.get_from_id(id_value=ADMIN_ROLE, parent=fixtures['organization'])
        maintainer_role = RoleDB.get_from_id(id_value=MAINTAINER_ROLE, parent=fixtures['organization'])
        collaborator = (CollaboratorDB.query().filter_by(organization_id=fixtures['organization'].organization_id,
                                                         user_id=fixtures['other_org_user'].user_id).first())
        ##############
        # Admin view #
        ##############
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_TASK, resource=fixtures['task'])
        response = client.send_request(method='GET', url=endpoint_url + '/permissions')
        assert response.status_code == HTTP_GET_STATUS_CODE
        expected_perms = {
            'users': [{
                'user':
                    fixtures['user'].public_id,
                'permissions': [
                    {
                        'resource_type': 'task',
                        'action': 'read',
                        'allow': True
                    },
                    {
                        'resource_type': 'task',
                        'action': 'delete',
                        'allow': False
                    },
                    {
                        'resource_uuid': fixtures['task'].uuid(),
                        'resource_type': 'task',
                        'action': 'delete',
                        'allow': True
                    },
                ]
            }, {
                'user':
                    fixtures['same_org_user'].public_id,
                'permissions': [
                    {
                        'resource_type': 'task',
                        'action': 'read',
                        'allow': True
                    },
                    {
                        'resource_type': 'task',
                        'action': 'delete',
                        'allow': False
                    },
                    {
                        'resource_uuid': fixtures['task'].uuid(),
                        'resource_type': 'task',
                        'action': 'update',
                        'allow': True
                    },
                ]
            }],
            'roles': [{
                'role':
                    admin_role.public_id,
                'permissions': [{
                    'resource_type': 'task',
                    'action': action.name.lower(),
                    'allow': True
                } for action in ResourceAction if action != ResourceAction.CREATE]
            }, {
                'role':
                    maintainer_role.public_id,
                'permissions': [{
                    'resource_type': 'task',
                    'action': action.name.lower(),
                    'allow': True
                } for action in ResourceAction if action != ResourceAction.CREATE]
            }, {
                'role':
                    fixtures['role'].public_id,
                'permissions': [
                    {
                        'resource_type': 'task',
                        'action': 'read',
                        'allow': True
                    },
                    {
                        'resource_type': 'task',
                        'action': 'update',
                        'allow': True
                    },
                    {
                        'resource_uuid': fixtures['task'].uuid(),
                        'resource_type': 'task',
                        'action': 'delete',
                        'allow': True
                    },
                    {
                        'resource_uuid': fixtures['task'].uuid(),
                        'resource_type': 'task',
                        'action': 'update',
                        'allow': False
                    },
                ]
            }],
            'collaborators': [{
                'collaborator':
                    collaborator.public_id,
                'permissions': [
                    {
                        'resource_uuid': fixtures['task'].uuid(),
                        'resource_type': 'task',
                        'action': 'update',
                        'allow': True
                    },
                    {
                        'resource_uuid': fixtures['task'].uuid(),
                        'resource_type': 'task',
                        'action': 'delete',
                        'allow': False
                    },
                ]
            }]
        }
        _check_permissions(actual_permissions=response.json(), expected_permissions=expected_perms)
        #####################
        # Regular user view #
        #####################
        cache.clear()
        fixtures['user'].roles.clear()
        save_to_db(fixtures['user'])
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_TASK, resource=fixtures['task'])
        response = client.send_request(method='GET', url=endpoint_url + '/permissions')
        assert response.status_code == HTTP_GET_STATUS_CODE
        for agents_key, agents_perms in expected_perms.items():
            for agent_perms in agents_perms:
                agent_perms['permissions'] = [perm for perm in agent_perms['permissions'] if perm['allow']]
        _check_permissions(actual_permissions=response.json(), expected_permissions=expected_perms)


class _TestAPIKeyScopes:

    def __init__(self, parameterized_endpoint: str, resource_type: Type[Resource], parents_types: List[Type[Resource]],
                 requests_json: dict):

        self._parameterized_endpoint = parameterized_endpoint
        self._resource_type = resource_type
        self._parents_types = parents_types
        self._requests_json = requests_json

    def _test_method(self, method: str, expected_status_code: int):
        method = method.lower().strip()
        if self._resource_type == Example and method == 'post':
            requests_json = {'batch': [self._requests_json]}
        else:
            requests_json = self._requests_json

        # Set endpoint
        if method == 'post':
            if not self._parents_types:
                return
            parent = load_default_resource(resource_type=self._parents_types[-1],
                                           parents_types=self._parents_types[:-1])
            endpoint = '/'.join(self._parameterized_endpoint.split('/')[:-1])
            endpoint_url = get_endpoint(parameterized_endpoint=endpoint, resource=parent)
        else:
            resource = load_default_resource(resource_type=self._resource_type, parents_types=self._parents_types)
            if method == 'put':
                resource.db_object().synced_by_clients = [_CLIENT_ID]
                resource.persist()
            endpoint_url = get_endpoint(parameterized_endpoint=self._parameterized_endpoint, resource=resource)

        # Set scopes
        resource_type_scopes = {
            Organization: 'organizations',
            Task: 'tasks',
            TaskFile: 'files',
            OrgFile: 'files',
            AIModel: 'models',
            Example: 'examples',
            PredictionLog: 'predictions'
        }

        scopes = {'delete': 'delete', 'get': 'read', 'post': 'create', 'put': 'update'}

        if self._resource_type in resource_type_scopes:
            scope_prefix = resource_type_scopes[self._resource_type] + '.'
            valid_scope = scope_prefix + scopes[method]
            valid_scopes = [valid_scope] if valid_scope in client_scopes else []
            invalid_scopes = [scope_prefix + scopes[x] for x in scopes if x != method]
        else:
            valid_scopes = []
            invalid_scopes = []

        if self._parents_types and self._parents_types[-1] in resource_type_scopes:
            parent_scope_prefix = resource_type_scopes[self._parents_types[-1]] + '.'
            # TODO: Not all parents require `update`. Only those that are touched by children requires it.
            parent_scopes = [parent_scope_prefix + 'read', parent_scope_prefix + 'update']
            valid_scopes += [parent_scope for parent_scope in parent_scopes if parent_scope in client_scopes]

        # Generate API key
        valid_api_key = generate_api_key(scopes=valid_scopes)
        if invalid_scopes:
            invalid_api_key = generate_api_key(scopes=[x for x in invalid_scopes if x in client_scopes])
        else:
            invalid_api_key = None

        # Forbidden request
        if invalid_api_key is not None:
            response = send_request(method=method, url=endpoint_url, api_key=invalid_api_key, json=requests_json)
            assert response.status_code in [HTTP_FORBIDDEN_STATUS_CODE, HTTP_METHOD_NOT_ALLOWED_STATUS_CODE]
        # Valid request
        response = send_request(method=method, url=endpoint_url, api_key=valid_api_key, json=requests_json)
        assert response.status_code in [expected_status_code, HTTP_METHOD_NOT_ALLOWED_STATUS_CODE]

    def test_delete(self, client: MockClient):
        self._test_method(method='DELETE', expected_status_code=HTTP_DELETE_STATUS_CODE)

    def test_get(self, client: MockClient):
        self._test_method(method='GET', expected_status_code=HTTP_GET_STATUS_CODE)

    def test_post(self, client: MockClient):
        self._test_method(method='POST', expected_status_code=HTTP_POST_STATUS_CODE)

    def test_put(self, client: MockClient):
        self._test_method(method='PUT', expected_status_code=HTTP_PUT_STATUS_CODE)


# TODO: implement `TestOrganizationsAPIKey`
#       Note: import `resources.organizations.Organization`
# class TestOrganizationsAPIKey:
#
#     @pytest.fixture(autouse=True)
#     def _api_key_scopes_tester(self):
#         mock_org = {'trn': 'mock_org_arn',
#                     'name': 'mock_org_name',
#                     'domain': 'mockorg.com',
#                     'address': 'mock_org_address'}
#         self._api_key_scopes_tester = _TestAPIKeyScopes(parameterized_endpoint=ENDPOINT_ORGANIZATION,
#                                                         resource_type=Organization,
#                                                         parents_types=[],
#                                                         requests_json=mock_org)
#
#     def test_delete(self, client: MockClient):
#         self._api_key_scopes_tester.test_delete(client=client)
#
#     def test_get(self, client: MockClient):
#         self._api_key_scopes_tester.test_get(client=client)
#
#     def test_post(self, client: MockClient):
#         self._api_key_scopes_tester.test_post(client=client)
#
#     def test_put(self, client: MockClient):
#         self._api_key_scopes_tester.test_put(client=client)


class TestTasksAPIKey:

    @pytest.fixture(autouse=True)
    def _api_key_scopes_tester(self):
        mock_task = {'name': 'Mock task name', 'description': 'Mock task description', 'icon': None}
        self._api_key_scopes_tester = _TestAPIKeyScopes(parameterized_endpoint=ENDPOINT_TASK,
                                                        resource_type=Task,
                                                        parents_types=[],
                                                        requests_json=mock_task)

    def test_delete(self, client: MockClient):
        self._api_key_scopes_tester.test_delete(client=client)

    def test_get(self, client: MockClient):
        self._api_key_scopes_tester.test_get(client=client)

    def test_post(self, client: MockClient):
        self._api_key_scopes_tester.test_post(client=client)

    def test_put(self, client: MockClient):
        self._api_key_scopes_tester.test_put(client=client)


# TODO: implement `TestFilesAPIKey`
# class TestFilesAPIKey:
#
#     @pytest.fixture(autouse=True)
#     def _api_key_scopes_tester(self):
#         mock_file = {}  # TODO: fill
#         self._api_key_scopes_tester = _TestAPIKeyScopes(parameterized_endpoint=ENDPOINT_TASK_FILE,
#                                                         resource_type=File,
#                                                         parents_types=[Task],
#                                                         requests_json=mock_file)
#
#     def test_delete(self, client: MockClient):
#         self._api_key_scopes_tester.test_delete(client=client)
#
#     def test_get(self, client: MockClient):
#         self._api_key_scopes_tester.test_get(client=client)
#
#     def test_post(self, client: MockClient):
#         self._api_key_scopes_tester.test_post(client=client)
#
#     def test_put(self, client: MockClient):
#         self._api_key_scopes_tester.test_put(client=client)

# TODO: implement `TestAIModelsAPIKey`
# class TestAIModelsAPIKey:
#
#     @pytest.fixture(autouse=True)
#     def _api_key_scopes_tester(self):
#         mock_ai_model = {}  # TODO: fill
#         self._api_key_scopes_tester = _TestAPIKeyScopes(parameterized_endpoint=ENDPOINT_AI_MODEL,
#                                                         resource_type=AIModel,
#                                                         parents_types=[Task],
#                                                         requests_json=mock_ai_model)
#
#     def test_delete(self, client: MockClient):
#         self._api_key_scopes_tester.test_delete(client=client)
#
#     def test_get(self, client: MockClient):
#         self._api_key_scopes_tester.test_get(client=client)
#
#     def test_post(self, client: MockClient):
#         self._api_key_scopes_tester.test_post(client=client)
#
#     def test_put(self, client: MockClient):
#         self._api_key_scopes_tester.test_put(client=client)


class TestExamplesAPIKey:

    @pytest.fixture(autouse=True)
    def _api_key_scopes_tester(self):
        self._api_key_scopes_tester = _TestAPIKeyScopes(parameterized_endpoint=ENDPOINT_EXAMPLE,
                                                        resource_type=Example,
                                                        parents_types=[Task],
                                                        requests_json=mock_element_values_json(task_id=1))

    def test_delete(self, client: MockClient):
        self._api_key_scopes_tester.test_delete(client=client)

    def test_get(self, client: MockClient):
        self._api_key_scopes_tester.test_get(client=client)

    def test_post(self, client: MockClient):
        self._api_key_scopes_tester.test_post(client=client)

    def test_put(self, client: MockClient):
        self._api_key_scopes_tester.test_put(client=client)


class TestTagsAPIKey:

    @pytest.fixture(autouse=True)
    def _api_key_scopes_tester(self):
        mock_tag = {'name': 'Mock tag name', 'description': 'Mock tag description', 'color': '0018FA'}
        self._api_key_scopes_tester = _TestAPIKeyScopes(parameterized_endpoint=ENDPOINT_TAG,
                                                        resource_type=Tag,
                                                        parents_types=[Task],
                                                        requests_json=mock_tag)

    def test_delete(self, client: MockClient):
        self._api_key_scopes_tester.test_delete(client=client)

    def test_get(self, client: MockClient):
        self._api_key_scopes_tester.test_get(client=client)

    def test_post(self, client: MockClient):
        self._api_key_scopes_tester.test_post(client=client)

    def test_put(self, client: MockClient):
        self._api_key_scopes_tester.test_put(client=client)


def _get_fixtures(session_user_id: str) -> dict:
    user = UserDB.get_from_uuid(session_user_id)
    organization = OrganizationDB.get(organization_id=user.organization_id)
    role = RoleDB.get_from_id(id_value='Dasci', parent=organization)
    task = load_default_resource(resource_type=Task)
    assert role.organization_id == organization.organization_id
    assert task.db_object().organization_id == organization.organization_id
    return {
        'organization': organization,
        'user': user,
        'role': role,
        'task': task,
        'same_org_user': (UserDB.query().filter(UserDB.user_id != user.user_id,
                                                UserDB.organization_id == organization.organization_id).first()),
        'other_org':
            (OrganizationDB.query().filter(OrganizationDB.organization_id != organization.organization_id).first()),
        'other_org_user': UserDB.query().filter(UserDB.organization_id != organization.organization_id).first(),
        'other_org_task': (TaskDB.query().filter(TaskDB.organization_id != organization.organization_id).first())
    }


def _set_permissions(session_user_id: str):
    fixtures = _get_fixtures(session_user_id=session_user_id)
    fixtures['user'].permissions.delete()
    user_perms = [
        # Permissions in session user's organization
        #    Generic permissions
        UserPermission(organization_id=fixtures['organization'].organization_id,
                       user_id=fixtures['user'].user_id,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.READ,
                       allow=True),
        UserPermission(organization_id=fixtures['organization'].organization_id,
                       user_id=fixtures['user'].user_id,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.CREATE,
                       allow=True),
        UserPermission(organization_id=fixtures['organization'].organization_id,
                       user_id=fixtures['other_org_user'].user_id,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.CREATE,
                       allow=True),
        UserPermission(organization_id=fixtures['organization'].organization_id,
                       user_id=fixtures['user'].user_id,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.DELETE,
                       allow=False),
        UserPermission(organization_id=fixtures['organization'].organization_id,
                       user_id=fixtures['same_org_user'].user_id,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.READ,
                       allow=True),
        UserPermission(organization_id=fixtures['organization'].organization_id,
                       user_id=fixtures['same_org_user'].user_id,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.DELETE,
                       allow=False),
        UserPermission(organization_id=fixtures['organization'].organization_id,
                       user_id=fixtures['user'].user_id,
                       resource_type=ResourceType.FILE,
                       action=ResourceAction.CREATE,
                       allow=True),
        UserPermission(organization_id=fixtures['organization'].organization_id,
                       user_id=fixtures['user'].user_id,
                       resource_type=ResourceType.AI_MODEL,
                       action=ResourceAction.DELETE,
                       allow=False),
        #    Resource-level permissions
        UserPermission(organization_id=fixtures['organization'].organization_id,
                       user_id=fixtures['user'].user_id,
                       resource_uuid=fixtures['task'].uuid(),
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.DELETE,
                       allow=True),
        UserPermission(organization_id=fixtures['organization'].organization_id,
                       user_id=fixtures['other_org_user'].user_id,
                       resource_uuid=fixtures['task'].uuid(),
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.UPDATE,
                       allow=True),
        UserPermission(organization_id=fixtures['organization'].organization_id,
                       user_id=fixtures['same_org_user'].user_id,
                       resource_uuid=fixtures['task'].uuid(),
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.UPDATE,
                       allow=True),
        UserPermission(organization_id=fixtures['organization'].organization_id,
                       user_id=fixtures['other_org_user'].user_id,
                       resource_uuid=fixtures['task'].uuid(),
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.DELETE,
                       allow=False),
        # Permissions in another organization
        #    Generic permissions
        UserPermission(organization_id=fixtures['other_org'].organization_id,
                       user_id=fixtures['user'].user_id,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.DELETE,
                       allow=True),
        #    Resource-level permissions
        UserPermission(organization_id=fixtures['other_org'].organization_id,
                       user_id=fixtures['user'].user_id,
                       resource_uuid=fixtures['other_org_task'].uuid,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.UPDATE,
                       allow=True),
        UserPermission(organization_id=fixtures['other_org'].organization_id,
                       user_id=fixtures['same_org_user'].user_id,
                       resource_uuid=fixtures['other_org_task'].uuid,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.READ,
                       allow=True),
        UserPermission(organization_id=fixtures['other_org'].organization_id,
                       user_id=fixtures['other_org_user'].user_id,
                       resource_uuid=fixtures['other_org_task'].uuid,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.DELETE,
                       allow=True)
    ]
    save_to_db(user_perms)
    fixtures['role'].permissions.delete()
    role_perms = [
        # Generic permissions
        RolePermission(role_id=fixtures['role'].role_id,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.READ,
                       allow=True),
        RolePermission(role_id=fixtures['role'].role_id,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.CREATE,
                       allow=False),
        RolePermission(role_id=fixtures['role'].role_id,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.UPDATE,
                       allow=True),
        RolePermission(role_id=fixtures['role'].role_id,
                       resource_type=ResourceType.FILE,
                       action=ResourceAction.CREATE,
                       allow=True),
        RolePermission(role_id=fixtures['role'].role_id,
                       resource_type=ResourceType.AI_MODEL,
                       action=ResourceAction.CREATE,
                       allow=False),
        # Resource-level permissions
        RolePermission(role_id=fixtures['role'].role_id,
                       resource_uuid=fixtures['task'].uuid(),
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.UPDATE,
                       allow=False),
        RolePermission(role_id=fixtures['role'].role_id,
                       resource_uuid=fixtures['task'].uuid(),
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.DELETE,
                       allow=True)
    ]
    save_to_db(role_perms)
