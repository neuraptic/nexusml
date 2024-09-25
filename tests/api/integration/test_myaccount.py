from datetime import datetime
from datetime import timedelta
from typing import Iterable, Union

import pytest

from nexusml.api.utils import API_DOMAIN
from nexusml.api.utils import config
from nexusml.constants import ENDPOINT_EXAMPLE
from nexusml.constants import ENDPOINT_MYACCOUNT
from nexusml.constants import ENDPOINT_MYACCOUNT_NOTIFICATIONS
from nexusml.constants import ENDPOINT_MYACCOUNT_ORGANIZATION
from nexusml.constants import ENDPOINT_MYACCOUNT_PERMISSIONS
from nexusml.constants import ENDPOINT_MYACCOUNT_ROLES
from nexusml.constants import ENDPOINT_MYACCOUNT_SETTINGS
from nexusml.constants import ENDPOINT_TASK
from nexusml.constants import ENDPOINT_TASK_FILE
from nexusml.constants import HTTP_DELETE_STATUS_CODE
from nexusml.constants import HTTP_GET_STATUS_CODE
from nexusml.constants import HTTP_PUT_STATUS_CODE
from nexusml.constants import HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
from nexusml.database.core import empty_table
from nexusml.database.core import save_to_db
from nexusml.database.examples import ExampleDB
from nexusml.database.files import TaskFileDB
from nexusml.database.myaccount import AccountClientSettings
from nexusml.database.myaccount import AccountSettings
from nexusml.database.notifications import AggregatedNotificationDB
from nexusml.database.notifications import NotificationDB
from nexusml.database.organizations import ClientDB
from nexusml.database.organizations import RoleDB
from nexusml.database.organizations import UserDB
from nexusml.database.permissions import RolePermission
from nexusml.database.permissions import UserPermission
from nexusml.database.tasks import TaskDB
from nexusml.enums import NotificationEvent
from nexusml.enums import NotificationSource
from nexusml.enums import ResourceAction
from nexusml.enums import ResourceType
from tests.api.integration.conftest import MockClient
from tests.api.integration.test_organizations import _get_role_json
from tests.api.integration.utils import get_endpoint
from tests.api.integration.utils import verify_response_jsons
from tests.api.utils import db_commit_and_expire
from tests.api.utils import get_json_from_db_object

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestMyAccount:

    def test_delete_last_admin(self, client: MockClient, session_user_id: str):
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_MYACCOUNT)
        response = client.send_request(method='DELETE', url=endpoint_url)
        assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
        assert response.json()['error']['message'] == 'There must be at least one admin user'
        db_commit_and_expire()
        assert UserDB.get_from_uuid(session_user_id) is not None

    def test_delete(self, mock_request_responses, mocker, client: MockClient, session_user_id: str):
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_MYACCOUNT)
        mocker.patch('nexusml.api.resources.organizations.get_user_roles', return_value=['not_admin'])

        response = client.send_request(method='DELETE', url=endpoint_url)

        assert response.status_code == HTTP_DELETE_STATUS_CODE
        db_commit_and_expire()
        assert UserDB.get_from_uuid(session_user_id) is None

    def test_get(self, mock_request_responses, client: MockClient, session_user_id: str):
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_MYACCOUNT)
        response = client.send_request(method='GET', url=endpoint_url)
        assert response.status_code == HTTP_GET_STATUS_CODE
        user = UserDB.get_from_uuid(session_user_id)
        user_auth0_data = {
            'email': 'test@testorg.com',
            'first_name': 'Test',
            'last_name': 'User',
        }
        assert response.json() == {
            'id': user.public_id,
            'uuid': user.uuid,
            'auth0_id': user.auth0_id,
            'email': user_auth0_data['email'],
            'first_name': user_auth0_data['first_name'],
            'last_name': user_auth0_data['last_name'],
            'email_verified': True
        }

    def test_put(self, mock_request_responses, client: MockClient):
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_MYACCOUNT)
        request_json: dict = {'first_name': 'new_f_name', 'last_name': 'new_l_name'}
        response = client.send_request(method='PUT', url=endpoint_url, json=request_json)
        assert response.status_code == HTTP_PUT_STATUS_CODE


class TestSettings:

    def test_get(self, client: MockClient):
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_MYACCOUNT_SETTINGS)
        # Make request and check response
        response = client.send_request(method='GET', url=endpoint_url)
        user_settings = AccountSettings.get(user_id=1)
        expected_data = {'notifications': user_settings.notifications.name.lower()}
        assert response.json() == expected_data

    def test_put(self, client: MockClient):
        # Make request and check response
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_MYACCOUNT_SETTINGS)
        new_settings = {'notifications': 'email'}
        response = client.send_request(method='PUT', url=endpoint_url, json=new_settings)
        expected_data = {**new_settings}
        assert response.status_code == HTTP_PUT_STATUS_CODE
        assert response.json() == expected_data
        db_commit_and_expire()
        user_settings = AccountSettings.get(user_id=1)
        user_settings = {'notifications': user_settings.notifications.name.lower()}
        assert user_settings == expected_data


class TestClientSettings:
    _CLIENT_VERSION = '0.1.0'

    def test_get(self, client: MockClient):
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_MYACCOUNT_SETTINGS)
        endpoint_url += f'/apps/{client.client_id}?version={self._CLIENT_VERSION}'
        response = client.send_request(method='GET', url=endpoint_url)
        target_client = ClientDB.get_from_uuid(client.client_id)
        user_client_settings = AccountClientSettings.get(user_id=1,
                                                         client_id=target_client.client_id,
                                                         client_version=self._CLIENT_VERSION)
        expected_data = {
            'client_id': client.client_id,
            'client_version': self._CLIENT_VERSION,
            'settings': user_client_settings.settings
        }
        assert response.json() == expected_data

    def test_put(self, client: MockClient):
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_MYACCOUNT_SETTINGS)
        endpoint_url += f'/apps/{client.client_id}?version={self._CLIENT_VERSION}'
        new_settings = {
            'client_version': self._CLIENT_VERSION,
            'settings': {
                'Parameter 1': 'New value 1',
                'Parameter 2': 'New value 2',
                'Parameter 3': 'New value 3'
            }
        }
        response = client.send_request(method='PUT', url=endpoint_url, json=new_settings)
        expected_data = {'client_id': client.client_id, **new_settings}
        assert response.json() == expected_data


class TestNotifications:

    def test_get(self, client: MockClient):
        _restore_notifications()
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_MYACCOUNT_NOTIFICATIONS)

        # Get all notifications
        response = client.send_request(method='GET', url=endpoint_url)
        assert response.status_code == HTTP_GET_STATUS_CODE
        all_user_notifications = NotificationDB.query().filter_by(recipient=1).all()
        all_user_notifications += AggregatedNotificationDB.query().filter_by(recipient=1).all()
        _verify_notifications(response_json=response.json(), expected_db_objects=all_user_notifications)

        # Filter by task
        response = client.send_request(method='GET', url=endpoint_url + f'?task_id={str(TaskDB.get(task_id=1).uuid)}')
        assert response.status_code == HTTP_GET_STATUS_CODE
        task_notifications = NotificationDB.filter_by_recipient_task(recipient=1, task_id=1)
        task_notifications += AggregatedNotificationDB.filter_by_recipient_task(recipient=1, task_id=1)
        _verify_notifications(response_json=response.json(), expected_db_objects=task_notifications)

        # Filter by source
        source_uuid = ExampleDB.get(example_id=1).uuid
        response = client.send_request(method='GET', url=endpoint_url + f'?source_uuid={str(source_uuid)}')
        assert response.status_code == HTTP_GET_STATUS_CODE
        source_notifications = NotificationDB.filter_by_recipient_source(recipient=1, source_uuid=source_uuid)
        _verify_notifications(response_json=response.json(), expected_db_objects=source_notifications)

    def test_delete(self, client: MockClient):
        _restore_notifications()
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_MYACCOUNT_NOTIFICATIONS)
        response = client.send_request(method='DELETE', url=endpoint_url)
        assert response.status_code == HTTP_DELETE_STATUS_CODE
        db_commit_and_expire()
        assert not NotificationDB.filter_by_recipient(recipient=1)
        assert not AggregatedNotificationDB.filter_by_recipient(recipient=1)


class TestNotification:

    def test_get(self, client: MockClient):
        _restore_notifications()
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_MYACCOUNT_NOTIFICATIONS)

        # Individual notification
        notification = NotificationDB.get(notification_id=1)
        response = client.send_request(method='GET', url=endpoint_url + '/' + str(notification.uuid))
        _verify_notifications(response_json=[response.json()], expected_db_objects=[notification])

        # Aggregated notification
        agg_notification = AggregatedNotificationDB.get(agg_notification_id=1)
        response = client.send_request(method='GET', url=endpoint_url + '/' + str(agg_notification.uuid))
        _verify_notifications(response_json=[response.json()], expected_db_objects=[agg_notification])

    def test_delete(self, client: MockClient):
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_MYACCOUNT_NOTIFICATIONS)

        # Individual notification
        _restore_notifications()
        notification = NotificationDB.get(notification_id=1)
        notification_uuid = notification.uuid
        response = client.send_request(method='DELETE', url=endpoint_url + '/' + str(notification.uuid))
        assert response.status_code == HTTP_DELETE_STATUS_CODE
        db_commit_and_expire()
        assert NotificationDB.get_from_uuid(notification_uuid) is None

        # Aggregated notification
        _restore_notifications()
        agg_notification = AggregatedNotificationDB.get(agg_notification_id=1)
        agg_notification_uuid = agg_notification.uuid
        response = client.send_request(method='DELETE', url=endpoint_url + '/' + str(agg_notification.uuid))
        assert response.status_code == HTTP_DELETE_STATUS_CODE
        db_commit_and_expire()
        assert AggregatedNotificationDB.get_from_uuid(agg_notification_uuid) is None


class TestOrganization:

    def test_get(self, client: MockClient, session_user_id: str):
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_MYACCOUNT_ORGANIZATION)
        response = client.send_request(method='GET', url=endpoint_url)
        assert response.status_code == HTTP_GET_STATUS_CODE
        user = UserDB.get_from_uuid(session_user_id)
        expected_json = get_json_from_db_object(db_object=user.organization)
        assert response.json() == expected_json


class TestRoles:

    def test_get(self, client: MockClient, session_user_id: str):
        user = UserDB.query().filter_by(uuid=session_user_id).first()
        # Make request
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_MYACCOUNT_ROLES)
        response = client.send_request(method='GET', url=endpoint_url)
        # Verify response
        assert response.status_code == HTTP_GET_STATUS_CODE
        user_roles = [_get_role_json(db_object=x) for x in user.roles]
        verify_response_jsons(actual_jsons=response.json()['roles'], expected_jsons=user_roles)


class TestPermissions:

    def test_get(self, client: MockClient, session_user_id: str):
        # Set user/role permissions
        user = UserDB.get_from_uuid(session_user_id)
        user.permissions.delete()
        same_org_task = TaskDB.query().filter(TaskDB.organization_id == user.organization_id).first()
        other_org_task = TaskDB.query().filter(TaskDB.organization_id != user.organization_id).first()
        user_perms = [
            # Permissions in session user's organization
            #    Generic permissions
            UserPermission(organization_id=user.organization_id,
                           user_id=user.user_id,
                           resource_type=ResourceType.TASK,
                           action=ResourceAction.CREATE,
                           allow=True),
            UserPermission(organization_id=user.organization_id,
                           user_id=user.user_id,
                           resource_type=ResourceType.TASK,
                           action=ResourceAction.DELETE,
                           allow=False),
            UserPermission(organization_id=user.organization_id,
                           user_id=user.user_id,
                           resource_type=ResourceType.FILE,
                           action=ResourceAction.READ,
                           allow=True),
            UserPermission(organization_id=user.organization_id,
                           user_id=user.user_id,
                           resource_type=ResourceType.AI_MODEL,
                           action=ResourceAction.DELETE,
                           allow=False),
            #    Resource-level permissions
            UserPermission(organization_id=user.organization_id,
                           user_id=user.user_id,
                           resource_uuid=same_org_task.uuid,
                           resource_type=ResourceType.TASK,
                           action=ResourceAction.UPDATE,
                           allow=True),
            # Permissions in another organization
            #    Generic permissions
            UserPermission(organization_id=other_org_task.organization_id,
                           user_id=user.user_id,
                           resource_type=ResourceType.FILE,
                           action=ResourceAction.CREATE,
                           allow=True),
            #    Resource-level permissions
            UserPermission(organization_id=other_org_task.organization_id,
                           user_id=user.user_id,
                           resource_uuid=other_org_task.uuid,
                           resource_type=ResourceType.TASK,
                           action=ResourceAction.READ,
                           allow=True)
        ]
        save_to_db(user_perms)
        role = RoleDB.get_from_id(id_value='Dasci', parent=user.organization)
        user.roles = [role]
        save_to_db(user)
        role.permissions.delete()
        role_perms = [
            # Generic permissions
            RolePermission(role_id=role.role_id,
                           resource_type=ResourceType.TASK,
                           action=ResourceAction.UPDATE,
                           allow=True),
            RolePermission(role_id=role.role_id,
                           resource_type=ResourceType.AI_MODEL,
                           action=ResourceAction.READ,
                           allow=True),
            # Resource-level permissions
            RolePermission(role_id=role.role_id,
                           resource_uuid=same_org_task.uuid,
                           resource_type=ResourceType.TASK,
                           action=ResourceAction.UPDATE,
                           allow=False),
            RolePermission(role_id=role.role_id,
                           resource_uuid=same_org_task.uuid,
                           resource_type=ResourceType.TASK,
                           action=ResourceAction.DELETE,
                           allow=True)
        ]
        save_to_db(role_perms)
        # Make request and verify response
        endpoint = get_endpoint(parameterized_endpoint=ENDPOINT_MYACCOUNT_PERMISSIONS)
        response = client.send_request(method='GET', url=endpoint + '?per_page=100')
        assert response.status_code == HTTP_GET_STATUS_CODE
        res_json = response.json()
        assert res_json['links']['previous'] is None
        assert res_json['links']['next'] is None
        expected_perms = [{
            'organization': user.organization.public_id,
            'resource_type': 'task',
            'action': 'create',
            'allow': True
        }, {
            'organization': user.organization.public_id,
            'resource_type': 'task',
            'action': 'update',
            'allow': True
        }, {
            'organization': user.organization.public_id,
            'resource_type': 'file',
            'action': 'read',
            'allow': True
        }, {
            'organization': user.organization.public_id,
            'resource_type': 'ai_model',
            'action': 'read',
            'allow': True
        }, {
            'organization': user.organization.public_id,
            'resource_uuid': same_org_task.uuid,
            'resource_type': 'task',
            'action': 'update',
            'allow': True
        }, {
            'organization': other_org_task.organization.public_id,
            'resource_type': 'file',
            'action': 'create',
            'allow': True
        }, {
            'organization': other_org_task.organization.public_id,
            'resource_uuid': other_org_task.uuid,
            'resource_type': 'task',
            'action': 'read',
            'allow': True
        }]
        actual_perms = res_json['data']
        assert len(actual_perms) == len(expected_perms)
        assert all(x in actual_perms for x in expected_perms)
        # Verify paging (default number of items per page for tests is 6)
        response = client.send_request(method='GET', url=endpoint)
        assert response.status_code == HTTP_GET_STATUS_CODE
        res_json = response.json()
        assert res_json['links']['previous'] is None
        assert res_json['links']['next'] is not None


def _restore_notifications():
    empty_table(NotificationDB)
    empty_table(AggregatedNotificationDB)

    api_url = config.get('server')['api_url']

    # Notification 1
    task_uuid = str(TaskDB.get(task_id=1).uuid)
    task_url = API_DOMAIN + api_url + ENDPOINT_TASK.replace('<task_id>', task_uuid)

    notification_1 = NotificationDB(notification_id=1,
                                    task_id=1,
                                    recipient=1,
                                    source_type=NotificationSource.TASK,
                                    event=NotificationEvent.UPDATE,
                                    created_at=datetime.utcnow(),
                                    source_uuid=task_uuid,
                                    source_url=task_url)

    next_dt = notification_1.created_at + timedelta(0, 3)

    # Notification 2
    example_uuid = str(ExampleDB.query().filter_by(task_id=1).first().uuid)
    example_url = (API_DOMAIN + api_url +
                   ENDPOINT_EXAMPLE.replace('<task_id>', task_uuid).replace('<example_id>', example_uuid))

    notification_2 = NotificationDB(notification_id=2,
                                    task_id=1,
                                    recipient=1,
                                    source_type=NotificationSource.EXAMPLE,
                                    event=NotificationEvent.UPDATE,
                                    created_at=next_dt,
                                    source_uuid=example_uuid,
                                    source_url=example_url)

    # Notification 3
    file_uuid = str(TaskFileDB.query().filter_by(task_id=1).first().uuid)
    file_url = (API_DOMAIN + api_url +
                ENDPOINT_TASK_FILE.replace('<task_id>', task_uuid).replace('<file_id>', example_uuid))

    notification_3 = NotificationDB(notification_id=3,
                                    task_id=1,
                                    recipient=5,
                                    source_type=NotificationSource.FILE,
                                    event=NotificationEvent.CREATION,
                                    created_at=next_dt,
                                    source_uuid=file_uuid,
                                    source_url=file_url)

    # Aggregated notification 1
    agg_notification_1 = AggregatedNotificationDB(agg_notification_id=1,
                                                  recipient=1,
                                                  source_type=NotificationSource.EXAMPLE,
                                                  event=NotificationEvent.CREATION,
                                                  since=datetime.utcnow(),
                                                  count=65)

    next_dt = notification_1.created_at + timedelta(0, 7)

    # Aggregated notification 2
    agg_notification_2 = AggregatedNotificationDB(agg_notification_id=2,
                                                  recipient=1,
                                                  source_type=NotificationSource.TAG,
                                                  event=NotificationEvent.DELETION,
                                                  since=next_dt,
                                                  count=8)

    # Save all notifications
    save_to_db(notification_1)
    save_to_db(notification_2)
    save_to_db(notification_3)
    save_to_db(agg_notification_1)
    save_to_db(agg_notification_2)


def _verify_notifications(response_json: Iterable[dict],
                          expected_db_objects: Iterable[Union[NotificationDB, AggregatedNotificationDB]]):

    expected_jsons = []

    for expected_db_object in expected_db_objects:
        # Get notification JSON
        notification_json = get_json_from_db_object(db_object=expected_db_object)

        # Remove private fields
        notification_json.pop('id')
        notification_json.pop('status')
        notification_json.pop('read')
        if 'source_uuid' in notification_json:
            notification_json.pop('source_uuid')  # source UUID is used internally for filtering purposes
        if isinstance(expected_db_object, NotificationDB):
            notification_json['datetime'] = notification_json.pop('created_at')

        # Add notification JSON
        expected_jsons.append(notification_json)

    verify_response_jsons(actual_jsons=response_json,
                          expected_jsons=expected_jsons,
                          optional_fields={'recipient', 'type'})
