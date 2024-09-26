from datetime import datetime
import os
from typing import List

import pytest

from nexusml.api.resources.examples import Example
from nexusml.api.resources.tasks import Task
from nexusml.api.utils import API_DOMAIN
from nexusml.api.utils import config
from nexusml.api.views.core import AUTH0_REQUIRED_ERR_MSG
from nexusml.constants import ENDPOINT_CLIENTS
from nexusml.constants import ENDPOINT_COLLABORATOR
from nexusml.constants import ENDPOINT_COLLABORATORS
from nexusml.constants import ENDPOINT_EXAMPLE
from nexusml.constants import ENDPOINT_MYACCOUNT
from nexusml.constants import ENDPOINT_MYACCOUNT_CLIENT_SETTINGS
from nexusml.constants import ENDPOINT_MYACCOUNT_NOTIFICATIONS
from nexusml.constants import ENDPOINT_MYACCOUNT_ORGANIZATION
from nexusml.constants import ENDPOINT_MYACCOUNT_PERMISSIONS
from nexusml.constants import ENDPOINT_MYACCOUNT_SETTINGS
from nexusml.constants import ENDPOINT_ORGANIZATIONS
from nexusml.constants import ENDPOINT_ROLE
from nexusml.constants import ENDPOINT_ROLES
from nexusml.constants import ENDPOINT_TASK
from nexusml.constants import ENDPOINT_TASKS
from nexusml.constants import ENDPOINT_USER
from nexusml.constants import ENDPOINT_USER_INVITE
from nexusml.constants import HTTP_DELETE_STATUS_CODE
from nexusml.constants import HTTP_FORBIDDEN_STATUS_CODE
from nexusml.constants import HTTP_GET_STATUS_CODE
from nexusml.constants import HTTP_POST_STATUS_CODE
from nexusml.constants import NUM_RESERVED_CLIENTS
from nexusml.database.core import delete_from_db
from nexusml.database.core import save_to_db
from nexusml.database.notifications import NotificationDB
from nexusml.database.organizations import ClientDB
from nexusml.database.organizations import CollaboratorDB
from nexusml.database.organizations import OrganizationDB
from nexusml.database.organizations import RoleDB
from nexusml.database.organizations import UserDB
from nexusml.database.tasks import TaskDB
from nexusml.enums import NotificationEvent
from nexusml.enums import NotificationSource
from nexusml.env import ENV_NOTIFICATION_EMAIL
from tests.api.integration.utils import get_endpoint
from tests.api.integration.utils import mock_element_values_json
from tests.api.integration.utils import send_request
from tests.api.integration.utils import verify_response_examples_or_prediction_logs
from tests.api.utils import db_commit_and_expire
from tests.api.utils import load_default_resource

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_CLIENT_ID = NUM_RESERVED_CLIENTS + 2  # First client IDs are reserved for official apps


class TestAPIKey:

    def test_delete(self):
        example = load_default_resource(resource_type=Example, parents_types=[Task])
        example_uuid = example.uuid()
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_EXAMPLE, resource=example)
        response = send_request(method='DELETE', url=endpoint_url, api_key=generate_api_key())
        assert response.status_code == HTTP_DELETE_STATUS_CODE
        db_commit_and_expire()
        assert Example.db_model().get_from_uuid(example_uuid) is None

    def test_get(self):
        example = load_default_resource(resource_type=Example, parents_types=[Task])
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_EXAMPLE, resource=example)
        response = send_request(method='GET', url=endpoint_url, api_key=generate_api_key())
        assert response.status_code == HTTP_GET_STATUS_CODE
        verify_response_examples_or_prediction_logs(actual_jsons=[response.json()],
                                                    expected_db_objects=[example.db_object()])

        # Check that tasks are returned with api key only if they belong to the organization of the client
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_TASKS)
        response = send_request(method='GET', url=endpoint_url, api_key=generate_api_key())
        client = ClientDB.get(client_id=_CLIENT_ID)
        org_tasks = TaskDB.query().filter_by(organization_id=client.organization_id).all()
        assert response.status_code == HTTP_GET_STATUS_CODE
        assert len(response.json()) == len(org_tasks)

    def test_post(self):
        task = load_default_resource(resource_type=Task)
        endpoint = get_endpoint(parameterized_endpoint=ENDPOINT_TASK, resource=task)
        response = send_request(method='POST',
                                url=(endpoint + '/examples'),
                                api_key=generate_api_key(),
                                json={'batch': [mock_element_values_json(task_id=task.db_object().task_id)]})
        assert response.status_code == HTTP_POST_STATUS_CODE
        pass  # TODO: verify the new example was created in the database

    def test_put(self):
        example = load_default_resource(resource_type=Example, parents_types=[Task])
        example.db_object().synced_by_clients = [_CLIENT_ID]
        example.persist()
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_EXAMPLE, resource=example)
        response = send_request(method='PUT',
                                url=endpoint_url,
                                api_key=generate_api_key(),
                                json=mock_element_values_json(task_id=example.db_object().task_id))
        assert response.status_code == HTTP_GET_STATUS_CODE
        verify_response_examples_or_prediction_logs(actual_jsons=[response.json()],
                                                    expected_db_objects=[example.db_object()])

    def test_unsupported_endpoints(self, session_user_id: str):

        def _test_method(method: str, url: str, api_key: str, json: dict = None):
            response = send_request(method=method, url=url, api_key=api_key, json=json)
            assert response.status_code == HTTP_FORBIDDEN_STATUS_CODE
            assert response.json()['error']['message'] == AUTH0_REQUIRED_ERR_MSG

        client = ClientDB.get(client_id=_CLIENT_ID)
        api_url = API_DOMAIN + config.get('server')['api_url']
        api_key = generate_api_key()

        ##############
        # My Account #
        ##############

        # DELETE|GET /myaccount
        myaccount_url = api_url + ENDPOINT_MYACCOUNT
        _test_method(method='DELETE', url=myaccount_url, api_key=api_key)
        _test_method(method='GET', url=myaccount_url, api_key=api_key)

        # DELETE|GET /myaccount/notifications
        notifications_url = api_url + ENDPOINT_MYACCOUNT_NOTIFICATIONS
        _test_method(method='DELETE', url=notifications_url, api_key=api_key)
        _test_method(method='GET', url=notifications_url, api_key=api_key)

        # DELETE|GET /myaccount/notifications/<notification_id>
        source_uuid = str(Task.db_model().get(task_id=1).uuid)
        source_url = api_url + ENDPOINT_TASK.replace('<task_id>', source_uuid)
        notification = NotificationDB(notification_id=1,
                                      task_id=1,
                                      recipient=1,
                                      source_type=NotificationSource.TASK,
                                      event=NotificationEvent.UPDATE,
                                      created_at=datetime.utcnow(),
                                      source_uuid=source_uuid,
                                      source_url=source_url)
        save_to_db(notification)
        notification_url = api_url + ENDPOINT_MYACCOUNT_NOTIFICATIONS + '/' + str(notification.uuid)
        _test_method(method='DELETE', url=notification_url, api_key=api_key)
        _test_method(method='GET', url=notification_url, api_key=api_key)

        # GET /myaccount/organization
        myorg_url = api_url + ENDPOINT_MYACCOUNT_ORGANIZATION
        _test_method(method='GET', url=myorg_url, api_key=api_key)

        # GET /myaccount/permissions
        myperms_url = api_url + ENDPOINT_MYACCOUNT_PERMISSIONS
        _test_method(method='GET', url=myperms_url, api_key=api_key)

        # GET|PUT /myaccount/settings
        mysettings_url = api_url + ENDPOINT_MYACCOUNT_SETTINGS
        new_settings = {'notifications': 'email'}
        _test_method(method='GET', url=mysettings_url, api_key=api_key)
        _test_method(method='PUT', url=mysettings_url, api_key=api_key, json=new_settings)

        # GET|PUT /myaccount/settings/apps/<client_id>
        myapp_url = (api_url + ENDPOINT_MYACCOUNT_CLIENT_SETTINGS.replace('<client_id>', str(client.uuid)))
        new_settings = {'param_1': 'value_1'}
        _test_method(method='GET', url=myapp_url, api_key=api_key)
        _test_method(method='PUT', url=myapp_url, api_key=api_key, json=new_settings)

        #################
        # Organizations #
        #################

        # POST /organizations
        orgs_url = api_url + ENDPOINT_ORGANIZATIONS
        session_user = UserDB.get_from_uuid(session_user_id)
        delete_from_db(session_user)
        new_org = {'trn': 'test_trn', 'name': 'test_name', 'domain': 'org2.com', 'address': 'test_address'}
        _test_method(method='POST', url=orgs_url, api_key=api_key, json=new_org)


def generate_api_key(scopes: List[str] = None, expire_at: datetime = None) -> str:
    return ClientDB.get(client_id=_CLIENT_ID).update_api_key(scopes=scopes, expire_at=expire_at)
