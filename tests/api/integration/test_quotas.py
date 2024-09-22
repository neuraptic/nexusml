from unittest.mock import MagicMock

import jwt
import pytest

from nexusml.api.resources.organizations import Organization
from nexusml.constants import ADMIN_ROLE
from nexusml.constants import ENDPOINT_MYACCOUNT
from nexusml.constants import ENDPOINT_ORGANIZATION
from nexusml.constants import ENDPOINT_TASKS
from nexusml.constants import HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
from nexusml.database.core import save_to_db
from nexusml.database.organizations import ClientDB
from nexusml.database.organizations import CollaboratorDB
from nexusml.database.organizations import OrganizationDB
from nexusml.database.organizations import RoleDB
from nexusml.database.organizations import user_roles
from nexusml.database.organizations import UserDB
from nexusml.database.services import Service
from nexusml.database.subscriptions import get_active_subscription
from nexusml.database.subscriptions import quotas
from nexusml.database.tasks import TaskDB
from nexusml.enums import ServiceType
from nexusml.statuses import inference_waiting_status
from nexusml.statuses import Status
from tests.api.integration.conftest import MockClient
from tests.api.integration.test_ai import mock_inference_request_json
from tests.api.integration.utils import get_endpoint
from tests.api.utils import db_commit_and_expire

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_ORG_ID = 2


class TestQuotas:

    def test_tasks_limit(self, client: MockClient):
        request_json = {'name': 'new task name', 'description': 'new task description'}
        _test_org_quota_limit(client=client, quota='tasks', request_json=request_json)

    def test_deployments_limit(self):
        pass  # TODO: add tests when AI deployment limit is implemented

    def test_predictions_limit(self, client: MockClient):
        task = TaskDB.query().filter_by(organization_id=_ORG_ID).first()

        inference_service = Service.filter_by_task_and_type(task_id=task.task_id, type_=ServiceType.INFERENCE)
        inference_service.set_status(status=Status(template=inference_waiting_status))

        request_json = {'batch': mock_inference_request_json(task_id=1)}

        _test_task_quota_limit(client=client, quota='predictions', request_json=request_json)

    def test_gpu_hours_limit(self, client: MockClient):
        pass  # TODO: uncomment line below
        # _test_task_quota_limit(client=client, quota='gpu', request_json=dict())

    def test_cpu_hours_limit(self, client: MockClient):
        pass  # TODO: uncomment line below
        # _test_task_quota_limit(client=client, quota='cpu', request_json=dict())

    def test_space_limit(self):
        """ Space quota limit is already tested in `test_files` and `test_examples`. """
        pass

    def test_users_limit(self, mocker, mock_request_responses, client: MockClient):
        mocker.patch('nexusml.api.views.myaccount.agent_from_token', side_effect=jwt.InvalidTokenError())
        mocker.patch('nexusml.api.views.myaccount.get_auth0_management_api_token', return_value=MagicMock())
        mock_user_invitation = MagicMock()
        mock_user_invitation.organization_id = 1
        mocker.patch('nexusml.api.views.myaccount.MyAccountView._get_user_invitation', return_value=MagicMock())

        user_db_obj: UserDB = UserDB.query().join(user_roles, user_roles.c.user_id == UserDB.user_id).join(
            RoleDB, user_roles.c.role_id == RoleDB.role_id).filter(RoleDB.name == ADMIN_ROLE,).first()
        mock_query = MagicMock()
        mock_query.join.return_value.join.return_value.filter.return_value.first.return_value = user_db_obj
        mocker.patch('nexusml.api.views.myaccount.UserDB.query', return_value=mock_query)
        mocker.patch('nexusml.api.resources.organizations.User.download_auth0_user_data',
                     side_effect=lambda auth0_id_or_email: None if '@' in str(auth0_id_or_email) else MagicMock())

        _test_org_quota_limit(client=client, quota='users', request_json={'email': 'extra_user@testorg.com'})

    def test_roles_limit(self, client: MockClient):
        request_json = {'name': 'new role name', 'description': 'new role description'}
        _test_org_quota_limit(client=client, quota='roles', request_json=request_json)

    def test_collaborators_limit(self, client: MockClient):
        _test_org_quota_limit(client=client, quota='collaborators', request_json={'email': 'extra_user@testorg.com'})

    def test_clients_limit(self, client: MockClient):
        request_json = {'name': 'new app name', 'description': 'new app description'}
        _test_org_quota_limit(client=client, quota='clients', request_json=request_json)


def _test_org_quota_limit(client: MockClient, quota: str, request_json: dict):
    assert quota in ['tasks', 'users', 'roles', 'collaborators', 'clients']
    dsp_name = quota if quota != 'clients' else 'apps'

    org_db_object = OrganizationDB.get(organization_id=_ORG_ID)
    account_email = 'extra_user@testorg.com'
    org_db_object.domain = account_email.split('@')[-1]
    save_to_db(org_db_object)

    db_models = {
        'tasks': TaskDB,
        'users': UserDB,
        'roles': RoleDB,
        'collaborators': CollaboratorDB,
        'clients': ClientDB
    }
    db_model = db_models[quota]
    num_objects = len(db_model.query().filter_by(organization_id=_ORG_ID).all())

    # Set database
    subscription = get_active_subscription(organization_id=_ORG_ID)
    quota_limit = getattr(subscription.plan, quotas[quota]['limit'])
    setattr(subscription, quotas[quota]['usage'], quota_limit)
    subscription.extras.clear()
    save_to_db(subscription)

    # Make request
    if quota == 'tasks':
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_TASKS)
    else:
        org = Organization.get(agent=UserDB.get(user_id=1), db_object_or_id=org_db_object)
        org_endpoint = get_endpoint(parameterized_endpoint=ENDPOINT_ORGANIZATION, resource=org)
        endpoint_url = org_endpoint + '/' + dsp_name

    if dsp_name == 'users':
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_MYACCOUNT)
        response = client.send_request(method='GET', url=endpoint_url, json=request_json)
    else:
        response = client.send_request(method='POST', url=endpoint_url, json=request_json)

    # Verify response and database
    assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
    assert response.json()['error']['message'] == f'Maximum number of {dsp_name} ({quota_limit}) exceeded'
    db_commit_and_expire()
    assert len(db_model.query().filter_by(organization_id=_ORG_ID).all()) == num_objects


def _test_task_quota_limit(client: MockClient, quota: str, request_json: dict):
    # Note: examples and space quota limits are already tested in `test_examples` and `test_files`
    assert quota in ['predictions', 'cpu', 'gpu']

    err_msgs = {'predictions': 'Maximum number of predictions', 'cpu': 'Maximum CPU hours', 'gpu': 'Maximum GPU hours'}

    subscription = get_active_subscription(organization_id=_ORG_ID)
    task = TaskDB.query().filter_by(organization_id=_ORG_ID).first()

    endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_TASKS) + '/' + task.public_id
    if quota == 'predictions':
        endpoint_url += '/predict'
    else:
        endpoint_url += '/train'
    ####################################
    # Try to exceed organization quota #
    ####################################
    # Set database
    quota_limit = getattr(subscription.plan, quotas[quota]['limit'])
    setattr(subscription, quotas[quota]['usage'], quota_limit)
    subscription.extras.clear()
    save_to_db(subscription)

    # Verify the request is rejected
    response = client.send_request(method='POST', url=endpoint_url, json=request_json)
    assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
    assert response.json()['error']['message'] == f'{err_msgs[quota]} ({quota_limit}) exceeded'
    db_commit_and_expire()
    assert getattr(subscription, quotas[quota]['usage']) == quota_limit
    ############################
    # Try to exceed task quota #
    ############################
    # Set database
    setattr(subscription, quotas[quota]['usage'], 0)
    quota_limit = getattr(task, quotas[quota]['limit'])
    setattr(task, quotas[quota]['usage'], quota_limit)
    save_to_db([subscription, task])

    # Verify the request is rejected
    response = client.send_request(method='POST', url=endpoint_url, json=request_json)
    assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
    assert response.json()['error']['message'] == f'{err_msgs[quota]} ({quota_limit}) exceeded'
    db_commit_and_expire()
    assert getattr(subscription, quotas[quota]['usage']) == 0
    assert getattr(task, quotas[quota]['usage']) == quota_limit
