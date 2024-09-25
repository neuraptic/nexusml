from datetime import datetime
from datetime import timedelta
import re
from typing import List, Tuple, Union

from flask import Response
import pytest

from nexusml.api.ext import cache
from nexusml.api.resources import ResourceNotFoundError
from nexusml.api.resources.organizations import Client
from nexusml.api.resources.organizations import Collaborator
from nexusml.api.resources.organizations import Organization
from nexusml.api.resources.organizations import User
from nexusml.api.utils import config
from nexusml.api.utils import decode_api_key
from nexusml.api.views.organizations import _invite_user
from nexusml.constants import ADMIN_ROLE
from nexusml.constants import DATETIME_FORMAT
from nexusml.constants import ENDPOINT_CLIENT
from nexusml.constants import ENDPOINT_CLIENT_API_KEY
from nexusml.constants import ENDPOINT_CLIENTS
from nexusml.constants import ENDPOINT_COLLABORATOR
from nexusml.constants import ENDPOINT_COLLABORATOR_PERMISSIONS
from nexusml.constants import ENDPOINT_COLLABORATORS
from nexusml.constants import ENDPOINT_ORGANIZATION
from nexusml.constants import ENDPOINT_ORGANIZATIONS
from nexusml.constants import ENDPOINT_ROLE
from nexusml.constants import ENDPOINT_ROLE_PERMISSIONS
from nexusml.constants import ENDPOINT_ROLE_USERS
from nexusml.constants import ENDPOINT_ROLES
from nexusml.constants import ENDPOINT_SUBSCRIPTION
from nexusml.constants import ENDPOINT_USER
from nexusml.constants import ENDPOINT_USER_INVITE
from nexusml.constants import ENDPOINT_USER_PERMISSIONS
from nexusml.constants import ENDPOINT_USER_ROLES
from nexusml.constants import ENDPOINT_USERS
from nexusml.constants import HTTP_BAD_REQUEST_STATUS_CODE
from nexusml.constants import HTTP_CONFLICT_STATUS_CODE
from nexusml.constants import HTTP_DELETE_STATUS_CODE
from nexusml.constants import HTTP_FORBIDDEN_STATUS_CODE
from nexusml.constants import HTTP_GET_STATUS_CODE
from nexusml.constants import HTTP_NOT_FOUND_STATUS_CODE
from nexusml.constants import HTTP_POST_STATUS_CODE
from nexusml.constants import HTTP_PUT_STATUS_CODE
from nexusml.constants import HTTP_SERVICE_UNAVAILABLE
from nexusml.constants import HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
from nexusml.constants import MAINTAINER_ROLE
from nexusml.constants import NULL_UUID
from nexusml.constants import NUM_RESERVED_CLIENTS
from nexusml.database.core import db_query
from nexusml.database.core import delete_from_db
from nexusml.database.core import save_to_db
from nexusml.database.files import OrgFileDB as FileDB
from nexusml.database.organizations import client_scopes
from nexusml.database.organizations import ClientDB
from nexusml.database.organizations import CollaboratorDB
from nexusml.database.organizations import OrganizationDB
from nexusml.database.organizations import RoleDB
from nexusml.database.organizations import user_roles as user_roles_table
from nexusml.database.organizations import UserDB
from nexusml.database.organizations import WaitList
from nexusml.database.permissions import RolePermission
from nexusml.database.permissions import UserPermission
from nexusml.database.services import Service
from nexusml.database.subscriptions import get_active_subscription
from nexusml.database.subscriptions import SubscriptionDB
from nexusml.database.tasks import TaskDB
from nexusml.enums import OrgFileUse
from nexusml.enums import ResourceAction
from nexusml.enums import ResourceType
from tests.api.conftest import restore_db
from tests.api.constants import BACKEND_JSON_FIELDS
from tests.api.integration.conftest import MockClient
from tests.api.integration.utils import get_endpoint
from tests.api.integration.utils import verify_resource_request
from tests.api.integration.utils import verify_response_json
from tests.api.integration.utils import verify_response_json_with_picture
from tests.api.integration.utils import verify_response_jsons
from tests.api.utils import db_commit_and_expire
from tests.api.utils import get_json_from_db_object
from tests.api.utils import get_role
from tests.api.utils import get_user

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_ORG_ID = 2  # 1 is reserved for the main organization
_USER_ID = 2  # 1 is session user in tests
_CLIENT_ID = NUM_RESERVED_CLIENTS + 2  # First client IDs are reserved for official apps


class TestOrganizations:

    def test_post(self,
                  mock_request_responses,
                  client: MockClient,
                  mock_client_id: str,
                  session_user_id: str,
                  session_user_auth0_id: str):
        """
        Valid request
        """
        # Delete session user from database to be able to create a new organization
        session_user = UserDB.query().filter_by(auth0_id=session_user_auth0_id).first()
        delete_from_db(session_user)

        # Make request and verify response
        valid_org_data = {'trn': 'test_trn', 'name': 'test_name', 'domain': 'testorg.com', 'address': 'test_address'}
        response = client.send_request(method='POST',
                                       url=get_endpoint(parameterized_endpoint=ENDPOINT_ORGANIZATIONS),
                                       json=valid_org_data)
        assert response.status_code == HTTP_POST_STATUS_CODE
        res_json = response.json()
        optional_fields = BACKEND_JSON_FIELDS.union({'logo'})
        verify_response_json(actual_json=res_json, expected_json=valid_org_data, optional_fields=optional_fields)

        # Check database
        db_commit_and_expire()
        session_user = UserDB.query().filter_by(auth0_id=session_user_auth0_id).first()
        assert session_user is not None
        user_roles = db_query(user_roles_table).filter(user_roles_table.c.user_id == session_user.user_id).all()
        assert len(user_roles) == 1  # the user creating the organization becomes the admin
        user_role = user_roles[0]
        assert RoleDB.get(role_id=user_role.role_id).name == ADMIN_ROLE

        # Check subscription to the Free Plan
        org_id = OrganizationDB.get_from_uuid(res_json['uuid']).organization_id
        org_subscriptions = SubscriptionDB.query().filter_by(plan_id=1, organization_id=org_id).all()
        assert len(org_subscriptions) == 1
        org_subscription = org_subscriptions[0]
        assert org_subscription.num_tasks == 0
        assert org_subscription.num_deployments == 0
        assert org_subscription.num_predictions == 0
        assert org_subscription.num_gpu_hours == 0
        assert org_subscription.num_cpu_hours == 0
        assert org_subscription.space_usage == 0
        assert org_subscription.num_users == 1
        assert org_subscription.num_roles == 0
        assert org_subscription.num_collaborators == 0
        assert org_subscription.num_clients == 0

        # Check demo tasks' copies
        # Note: demo tasks are populated asynchronously with a Celery task. Only the schema is copied upon the request.
        pass  # TODO
        ####################################################
        # Try to create another organization (not allowed) #
        ####################################################
        org_trn = 'test_trn_2'
        other_org_data = {
            'trn': org_trn,
            'name': 'test_name_2',
            'domain': 'testdomain2.com',
            'address': 'test_address_2'
        }
        response = client.send_request(method='POST',
                                       url=get_endpoint(parameterized_endpoint=ENDPOINT_ORGANIZATIONS),
                                       json=other_org_data)
        assert response.status_code == HTTP_CONFLICT_STATUS_CODE
        assert response.json()['error']['message'] == 'You already belong to another organization'
        db_commit_and_expire()
        assert OrganizationDB.get_from_id(id_value=org_trn) is None
        #################
        # Duplicate TRN #
        #################
        delete_from_db(UserDB.query().filter_by(auth0_id=session_user_auth0_id).first())
        assert UserDB.query().filter_by(auth0_id=session_user_auth0_id).first() is None
        duplicate_trn = 'org_2_trn'
        assert OrganizationDB.get_from_id(id_value=duplicate_trn) is not None
        invalid_org_data = {
            'trn': duplicate_trn,
            'name': 'test_name_2',
            'domain': 'testdomain2.com',
            'address': 'test_address_2'
        }
        response = client.send_request(method='POST',
                                       url=get_endpoint(parameterized_endpoint=ENDPOINT_ORGANIZATIONS),
                                       json=invalid_org_data)
        assert response.status_code == HTTP_CONFLICT_STATUS_CODE
        assert response.json()['error']['message'] == f'Organization "{duplicate_trn}" already exists'
        ###############################################
        # Domain mismatch (user's and organization's) #
        ###############################################
        org_trn = 'test_trn_2'
        assert UserDB.query().filter_by(auth0_id=session_user_auth0_id).first() is None
        invalid_org_data = {
            'trn': org_trn,
            'name': 'test_name_2',
            'domain': 'invaliddomain.com',
            'address': 'test_address_2'
        }
        response = client.send_request(method='POST',
                                       url=get_endpoint(parameterized_endpoint=ENDPOINT_ORGANIZATIONS),
                                       json=invalid_org_data)
        assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
        assert response.json()['error']['message'] == "Domains don't match"
        db_commit_and_expire()
        assert OrganizationDB.get_from_id(id_value=org_trn) is None
        #####################################################
        # Try to exceed the maximum number of organizations #
        #####################################################
        restore_db(mock_client_id=mock_client_id,
                   session_user_id=session_user_id,
                   session_user_auth0_id=session_user_auth0_id)

        init_waitlist = WaitList.query().all()
        oldest_waitlist_entry = WaitList.query().order_by(WaitList.id_).first()
        assert oldest_waitlist_entry.id_ == min(x.id_ for x in init_waitlist)

        new_config = config.get()
        new_config['limits']['organizations']['num_organizations'] = 3
        new_config['limits']['organizations']['waitlist'] = len(init_waitlist)
        config.set(new_config)

        response = client.send_request(method='POST',
                                       url=get_endpoint(parameterized_endpoint=ENDPOINT_ORGANIZATIONS),
                                       json=valid_org_data)
        assert response.status_code == HTTP_SERVICE_UNAVAILABLE
        err_msg = ('System capacity exceeded due to unexpectedly high demand. '
                   'We are working hard to scale our system to better serve you. '
                   'In the meantime, you have been added to our wait list and '
                   'will be notified as soon as we can accommodate your request.')
        assert response.json()['error']['message'] == err_msg

        db_commit_and_expire()
        assert OrganizationDB.query().count() == new_config['limits']['organizations']['num_organizations']
        new_waitlist = WaitList.query().all()
        assert oldest_waitlist_entry not in new_waitlist
        assert len(new_waitlist) == len(init_waitlist) == new_config['limits']['organizations']['waitlist']


class TestOrganization:

    @staticmethod
    def _test_org_method(client: MockClient,
                         method: str,
                         request_json: dict = None,
                         expected_status_code: int = None) -> Response:

        method = method.lower().strip()

        org_db_object = OrganizationDB.get(organization_id=_ORG_ID)
        org = Organization.get(agent=UserDB.get(user_id=1), db_object_or_id=org_db_object)

        if method == 'delete':
            expected_jsons = None
        elif request_json:
            expected_json = dict(request_json)
            expected_json.pop('logo', None)  # the logo is verified outside
            expected_jsons = [expected_json]
        else:
            expected_org_json = get_json_from_db_object(db_object=org.db_object())
            expected_jsons = [expected_org_json]

        return verify_resource_request(client=client,
                                       method=method,
                                       endpoint=ENDPOINT_ORGANIZATION,
                                       resource=org,
                                       request_json=request_json,
                                       expected_jsons=expected_jsons,
                                       expected_status_code=expected_status_code)

    def test_delete(self,
                    client: MockClient,
                    mock_s3,
                    mock_client_id: str,
                    session_user_id: str,
                    session_user_auth0_id: str):
        """
        Valid request
        """
        self._test_org_method(client=client, method='DELETE')
        ###################################################################
        # Invalid request: try to delete the Organization as a Maintainer #
        ###################################################################
        restore_db(mock_client_id=mock_client_id,
                   session_user_id=session_user_id,
                   session_user_auth0_id=session_user_auth0_id)
        cache.clear()
        session_user = get_user(user_id=1)
        maintainer_role = RoleDB.get_from_id(id_value=MAINTAINER_ROLE, parent=session_user.db_object().organization)
        session_user.db_object().roles = [maintainer_role]
        save_to_db(session_user.db_object())
        self._test_org_method(client=client, method='DELETE', expected_status_code=HTTP_FORBIDDEN_STATUS_CODE)

    def test_get(self, client: MockClient, mock_s3):
        self._test_org_method(client=client, method='GET')

    def test_put(self,
                 client: MockClient,
                 mock_s3,
                 mock_client_id: str,
                 session_user_id: str,
                 session_user_auth0_id: str):
        #################
        # Valid request #
        #################
        # Save logo file database object
        logo_file = FileDB(organization_id=_ORG_ID, filename='org_logo', size='30', use_for=OrgFileUse.PICTURE)
        save_to_db(logo_file)
        # Make request
        org_data = {
            'trn': 'modified_trn',
            'name': 'modified_name',
            'address': 'modified_address',
            'logo': logo_file.public_id
        }
        response = self._test_org_method(client=client, method='PUT', request_json=org_data)
        # Check returned logo file
        res_logo_file = dict(response.json()['logo'])
        res_logo_file.pop('upload_url')  # the file was not uploaded to S3
        res_logo_file.pop('created_by')
        expected_logo_file_json = get_json_from_db_object(db_object=logo_file,
                                                          ignore_columns=['format_', 'type_', 'use_for'])
        assert res_logo_file == expected_logo_file_json
        ###############################################
        # Invalid request: try to use an existing TRN #
        ###############################################
        restore_db(mock_client_id=mock_client_id,
                   session_user_id=session_user_id,
                   session_user_auth0_id=session_user_auth0_id)
        org_data = {'trn': 'org_3_trn', 'name': 'modified_name', 'address': 'modified_address'}
        response = self._test_org_method(client=client,
                                         method='PUT',
                                         request_json=org_data,
                                         expected_status_code=HTTP_CONFLICT_STATUS_CODE)
        assert response.json()['error']['message'] == 'TRN already in use by another organization'
        #########################################
        # Invalid request: try to modify domain #
        #########################################
        restore_db(mock_client_id=mock_client_id,
                   session_user_id=session_user_id,
                   session_user_auth0_id=session_user_auth0_id)
        org_data = {
            'trn': 'modified_trn',
            'name': 'modified_name',
            'domain': 'modifieddomain.com',
            'address': 'modified_address'
        }
        self._test_org_method(client=client,
                              method='PUT',
                              request_json=org_data,
                              expected_status_code=HTTP_BAD_REQUEST_STATUS_CODE)
        ###################################################################
        # Invalid request: try to modify the Organization as a Maintainer #
        ###################################################################
        restore_db(mock_client_id=mock_client_id,
                   session_user_id=session_user_id,
                   session_user_auth0_id=session_user_auth0_id)
        cache.clear()
        session_user = get_user(user_id=1)
        maintainer_role = RoleDB.get_from_id(id_value=MAINTAINER_ROLE, parent=session_user.db_object().organization)
        session_user.db_object().roles = [maintainer_role]
        save_to_db(session_user.db_object())
        org_data = {'trn': 'modified_trn', 'name': 'modified_name', 'address': 'modified_address'}
        self._test_org_method(client=client,
                              method='PUT',
                              request_json=org_data,
                              expected_status_code=HTTP_FORBIDDEN_STATUS_CODE)


class TestSubscription:

    def test_get(self, client: MockClient):
        """
        One subscription, one discount, and one extra
        """
        org_db_object = OrganizationDB.get(organization_id=_ORG_ID)
        org = Organization.get(agent=UserDB.get(user_id=1), db_object_or_id=org_db_object)
        endpoint = get_endpoint(parameterized_endpoint=ENDPOINT_SUBSCRIPTION, resource=org)
        response = client.send_request(method='GET', url=endpoint)
        assert response.status_code == HTTP_GET_STATUS_CODE
        expected_subscription = {
            'plan': {
                'name': 'Test Plan Name',
                'price': 12000.0,
                'currency': 'dollar',
                'billing_cycle': 'annual',
                'max_tasks': 20,
                'max_deployments': 500,
                'max_users': 100,
                'max_roles': 10,
                'max_collaborators': 50,
                'max_apps': 3,
            },
            'end_at': None,
            'cancel_at': None,
            'usage': {
                'num_tasks': 3,
                'num_deployments': 0,
                'num_users': 8,
                'num_roles': 2,  # Admin and Maintainer don't count
                'num_collaborators': 4,
                'num_apps': 5,
            },
            'extras': [{
                'price': 2200.0,
                'end_at': None,
                'cancel_at': None,
                'extra_tasks': 3,
                'extra_deployments': 0,
                'extra_space': 0,
                'extra_users': 5,
                'extra_roles': 3,
                'extra_collaborators': 8,
                'extra_apps': 2,
            }],
            'discounts': [{
                'percentage': 20,
                'end_at': None,
                'cancel_at': None,
            }]
        }
        # TODO: check 'space_usage', 'space_limit' and 'start_at' as well
        verify_response_json(actual_json=response.json(),
                             expected_json=expected_subscription,
                             optional_fields=BACKEND_JSON_FIELDS.union({'space_usage', 'space_limit', 'start_at'}))
        #######################################################
        # One subscription, multiple discounts, and one extra #
        #######################################################
        pass  # TODO: 200 response
        #######################################################
        # One subscription, one discount, and multiple extras #
        #######################################################
        pass  # TODO: 200 response
        ############################################################################
        # Multiple active subscriptions (not allowed), one discount, and one extra #
        ############################################################################
        pass  # TODO: 400 response
        ####################
        # Test quota usage #
        ####################
        pass  # TODO


class TestUsers:

    def test_get(self, mock_request_responses, client: MockClient):
        endpoint = get_endpoint(parameterized_endpoint=ENDPOINT_USERS, resource=_get_organization())
        response = client.send_request(method='GET', url=endpoint)
        assert response.status_code == HTTP_GET_STATUS_CODE
        expected_users = [
            get_json_from_db_object(db_object=x) for x in UserDB.query().filter_by(organization_id=_ORG_ID).all()
        ]
        optional_fields = BACKEND_JSON_FIELDS.union({'auth0_id', 'email', 'first_name', 'last_name'}) - {'id', 'uuid'}
        verify_response_jsons(actual_jsons=response.json()['data'],
                              expected_jsons=expected_users,
                              optional_fields=optional_fields)


class TestUser:

    def test_delete(self, client: MockClient, mock_client_id: str, session_user_id: str, session_user_auth0_id: str):

        def _test_delete(user_roles: List[str],
                         session_user_roles: List[str],
                         expected_code: int,
                         expected_msg: str = None):
            # Restore database and cache
            restore_db(mock_client_id=mock_client_id,
                       session_user_id=session_user_id,
                       session_user_auth0_id=session_user_auth0_id)
            cache.clear()

            # Assign roles to target user and session user
            user = get_user(user_id=_USER_ID)
            session_user = get_user(user_id=1)
            user.db_object().roles = [
                RoleDB.get_from_id(id_value=x, parent=user.db_object().organization) for x in user_roles
            ]
            session_user.db_object().roles = [
                RoleDB.get_from_id(id_value=x, parent=user.db_object().organization) for x in session_user_roles
            ]
            save_to_db([user.db_object(), session_user.db_object()])

            # Get current user count
            subscription = get_active_subscription(organization_id=_ORG_ID)
            num_users = subscription.num_users

            # Make request and verify response
            user_uuid = user.uuid()
            endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_USER, resource=user)
            response = client.send_request(method='DELETE', url=endpoint_url)
            assert response.status_code == expected_code
            if expected_msg:
                assert response.json()['error']['message'] == expected_msg

            # Verify database
            db_commit_and_expire()
            if expected_code == HTTP_DELETE_STATUS_CODE:
                assert UserDB.get_from_uuid(user_uuid) is None
                assert subscription.num_users == num_users - 1
            else:
                assert UserDB.get_from_uuid(user_uuid) is not None
                assert subscription.num_users == num_users

        ########################################
        # Valid request: delete a regular user #
        ########################################
        _test_delete(user_roles=['dasci'], session_user_roles=[MAINTAINER_ROLE], expected_code=HTTP_DELETE_STATUS_CODE)
        ##############################################################
        # Valid request: delete a Maintainer user with an Admin user #
        ##############################################################
        _test_delete(user_roles=[MAINTAINER_ROLE, 'dasci'],
                     session_user_roles=[ADMIN_ROLE],
                     expected_code=HTTP_DELETE_STATUS_CODE)
        ######################################################################################
        # Invalid request: try to delete an Admin user with another Admin user (not allowed) #
        ######################################################################################
        _test_delete(user_roles=[ADMIN_ROLE, 'dasci'],
                     session_user_roles=[ADMIN_ROLE],
                     expected_code=HTTP_FORBIDDEN_STATUS_CODE,
                     expected_msg='Admins can only be deleted by themselves')
        #####################################################################################
        # Invalid request: try to delete an Admin user with a Maintainer user (not allowed) #
        #####################################################################################
        _test_delete(user_roles=[ADMIN_ROLE, 'dasci'],
                     session_user_roles=[MAINTAINER_ROLE],
                     expected_code=HTTP_FORBIDDEN_STATUS_CODE,
                     expected_msg='Admins can only be deleted by themselves')
        ###############################################################################################
        # Invalid request: try to delete a Maintainer user with another Maintainer user (not allowed) #
        ###############################################################################################
        _test_delete(user_roles=[MAINTAINER_ROLE, 'dasci'],
                     session_user_roles=[MAINTAINER_ROLE],
                     expected_code=HTTP_FORBIDDEN_STATUS_CODE,
                     expected_msg='Maintainers can only be deleted by an admin or by themselves')

    def test_get(self, mock_request_responses, client: MockClient):
        user = get_user(user_id=_USER_ID)
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_USER, resource=user)
        response = client.send_request(method='GET', url=endpoint_url)
        assert response.status_code == HTTP_GET_STATUS_CODE
        expected_json = {'email': 'user_2@org2.com', 'first_name': 'User', 'last_name': '2', 'email_verified': True}
        optional_fields = BACKEND_JSON_FIELDS.union({'auth0_id'})
        verify_response_json(actual_json=response.json(), expected_json=expected_json, optional_fields=optional_fields)


class TestUserRoles:

    def test_delete(self, client: MockClient):
        _test_delete_user_roles(client=client, delete_single_role=False)

    def test_get(self, client: MockClient):
        user = get_user(user_id=_USER_ID)
        # Make request
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_USER_ROLES, resource=user)
        response = client.send_request(method='GET', url=endpoint_url)
        # Verify response
        assert response.status_code == HTTP_GET_STATUS_CODE
        res_json = response.json()
        assert res_json['user'] == user.public_id()
        user_roles = [_get_role_json(db_object=x) for x in user.db_object().roles]
        verify_response_jsons(actual_jsons=res_json['roles'], expected_jsons=user_roles)

    def test_post(self, client: MockClient):

        def _restore_user_roles():
            user = UserDB.get(user_id=_USER_ID)
            user.roles.clear()
            db_commit_and_expire()
            assert not user.roles
            cache.clear()

        def _verify_cleared_perms(role_name: str):
            assert role_name in [ADMIN_ROLE, MAINTAINER_ROLE]
            _restore_user_roles()
            user = UserDB.get(user_id=_USER_ID)
            user_perms = [
                UserPermission(organization_id=_ORG_ID,
                               user_id=_USER_ID,
                               resource_type=ResourceType.TASK,
                               action=ResourceAction.DELETE,
                               allow=True),
                UserPermission(organization_id=_ORG_ID,
                               user_id=_USER_ID,
                               resource_type=ResourceType.FILE,
                               action=ResourceAction.CREATE,
                               allow=False)
            ]
            save_to_db(user_perms)
            assert user.permissions.all() == user_perms
            role = get_role(organization_id=_ORG_ID, name=role_name).db_object()
            request_json = {'roles': [role.public_id]}
            response = client.send_request(method='POST', url=endpoint_url, json=request_json)
            assert response.status_code == HTTP_POST_STATUS_CODE
            db_commit_and_expire()
            assert len(user.roles) == 1
            assert role in user.roles
            assert not user.permissions.all()

        user = get_user(user_id=_USER_ID)
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_USER_ROLES, resource=user)
        ###############################
        # Assign Dasci and Devs roles #
        ###############################
        _restore_user_roles()
        dasci_role = get_role(organization_id=_ORG_ID, name='dasci').db_object()
        devs_role = get_role(organization_id=_ORG_ID, name='devs').db_object()
        request_json = {'roles': [dasci_role.public_id, devs_role.public_id]}
        response = client.send_request(method='POST', url=endpoint_url, json=request_json)
        assert response.status_code == HTTP_POST_STATUS_CODE
        db_commit_and_expire()
        assert len(user.db_object().roles) == 2
        assert dasci_role in user.db_object().roles and devs_role in user.db_object().roles
        ###################################################################
        # Verify user permissions are cleared when Admin role is assigned #
        ###################################################################
        _verify_cleared_perms(role_name=ADMIN_ROLE)
        ########################################################################
        # Verify user permissions are cleared when Maintainer role is assigned #
        ########################################################################
        _verify_cleared_perms(role_name=MAINTAINER_ROLE)
        ####################################################################
        # Try to assign an Admin role with a Maintainer user (not allowed) #
        ####################################################################
        _restore_user_roles()
        session_user = UserDB.get(user_id=1)
        session_user.roles = [RoleDB.get_from_id(id_value=MAINTAINER_ROLE, parent=session_user.organization)]
        save_to_db(session_user)
        admin_role = RoleDB.get_from_id(id_value=ADMIN_ROLE, parent=session_user.organization)
        request_json = {'roles': [admin_role.public_id]}
        response = client.send_request(method='POST', url=endpoint_url, json=request_json)
        assert response.status_code == HTTP_FORBIDDEN_STATUS_CODE
        assert response.json()['error']['message'] == 'Admin roles can only be assigned by an admin'
        db_commit_and_expire()
        assert not user.db_object().roles


class TestUserRole:

    def test_delete(self, client: MockClient):
        _test_delete_user_roles(client=client, delete_single_role=True)


class TestUserPermissions:

    @pytest.fixture(autouse=True)
    def _agent_tests(self):
        self._agent_tests = _TestAgentPermissions(agent_type='user', user_id=3)

    def test_delete(self, client: MockClient):
        self._agent_tests.test_delete(client=client)

    def test_get(self, client: MockClient):
        self._agent_tests.test_get(client=client)

    def test_post(self, client: MockClient):
        self._agent_tests.test_post(client=client)


class TestUserInvite:

    def mock_download_auth0_user_data(self, auth0_id_or_email):
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        if email_pattern.match(auth0_id_or_email):
            raise ResourceNotFoundError
        else:
            return {
                'auth0_id': auth0_id_or_email,
                'email': 'testemail@email.com',
                'email_verified': True,
                'first_name': 'Test first name',
                'last_name': 'Test last name'
            }

    def test_post(self, client: MockClient, mock_request_responses, mocker):
        org = _get_organization()
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_USER_INVITE, resource=org)
        account_email = 'invitation_test@testorg.com'
        org_domain = org.db_object().domain
        org.db_object().domain = account_email.split('@')[-1]
        save_to_db(org.db_object())
        invite_json = {'email': account_email}

        mocker.patch.object(User, 'download_auth0_user_data', side_effect=self.mock_download_auth0_user_data)

        mock_invite_user = mocker.patch('nexusml.api.views.organizations._invite_user')
        mock_invite_user.delay.side_effect = _invite_user.__wrapped__

        #################
        # Valid request #
        #################
        response = client.send_request(method='POST', url=endpoint_url, json=invite_json)
        assert response.status_code == 204
        #####################################################################################################
        # Try to send invitation to an email that doesn't belong to the organization's domain (not allowed) #
        #####################################################################################################
        cache.clear()
        org.db_object().domain = org_domain
        save_to_db(org.db_object())
        response = client.send_request(method='POST', url=endpoint_url, json=invite_json)
        assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
        err_msg = ("All members must belong to the organization's domain. "
                   'You can add users outside your organization as collaborators.')
        assert response.json()['error']['message'] == err_msg
        #################################################################################################
        # Try to send invitation to an email associated with a member of the organization (not allowed) #
        #################################################################################################
        db_commit_and_expire()
        user_email = 'user_3@org3.com'
        invite_json = {'email': user_email}
        response = client.send_request(method='POST', url=endpoint_url, json=invite_json)
        assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
        err_msg = ("All members must belong to the organization's domain. "
                   'You can add users outside your organization as collaborators.')
        assert response.json()['error']['message'] == err_msg


class TestRoles:

    def test_get(self, client: MockClient):
        org = _get_organization()
        endpoint = get_endpoint(parameterized_endpoint=ENDPOINT_ROLES, resource=org)
        response = client.send_request(method='GET', url=endpoint)
        assert response.status_code == HTTP_GET_STATUS_CODE
        expected_roles = [
            _get_role_json(db_object=x)
            for x in RoleDB.query().filter_by(organization_id=org.db_object().organization_id).all()
        ]
        verify_response_jsons(actual_jsons=response.json(), expected_jsons=expected_roles)

    def test_post(self, client: MockClient):
        organization = _get_organization()
        subscription = get_active_subscription(organization_id=_ORG_ID)
        prev_num_roles = subscription.num_roles
        new_role = {'name': 'New Test Role', 'description': 'New test role description'}
        endpoint = get_endpoint(parameterized_endpoint=ENDPOINT_ROLES, resource=organization)
        response = client.send_request(method='POST', url=endpoint, json=new_role)
        assert response.status_code == HTTP_POST_STATUS_CODE
        verify_response_json(actual_json=response.json(), expected_json=new_role, optional_fields=BACKEND_JSON_FIELDS)
        db_commit_and_expire()
        assert RoleDB.query().filter_by(organization_id=_ORG_ID, name=new_role['name']).first() is not None
        assert subscription.num_roles == prev_num_roles + 1


class TestRole:

    def test_delete(self, client: MockClient):
        # Valid request
        role = get_role(organization_id=_ORG_ID, name='dasci')
        subscription = get_active_subscription(organization_id=_ORG_ID)
        subscription.num_roles = 1
        save_to_db(subscription)
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_ROLE, resource=role)
        response = client.send_request(method='DELETE', url=endpoint_url)
        assert response.status_code == HTTP_DELETE_STATUS_CODE
        db_commit_and_expire()
        assert RoleDB.query().filter_by(organization_id=_ORG_ID, name='dasci').first() is None
        assert subscription.num_roles == 0

        # Try to delete predefined roles (not allowed)
        predefined_roles = [ADMIN_ROLE, MAINTAINER_ROLE]
        for role_name in predefined_roles:
            role = get_role(organization_id=_ORG_ID, name=role_name)
            endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_ROLE, resource=role)
            response = client.send_request(method='DELETE', url=endpoint_url)
            assert response.status_code == HTTP_FORBIDDEN_STATUS_CODE
            assert response.json()['error']['message'] == 'Admin and Maintainer roles cannot be deleted'
            db_commit_and_expire()
            assert RoleDB.query().filter_by(organization_id=_ORG_ID, name=role_name).first() is not None

    def test_get(self, client: MockClient):
        role = get_role(organization_id=_ORG_ID, name='dasci')
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_ROLE, resource=role)
        response = client.send_request(method='GET', url=endpoint_url)
        assert response.status_code == HTTP_GET_STATUS_CODE
        verify_response_json(actual_json=response.json(), expected_json=_get_role_json(db_object=role.db_object()))

    def test_put(self, client: MockClient, mock_client_id: str, session_user_id: str, session_user_auth0_id: str):
        # Valid request
        role = get_role(organization_id=_ORG_ID, name='dasci')
        modified_role = {'name': 'Modified Test Role', 'description': 'Modified test role description'}
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_ROLE, resource=role)
        response = client.send_request(method='PUT', url=endpoint_url, json=modified_role)
        assert response.status_code == HTTP_PUT_STATUS_CODE
        db_commit_and_expire()
        assert all(getattr(role.db_object(), field) == modified_role[field] for field in modified_role)
        optional_fields = BACKEND_JSON_FIELDS - {'id', 'uuid'}
        verify_response_json(actual_json=response.json(),
                             expected_json=_get_role_json(db_object=role.db_object()),
                             optional_fields=optional_fields)

        # Try to modify predefined roles (not allowed)
        restore_db(mock_client_id=mock_client_id,
                   session_user_id=session_user_id,
                   session_user_auth0_id=session_user_auth0_id)
        predefined_roles = [ADMIN_ROLE, MAINTAINER_ROLE]
        for role_name in predefined_roles:
            role = get_role(organization_id=_ORG_ID, name=role_name)
            role_dict = role.db_object().to_dict()
            endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_ROLE, resource=role)
            response = client.send_request(method='PUT', url=endpoint_url, json=modified_role)
            assert response.status_code == HTTP_FORBIDDEN_STATUS_CODE
            assert response.json()['error']['message'] == 'Admin and Maintainer roles cannot be modified'
            db_commit_and_expire()
            assert RoleDB.query().filter_by(organization_id=_ORG_ID, name=modified_role['name']).first() is None
            role_db_object = RoleDB.query().filter_by(organization_id=_ORG_ID, name=role_name).first()
            assert role_db_object.to_dict() == role_dict


class TestRoleUsers:

    def test_get(self, client: MockClient, mock_request_responses):
        role = get_role(organization_id=_ORG_ID, name='dasci')
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_ROLE_USERS, resource=role)
        response = client.send_request(method='GET', url=endpoint_url)
        assert response.status_code == HTTP_GET_STATUS_CODE
        res_json = response.json()
        role_users = [get_json_from_db_object(db_object=x) for x in role.db_object().users.all()]
        optional_fields = BACKEND_JSON_FIELDS.union({'auth0_id', 'first_name', 'last_name', 'email'}) - {'id', 'uuid'}
        verify_response_jsons(actual_jsons=res_json['data'], expected_jsons=role_users, optional_fields=optional_fields)


class TestRolePermissions:

    def test_delete(self, client: MockClient):
        role = get_role(organization_id=_ORG_ID, name='dasci')
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_ROLE_PERMISSIONS, resource=role)
        ##########################
        # Delete all permissions #
        ##########################
        _set_role_permissions(role=role.db_object())
        response = client.send_request(method='DELETE', url=endpoint_url)
        assert response.status_code == HTTP_DELETE_STATUS_CODE
        db_commit_and_expire()
        assert not role.db_object().permissions.all()
        ###########################
        # Delete file permissions #
        ###########################
        _set_role_permissions(role=role.db_object())
        query = '?resource_type=file'
        response = client.send_request(method='DELETE', url=endpoint_url + query)
        assert response.status_code == HTTP_DELETE_STATUS_CODE
        db_commit_and_expire()
        role_perms = role.db_object().permissions.all()
        assert len(role_perms) == 1
        assert role_perms[0].resource_type != ResourceType.FILE
        #############################
        # Filter action permissions #
        #############################
        _set_role_permissions(role=role.db_object())
        query = '?action=read'
        response = client.send_request(method='DELETE', url=endpoint_url + query)
        assert response.status_code == HTTP_DELETE_STATUS_CODE
        db_commit_and_expire()
        role_perms = role.db_object().permissions.all()
        assert len(role_perms) == 2
        assert all(perm.action != ResourceAction.READ for perm in role_perms)

    def test_get(self, client: MockClient):
        role = get_role(organization_id=_ORG_ID, name='dasci')
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_ROLE_PERMISSIONS, resource=role)
        #######################
        # Get all permissions #
        #######################
        _set_role_permissions(role=role.db_object())
        response = client.send_request(method='GET', url=endpoint_url)
        assert response.status_code == HTTP_GET_STATUS_CODE
        res_json = response.json()
        expected_perms = [{
            'resource_type': 'file',
            'action': 'read',
            'allow': True
        }, {
            'resource_type': 'file',
            'action': 'create',
            'allow': True
        }, {
            'resource_type': 'file',
            'action': 'update',
            'allow': True
        }, {
            'resource_type': 'ai_model',
            'action': 'read',
            'allow': True
        }]
        assert all(x in res_json['data'] for x in expected_perms)
        ###########################
        # Filter file permissions #
        ###########################
        query = '?resource_type=file'
        response = client.send_request(method='GET', url=endpoint_url + query)
        assert response.status_code == HTTP_GET_STATUS_CODE
        res_json = response.json()
        expected_perms = [{
            'resource_type': 'file',
            'action': 'read',
            'allow': True
        }, {
            'resource_type': 'file',
            'action': 'create',
            'allow': True
        }, {
            'resource_type': 'file',
            'action': 'update',
            'allow': True
        }]
        assert all(x in res_json['data'] for x in expected_perms)
        ######################################
        # Filter wildcard-action permissions #
        ######################################
        query = '?action=read'
        response = client.send_request(method='GET', url=endpoint_url + query)
        assert response.status_code == HTTP_GET_STATUS_CODE
        res_json = response.json()
        expected_perms = [{
            'resource_type': 'file',
            'action': 'read',
            'allow': True
        }, {
            'resource_type': 'ai_model',
            'action': 'read',
            'allow': True
        }]
        assert all(x in res_json['data'] for x in expected_perms)

    def test_post(self, client: MockClient):

        def _deny_generic_permission(role, provide_rsrc_uuid: bool):
            delete_from_db(role.db_object().permissions.all())
            assert not role.db_object().permissions.all()
            perm = {'resource_type': 'file', 'action': 'create', 'allow': False}
            if provide_rsrc_uuid:
                perm['resource_uuid'] = None
            endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_ROLE_PERMISSIONS, resource=role)
            response = client.send_request(method='POST', url=endpoint_url, json=perm)
            assert response.status_code == HTTP_BAD_REQUEST_STATUS_CODE
            assert response.json()['error']['message'] == 'Cannot deny generic permissions to roles'
            db_commit_and_expire()
            assert not role.db_object().permissions.all()

        #################
        # Valid request #
        #################
        role = get_role(organization_id=_ORG_ID, name='dasci')
        delete_from_db(role.db_object().permissions.all())
        assert not role.db_object().permissions.all()
        perm = {'resource_type': 'task', 'action': 'create', 'allow': True}
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_ROLE_PERMISSIONS, resource=role)
        response = client.send_request(method='POST', url=endpoint_url, json=perm)
        assert response.status_code == HTTP_POST_STATUS_CODE
        db_commit_and_expire()
        role_perms = role.db_object().permissions.all()
        assert len(role_perms) == 1
        assert role_perms[0].resource_type == ResourceType.TASK
        assert role_perms[0].action == ResourceAction.CREATE
        assert role_perms[0].allow


class TestCollaborators:

    def test_get(self, mock_request_responses, client: MockClient):
        endpoint = get_endpoint(parameterized_endpoint=ENDPOINT_COLLABORATORS, resource=_get_organization())
        response = client.send_request(method='GET', url=endpoint)
        assert response.status_code == HTTP_GET_STATUS_CODE
        collaborators_q_res = CollaboratorDB.query().filter(CollaboratorDB.user_id.in_([5, 6, 7, 8])).all()
        collaborators = dict()
        for q_res in collaborators_q_res:
            collaborators[q_res.user_id] = q_res
        expected_jsons = [
            {
                'uuid': str(collaborators[5].uuid),
                'id': collaborators[5].public_id,
                'email': 'user_5@org3.com',
                'organization': 'Organization 3',
                'first_name': 'User',
                'last_name': '5',
            },
            {
                'uuid': str(collaborators[6].uuid),
                'id': collaborators[6].public_id,
                'email': 'user_6@org3.com',
                'organization': 'Organization 3',
                'first_name': 'User',
                'last_name': '6',
            },
            {
                'uuid': str(collaborators[7].uuid),
                'id': collaborators[7].public_id,
                'email': 'user_7@org3.com',
                'organization': 'Organization 3',
                'first_name': 'User',
                'last_name': '7',
            },
            {
                'uuid': str(collaborators[8].uuid),
                'id': collaborators[8].public_id,
                'email': 'user_8@org3.com',
                'organization': 'Organization 3',
                'first_name': 'User',
                'last_name': '8',
            },
        ]
        optional_fields = BACKEND_JSON_FIELDS - {'id', 'uuid'}
        verify_response_jsons(actual_jsons=response.json()['data'],
                              expected_jsons=expected_jsons,
                              optional_fields=optional_fields)

    def test_post(self, mocker, mock_request_responses, client: MockClient):
        organization = _get_organization()
        subscription = get_active_subscription(organization_id=_ORG_ID)
        prev_num_collaborators = subscription.num_collaborators
        user = UserDB.get(user_id=8)  # collaborator's user
        endpoint = get_endpoint(parameterized_endpoint=ENDPOINT_COLLABORATORS, resource=organization)
        #################
        # Valid request #
        #################
        delete_from_db(_get_collaborator(organization_id=_ORG_ID, user_id=user.user_id).db_object())
        new_collaborator = {'email': f'user_{user.user_id}@org3.com'}
        response = client.send_request(method='POST', url=endpoint, json=new_collaborator)
        assert response.status_code == HTTP_POST_STATUS_CODE
        account = {
            'email': f'user_{user.user_id}@org3.com',
            'first_name': 'User',
            'last_name': '8',
        }
        expected_json = {
            'email': account['email'],
            'organization': user.organization.name,
            'first_name': account['first_name'],
            'last_name': account['last_name']
        }
        verify_response_json(actual_json=response.json(),
                             expected_json=expected_json,
                             optional_fields=BACKEND_JSON_FIELDS)
        db_commit_and_expire()
        assert _get_collaborator(organization_id=_ORG_ID, user_id=user.user_id) is not None
        assert subscription.num_collaborators == prev_num_collaborators + 1
        ###################################
        # Try to add a collaborator again #
        ###################################
        response = client.send_request(method='POST', url=endpoint, json=new_collaborator)
        assert response.status_code == HTTP_CONFLICT_STATUS_CODE
        assert response.json()['error']['message'] == 'Resource already exists'
        db_commit_and_expire()
        assert subscription.num_collaborators == prev_num_collaborators + 1
        #########################################
        # Try to add a member as a collaborator #
        #########################################
        existing_user = {'email': 'user_2@org2.com'}
        response = client.send_request(method='POST', url=endpoint, json=existing_user)
        assert response.status_code == HTTP_CONFLICT_STATUS_CODE
        err_msg = f'User with email address "{existing_user["email"]}" is a member of the organization'
        assert response.json()['error']['message'] == err_msg
        db_commit_and_expire()
        assert subscription.num_collaborators == prev_num_collaborators + 1
        ################################################################
        # Try to add a collaborator that is not registered in NexusML yet #
        ################################################################
        mock_exception = ResourceNotFoundError
        mocker.patch.object(User, 'download_auth0_user_data', side_effect=mock_exception)
        invalid_collaborator = {'email': 'random@email.com'}
        response = client.send_request(method='POST', url=endpoint, json=invalid_collaborator)
        assert response.status_code == HTTP_NOT_FOUND_STATUS_CODE
        err_msg = f'User with email "{invalid_collaborator["email"]}" not registered'
        assert response.json()['error']['message'] == err_msg
        db_commit_and_expire()
        assert subscription.num_collaborators == prev_num_collaborators + 1


class TestCollaborator:

    def test_delete(self, client: MockClient):
        collaborator = _get_collaborator(organization_id=_ORG_ID, user_id=8)
        subscription = get_active_subscription(organization_id=_ORG_ID)
        subscription.num_collaborators = 1
        save_to_db(subscription)
        collaborator_uuid = collaborator.uuid()
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_COLLABORATOR, resource=collaborator)
        response = client.send_request(method='DELETE', url=endpoint_url)
        assert response.status_code == HTTP_DELETE_STATUS_CODE
        db_commit_and_expire()
        assert CollaboratorDB.get_from_uuid(collaborator_uuid) is None
        assert subscription.num_collaborators == 0

    def test_get(self, mock_request_responses, client: MockClient):
        collaborator = _get_collaborator(organization_id=_ORG_ID, user_id=8)
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_COLLABORATOR, resource=collaborator)
        response = client.send_request(method='GET', url=endpoint_url)
        assert response.status_code == HTTP_GET_STATUS_CODE
        expected_json = {
            'id': collaborator.public_id(),
            'uuid': collaborator.uuid(),
            'email': 'user_8@org3.com',
            'organization': 'Organization 3',
            'first_name': 'User',
            'last_name': '8',
        }
        optional_fields = BACKEND_JSON_FIELDS - {'id', 'uuid'}
        verify_response_json(actual_json=response.json(), expected_json=expected_json, optional_fields=optional_fields)


class TestCollaboratorPermissions:

    @pytest.fixture(autouse=True)
    def _agent_tests(self):
        self._agent_tests = _TestAgentPermissions(agent_type='collaborator', user_id=8)

    def test_delete(self, client: MockClient):
        self._agent_tests.test_delete(client=client)

    def test_get(self, client: MockClient):
        self._agent_tests.test_get(client=client)

    def test_post(self, client: MockClient):
        self._agent_tests.test_post(client=client)


class TestClients:

    def test_get(self, client: MockClient, mock_s3):
        org = _get_organization()

        endpoint = get_endpoint(parameterized_endpoint=ENDPOINT_CLIENTS, resource=org)
        response = client.send_request(method='GET', url=endpoint)
        assert response.status_code == HTTP_GET_STATUS_CODE

        all_org_clients = ClientDB.query().filter_by(organization_id=org.db_object().organization_id).all()

        expected_clients = [
            _get_client_json(db_object=x)
            for x in all_org_clients
            if Service.query().filter_by(client_id=x.client_id).first() is None
        ]

        verify_response_jsons(actual_jsons=response.json(),
                              expected_jsons=expected_clients,
                              optional_fields={'auth0_id'})

    def test_post(self, client: MockClient, mock_s3):
        # Set database
        old_client = ClientDB.query().filter_by(organization_id=_ORG_ID).first()
        delete_from_db(old_client)

        subscription = get_active_subscription(organization_id=_ORG_ID)
        subscription.num_clients -= 1
        save_to_db(subscription)
        prev_num_clients = subscription.num_clients

        icon_file = FileDB(organization_id=_ORG_ID, filename='app_icon', size=45, use_for=OrgFileUse.PICTURE)
        save_to_db(icon_file)

        # Make request
        req_json = {
            'name': 'New Test Client',
            'description': 'New test client description',
            'icon': str(icon_file.public_id)
        }
        endpoint = get_endpoint(parameterized_endpoint=ENDPOINT_CLIENTS, resource=_get_organization())
        response = client.send_request(method='POST', url=endpoint, json=req_json)

        # Verify response
        assert response.status_code == HTTP_POST_STATUS_CODE
        verify_response_json_with_picture(request_json=req_json,
                                          response_json=response.json(),
                                          picture_db_object=icon_file,
                                          picture_field='icon')
        db_commit_and_expire()
        assert subscription.num_clients == prev_num_clients + 1


class TestClient:

    def test_delete(self, client: MockClient, mock_s3):
        target_client = _get_client()
        client_uuid = target_client.uuid()
        subscription = get_active_subscription(organization_id=_ORG_ID)
        subscription.num_clients = 1
        save_to_db(subscription)
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_CLIENT, resource=target_client)
        response = client.send_request(method='DELETE', url=endpoint_url)
        assert response.status_code == HTTP_DELETE_STATUS_CODE
        db_commit_and_expire()
        assert ClientDB.get_from_uuid(client_uuid) is None
        assert subscription.num_clients == 0

    def test_get(self, client: MockClient, mock_s3):
        # Regular query
        target_client = _get_client()
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_CLIENT, resource=target_client)
        response = client.send_request(method='GET', url=endpoint_url)
        assert response.status_code == HTTP_GET_STATUS_CODE
        verify_response_json(actual_json=response.json(),
                             expected_json=_get_client_json(db_object=target_client.db_object()),
                             optional_fields={'auth0_id'})
        # Try to get a service client (not allowed)
        all_org_clients = ClientDB.query().filter_by(organization_id=_ORG_ID).all()
        service_clients = [
            x for x in all_org_clients if Service.query().filter_by(client_id=x.client_id).first() is not None
        ]
        endpoint_url = endpoint_url[:endpoint_url.rindex('/') + 1] + service_clients[0].public_id
        response = client.send_request(method='GET', url=endpoint_url)
        assert response.status_code == HTTP_NOT_FOUND_STATUS_CODE
        assert response.json()['error']['message'] == f'Resource not found: "{service_clients[0].public_id}"'

    def test_put(self, client: MockClient, mock_s3):
        icon_file = FileDB(organization_id=_ORG_ID, filename='updated_app_icon', size=62, use_for=OrgFileUse.PICTURE)
        save_to_db(icon_file)
        request_json = {
            'name': 'Modified Test Client',
            'description': 'Modified test client description',
            'icon': str(icon_file.public_id)
        }
        endpoint = get_endpoint(parameterized_endpoint=ENDPOINT_CLIENT, resource=_get_client())
        response = client.send_request(method='PUT', url=endpoint, json=request_json)
        assert response.status_code == HTTP_PUT_STATUS_CODE
        verify_response_json_with_picture(request_json=request_json,
                                          response_json=response.json(),
                                          picture_db_object=icon_file,
                                          picture_field='icon')


class TestClientAPIKey:

    def test_get(self, client: MockClient):
        target_client = _get_client()
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_CLIENT_API_KEY, resource=target_client)
        response = client.send_request(method='GET', url=endpoint_url)
        assert response.status_code == HTTP_GET_STATUS_CODE
        verify_response_json(actual_json=response.json(),
                             expected_json=_get_api_key_json(client=target_client.db_object()))

    def test_put(self, client: MockClient):

        def _make_request_and_verify_response(target_client: Client, request_json: dict):
            endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_CLIENT_API_KEY, resource=target_client)
            response = client.send_request(method='PUT', url=endpoint_url, json=request_json)
            assert response.status_code == HTTP_PUT_STATUS_CODE
            response_json = response.json()
            db_commit_and_expire()
            db_dec_token = decode_api_key(api_key=target_client.db_object().api_key)
            for field, response_value in response_json.items():
                if field == 'token':
                    assert response_value == target_client.db_object().api_key
                elif field == 'scopes':
                    # Compare with database object
                    assert response_value == db_dec_token['scope'].split(' ')
                    # Compare with request JSON
                    assert response_value == sorted(request_json.get('scopes') or client_scopes)
                elif field == 'expire_at' and 'expire_at' in request_json:
                    expected_exp = request_json['expire_at']
                    # Check database object
                    if expected_exp is None:
                        assert 'exp' not in db_dec_token
                    else:
                        db_dec_token_exp = datetime.utcfromtimestamp(db_dec_token['exp'])
                        db_expire_at = db_dec_token_exp.strftime(DATETIME_FORMAT)
                        assert db_expire_at == expected_exp
                    # Check response JSON
                    res_dec_token = decode_api_key(api_key=response_json['token'])
                    if expected_exp is None:
                        assert 'exp' not in res_dec_token
                        assert response_value is None
                    else:
                        res_dec_token_exp = datetime.utcfromtimestamp(res_dec_token['exp'])
                        res_expire_at = res_dec_token_exp.strftime(DATETIME_FORMAT) + 'Z'
                        assert res_expire_at == expected_exp or res_expire_at == (expected_exp + 'Z')
                        assert response_value == res_expire_at

        target_client = _get_client()
        ##################################################
        # Test 1: provide scopes and expiration datetime #
        ##################################################
        api_key_exp = datetime.utcnow() + timedelta(minutes=23)
        test_1_json = {
            'scopes': ['tasks.read', 'files.create', 'models.read', 'examples.read'],
            'expire_at': api_key_exp.strftime(DATETIME_FORMAT)
        }
        _make_request_and_verify_response(target_client=target_client, request_json=test_1_json)
        ################################################################################
        # Test 2: provide scopes and null expiration datetime (never expiring API key) #
        ################################################################################
        test_2_json = {'scopes': ['tasks.read', 'files.create', 'models.read', 'examples.read'], 'expire_at': None}
        _make_request_and_verify_response(target_client=target_client, request_json=test_2_json)
        ##########################################################################
        # Test 3: provide scopes and no expiration datetime (default one is set) #
        ##########################################################################
        test_3_json = {'scopes': ['tasks.read', 'files.create', 'models.read', 'examples.read']}
        _make_request_and_verify_response(target_client=target_client, request_json=test_3_json)
        ############################################################################
        # Test 4: don't provide scopes and expiration datetime                     #
        #         (all scopes are included and default expiration datetime is set) #
        ############################################################################
        test_4_json = {}
        _make_request_and_verify_response(target_client=target_client, request_json=test_4_json)
        dec_token = decode_api_key(api_key=target_client.db_object().api_key)
        assert sorted(dec_token['scope'].split(' ')) == client_scopes
        ################################################################
        # Test 5: provide scopes and expiration datetime from the past #
        ################################################################
        api_key_exp = datetime.utcnow() - timedelta(minutes=23)
        test_5_json = {
            'scopes': ['tasks.read', 'files.create', 'models.read', 'examples.read'],
            'expire_at': api_key_exp.strftime(DATETIME_FORMAT)
        }
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_CLIENT_API_KEY, resource=target_client)
        response = client.send_request(method='PUT', url=endpoint_url, json=test_5_json)
        assert response.status_code == HTTP_BAD_REQUEST_STATUS_CODE
        assert response.json()['errors']['json']['expire_at'] == ['Expiration datetime must be after current datetime']


class _TestAgentPermissions:

    def __init__(self, agent_type: str, user_id: int):
        assert agent_type in ['user', 'collaborator']
        self.agent_type = agent_type
        self.user_id = user_id

    def _get_agent(self) -> Tuple[Union[User, Collaborator], UserDB]:
        user_db_object = UserDB.get(user_id=self.user_id)
        if self.agent_type == 'user':
            agent = get_user(user_id=self.user_id)
        else:
            agent = _get_collaborator(organization_id=_ORG_ID, user_id=self.user_id)
        return agent, user_db_object

    def _get_endpoint(self) -> str:
        if self.agent_type == 'user':
            return ENDPOINT_USER_PERMISSIONS
        else:
            return ENDPOINT_COLLABORATOR_PERMISSIONS

    def test_delete(self, client: MockClient):
        agent, user = self._get_agent()
        endpoint_url = get_endpoint(parameterized_endpoint=self._get_endpoint(), resource=agent)
        ##########################
        # Delete all permissions #
        ##########################
        _set_user_permissions(organization_id=_ORG_ID, user_id=self.user_id)
        response = client.send_request(method='DELETE', url=endpoint_url)
        assert response.status_code == HTTP_DELETE_STATUS_CODE
        db_commit_and_expire()
        assert not user.permissions.all()
        ####################################
        # Delete specific task permissions #
        ####################################
        _set_user_permissions(organization_id=_ORG_ID, user_id=self.user_id)
        task_uuid = TaskDB.get(task_id=1).uuid
        query = '?resource_uuid=' + task_uuid
        response = client.send_request(method='DELETE', url=endpoint_url + query)
        assert response.status_code == HTTP_DELETE_STATUS_CODE
        db_commit_and_expire()
        user_perms = user.permissions.all()
        assert len(user_perms) == 2
        assert user_perms[0].resource_uuid != task_uuid
        assert user_perms[1].resource_uuid != task_uuid
        ###############################
        # Delete deletion permissions #
        ###############################
        _set_user_permissions(organization_id=_ORG_ID, user_id=self.user_id)
        query = '?action=delete'
        response = client.send_request(method='DELETE', url=endpoint_url + query)
        assert response.status_code == HTTP_DELETE_STATUS_CODE
        db_commit_and_expire()
        user_perms = user.permissions.all()
        assert len(user_perms) == 2
        assert user_perms[0].action != ResourceAction.DELETE
        assert user_perms[1].action != ResourceAction.DELETE

    def test_get(self, client: MockClient):
        agent, user = self._get_agent()
        endpoint_url = get_endpoint(parameterized_endpoint=self._get_endpoint(), resource=agent)
        #######################
        # Get all permissions #
        #######################
        _set_user_permissions(organization_id=_ORG_ID, user_id=self.user_id)
        response = client.send_request(method='GET', url=endpoint_url)
        assert response.status_code == HTTP_GET_STATUS_CODE
        res_json = response.json()
        expected_perms = []
        db_commit_and_expire()
        for permission in user.permissions.all():
            perm_json = {
                'resource_type': permission.resource_type.name.lower(),
                'action': permission.action.name.lower(),
                'allow': permission.allow
            }
            if permission.resource_uuid != NULL_UUID:
                perm_json['resource_uuid'] = permission.resource_uuid
            expected_perms.append(perm_json)
        assert all(x in res_json['data'] for x in expected_perms)
        ####################################
        # Filter specific task permissions #
        ####################################
        task_uuid = TaskDB.get(task_id=1).uuid
        query = '?resource_uuid=' + task_uuid
        response = client.send_request(method='GET', url=endpoint_url + query)
        assert response.status_code == HTTP_GET_STATUS_CODE
        res_json = response.json()
        expected_perms = [{
            'resource_uuid': task_uuid,
            'resource_type': 'task',
            'action': 'read',
            'allow': True
        }, {
            'resource_uuid': task_uuid,
            'resource_type': 'task',
            'action': 'update',
            'allow': False
        }]
        assert all(x in res_json['data'] for x in expected_perms)
        ###############################
        # Filter deletion permissions #
        ###############################
        query = '?action=delete'
        response = client.send_request(method='GET', url=endpoint_url + query)
        assert response.status_code == HTTP_GET_STATUS_CODE
        res_json = response.json()
        expected_perms = [{
            'resource_type': 'file',
            'action': 'delete',
            'allow': True
        }, {
            'resource_type': 'ai_model',
            'action': 'delete',
            'allow': False
        }]
        assert all(x in res_json['data'] for x in expected_perms)

    def test_post(self, client: MockClient):
        #################
        # Valid request #
        #################
        agent, user = self._get_agent()
        endpoint_url = get_endpoint(parameterized_endpoint=self._get_endpoint(), resource=agent)
        db_commit_and_expire()
        assert not user.permissions.all()
        perm = {'resource_type': 'task', 'action': 'create', 'allow': True}
        response = client.send_request(method='POST', url=endpoint_url, json=perm)
        assert response.status_code == HTTP_POST_STATUS_CODE
        verify_response_json(actual_json=response.json(),
                             expected_json={
                                 **perm, 'resource_uuid': None
                             },
                             optional_fields={'resource_uuid'})
        db_commit_and_expire()
        user_perms = user.permissions.all()
        assert len(user_perms) == 1
        assert user_perms[0].resource_type == ResourceType.TASK
        assert user_perms[0].action == ResourceAction.CREATE
        assert user_perms[0].allow
        ###################################################################
        # Try to assign a creation permission on a resource (not allowed) #
        ###################################################################
        delete_from_db(user.permissions.all())
        assert not user.permissions.all()
        file = FileDB.get(file_id=1)
        perm = {'resource_uuid': file.uuid, 'resource_type': 'file', 'action': 'create', 'allow': True}
        response = client.send_request(method='POST', url=endpoint_url, json=perm)
        assert response.status_code == HTTP_BAD_REQUEST_STATUS_CODE
        assert response.json()['error']['message'] == 'Cannot assign creation permissions at resource level'
        db_commit_and_expire()
        assert not user.permissions.all()


def _get_organization() -> Organization:
    org_db_object = OrganizationDB.get(organization_id=_ORG_ID)
    return Organization.get(agent=UserDB.get(user_id=1), db_object_or_id=org_db_object)


def _get_collaborator(organization_id: int, user_id: int) -> Collaborator:
    session_user = UserDB.get(user_id=1)
    collaborator = CollaboratorDB.query().filter_by(organization_id=organization_id, user_id=user_id).first()
    org = Organization.get(agent=session_user, db_object_or_id=OrganizationDB.get(organization_id=organization_id))
    return Collaborator.get(agent=session_user, db_object_or_id=collaborator, parents=[org])


def _get_client() -> Client:
    session_user = UserDB.get(user_id=1)
    client = ClientDB.get(client_id=_CLIENT_ID)
    org = Organization.get(agent=session_user, db_object_or_id=OrganizationDB.get(organization_id=_ORG_ID))
    assert client.organization_id == org.db_object().organization_id
    return Client.get(agent=session_user, db_object_or_id=client, parents=[org])


def _get_role_json(db_object: RoleDB) -> dict:
    return get_json_from_db_object(db_object=db_object, ignore_columns=(BACKEND_JSON_FIELDS - {'uuid', 'id'}))


def _get_client_json(db_object: ClientDB) -> dict:
    return get_json_from_db_object(db_object=db_object, ignore_columns=['api_key'])


def _get_api_key_json(client: ClientDB) -> dict:
    dec_token = decode_api_key(api_key=client.api_key)
    api_key = {'scopes': sorted(dec_token['scope'].split(' ')), 'token': client.api_key}
    if 'exp' in dec_token:
        exp_at = datetime.utcfromtimestamp(dec_token['exp'])
        api_key['expire_at'] = exp_at.strftime(DATETIME_FORMAT) + 'Z'
    else:
        api_key['expire_at'] = None
    return api_key


def _set_user_permissions(organization_id: int, user_id: int):
    task_uuid = TaskDB.get(task_id=1).uuid
    user_perms = UserPermission.query().filter_by(organization_id=organization_id, user_id=user_id).all()
    delete_from_db(user_perms)
    user_perms = [
        UserPermission(organization_id=organization_id,
                       user_id=user_id,
                       resource_uuid=task_uuid,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.READ,
                       allow=True),
        UserPermission(organization_id=organization_id,
                       user_id=user_id,
                       resource_uuid=task_uuid,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.UPDATE,
                       allow=False),
        UserPermission(organization_id=organization_id,
                       user_id=user_id,
                       resource_type=ResourceType.FILE,
                       action=ResourceAction.DELETE,
                       allow=True),
        UserPermission(organization_id=organization_id,
                       user_id=user_id,
                       resource_type=ResourceType.AI_MODEL,
                       action=ResourceAction.DELETE,
                       allow=False)
    ]
    save_to_db(user_perms)


def _set_role_permissions(role: RoleDB):
    role.permissions.delete()
    role_perms = [
        RolePermission(role_id=role.role_id, resource_type=ResourceType.FILE, action=ResourceAction.READ, allow=True),
        RolePermission(role_id=role.role_id, resource_type=ResourceType.FILE, action=ResourceAction.CREATE, allow=True),
        RolePermission(role_id=role.role_id, resource_type=ResourceType.FILE, action=ResourceAction.UPDATE, allow=True),
        RolePermission(role_id=role.role_id,
                       resource_type=ResourceType.AI_MODEL,
                       action=ResourceAction.READ,
                       allow=True)
    ]
    save_to_db(role_perms)


def _test_delete_user_roles(client: MockClient, delete_single_role: bool):
    user = get_user(user_id=_USER_ID)

    admin_role = RoleDB.get_from_id(id_value=ADMIN_ROLE, parent=user.db_object().organization)
    maintainer_role = RoleDB.get_from_id(id_value=MAINTAINER_ROLE, parent=user.db_object().organization)
    dasci_role = RoleDB.get_from_id(id_value='dasci', parent=user.db_object().organization)
    devs_role = RoleDB.get_from_id(id_value='devs', parent=user.db_object().organization)

    user_roles = [dasci_role, devs_role]
    user.db_object().roles = user_roles
    save_to_db(user.db_object())

    roles_url = get_endpoint(parameterized_endpoint=ENDPOINT_USER_ROLES, resource=user)
    dasci_url = (roles_url + '/' + dasci_role.public_id) if delete_single_role else None
    admin_role_url = (roles_url + '/' + admin_role.public_id) if delete_single_role else None
    maintainer_url = (roles_url + '/' + maintainer_role.public_id) if delete_single_role else None
    #################
    # Valid request #
    #################
    assert len(user.db_object().roles) > 0
    assert dasci_role in user.db_object().roles
    response = client.send_request(method='DELETE', url=(dasci_url or roles_url))
    assert response.status_code == HTTP_DELETE_STATUS_CODE
    db_commit_and_expire()
    if delete_single_role:
        assert dasci_role not in user.db_object().roles
    else:
        assert not user.db_object().roles
    ###################################################################################
    # Invalid request: try to remove Admin role with another Admin user (not allowed) #
    ###################################################################################
    cache.clear()
    user.db_object().roles = user_roles + [admin_role]
    save_to_db(user.db_object())
    response = client.send_request(method='DELETE', url=(admin_role_url or roles_url))
    assert response.status_code == HTTP_FORBIDDEN_STATUS_CODE
    assert response.json()['error']['message'] == 'Admin role can only be removed by the corresponding user'
    db_commit_and_expire()
    assert set(user.db_object().roles) == set(user_roles + [admin_role])
    ##################################################################################
    # Invalid request: try to remove Admin role with a Maintainer user (not allowed) #
    ##################################################################################
    cache.clear()
    session_user = UserDB.get(user_id=1)
    session_user.roles = [maintainer_role]
    save_to_db(session_user)
    response = client.send_request(method='DELETE', url=(admin_role_url or roles_url))
    assert response.status_code == HTTP_FORBIDDEN_STATUS_CODE
    assert response.json()['error']['message'] == 'Admin role can only be removed by the corresponding user'
    db_commit_and_expire()
    assert set(user.db_object().roles) == set(user_roles + [admin_role])
    #############################################################################################
    # Invalid request: try to remove Maintainer role with another Maintainer user (not allowed) #
    #############################################################################################
    cache.clear()
    user.db_object().roles = user_roles + [maintainer_role]
    save_to_db(user.db_object())
    response = client.send_request(method='DELETE', url=(maintainer_url or roles_url))
    assert response.status_code == HTTP_FORBIDDEN_STATUS_CODE
    assert response.json()['error']['message'] == ('Maintainer role can only be removed by '
                                                   'an admin or by the corresponding user')
    db_commit_and_expire()
    assert set(user.db_object().roles) == set(user_roles + [maintainer_role])
