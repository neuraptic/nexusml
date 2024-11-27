from datetime import datetime
import functools
import os
from typing import Iterable, List, Optional, Union

from flask import g
import requests
from sqlalchemy.exc import OperationalError

from nexusml.api.endpoints import ENDPOINT_CLIENT
from nexusml.api.endpoints import ENDPOINT_COLLABORATOR
from nexusml.api.endpoints import ENDPOINT_ORGANIZATION
from nexusml.api.endpoints import ENDPOINT_ROLE
from nexusml.api.endpoints import ENDPOINT_USER
from nexusml.api.ext import cache
from nexusml.api.resources.base import DuplicateResourceError
from nexusml.api.resources.base import InvalidDataError
from nexusml.api.resources.base import PermissionDeniedError
from nexusml.api.resources.base import Resource
from nexusml.api.resources.base import ResourceError
from nexusml.api.resources.base import ResourceNotFoundError
from nexusml.api.resources.base import UnprocessableRequestError
from nexusml.api.resources.files import OrgFile as File
from nexusml.api.resources.utils import check_quota_usage as check_quota_usage_
from nexusml.api.schemas.base import ResourceResponseSchema
from nexusml.api.schemas.organizations import AppRequestSchema
from nexusml.api.schemas.organizations import AppResponseSchema
from nexusml.api.schemas.organizations import CollaboratorRequestSchema
from nexusml.api.schemas.organizations import CollaboratorResponseSchema
from nexusml.api.schemas.organizations import OrganizationPUTRequestSchema
from nexusml.api.schemas.organizations import OrganizationResponseSchema
from nexusml.api.schemas.organizations import RoleRequestSchema
from nexusml.api.schemas.organizations import RoleResponseSchema
from nexusml.api.schemas.organizations import UserRequestSchema
from nexusml.api.schemas.organizations import UserResponseSchema
from nexusml.api.utils import delete_auth0_user
from nexusml.api.utils import get_auth0_management_api_token
from nexusml.api.utils import get_auth0_user_data
from nexusml.constants import ADMIN_ROLE
from nexusml.constants import API_NAME
from nexusml.constants import MAINTAINER_ROLE
from nexusml.database.base import Entity
from nexusml.database.core import save_to_db
from nexusml.database.files import OrgFileDB as FileDB
from nexusml.database.organizations import Agent
from nexusml.database.organizations import ClientDB
from nexusml.database.organizations import CollaboratorDB
from nexusml.database.organizations import KNOWN_CLIENT_IDS
from nexusml.database.organizations import OrganizationDB
from nexusml.database.organizations import RoleDB
from nexusml.database.organizations import user_roles as user_roles_table
from nexusml.database.organizations import UserDB
from nexusml.database.permissions import RolePermission
from nexusml.database.permissions import UserPermission
from nexusml.database.services import Service
from nexusml.database.subscriptions import get_active_subscription as get_active_subscription_
from nexusml.database.subscriptions import quotas
from nexusml.database.subscriptions import SubscriptionDB
from nexusml.database.subscriptions import SubscriptionExtra
from nexusml.enums import NotificationSource
from nexusml.enums import OrgFileUse
from nexusml.enums import ResourceAction
from nexusml.enums import ResourceType
from nexusml.env import ENV_AUTH0_DOMAIN
from nexusml.env import ENV_SUPPORT_EMAIL

_CONTACT_MSG = f'Please, contact {os.environ[ENV_SUPPORT_EMAIL]}'

# TODO: Place all Auth0 functions and classes in a single module.


def _update_quota_usage(name: str, description: str = None, delta: Union[int, float] = 1):
    """Decorator for updating quota usage in POST and DELETE.

    Args:
        name (str): The name of the quota.
        description (str, optional): The description of the quota.
        delta (Union[int, float], optional): The change in quota usage. Defaults to 1.

    Returns:
        function: The wrapped function with quota update logic.
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            assert func.__name__ in ['post', 'delete']
            is_post = func.__name__ == 'post'
            org = kwargs['parents'][0] if is_post else args[0].parents()[0]
            assert isinstance(org, Organization)
            if is_post:
                org.check_quota_usage(name=name, description=description, delta=delta)
            result = func(*args, **kwargs)
            org.update_quota_usage(name=name, delta=(1 if is_post else -1))
            return result

        return wrapper

    return decorator


class User(Resource):

    @classmethod
    def db_model(cls):
        return UserDB

    @classmethod
    def load_schema(cls):
        return UserRequestSchema

    @classmethod
    def dump_schema(cls):
        return UserResponseSchema

    @classmethod
    def location(cls) -> str:
        return ENDPOINT_USER

    @classmethod
    @_update_quota_usage(name='users', description='Maximum number of users')
    def post(cls,
             agent: Agent,
             data: dict,
             parents: list = None,
             check_permissions: bool = True,
             check_parents: bool = True,
             notify_to: Iterable[UserDB] = None):
        """Handles POST request for creating a new user.

        Args:
            agent (Agent): The agent performing the action.
            data (dict): The data for the new user.
            parents (list, optional): The parent resources. Defaults to None.
            check_permissions (bool, optional): Whether to check permissions. Defaults to True.
            check_parents (bool, optional): Whether to check parents. Defaults to True.
            notify_to (Iterable[UserDB], optional): Users to notify. Defaults to None.

        Returns:
            User: The newly created user resource.

        Raises:
            UnprocessableRequestError: If the email domain does not match the organization's domain.
            DuplicateResourceError: If the user already exists in the organization.
        """
        assert parents and isinstance(parents[0], Organization)
        org = parents[0].db_object()

        assert isinstance(agent, UserDB)

        # Check if the user already exists in DB
        new_user_db_obj = UserDB.query().filter_by(auth0_id=data['auth0_id']).first()
        if new_user_db_obj:
            org_msg = 'this' if new_user_db_obj.organization_id == org.organization_id else 'another'
            raise DuplicateResourceError(f'User already exists in {org_msg} organization')

        # Check user domain
        if data['email'].split('@')[-1] != org.domain:
            raise UnprocessableRequestError("Email address doesn't belong to organization's domain")

        # Create user
        new_user = super().post(agent=agent,
                                data={'auth0_id': data['auth0_id']},
                                parents=parents,
                                check_permissions=check_permissions,
                                check_parents=check_parents,
                                notify_to=notify_to)

        # Assign read permissions on session user's organization to the new user
        new_user_perm = UserPermission(user_id=new_user.db_object().user_id,
                                       organization_id=org.organization_id,
                                       resource_uuid=org.uuid,
                                       resource_type=ResourceType.ORGANIZATION,
                                       action=ResourceAction.READ,
                                       allow=True)
        save_to_db(new_user_perm)

        return new_user

    @_update_quota_usage(name='users')
    def delete(self, notify_to: Iterable[UserDB] = None):
        """Handles DELETE request for deleting a user.

        Args:
            notify_to (Iterable[UserDB], optional): Users to notify. Defaults to None.

        Raises:
            PermissionDeniedError: If an admin or maintainer tries to delete a user without proper permissions.
        """
        user_roles = get_user_roles(user=self.db_object())
        session_user_roles = get_user_roles(user=self.user())
        is_session_user = self.db_object().user_id == self.user().user_id
        if ADMIN_ROLE in user_roles and not is_session_user:
            raise PermissionDeniedError('Admins can only be deleted by themselves')
        elif MAINTAINER_ROLE in user_roles and not (ADMIN_ROLE in session_user_roles or is_session_user):
            raise PermissionDeniedError('Maintainers can only be deleted by an admin or by themselves')

        check_last_admin_deletion(user=self, user_roles=user_roles)

        delete_auth0_user(auth0_id=self.db_object().auth0_id)
        super().delete(notify_to=notify_to)

    @staticmethod
    def download_auth0_user_data(auth0_id_or_email: str) -> dict:
        """
        Downloads user's data from Auth0.

        Different info is returned depending on session user's roles:

            - Regular users can query only public user info.
            - Admins and maintainers can query all user info.

        Apps (clients) can query all user info.

        Args:
            auth0_id_or_email (str): The user's `auth0_id` or email.

        Returns:
            dict: The account data.

        Raises:
            ResourceNotFoundError: If no account is found for the given UUID or email.
        """
        mgmt_api_access_token: str = get_auth0_management_api_token()
        auth0_user_data: dict = get_auth0_user_data(access_token=mgmt_api_access_token,
                                                    auth0_id_or_email=auth0_id_or_email)

        if not auth0_user_data:
            raise ResourceNotFoundError(f'No Auth0 user found for {auth0_id_or_email}')

        public_user_info: dict = {
            'auth0_id': auth0_user_data['user_id'],
            'email': auth0_user_data['email'],
            'email_verified': auth0_user_data.get('email_verified', False),
            'first_name': auth0_user_data.get('given_name'),
            'last_name': auth0_user_data.get('family_name'),
        }

        return public_user_info

    def put(self, data: dict, notify_to: Iterable[UserDB] = None) -> None:
        self._check_permissions(action=ResourceAction.UPDATE)

        fields_map: dict = {'first_name': 'given_name', 'last_name': 'family_name'}
        updated_data: dict = {fields_map.get(key, key): value for key, value in data.items()}

        mgmt_api_access_token = get_auth0_management_api_token()
        url = f'https://{os.environ[ENV_AUTH0_DOMAIN]}/api/v2/users/{g.user_auth0_id}'
        headers = {'Authorization': f'Bearer {mgmt_api_access_token}', 'content-type': 'application/json'}

        response = requests.patch(url, json=updated_data, headers=headers)
        if response.status_code < 200 or response.status_code >= 300:
            # TODO: Replace this and other 'ifs' with response http error handler method 'raise_for_status'
            raise requests.HTTPError('Auth0 patch request error')

    def dump(
        self,
        serialize=True,
        expand_associations=False,
        reference_parents=False,
        update_sync_state: bool = True,
    ) -> Union[ResourceResponseSchema, dict]:

        db_user_data: dict = super().dump(serialize=False,
                                          expand_associations=expand_associations,
                                          reference_parents=reference_parents,
                                          update_sync_state=update_sync_state)

        db_user_data.pop('organization_id')
        db_user_data.pop('user_id')
        db_user_data.pop('public_id')

        auth0_id: str = self.db_object().auth0_id
        auth0_user_data: dict = self.download_auth0_user_data(auth0_id_or_email=auth0_id)
        public_auth0_user_data: dict = {
            'email': auth0_user_data['email'],
            'first_name': auth0_user_data['first_name'],
            'last_name': auth0_user_data['last_name'],
            'email_verified': auth0_user_data['email_verified']
        }

        public_user_data: dict = db_user_data | public_auth0_user_data
        response_data: Union[ResourceResponseSchema, dict] = self.dump_data(data=public_user_data, serialize=serialize)

        return response_data


class Role(Resource):

    @classmethod
    def db_model(cls):
        return RoleDB

    @classmethod
    def load_schema(cls):
        return RoleRequestSchema

    @classmethod
    def dump_schema(cls):
        return RoleResponseSchema

    @classmethod
    def location(cls) -> str:
        return ENDPOINT_ROLE

    @classmethod
    @_update_quota_usage(name='roles', description='Maximum number of roles')
    def post(cls,
             agent: Agent,
             data: dict,
             parents: list = None,
             check_permissions: bool = True,
             check_parents: bool = True,
             notify_to: Iterable[UserDB] = None):
        """Handles POST request for creating a new role.

        Args:
            agent (Agent): The agent performing the action.
            data (dict): The data for the new role.
            parents (list, optional): The parent resources. Defaults to None.
            check_permissions (bool, optional): Whether to check permissions. Defaults to True.
            check_parents (bool, optional): Whether to check parents. Defaults to True.
            notify_to (Iterable[UserDB], optional): Users to notify. Defaults to None.

        Returns:
            Role: The newly created role resource.
        """
        assert parents and isinstance(parents[0], Organization)
        org = parents[0].db_object()

        # Create role
        new_role = super().post(agent=agent,
                                data=data,
                                parents=parents,
                                check_permissions=check_permissions,
                                check_parents=check_parents,
                                notify_to=notify_to)

        # Assign read permissions to the new role on session user's organization
        new_role_perm = RolePermission(role_id=new_role.db_object().role_id,
                                       resource_uuid=org.uuid,
                                       resource_type=ResourceType.ORGANIZATION,
                                       action=ResourceAction.READ,
                                       allow=True)
        save_to_db(new_role_perm)

        return new_role

    def put(self, data: dict, notify_to: Iterable[User] = None):
        """Handles PUT request for updating a role.

        Args:
            data (dict): The updated data for the role.
            notify_to (Iterable[User], optional): Users to notify. Defaults to None.

        Raises:
            PermissionDeniedError: If an attempt is made to modify admin or maintainer roles.
        """
        if self.db_object().name in [ADMIN_ROLE, MAINTAINER_ROLE]:
            raise PermissionDeniedError('Admin and Maintainer roles cannot be modified')
        super().put(data=data, notify_to=notify_to)

    @_update_quota_usage(name='roles')
    def delete(self, notify_to: Iterable[UserDB] = None):
        """Handles DELETE request for deleting a role.

        Args:
            notify_to (Iterable[UserDB], optional): Users to notify. Defaults to None.

        Raises:
            PermissionDeniedError: If an attempt is made to delete admin or maintainer roles.
        """
        if self.db_object().name in [ADMIN_ROLE, MAINTAINER_ROLE]:
            raise PermissionDeniedError('Admin and Maintainer roles cannot be deleted')
        super().delete(notify_to=notify_to)


class Collaborator(Resource):

    @classmethod
    def db_model(cls):
        return CollaboratorDB

    @classmethod
    def load_schema(cls):
        return CollaboratorRequestSchema

    @classmethod
    def dump_schema(cls):
        return CollaboratorResponseSchema

    @classmethod
    def location(cls) -> str:
        return ENDPOINT_COLLABORATOR

    @classmethod
    @_update_quota_usage(name='collaborators', description='Maximum number of collaborators')
    def post(cls,
             agent: Agent,
             data: dict,
             parents: list = None,
             check_permissions: bool = True,
             check_parents: bool = True,
             notify_to: Iterable[UserDB] = None):
        """Handles POST request for creating a new collaborator.

        Args:
            agent (Agent): The agent performing the action.
            data (dict): The data for the new collaborator.
            parents (list, optional): The parent resources. Defaults to None.
            check_permissions (bool, optional): Whether to check permissions. Defaults to True.
            check_parents (bool, optional): Whether to check parents. Defaults to True.
            notify_to (Iterable[UserDB], optional): Users to notify. Defaults to None.

        Returns:
            Collaborator: The newly created collaborator resource.

        Raises:
            UnprocessableRequestError: If the email domain matches the organization's domain.
            ResourceNotFoundError: If the email address is not registered in the API.
            DuplicateResourceError: If the user is already a member of the organization.
        """
        assert parents and isinstance(parents[0], Organization)

        # Get or create collaborator's user data from Auth0
        email = data.pop('email')
        assert isinstance(agent, UserDB)

        try:
            auth0_user_data: dict = User.download_auth0_user_data(auth0_id_or_email=email)
        except ResourceNotFoundError:
            raise ResourceNotFoundError(f'User with email "{email}" not registered')

        # Get collaborator's user
        collaborator = UserDB.query().filter_by(auth0_id=auth0_user_data['auth0_id']).first()
        assert collaborator is not None

        if collaborator.organization_id == agent.organization_id:
            raise DuplicateResourceError(f'User with email address "{email}" is a member of the organization')

        # Fill data
        data['user_id'] = collaborator.user_id
        data['organization_id'] = agent.organization_id

        return super().post(agent=agent,
                            data=data,
                            parents=parents,
                            check_permissions=check_permissions,
                            check_parents=check_parents,
                            notify_to=notify_to)

    @_update_quota_usage(name='collaborators')
    def delete(self, notify_to: Iterable[UserDB] = None):
        """Handles DELETE request for deleting a collaborator.

        Args:
            notify_to (Iterable[UserDB], optional): Users to notify. Defaults to None.
        """
        super().delete(notify_to=notify_to)

    def dump(self,
             serialize=True,
             expand_associations=False,
             reference_parents=False,
             update_sync_state: bool = True) -> Union[ResourceResponseSchema, dict]:
        """
        Dumps the collaborator data.

        Args:
            serialize (bool, optional): Whether to serialize the data. Defaults to True.
            expand_associations (bool, optional): Whether to expand associations. Defaults to False.
            reference_parents (bool, optional): Whether to reference parents. Defaults to False.
            update_sync_state (bool, optional): Whether to update the sync state. Defaults to True.

        Returns:
            Union[ResourceResponseSchema, dict]: The dumped collaborator data.
        """
        # Download Auth0 user data
        user_db_obj = UserDB.get(user_id=self.db_object().user_id)
        auth0_user_data: dict = User.download_auth0_user_data(auth0_id_or_email=user_db_obj.auth0_id)

        # Fill collaborator's info
        data = {
            'id': self.db_object().public_id,
            'uuid': self.db_object().uuid,
            'organization': user_db_obj.organization.name,
        }
        data.update(auth0_user_data)

        # Serialize data
        return self.dump_schema()().dump(data) if serialize else data


class Client(Resource):

    @classmethod
    def db_model(cls):
        return ClientDB

    @classmethod
    def load_schema(cls):
        return AppRequestSchema

    @classmethod
    def dump_schema(cls):
        return AppResponseSchema

    @classmethod
    def location(cls) -> str:
        return ENDPOINT_CLIENT

    @_update_quota_usage(name='clients')
    def delete(self, notify_to: Iterable[UserDB] = None):
        """Handles DELETE request for deleting a client.

        Args:
            notify_to (Iterable[UserDB], optional): Users to notify. Defaults to None.

        Raises:
            PermissionDeniedError: If the client is a default application or service.
        """
        is_service = Service.query().filter_by(client_id=self.db_object().client_id).first() is not None
        if is_service or self.db_object().client_id in KNOWN_CLIENT_IDS.values():
            raise PermissionDeniedError(f'Default {API_NAME} applications and services cannot be deleted')
        super().delete(notify_to=notify_to)

    @classmethod
    def get(cls,
            agent: Agent,
            db_object_or_id: Union[Entity, str],
            parents: list = None,
            cache: bool = False,
            check_permissions: bool = True,
            check_parents: bool = True,
            remove_notifications: bool = False):
        """Handles GET request for retrieving a client.

        Args:
            agent (Agent): The agent performing the action.
            db_object_or_id (Union[Entity, str]): The database object or ID.
            parents (list, optional): The parent resources. Defaults to None.
            cache (bool, optional): Whether to use cache. Defaults to False.
            check_permissions (bool, optional): Whether to check permissions. Defaults to True.
            check_parents (bool, optional): Whether to check parents. Defaults to True.
            remove_notifications (bool, optional): Whether to remove notifications. Defaults to False.

        Returns:
            Client: The retrieved client resource.

        Raises:
            ResourceNotFoundError: If the client is running a service.
        """
        # Get client
        client = super().get(agent=agent,
                             db_object_or_id=db_object_or_id,
                             parents=parents,
                             cache=cache,
                             check_permissions=check_permissions,
                             check_parents=check_parents,
                             remove_notifications=remove_notifications)
        # Verify the client is not running a service (service clients are not visible in the API)
        if Service.query().filter_by(client_id=client.db_object().client_id).first() is not None:
            resource_id = db_object_or_id if isinstance(db_object_or_id, str) else client.public_id()
            raise ResourceNotFoundError(resource_id=resource_id)
        return client

    @classmethod
    @_update_quota_usage(name='clients', description='Maximum number of apps')
    def post(cls,
             agent: Agent,
             data: dict,
             parents: list = None,
             check_permissions: bool = True,
             check_parents: bool = True,
             notify_to: Iterable[UserDB] = None):
        """Handles POST request for creating a new client.

        Args:
            agent (Agent): The agent performing the action.
            data (dict): The data for the new client.
            parents (list, optional): The parent resources. Defaults to None.
            check_permissions (bool, optional): Whether to check permissions. Defaults to True.
            check_parents (bool, optional): Whether to check parents. Defaults to True.
            notify_to (Iterable[UserDB], optional): Users to notify. Defaults to None.

        Returns:
            Client: The newly created client resource.
        """
        return super().post(agent=agent,
                            data=data,
                            parents=parents,
                            check_permissions=check_permissions,
                            check_parents=check_parents,
                            notify_to=notify_to)

    def put(self, data: dict, notify_to: Iterable[User] = None):
        """Handles PUT request for updating a client.

        Args:
            data (dict): The updated data for the client.
            notify_to (Iterable[User], optional): Users to notify. Defaults to None.

        Raises:
            PermissionDeniedError: If the client is a default application or service.
        """
        is_service = Service.query().filter_by(client_id=self.db_object().client_id).first() is not None
        if is_service or self.db_object().client_id in KNOWN_CLIENT_IDS.values():
            raise PermissionDeniedError(f'Default {API_NAME} applications and services cannot be modified')
        super().put(data=data, notify_to=notify_to)

    def dump(self,
             serialize=True,
             expand_associations=False,
             reference_parents=False,
             update_sync_state: bool = True) -> Union[ResourceResponseSchema, dict]:
        """
        Dumps the client data.

        WARNING: `icon` field is always expanded, even if `expand_associations=False`.

        Args:
            serialize (bool, optional): Whether to serialize the data. Defaults to True.
            expand_associations (bool, optional): Whether to expand associations. Defaults to False.
            reference_parents (bool, optional): Whether to reference parents. Defaults to False.
            update_sync_state (bool, optional): Whether to update the sync state. Defaults to True.

        Returns:
            Union[ResourceResponseSchema, dict]: The dumped client data.
        """
        # Dump resource data
        dumped_data = super().dump(serialize=False,
                                   expand_associations=expand_associations,
                                   reference_parents=reference_parents,
                                   update_sync_state=update_sync_state)
        # Get icon file
        if dumped_data['icon'] is not None:
            icon_file = File.get(agent=self.agent(),
                                 db_object_or_id=FileDB.get(file_id=self.db_object().icon),
                                 parents=self.parents(),
                                 check_permissions=False,
                                 check_parents=False)
            dumped_data['icon'] = icon_file.dump(serialize=False)
        # Return dumped data
        return self.dump_data(dumped_data, serialize=serialize)

    def _set_data(self, data: dict, notify_to: Iterable[User] = None):
        """
        Sets the client data.

        Args:
            data (dict): The data to set.
            notify_to (Iterable[User], optional): Users to notify. Defaults to None.

        Raises:
            InvalidDataError: If the icon file is not valid.
        """
        # Get icon file
        icon_file_id = data.get('icon', None)
        if icon_file_id:
            icon_file = File.get(agent=self.agent(),
                                 db_object_or_id=icon_file_id,
                                 parents=self.parents(),
                                 check_permissions=False,
                                 check_parents=False)
            if icon_file.db_object().use_for != OrgFileUse.PICTURE:
                raise InvalidDataError('Invalid app icon')
            data['icon'] = icon_file.db_object().file_id
        # Set resource data
        super()._set_data(data=data, notify_to=notify_to)


class Organization(Resource):

    @classmethod
    def db_model(cls):
        return OrganizationDB

    @classmethod
    def load_schema(cls):
        return OrganizationPUTRequestSchema

    @classmethod
    def dump_schema(cls):
        return OrganizationResponseSchema

    @classmethod
    def location(cls) -> str:
        return ENDPOINT_ORGANIZATION

    @classmethod
    def collections(cls) -> dict:
        return {'users': User, 'roles': Role, 'collaborators': Collaborator, 'apps': Client}

    @classmethod
    def permission_resource_type(cls) -> ResourceType:
        return ResourceType.ORGANIZATION

    @classmethod
    def notification_source_type(cls) -> NotificationSource:
        return NotificationSource.ORGANIZATION

    @classmethod
    def post(cls,
             agent: Agent,
             data: dict,
             parents: list = None,
             check_permissions: bool = True,
             check_parents: bool = True,
             notify_to: Iterable[UserDB] = None):
        """Handles POST request for creating an organization.

        Args:
            agent (Agent): The agent performing the action.
            data (dict): The data for the new organization.
            parents (list, optional): The parent resources. Defaults to None.
            check_permissions (bool, optional): Whether to check permissions. Defaults to True.
            check_parents (bool, optional): Whether to check parents. Defaults to True.
            notify_to (Iterable[UserDB], optional): Users to notify. Defaults to None.

        Raises:
            NotImplementedError: This method is not implemented due to circular dependency.
        """
        # Since `UserDB` database model references `OrganizationDB`, there would be a circular dependency
        raise NotImplementedError()

    def put(self, data: dict, notify_to: Iterable[UserDB] = None):
        """Handles PUT request for updating an organization.

        Args:
            data (dict): The updated data for the organization.
            notify_to (Iterable[UserDB], optional): Users to notify. Defaults to None.

        Raises:
            PermissionDeniedError: If the user is not an admin.
            DuplicateResourceError: If the TRN is already in use by another organization.
            InvalidDataError: If the logo file is not valid.
        """
        # An Organization can only be modified by its Admin(s)
        # Note: admin role is already required by `PUT /<organization_id>` endpoint. This is a double check
        if ADMIN_ROLE not in get_user_roles(user=self.user()):
            raise PermissionDeniedError('An organization can only be modified by its admin(s)')
        # Check whether provided TRN has been modified and is already in use by another organization
        if data['trn'] != self.db_object().trn and OrganizationDB.get_from_id(id_value=data['trn']) is not None:
            raise DuplicateResourceError('TRN already in use by another organization')
        # Get logo file
        logo_file_id = data.get('logo', None)
        if logo_file_id:
            logo_file = File.get(agent=self.agent(),
                                 db_object_or_id=logo_file_id,
                                 parents=[self],
                                 check_permissions=False,
                                 check_parents=False)
            if logo_file.db_object().use_for != OrgFileUse.PICTURE:
                raise InvalidDataError('Invalid logo image')
            data['logo'] = logo_file.db_object().file_id
        # Update organization
        super().put(data=data, notify_to=notify_to)

    def delete(self, notify_to: Iterable[User] = None):
        """Handles DELETE request for deleting an organization.

        Args:
            notify_to (Iterable[User], optional): Users to notify. Defaults to None.

        Raises:
            PermissionDeniedError: If the user is not an admin.
        """
        # An Organization can only be deleted by its Admin(s)
        # Note: admin role is already required by `DELETE /<organization_id>` endpoint. This is a double check
        if ADMIN_ROLE not in get_user_roles(user=self.user()):
            raise PermissionDeniedError('An organization can only be deleted by its admin(s)')
        super().delete(notify_to=notify_to)

    def dump(self,
             serialize=True,
             expand_associations=False,
             reference_parents=False,
             update_sync_state: bool = True) -> Union[ResourceResponseSchema, dict]:
        """Dumps the organization data.

        WARNING: `logo` field is always expanded, even if `expand_associations=False`.

        Args:
            serialize (bool, optional): Whether to serialize the data. Defaults to True.
            expand_associations (bool, optional): Whether to expand associations. Defaults to False.
            reference_parents (bool, optional): Whether to reference parents. Defaults to False.
            update_sync_state (bool, optional): Whether to update the sync state. Defaults to True.

        Returns:
            Union[ResourceResponseSchema, dict]: The dumped organization data.
        """
        # Dump resource data
        dumped_data = super().dump(serialize=False,
                                   expand_associations=expand_associations,
                                   reference_parents=reference_parents,
                                   update_sync_state=update_sync_state)
        # Get logo file
        if dumped_data['logo'] is not None:
            logo_file = File.get(agent=self.agent(),
                                 db_object_or_id=FileDB.get(file_id=self.db_object().logo),
                                 parents=[self],
                                 check_permissions=False,
                                 check_parents=False)
            dumped_data['logo'] = logo_file.dump(serialize=False)
        # Return dumped data
        return self.dump_data(dumped_data, serialize=serialize)

    def check_quota_usage(self, name: str, description: str = None, cache: bool = False, delta: Union[int, float] = 0):
        """Checks the subscription's quota usage.

        Args:
            name (str): The name of the quota.
            description (str, optional): The description of the quota. Defaults to None.
            cache (bool, optional): Whether to use cache. Defaults to False.
            delta (Union[int, float], optional): The virtual increase/decrease in usage. Defaults to 0.
        """
        quota = quotas[name]
        if cache:
            subscription = get_active_subscription(organization_id=self.db_object().organization_id)
            extras = get_active_subscription_extras(subscription_id=subscription.subscription_id)
        else:
            subscription = load_active_subscription(organization_id=self.db_object().organization_id)
            extras = load_active_subscription_extras(subscription_id=subscription.subscription_id)
        plan_quota = getattr(subscription.plan, quota['limit'])
        extra_quota = sum(getattr(x.extra, quota['extra']) for x in extras)
        check_quota_usage_(name=name,
                           usage=(getattr(subscription, quota['usage']) + delta),
                           limit=(plan_quota + extra_quota),
                           description=description)

    def update_quota_usage(self, name: str, delta: Union[int, float]):
        """Updates the subscription's quota usage.

        Args:
            name (str): The name of the quota.
            delta (Union[int, float]): The change in quota usage.

        Raises:
            ResourceError: If there is a problem with the quota update.
        """
        org_id = self.db_object().organization_id
        if self.cached():
            subscription = get_active_subscription(organization_id=org_id)
            cache.delete_memoized(get_active_subscription, org_id)  # remove active subscription from cache
        else:
            subscription = load_active_subscription(organization_id=org_id)
        try:
            subscription.update_numeric_value(column=quotas[name]['usage'], delta=delta)
        except OperationalError:
            raise ResourceError('There seems to be a problem with your quota. ' + _CONTACT_MSG)


def load_active_subscription(organization_id: int) -> SubscriptionDB:
    """Loads the active subscription for the given organization ID.

    Args:
        organization_id (int): The organization ID.

    Returns:
        SubscriptionDB: The active subscription.

    Raises:
        ResourceNotFoundError: If no active subscriptions are found.
        ResourceError: If there is a problem with the subscription.
    """
    try:
        subscription = get_active_subscription_(organization_id=organization_id)
        if subscription is None:
            raise ResourceNotFoundError('No active subscriptions found')
        subscription.force_relationship_loading()
    except AssertionError:
        raise ResourceError('There seems to be a problem with your subscription. ' + _CONTACT_MSG)
    return subscription


def load_active_subscription_extras(subscription_id: int) -> List[SubscriptionExtra]:
    """Loads the active subscription extras for the given subscription ID.

    Args:
        subscription_id (int): The subscription ID.

    Returns:
        List[SubscriptionExtra]: The list of active subscription extras.
    """

    def _is_active(extra: SubscriptionExtra) -> bool:
        started = extra.start_at <= datetime.utcnow()
        ended = extra.end_at is not None and extra.end_at <= datetime.utcnow()
        canceled = extra.cancel_at is not None and extra.cancel_at <= datetime.utcnow()
        return started and not (ended or canceled)

    extras = SubscriptionExtra.query().filter_by(subscription_id=subscription_id).all()
    active_extras = [x for x in extras if _is_active(extra=x)]
    for active_extra in active_extras:
        active_extra.subscription.force_relationship_loading()
    return active_extras


@cache.memoize()
def get_active_subscription(organization_id: int) -> SubscriptionDB:
    """Gets the active subscription for the given organization ID, using cache.

    Args:
        organization_id (int): The organization ID.

    Returns:
        SubscriptionDB: The active subscription.
    """
    return load_active_subscription(organization_id=organization_id)


@cache.memoize()
def get_active_subscription_extras(subscription_id: int) -> List[SubscriptionExtra]:
    """Gets the active subscription extras for the given subscription ID, using cache.

    Args:
        subscription_id (int): The subscription ID.

    Returns:
        List[SubscriptionExtra]: The list of active subscription extras.
    """
    return load_active_subscription_extras(subscription_id=subscription_id)


@cache.memoize()
def get_user_roles(user: UserDB) -> List[str]:
    """Gets the roles for the given user, using cache.

    Args:
        user (UserDB): The user.

    Returns:
        List[str]: The list of user roles.
    """
    return [role.name for role in user.roles]


def get_resource_organization(resource: Resource) -> Optional[Organization]:
    """Gets the organization for the given resource, looking at parent relationships.

    WARNING: even if `resource` may not have any parent of type `Organization`,
             its database object may still have a parent relationship instance of `Organization`'s database model.
             For example, tasks are accessed directly from `/tasks` endpoint instead of `/<organization_id>/tasks`,
             so they do not have any parent resource, while having an (optional) `organization` parent relationship
             at database level. We chose this design because we are supporting orphan tasks that don't belong to
             any organization.

    Args:
        resource (Resource): The resource.

    Returns:
        Optional[Organization]: The organization, if found.
    """
    # Look at current resource and direct parents
    for rsrc in [resource] + resource.parents():
        if isinstance(rsrc, Organization):
            return rsrc
        db_object = rsrc.db_object()
        for db_relationship in db_object.relationships():
            db_rel_object = getattr(db_object, db_relationship)
            if isinstance(db_rel_object, OrganizationDB):
                return Organization.get(agent=resource.agent(), db_object_or_id=db_rel_object, check_permissions=False)

    # Recursively look at parents of parents
    for parent in resource.parents():
        organization = get_resource_organization(parent)
        if isinstance(organization, Organization):
            return organization
    return


def check_last_admin_deletion(user, user_roles: list) -> None:
    admins_count: int = user.db_model().query().join(
        user_roles_table, user_roles_table.c.user_id == user.db_model().user_id).join(
            RoleDB, user_roles_table.c.role_id == RoleDB.role_id).filter(
                user.db_model().organization_id == user.db_object().organization_id,
                RoleDB.name == ADMIN_ROLE,
            ).count()
    if ADMIN_ROLE in user_roles and admins_count <= 1:
        raise ResourceError('There must be at least one admin user')
