from typing import List, Union

from flask import g
from flask import jsonify
from flask import Response
from flask_apispec import doc
from flask_apispec import marshal_with
from flask_apispec import use_kwargs
import jwt
from marshmallow import fields

from nexusml.api.external.auth0 import Auth0Manager
from nexusml.api.resources.base import collaborators_permissions
from nexusml.api.resources.base import dump
from nexusml.api.resources.base import ResourceNotFoundError
from nexusml.api.resources.base import users_permissions
from nexusml.api.resources.myaccount import AggregatedNotification
from nexusml.api.resources.myaccount import Notification
from nexusml.api.resources.organizations import Organization
from nexusml.api.resources.organizations import Role
from nexusml.api.resources.organizations import User
from nexusml.api.resources.tasks import Task
from nexusml.api.schemas.myaccount import ClientSettingsRequestSchema
from nexusml.api.schemas.myaccount import ClientSettingsResponseSchema
from nexusml.api.schemas.myaccount import MyAccountRolesSchema
from nexusml.api.schemas.myaccount import NotificationSchema
from nexusml.api.schemas.myaccount import SettingsSchema
from nexusml.api.schemas.organizations import OrganizationPermissionsPage
from nexusml.api.schemas.organizations import OrganizationResponseSchema
from nexusml.api.schemas.organizations import UserResponseSchema
from nexusml.api.schemas.organizations import UserUpdateSchema
from nexusml.api.views.base import create_view
from nexusml.api.views.common import paginated_response
from nexusml.api.views.common import permissions_jsons
from nexusml.api.views.core import agent_from_token
from nexusml.api.views.core import error_response
from nexusml.api.views.core import process_delete_request
from nexusml.api.views.core import process_get_request
from nexusml.api.views.core import process_post_or_put_request
from nexusml.api.views.utils import paging_url_params
from nexusml.constants import ADMIN_ROLE
from nexusml.constants import HTTP_DELETE_STATUS_CODE
from nexusml.constants import HTTP_FORBIDDEN_STATUS_CODE
from nexusml.constants import SWAGGER_TAG_MYACCOUNT
from nexusml.database.core import delete_from_db
from nexusml.database.core import save_to_db
from nexusml.database.myaccount import AccountClientSettings
from nexusml.database.myaccount import AccountSettings
from nexusml.database.notifications import AggregatedNotificationDB
from nexusml.database.notifications import NotificationDB
from nexusml.database.organizations import ClientDB
from nexusml.database.organizations import InvitationDB
from nexusml.database.organizations import RoleDB
from nexusml.database.organizations import user_roles
from nexusml.database.organizations import UserDB
from nexusml.enums import InviteStatus

################
# Define views #
################

_MyAccountView = create_view(reject_api_keys=['DELETE', 'GET', 'POST', 'PUT'])

_notifications_query_params = {
    'task_id': fields.String(description='Public ID or UUID of the task'),
    'source_uuid': fields.String(description='UUID of the source')
}


class MyAccountView(_MyAccountView):
    """
    View for handling user account related requests, including retrieving and deleting user data.
    """

    def _get_user_from_token(self) -> User:
        """
        Retrieve user object from the authentication token.

        Returns:
            User: The user object associated with the authentication token.
        """
        # TODO: Check this try except (except is a good path). Non exception logic should not be inside an except.
        try:
            user_db_object: UserDB = agent_from_token()
        except jwt.InvalidTokenError as token_error:
            auth0_manager: Auth0Manager = Auth0Manager()
            auth0_user_data: dict = auth0_manager.get_auth0_user_data(auth0_id_or_email=g.user_auth0_id)
            user_invitation: InvitationDB = self._get_user_invitation(token_error=token_error,
                                                                      email=auth0_user_data['email'])
            new_user: User = self._create_new_user_from_invitation(user_invitation=user_invitation,
                                                                   user_auth0_id=auth0_user_data['user_id'])
            return new_user

        organization = Organization.get(agent=user_db_object,
                                        db_object_or_id=user_db_object.organization,
                                        check_permissions=False)
        return User.get(agent=user_db_object,
                        db_object_or_id=user_db_object,
                        parents=[organization],
                        check_permissions=False,
                        check_parents=False)

    def _get_user_invitation(self, token_error: jwt.InvalidTokenError, email: str) -> InvitationDB:
        """
        Retrieves the user invitation from the database based on the Auth0 user data.

        This function fetches an Auth0 management API token and uses it to obtain user data
        either by Auth0 ID or email. It then queries the database for a pending user
        invitation corresponding to the email found in the Auth0 user data. If no such
        invitation is found, it raises the provided token error.

        Args:
            token_error (jwt.InvalidTokenError): The error to raise if the user invitation is not found.

        Returns:
            InvitationDB: The user invitation if found.

        Raises:
            jwt.InvalidTokenError: If the user invitation is not found.
        """
        user_invitation: InvitationDB = InvitationDB.query().filter_by(email=email, status=InviteStatus.PENDING).first()
        if not user_invitation:
            raise token_error

        return user_invitation

    def _create_new_user_from_invitation(self, user_invitation: InvitationDB, user_auth0_id: str) -> User:
        """
        Creates a new user from the provided user invitation.

        This function checks for an existing admin user in the same organization as the
        user invitation since an admin user is needed to create another user.
        It then creates a new user using the email from the user invitation
        and associates it with the organization. The status of the user invitation is
        updated to accepted and saved to the database.

        Args:
            user_invitation (InvitationDB): The user invitation to create a new user from.

        Returns:
            User: The newly created user.
        """
        admin_user_db_obj: UserDB = UserDB.query().join(user_roles, user_roles.c.user_id == UserDB.user_id).join(
            RoleDB, user_roles.c.role_id == RoleDB.role_id).filter(
                UserDB.organization_id == user_invitation.organization_id,
                RoleDB.name == ADMIN_ROLE,
            ).first()

        organization: Organization = Organization.get(agent=admin_user_db_obj,
                                                      db_object_or_id=admin_user_db_obj.organization,
                                                      check_permissions=False)

        new_user = User()
        new_user.post(agent=admin_user_db_obj,
                      data={
                          'email': user_invitation.email,
                          'auth0_id': user_auth0_id
                      },
                      parents=[organization])

        user_invitation.status = InviteStatus.ACCEPTED
        save_to_db(user_invitation)

        return new_user

    @doc(tags=[SWAGGER_TAG_MYACCOUNT])
    def delete(self):
        """
        Handle DELETE request to remove user account.

        Returns:
            Response: The response object indicating the result of the delete operation.
        """
        return process_delete_request(resource=self._get_user_from_token())

    @doc(tags=[SWAGGER_TAG_MYACCOUNT])
    def get(self):
        """
        Handle GET request to retrieve user account data.

        Returns:
            dict: The JSON representation of the user account data.
        """
        user: User = self._get_user_from_token()
        user_data: dict = user.dump(serialize=False)
        if not user_data['email_verified']:
            return error_response(code=HTTP_FORBIDDEN_STATUS_CODE, message='Email not verified')

        response = jsonify(user_data)
        response.headers['Location'] = user.url()
        return response

    @doc(tags=[SWAGGER_TAG_MYACCOUNT])
    @use_kwargs(UserUpdateSchema, location='json')
    @marshal_with(UserResponseSchema)
    def put(self, **kwargs):
        """
        Handle PUT request to update user account data.

        This function allows the user to update their account information. It retrieves the current
        user from the authentication token and processes the request to update the user's details
        based on the provided JSON payload.

        Args:
            **kwargs: JSON payload with fields to update.

        Returns:
            Response: The response object with the updated user data.
        """
        user: User = self._get_user_from_token()
        response: Response = process_post_or_put_request(agent=user.db_object(), resource_or_model=user, json=kwargs)
        return response


class SettingsView(_MyAccountView):
    """
    View for handling user settings related requests, including retrieving and updating user settings.
    """

    @doc(tags=[SWAGGER_TAG_MYACCOUNT], description='Returns a JSON with user settings')
    @marshal_with(SettingsSchema)
    def get(self):
        """
        Handle GET request to retrieve user settings.

        Returns:
            dict: The JSON representation of the user settings.
        """
        user = agent_from_token()
        user_settings = AccountSettings.get(user_id=user.user_id) or AccountSettings(user_id=user.user_id)
        response_json = user_settings.to_dict()
        return response_json

    @doc(tags=[SWAGGER_TAG_MYACCOUNT], description='Update user settings from a JSON')
    @use_kwargs(SettingsSchema, location='json')
    @marshal_with(SettingsSchema)
    def put(self, **kwargs):
        """
        Handle PUT request to update user settings.

        Args:
            kwargs (dict): The new settings data.

        Returns:
            dict: The JSON representation of the updated user settings.
        """
        user = agent_from_token()
        user_settings = AccountSettings.get(user_id=user.user_id) or AccountSettings(user_id=user.user_id)
        user_settings.notifications = kwargs['notifications']  # TODO: use dynamic field assignment
        save_to_db(user_settings)
        response_json = user_settings.to_dict()
        return response_json


class ClientSettingsView(_MyAccountView):
    """
    View for handling client-specific user settings, including retrieving and updating settings.
    """

    @doc(tags=[SWAGGER_TAG_MYACCOUNT],
         description='Returns a JSON with the user settings for the specified client version')
    @use_kwargs({'version': fields.String(required=True, description='Client version')}, location='query')
    @marshal_with(ClientSettingsResponseSchema)
    def get(self, client_id: str, **kwargs):
        """
        Handle GET request to retrieve user settings for a specific client version.

        Args:
            client_id (str): The ID of the client.
            kwargs (dict): Additional query parameters.

        Returns:
            dict: The JSON representation of the client-specific user settings.
        """
        user = agent_from_token()
        client = ClientDB.get_from_id(id_value=client_id, parent=user.organization)
        if client is None:
            raise ResourceNotFoundError(resource_id=client_id)

        user_client_settings = AccountClientSettings.get(user_id=user.user_id,
                                                         client_id=client.client_id,
                                                         client_version=kwargs['version'])
        if user_client_settings is None:
            user_client_settings = AccountClientSettings(user_id=user.user_id,
                                                         client_id=client.client_id,
                                                         client_version=kwargs['version'])
            save_to_db(user_client_settings)

        return {
            'client_id': client_id,
            'client_version': user_client_settings.client_version,
            'settings': user_client_settings.settings
        }

    @doc(tags=[SWAGGER_TAG_MYACCOUNT], description='Update the user settings for the specified client version')
    @use_kwargs(ClientSettingsRequestSchema, location='json')
    @marshal_with(ClientSettingsResponseSchema)
    def put(self, client_id: str, **kwargs):
        """
        Handle PUT request to update user settings for a specific client version.

        Args:
            client_id (str): The ID of the client.
            kwargs (dict): The new settings data.

        Returns:
            dict: The JSON representation of the updated client-specific user settings.
        """
        user = agent_from_token()
        client = ClientDB.get_from_id(id_value=client_id, parent=user.organization)
        if client is None:
            raise ResourceNotFoundError(resource_id=client_id)

        user_client_settings = AccountClientSettings.get(user_id=user.user_id,
                                                         client_id=client.client_id,
                                                         client_version=kwargs['client_version'])
        if user_client_settings is None:
            user_client_settings = AccountClientSettings(user_id=user.user_id,
                                                         client_id=client.client_id,
                                                         client_version=kwargs['client_version'])

        user_client_settings.settings = kwargs['settings']

        save_to_db(user_client_settings)

        return {
            'client_id': client_id,
            'client_version': user_client_settings.client_version,
            'settings': user_client_settings.settings
        }


class NotificationsView(_MyAccountView):
    """
    View for handling notifications related requests, including retrieving and deleting notifications.
    """

    @doc(tags=[SWAGGER_TAG_MYACCOUNT])
    @use_kwargs(_notifications_query_params, location='query')
    def delete(self, **kwargs):
        """
        Handle DELETE request to remove notifications based on query parameters.

        Args:
            kwargs (dict): The query parameters for identifying notifications.

        Returns:
            Response: The response object indicating the result of the delete operation.
        """
        notifications = _get_notifications(query_params=kwargs)

        # We don't call `resources.base.Resource.delete()` because it commits the transaction in each call
        # `delete_from_db()` commits the transaction only once, after deleting all objects
        delete_from_db(n.db_object() for n in notifications)

        return {}, HTTP_DELETE_STATUS_CODE

    @doc(tags=[SWAGGER_TAG_MYACCOUNT])
    @use_kwargs(_notifications_query_params, location='query')
    @marshal_with(NotificationSchema(many=True))
    def get(self, **kwargs):
        """
        Handle GET request to retrieve notifications based on query parameters.

        Args:
            kwargs (dict): The query parameters for identifying notifications.

        Returns:
            dict: The JSON representation of the notifications.
        """
        notifications = _get_notifications(query_params=kwargs)

        return jsonify(dump(notifications))


class NotificationView(_MyAccountView):
    """
    View for handling individual notification requests, including retrieving and deleting a notification.
    """

    @staticmethod
    def _get_notification(notification_id: str) -> Union[Response, Notification, AggregatedNotification]:
        """
        Retrieve a notification object based on the notification ID.

        Args:
            notification_id (str): The ID of the notification.

        Returns:
            Union[Response, Notification, AggregatedNotification]: The notification object or a response.
        """
        user = agent_from_token()
        try:
            return Notification.get(agent=user, db_object_or_id=notification_id)
        except ResourceNotFoundError:
            return AggregatedNotification.get(agent=user, db_object_or_id=notification_id)

    @doc(tags=[SWAGGER_TAG_MYACCOUNT])
    def delete(self, notification_id: str):
        """
        Handle DELETE request to remove a specific notification.

        Args:
            notification_id (str): The ID of the notification.

        Returns:
            Response: The response object indicating the result of the delete operation.
        """
        return process_delete_request(resource=self._get_notification(notification_id))

    @doc(tags=[SWAGGER_TAG_MYACCOUNT])
    @marshal_with(NotificationSchema)
    def get(self, notification_id: str):
        """
        Handle GET request to retrieve a specific notification.

        Args:
            notification_id (str): The ID of the notification.

        Returns:
            dict: The JSON representation of the notification.
        """
        return process_get_request(resource=self._get_notification(notification_id))


class OrganizationView(_MyAccountView):
    """
    View for handling organization related requests, including retrieving organization data.
    """

    @doc(tags=[SWAGGER_TAG_MYACCOUNT])
    @marshal_with(OrganizationResponseSchema)
    def get(self):
        """
        Handle GET request to retrieve organization data.

        Returns:
            dict: The JSON representation of the organization data.
        """
        user = agent_from_token()
        return process_get_request(resource=Organization.get(agent=user, db_object_or_id=user.organization))


class RolesView(_MyAccountView):
    """
    View for handling user roles related requests, including retrieving user roles.
    """

    @doc(tags=[SWAGGER_TAG_MYACCOUNT])
    @marshal_with(MyAccountRolesSchema)
    def get(self, **kwargs):
        """
        Handle GET request to retrieve user roles.

        Args:
            kwargs (dict): Additional query parameters.

        Returns:
            dict: The JSON representation of the user roles.
        """
        user = agent_from_token()
        org = Organization.get(agent=user, db_object_or_id=user.organization)
        user_roles = [Role.get(agent=user, db_object_or_id=x, parents=[org]) for x in user.roles]
        res_json = {'roles': dump(user_roles)}
        return jsonify(MyAccountRolesSchema().dump(res_json))


class PermissionsView(_MyAccountView):
    """
    View for handling user permissions related requests, including retrieving user permissions.
    """

    @doc(tags=[SWAGGER_TAG_MYACCOUNT])
    @use_kwargs(paging_url_params(collection_name='permissions'), location='query')
    @marshal_with(OrganizationPermissionsPage)
    def get(self, **kwargs):
        """
        Handle GET request to retrieve user permissions.

        Args:
            kwargs (dict): Additional query parameters.

        Returns:
            dict: The JSON representation of the user permissions.
        """
        user = agent_from_token()
        user_perms = []

        # Get user permissions in their own organization
        users_perms = users_permissions(organization=user.organization, user_ids=[user.user_id], allow=True)
        if users_perms:
            user_, user_perms_ = users_perms.popitem()  # there should be only one item
            assert user_.user_id == user.user_id
            user_perms += permissions_jsons(permissions=user_perms_, organization=user.organization)

        # Get user permissions in external organizations
        collaborators_perms = collaborators_permissions(user_ids=[user.user_id], allow=True)
        if collaborators_perms:
            user_, collaborator_perms = collaborators_perms.popitem()  # there should be only one item
            assert user_.user_id == user.user_id
            for org, perms in collaborator_perms.items():
                user_perms += permissions_jsons(permissions=perms, organization=org)

        return paginated_response(data=user_perms, schema=OrganizationPermissionsPage, **kwargs)


def _get_notifications(query_params: dict) -> List[Union[Notification, AggregatedNotification]]:
    """
    Retrieve notifications based on query parameters.

    Args:
        query_params (dict): The query parameters for identifying notifications.

    Returns:
        List[Union[Notification, AggregatedNotification]]: The list of notification objects.
    """

    def _to_resource(user, db_object):
        if isinstance(db_object, NotificationDB):
            return Notification.get(agent=user, db_object_or_id=db_object)
        else:
            return AggregatedNotification.get(agent=user, db_object_or_id=db_object)

    user = agent_from_token()
    task_id = query_params.get('task_id')
    source_uuid = query_params.get('source_uuid')

    if source_uuid:
        db_objects = NotificationDB.filter_by_recipient_source(recipient=user.user_id, source_uuid=source_uuid)
    elif task_id:
        task = Task.get(agent=user, db_object_or_id=task_id).db_object()
        db_objects = NotificationDB.filter_by_recipient_task(recipient=user.user_id, task_id=task.task_id)
        db_objects += AggregatedNotificationDB.filter_by_recipient_task(recipient=user.user_id, task_id=task.task_id)
    else:
        db_objects = NotificationDB.filter_by_recipient(recipient=user.user_id)
        db_objects += AggregatedNotificationDB.filter_by_recipient(recipient=user.user_id)

    return [_to_resource(user, db_object) for db_object in db_objects]
