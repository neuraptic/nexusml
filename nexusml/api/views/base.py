import functools
from typing import Iterable, List, Type

from flask import g
from flask import jsonify
from flask import request
from flask import Response
from flask_apispec import doc
from flask_apispec import marshal_with
from flask_apispec import MethodResource

from nexusml.api.resources.base import filter_effective_permissions
from nexusml.api.resources.base import InvalidDataError
from nexusml.api.resources.base import Permission
from nexusml.api.resources.base import Resource
from nexusml.api.resources.base import users_permissions
from nexusml.api.resources.organizations import get_user_roles
from nexusml.api.resources.organizations import Organization
from nexusml.api.resources.tasks import Task
from nexusml.api.schemas.organizations import ResourceLevelPermissionsSchema
from nexusml.api.views.core import agent_from_token
from nexusml.api.views.core import capture_request_errors
from nexusml.api.views.core import capture_schema_errors
from nexusml.api.views.core import error_response
from nexusml.api.views.core import limiter
from nexusml.api.views.core import load_url_resources
from nexusml.api.views.core import rate_limits
from nexusml.api.views.core import roles_required
from nexusml.api.views.core import validate_payload_size
from nexusml.api.views.core import validate_token
from nexusml.constants import ADMIN_ROLE
from nexusml.constants import HTTP_BAD_REQUEST_STATUS_CODE
from nexusml.constants import HTTP_DELETE_STATUS_CODE
from nexusml.constants import HTTP_FORBIDDEN_STATUS_CODE
from nexusml.constants import MAINTAINER_ROLE
from nexusml.constants import NULL_UUID
from nexusml.database.core import db_query
from nexusml.database.core import delete_from_db
from nexusml.database.organizations import ClientDB
from nexusml.database.organizations import CollaboratorDB
from nexusml.database.organizations import KNOWN_CLIENT_UUIDS
from nexusml.database.organizations import RoleDB
from nexusml.database.organizations import UserDB
from nexusml.database.permissions import RolePermission
from nexusml.database.permissions import UserPermission
from nexusml.enums import ResourceAction

__all__ = [
    'create_view',
]


def create_view(resource_types: List[Type[Resource]] = None,
                load_resources: bool = True,
                reject_api_keys: Iterable[str] = None,
                permission_management: bool = False,
                swagger_tags: List[str] = None) -> MethodResource:
    """
    Factory function to create a `View` class implementing basic endpoint functionality
    such as request/token validation, API rate limiting, and error handling. If permission
    management is enabled, the view also provides endpoints for managing user/role
    permissions on resources.

    Args:
        resource_types (List[Type[Resource]], optional): A list of resource types involved in the endpoint.
            The number and order of these types must match the resource IDs specified in the URLs.
        load_resources (bool, optional): If `True` and `resource_types` is provided, resources specified
            in the URL are automatically loaded and injected into the view function via the `resources` argument.
        reject_api_keys (Iterable[str], optional): Specifies which HTTP methods should reject API keys
            (i.e., only allow Auth0 access tokens).
        permission_management (bool, optional): Requires `load_resources=True`. If `True`, enables
            permission management endpoints (`GET|DELETE /<resource_id>/permissions`).
        swagger_tags (List[str], optional): Tags for the Swagger documentation of permission management
            endpoints (only used when `permission_management` is `True`).

    Returns:
        Type[MethodResource]: A subclass of `flask_apispec.views.MethodResource` with the specified configurations.
    """

    if permission_management:
        assert load_resources

    ##############
    # Decorators #
    ##############

    def _validate_api_key_organization(func):
        """
        Decorator that verifies the API key belongs to the same organization as the resource
        being accessed. Skips validation if an Auth0 token is used.

        Returns:
            Callable: The wrapped function with API key organization validation.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """
            Verifies the API key used for the request belongs to the same organization as the resource being accessed.
            If an Auth0 token is used, the validation process is skipped.

            NOTE: Among the resources supporting API-key-based access, only tasks may exist outside organizations.
                  Instead of belonging to an organization, a task is administered/owned by (at most) one organization.
                  A task may become orphaned when such an organization is deleted.
            """
            if g.token_type == 'api_key' and g.client_uuid not in KNOWN_CLIENT_UUIDS.values():
                # Get client
                api_key_client = ClientDB.get_from_uuid(uuid=g.client_uuid)
                if api_key_client is None:
                    return error_response(code=HTTP_BAD_REQUEST_STATUS_CODE,
                                          message='Invalid API key: client not found')
                # Check client's organization
                if 'organization_id' in kwargs or 'task_id' in kwargs:
                    # Get the organization associated with the resource(s) being accessed
                    if 'organization_id' in kwargs:
                        org = (Organization.get(agent=api_key_client,
                                                db_object_or_id=kwargs['organization_id']).db_object())
                    else:
                        org = (Task.get(agent=api_key_client,
                                        db_object_or_id=kwargs['task_id']).db_object().organization)
                    # Verify client organization
                    if org.organization_id != api_key_client.organization_id:
                        return error_response(code=HTTP_FORBIDDEN_STATUS_CODE, message='Forbidden')
            # Call wrapped function
            return func(*args, **kwargs)

        return wrapper

    def _load_resources(func):
        """
        Decorator that automatically loads the resources specified in the URL parameters
        and injects them into the `resources` argument of the view function.

        Returns:
            Callable: The wrapped function with resources loaded and injected.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # WARNING: this decorator requires Python 3.6+, as the order of arguments passed to the
            #          wrapped function must be preserved. The order is guaranteed since version 3.6.
            #          See https://docs.python.org/3.6/whatsnew/3.6.html#whatsnew36-pep468

            # If no resource IDs were provided, just call wrapped function
            if not kwargs:
                return func(*args, resources=[])

            # Get resources' IDs
            resources_ids = [v for k, v in kwargs.items() if k in request.view_args]

            # Get involved resource models
            assert len(resources_ids) in [len(resource_types) - 1, len(resource_types)]

            resource_types_ = resource_types[:len(resources_ids)]

            # Load resources
            try:
                resources = load_url_resources(agent=agent_from_token(),
                                               resource_ids=resources_ids,
                                               resource_types=resource_types_)
            except InvalidDataError:
                return error_response(code=HTTP_BAD_REQUEST_STATUS_CODE, message='Invalid resource identifier(s)')

            # Inject resources into `resources` keyword argument
            kwargs.update({'resources': resources})

            assert len(kwargs['resources']) == len(request.view_args)
            assert all(isinstance(r, rt) for r, rt in zip(kwargs['resources'], resource_types_))

            # Call wrapped function
            return func(*args, **kwargs)

        return wrapper

    _decorators = [
        _validate_api_key_organization, capture_schema_errors, capture_request_errors,
        limiter.limit(rate_limits),
        validate_token(resource_types, reject_api_keys), validate_payload_size
    ]

    if load_resources and resource_types:
        _decorators.insert(0, _load_resources)

    #######################################
    # Endpoints for permission management #
    #######################################
    if permission_management:

        class PermissionsView(create_view(resource_types=resource_types)):
            """
            Class implementing permission management endpoints (`GET|DELETE`) for the resources.
            Provides functionality for querying and removing user/role-level permissions for
            a given resource.
            """

            @staticmethod
            def _get_organization(resources: List[Resource]) -> Organization:
                """
                Retrieves the organization associated with a resource, either directly from the resource
                or from its associated task.

                Args:
                    resources (List[Resource]): A list of resources, with the first resource typically being
                    the root resource (e.g., task or organization).

                Returns:
                    Organization: The organization associated with the resource.
                """
                root_resource = resources[0]
                resource = resources[-1]

                if isinstance(root_resource, Task):
                    organization = Organization.get(agent=resource.agent(),
                                                    db_object_or_id=root_resource.db_object().organization,
                                                    check_permissions=False)
                else:
                    organization = root_resource
                    assert isinstance(organization, Organization)

                return organization

            @doc(tags=swagger_tags,
                 description='Removes all user/role/collaborator permissions on the resource. '
                 'Only for admins and maintainers. Admin and maintainer permissions cannot be deleted')
            @roles_required(roles=[ADMIN_ROLE, MAINTAINER_ROLE], require_all=False)
            def delete(self, resources: List[Resource], **kwargs):
                """
                Deletes all permissions (user/role/collaborator) for the given resource, except
                for admin and maintainer permissions which cannot be deleted.

                Args:
                    resources (List[Resource]): A list of resources, with the resource to delete permissions
                    for being the last element in the list.
                """
                resource = resources[-1]
                organization = self._get_organization(resources=resources)
                org_id = organization.db_object().organization_id
                user_permissions = (UserPermission.query().filter_by(organization_id=org_id,
                                                                     resource_uuid=resource.uuid()).all())
                role_permissions = (RolePermission.query().join(RoleDB).filter(
                    RolePermission.resource_uuid == resource.uuid(), RoleDB.organization_id == org_id,
                    RoleDB.name.notin_([ADMIN_ROLE, MAINTAINER_ROLE])).all())
                delete_from_db(user_permissions + role_permissions)
                return Response(status=HTTP_DELETE_STATUS_CODE)

            @doc(tags=swagger_tags, description='Gets all user/role/collaborator permissions on the resource')
            @marshal_with(ResourceLevelPermissionsSchema)
            def get(self, resources: List[Resource], **kwargs):
                """
                Retrieves all permissions (user/role/collaborator) for the specified resource,
                with the ability to filter permissions based on the current user's roles.

                WARNING: regular users can query only granted permissions.

                Args:
                    resources (List[Resource]): A list of resources, with the resource to query permissions
                    for being the last element in the list.
                """

                def _perm_to_dict(perm: Permission) -> dict:
                    perm_dict = {'resource_type': perm.resource_type, 'action': perm.action, 'allow': perm.allow}
                    if perm.resource_uuid and perm.resource_uuid != NULL_UUID:
                        perm_dict['resource_uuid'] = perm.resource_uuid
                    return perm_dict

                # Get resource and parent organization
                resource = resources[-1]
                organization = self._get_organization(resources=resources)
                org_id = organization.db_object().organization_id

                # Determine which permissions can be queried
                session_agent = agent_from_token()
                if isinstance(session_agent, UserDB):
                    session_user_roles = get_user_roles(user=session_agent)
                    is_admin = ADMIN_ROLE in session_user_roles
                    is_maintainer = MAINTAINER_ROLE in session_user_roles
                    only_granted = not (is_admin or is_maintainer)
                else:
                    only_granted = False  # apps (clients) can query all permissions

                # User permissions
                perms_by_user = users_permissions(organization=organization.db_object(),
                                                  resource_type=resource.permission_resource_type(),
                                                  resource_uuid=resource.uuid(),
                                                  allow=(True if only_granted else None),
                                                  inheritance=False)

                user_perms = []
                collaborator_perms = []

                for user, perms in perms_by_user.items():
                    # Note: ignore generic creation permissions
                    perms_dicts = [_perm_to_dict(perm=perm) for perm in perms if perm.action != ResourceAction.CREATE]
                    if user.organization_id == org_id:
                        user_perms.append({'user': user.public_id, 'permissions': perms_dicts})
                    else:
                        collaborator = (CollaboratorDB.query().filter_by(organization_id=org_id,
                                                                         user_id=user.user_id).first())
                        assert collaborator is not None
                        collaborator_perms.append({'collaborator': collaborator.public_id, 'permissions': perms_dicts})

                # Role permissions
                role_filters = (RoleDB.organization_id == org_id, RoleDB.role_id == RolePermission.role_id)

                role_gen_filters = [
                    RolePermission.resource_uuid == NULL_UUID, RolePermission.action != ResourceAction.CREATE
                ]  # ignore generic creation permissions
                role_gen_perms = (db_query(RoleDB,
                                           RolePermission).filter(*role_filters).filter(*role_gen_filters).all())

                role_rsrc_filters = [RolePermission.resource_uuid == resource.uuid()]
                role_rsrc_perms = (db_query(RoleDB,
                                            RolePermission).filter(*role_filters).filter(*role_rsrc_filters).all())

                perms_by_role = dict()

                for row in role_gen_perms + role_rsrc_perms:
                    if row.RoleDB not in perms_by_role:
                        perms_by_role[row.RoleDB] = []
                    perms_by_role[row.RoleDB].append(row.RolePermission)

                role_perms = []

                for role, perms in perms_by_role.items():
                    effective_perms = filter_effective_permissions(role_permissions=perms)
                    if only_granted:
                        effective_perms = [perm for perm in effective_perms if perm.allow]
                    perms_dicts = [_perm_to_dict(perm=perm) for perm in effective_perms]
                    assert role.organization_id == org_id
                    role_perms.append({'role': role.public_id, 'permissions': perms_dicts})

                # Admin/Maintainer's permissions
                admin = RoleDB.query().filter_by(organization_id=org_id, name=ADMIN_ROLE).first()
                maintainer = RoleDB.query().filter_by(organization_id=org_id, name=MAINTAINER_ROLE).first()
                all_perms = [
                    Permission(resource_uuid=NULL_UUID,
                               resource_type=resource.permission_resource_type(),
                               action=action,
                               allow=True) for action in ResourceAction if action != ResourceAction.CREATE
                ]
                all_perms_dicts = [_perm_to_dict(perm=perm) for perm in all_perms]
                role_perms.append({'role': admin.public_id, 'permissions': all_perms_dicts})
                role_perms.append({'role': maintainer.public_id, 'permissions': all_perms_dicts})

                # Build response
                res_json = {'users': user_perms, 'roles': role_perms, 'collaborators': collaborator_perms}
                return jsonify(ResourceLevelPermissionsSchema().dump(res_json))

    else:
        PermissionsView = None

    ###############
    # Constructor #
    ###############

    return type('View', (MethodResource,), {
        'decorators': _decorators,
        'resource_types': resource_types,
        'permissions_view': PermissionsView
    })
