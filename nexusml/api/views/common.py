import math
from typing import List, Type, Union

from flask import jsonify
from flask import request
from flask import Response
from flask_apispec import doc
from flask_apispec import marshal_with
from flask_apispec import use_kwargs
from marshmallow import fields
from marshmallow import validate
from sqlalchemy import and_ as sql_and
from sqlalchemy.exc import IntegrityError

from nexusml.api.resources.base import Permission
from nexusml.api.resources.base import PermissionDeniedError
from nexusml.api.resources.base import Resource
from nexusml.api.resources.base import ResourceNotFoundError
from nexusml.api.schemas.base import BaseSchema
from nexusml.api.schemas.organizations import PermissionSchema
from nexusml.api.schemas.organizations import PermissionsPage
from nexusml.api.views.core import agent_from_token
from nexusml.api.views.core import get_page_resources
from nexusml.api.views.core import page_url
from nexusml.api.views.core import roles_required
from nexusml.api.views.utils import paging_url_params
from nexusml.constants import ADMIN_ROLE
from nexusml.constants import HTTP_DELETE_STATUS_CODE
from nexusml.constants import HTTP_POST_STATUS_CODE
from nexusml.constants import MAINTAINER_ROLE
from nexusml.constants import NULL_UUID
from nexusml.constants import SWAGGER_TAG_ORGANIZATIONS
from nexusml.database.ai import AIModelDB
from nexusml.database.ai import PredictionDB
from nexusml.database.core import db_commit
from nexusml.database.core import db_rollback
from nexusml.database.core import save_to_db
from nexusml.database.examples import ExampleDB
from nexusml.database.files import TaskFileDB
from nexusml.database.organizations import OrganizationDB
from nexusml.database.organizations import RoleDB
from nexusml.database.organizations import UserDB
from nexusml.database.permissions import RolePermission
from nexusml.database.permissions import UserPermission
from nexusml.database.tasks import TaskDB
from nexusml.enums import ResourceAction
from nexusml.enums import ResourceType

_ADMIN_MAINTAINER_PERM_ERR = 'Admin/Maintainer permissions cannot be removed or modified'


def _is_admin_or_maintainer(agent: Union[UserDB, RoleDB], organization: OrganizationDB) -> bool:
    """
    Determines if the given agent is an admin or maintainer of the specified organization.

    Args:
        agent (Union[UserDB, RoleDB]): The user or role to check.
        organization (OrganizationDB): The organization to check against.

    Returns:
        bool: True if the agent is an admin or maintainer of the organization, False otherwise.
    """
    if agent.organization_id != organization.organization_id:
        return False

    if isinstance(agent, UserDB):
        if any(role.name in [ADMIN_ROLE, MAINTAINER_ROLE] for role in agent.roles):
            return True
    else:
        assert isinstance(agent, RoleDB)
        if agent.name in [ADMIN_ROLE, MAINTAINER_ROLE]:
            return True

    return False


class PermissionAssignmentView:
    """ Endpoints for user/role permission management. """

    # TODO: subclasses are calling `delete()`, `get()`, and `post()` breaking parent-child relationships,
    #       because they are keeping the `organization_id` specified in the request URL while modifying `agent_id`.
    #       This is a bad design. Refactor it

    db_model: Type[Union[UserPermission, RolePermission]] = None  # set in subclass

    _allowed_rsrc_types = [x.name.lower() for x in ResourceType]
    _allowed_rsrc_types_str = ' | '.join([f'"{x}"' for x in _allowed_rsrc_types])

    _allowed_actions = [x.name.lower() for x in ResourceAction]
    _allowed_actions_str = ' | '.join([f'"{x}"' for x in _allowed_actions])

    _url_params = {
        'resource_uuid':
            fields.String(description='Resource UUID (only for resource-level permissions)'),
        'resource_type':
            fields.String(validate=validate.OneOf(_allowed_rsrc_types),
                          description='Resource type: ' + _allowed_rsrc_types_str),
        'action':
            fields.String(validate=validate.OneOf(_allowed_actions), description='Action: ' + _allowed_actions_str),
        'allow':
            fields.Boolean(description='Filter granted/denied permissions (only for admins and maintainers): '
                           'true (granted) | false (denied)')
    }
    _paging_url_params = {**_url_params, **paging_url_params(collection_name='permissions')}

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS], description='Note: only available to admins and maintainers')
    @use_kwargs(_url_params, location='query')
    @roles_required(roles=[ADMIN_ROLE, MAINTAINER_ROLE], require_all=False)
    def delete(self, organization_id: str, agent_id: str, resources: List[Resource], **kwargs):
        """
        Deletes permissions for the specified user or role.

        Args:
            organization_id (str): The ID of the organization.
            agent_id (str): The ID of the agent (user or role).
            resources (List[Resource]): List of resources, including the organization and agent.
            **kwargs: Additional keyword arguments for filtering permissions.

        Returns:
            Response: The response with delete status.
        """
        # Get user or role
        org = resources[0].db_object()
        agent = resources[1]
        agent_db = agent.db_object()

        # If they're an admin or a maintainer, reject request
        if _is_admin_or_maintainer(agent=agent_db, organization=org):
            raise PermissionDeniedError(_ADMIN_MAINTAINER_PERM_ERR)

        # Filter permissions
        perm_filters = self._permission_filters(organization_id=resources[0].db_object().organization_id,
                                                resource_uuid=kwargs.get('resource_uuid'),
                                                resource_type=kwargs.get('resource_type'),
                                                action=kwargs.get('action'),
                                                allow=kwargs.get('allow'))
        perms_to_remove = agent_db.permissions.filter(sql_and(*perm_filters)) if perm_filters else agent_db.permissions

        # Delete permissions
        perms_to_remove.delete()
        agent.persist()

        return Response(status=HTTP_DELETE_STATUS_CODE)

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS])
    @use_kwargs(_paging_url_params, location='query')
    @marshal_with(PermissionsPage)
    def get(self, organization_id: str, agent_id: str, resources: List[Resource], **kwargs):
        """
        Retrieves permissions for the specified user or role.

        Args:
            organization_id (str): The ID of the organization.
            agent_id (str): The ID of the agent (user or role).
            resources (List[Resource]): List of resources, including the organization and agent.
            **kwargs: Additional keyword arguments for filtering and pagination.

        Returns:
            Response: The response with a paginated list of permissions.
        """
        # Get user or role to be queried
        org = resources[0].db_object()
        agent = resources[1].db_object()

        # Regular users can only query their own permissions
        session_agent = agent_from_token()
        if isinstance(session_agent, UserDB) and not _is_admin_or_maintainer(agent=session_agent, organization=org):
            if isinstance(agent, UserDB) and agent != session_agent:
                raise PermissionDeniedError()
            elif isinstance(agent, RoleDB) and agent not in session_agent.roles:
                raise PermissionDeniedError()

        # If queried user is an admin or a maintainer, return all permissions
        if _is_admin_or_maintainer(agent=agent, organization=org):
            all_permissions = [
                Permission(resource_type=rt, resource_uuid=NULL_UUID, action=a, allow=True)
                for rt in ResourceType
                for a in ResourceAction
            ]
            perms_jsons = permissions_jsons(permissions=all_permissions)
            return paginated_response(data=perms_jsons, schema=PermissionsPage, **kwargs)

        # Filter user/role's permissions
        # WARNING: we don't apply inheritance here (user permissions correspond to those assigned to the user directly)
        agent_permissions = agent.permissions
        perm_filters = self._permission_filters(organization_id=org.organization_id,
                                                resource_uuid=kwargs.get('resource_uuid'),
                                                resource_type=kwargs.get('resource_type'),
                                                action=kwargs.get('action'),
                                                allow=kwargs.get('allow'))
        query = agent_permissions.filter(sql_and(*perm_filters)) if perm_filters else agent_permissions

        # Return permissions
        return get_page_resources(query=query,
                                  page_number=kwargs['page'],
                                  per_page=kwargs['per_page'],
                                  total_count=kwargs['total_count'])

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS], description='Note: only available to admins and maintainers')
    @use_kwargs(PermissionSchema, location='json')
    @marshal_with(PermissionSchema)
    @roles_required(roles=[ADMIN_ROLE, MAINTAINER_ROLE], require_all=False)
    def post(self, organization_id: str, agent_id: str, resources: List[Resource], **kwargs):
        """
        Assigns a permission to a specified user or role.

        Args:
            organization_id (str): The ID of the organization.
            agent_id (str): The ID of the agent (user or role).
            resources (List[Resource]): List of resources, including the organization and agent.
            **kwargs: Additional keyword arguments for permission details.

        Returns:
            Response: The response with the newly assigned permission.
        """
        org = resources[0].db_object()
        rsrc_uuid = kwargs.get('resource_uuid') or NULL_UUID
        rsrc_type = kwargs['resource_type']
        action = kwargs['action']
        allow = kwargs['allow']

        # Get user or role
        agent = resources[1]
        agent_db = agent.db_object()

        # If they're an admin or a maintainer, reject request
        if _is_admin_or_maintainer(agent=agent_db, organization=org):
            raise PermissionDeniedError(_ADMIN_MAINTAINER_PERM_ERR)

        # Don't allow assigning creation/deletion permissions on Organizations. They are reserved for admins.
        if rsrc_type == ResourceType.ORGANIZATION and action in [ResourceAction.CREATE, ResourceAction.DELETE]:
            raise PermissionDeniedError('Cannot assign creation/deletion permissions on organizations')

        # Verify that the specified resource exists and belongs to the organization
        if rsrc_uuid != NULL_UUID:
            rsrc_db_models = {
                ResourceType.ORGANIZATION: OrganizationDB,
                ResourceType.TASK: TaskDB,
                ResourceType.FILE: TaskFileDB,
                ResourceType.AI_MODEL: AIModelDB,
                ResourceType.EXAMPLE: ExampleDB,
                ResourceType.PREDICTION: PredictionDB
            }
            rsrc_db_model = rsrc_db_models[rsrc_type]

            rsrc_db_object = rsrc_db_model.get_from_uuid(rsrc_uuid)
            if rsrc_db_object is None:
                belongs_to_org = False
            else:
                if hasattr(rsrc_db_model, 'organization_id'):
                    belongs_to_org = rsrc_db_object.organization_id == org.organization_id
                else:
                    belongs_to_org = rsrc_db_object.task.organization_id == org.organization_id

            if not belongs_to_org:
                raise ResourceNotFoundError(resource_id=rsrc_uuid)

        # Assign permission
        fks = {col: getattr(agent_db, col) for col in self.db_model.foreign_keys_columns() - {'organization_id'}}
        org_id = {'organization_id': org.organization_id} if hasattr(self.db_model, 'organization_id') else dict()
        perm_attrs = {
            **fks,
            **org_id, 'resource_uuid': rsrc_uuid,
            'resource_type': rsrc_type,
            'action': action,
            'allow': allow
        }
        # BugNote: Fix this? Why is this `not-callable`?
        # pylint: disable-next=not-callable
        new_permission = self.db_model(**perm_attrs)
        try:
            save_to_db(new_permission)
        except IntegrityError as e:
            db_rollback()
            if e.orig.args[0] == 1062:
                self.db_model.query().filter_by(**{c: v for c, v in perm_attrs.items() if c != 'allow'}).delete()
                db_commit()
                save_to_db(new_permission)
        agent.uncache()

        # Build response
        response = jsonify(PermissionSchema().dump(kwargs))
        response.status_code = HTTP_POST_STATUS_CODE
        response.headers['Location'] = request.url
        return response

    @classmethod
    def _permission_filters(cls,
                            organization_id: int,
                            resource_uuid: str = None,
                            resource_type: str = None,
                            action: str = None,
                            allow: bool = None) -> list:
        """
        Builds a list of filters for querying permissions.

        Args:
            organization_id (int): The ID of the organization.
            resource_uuid (str, optional): The UUID of the resource.
            resource_type (str, optional): The type of the resource.
            action (str, optional): The action associated with the permission.
            allow (bool, optional): The allow status of the permission.

        Returns:
            list: A list of filters for querying permissions.
        """
        filters = [cls.db_model.organization_id == organization_id] if hasattr(cls.db_model, 'organization_id') else []
        if resource_uuid is not None:
            filters.append(cls.db_model.resource_uuid == resource_uuid)
        if resource_type is not None:
            filters.append(cls.db_model.resource_type == resource_type)
        if action is not None:
            filters.append(cls.db_model.action == action)
        if allow is not None:
            filters.append(cls.db_model.allow == allow)
        return filters


def permissions_jsons(permissions: List[Permission], organization: OrganizationDB = None) -> list:
    """
    Converts a list of Permission objects to JSON-serializable dictionaries.

    Args:
        permissions (List[Permission]): A list of Permission objects.
        organization (OrganizationDB, optional): The organization to include in the JSON.

    Returns:
        list: A list of JSON-serializable dictionaries representing the permissions.
    """
    return [{
        **({
            'organization': organization.public_id
        } if organization is not None else dict()), 'resource_type':
            perm.resource_type,
        'resource_uuid':
            perm.resource_uuid,
        'action':
            perm.action,
        'allow':
            perm.allow
    } for perm in permissions]


def paginated_response(data: List[dict], schema: BaseSchema, **url_query) -> Response:
    """
    Returns a paginated response.

    Args:
        data (list): All data to be paginated.
        schema (BaseSchema): Schema to be used for dumping the JSON.
        **url_query: URL query parameters.

    Returns:
        Response: The response with paginated data.
    """
    previous_page = url_query['page'] - 1
    current_page = url_query['page']
    next_page = url_query['page'] + 1

    if previous_page >= 1:
        previous_url = page_url(page_number=previous_page, current_page_number=current_page)
    else:
        previous_url = None

    current_url = page_url(page_number=current_page, current_page_number=current_page)

    if next_page <= math.ceil(len(data) / url_query['per_page']):
        next_url = page_url(page_number=next_page, current_page_number=current_page)
    else:
        next_url = None

    res_json = {
        'data': data[(previous_page * url_query['per_page']):(current_page * url_query['per_page'])],
        'links': {
            'previous': previous_url,
            'current': current_url,
            'next': next_url,
        }
    }

    if url_query['total_count']:
        res_json['total_count'] = len(data)

    return jsonify(schema().dump(res_json))
