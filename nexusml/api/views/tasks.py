from copy import deepcopy
from typing import Iterable, List

from flask import g
from flask import jsonify
from flask import Response
from flask_apispec import doc
from flask_apispec import marshal_with
from flask_apispec import use_kwargs

from nexusml.api.resources.base import dump
from nexusml.api.resources.base import InvalidDataError
from nexusml.api.resources.base import PermissionDeniedError
from nexusml.api.resources.base import Resource
from nexusml.api.resources.organizations import get_user_roles
from nexusml.api.resources.tasks import InputCategory
from nexusml.api.resources.tasks import InputElement
from nexusml.api.resources.tasks import MetadataCategory
from nexusml.api.resources.tasks import MetadataElement
from nexusml.api.resources.tasks import OutputCategory
from nexusml.api.resources.tasks import OutputElement
from nexusml.api.resources.tasks import Task
from nexusml.api.schemas.base import StatusRequestSchema
from nexusml.api.schemas.base import StatusResponseSchema
from nexusml.api.schemas.tasks import CategoryRequest
from nexusml.api.schemas.tasks import CategoryResponse
from nexusml.api.schemas.tasks import ElementRequest
from nexusml.api.schemas.tasks import ElementResponse
from nexusml.api.schemas.tasks import TaskPOSTRequest
from nexusml.api.schemas.tasks import TaskPUTRequest
from nexusml.api.schemas.tasks import TaskQuotaUsageRequest
from nexusml.api.schemas.tasks import TaskQuotaUsageResponse
from nexusml.api.schemas.tasks import TaskResponse
from nexusml.api.schemas.tasks import TaskSchemaResponse
from nexusml.api.schemas.tasks import TaskSettingsSchema
from nexusml.api.views.base import create_view
from nexusml.api.views.core import agent_from_token
from nexusml.api.views.core import allowed_clients
from nexusml.api.views.core import error_response
from nexusml.api.views.core import get_page_resources
from nexusml.api.views.core import process_delete_request
from nexusml.api.views.core import process_get_request
from nexusml.api.views.core import process_post_or_put_request
from nexusml.api.views.utils import paging_url_params
from nexusml.constants import ADMIN_ROLE
from nexusml.constants import HTTP_BAD_REQUEST_STATUS_CODE
from nexusml.constants import HTTP_NOT_FOUND_STATUS_CODE
from nexusml.constants import HTTP_POST_STATUS_CODE
from nexusml.constants import HTTP_PUT_STATUS_CODE
from nexusml.constants import MAINTAINER_ROLE
from nexusml.constants import NULL_UUID
from nexusml.constants import SWAGGER_TAG_TASKS
from nexusml.database.core import db_query
from nexusml.database.core import save_to_db
from nexusml.database.organizations import Agent
from nexusml.database.organizations import ClientDB
from nexusml.database.organizations import KNOWN_CLIENT_UUIDS
from nexusml.database.organizations import user_roles
from nexusml.database.organizations import UserDB
from nexusml.database.permissions import RolePermission
from nexusml.database.permissions import UserPermission
from nexusml.database.services import Service
from nexusml.database.tasks import TaskDB
from nexusml.enums import ResourceAction
from nexusml.enums import ResourceType
from nexusml.enums import ServiceType
from nexusml.statuses import al_stopped_status
from nexusml.statuses import al_waiting_status
from nexusml.statuses import cl_stopped_status
from nexusml.statuses import cl_waiting_status
from nexusml.statuses import inference_stopped_status
from nexusml.statuses import inference_waiting_status
from nexusml.statuses import monitoring_stopped_status
from nexusml.statuses import monitoring_waiting_status
from nexusml.statuses import Status
from nexusml.statuses import testing_stopped_status
from nexusml.statuses import testing_waiting_status

################
# Define views #
################

_TasksView = create_view()
_TaskView = create_view(resource_types=[Task], permission_management=True, swagger_tags=[SWAGGER_TAG_TASKS])
_InputElementView = create_view(resource_types=[Task, InputElement])
_OutputElementView = create_view(resource_types=[Task, OutputElement])
_MetadataElementView = create_view(resource_types=[Task, MetadataElement])
_InputCategoryView = create_view(resource_types=[Task, InputElement, InputCategory])
_OutputCategoryView = create_view(resource_types=[Task, OutputElement, OutputCategory])
_MetadataCategoryView = create_view(resource_types=[Task, MetadataElement, MetadataCategory])

_category_url_params = paging_url_params(collection_name='categories')


class TasksView(_TasksView):

    @doc(tags=[SWAGGER_TAG_TASKS], description='WARNING: results will be paginated in the near future')
    @marshal_with(TaskResponse(many=True))
    def get(self) -> Response:
        """
        Retrieves a list of tasks accessible to the current session agent based on their organization and permissions.
        This method checks the agent's organization, permissions, and roles to determine which tasks can be accessed.

        Returns:
            Response: The response containing the accessible tasks.
        """

        # TODO: paginate results

        def _return_tasks(session_agent: Agent, db_objects: Iterable[TaskDB]) -> Response:
            # Double check that session user is somehow related to tasks' organization
            ext_orgs = {x.organization_id for x in db_objects if x.organization_id != session_agent.organization_id}
            if ext_orgs:
                assert isinstance(session_agent, UserDB)
                # If the user was invited to collaborate in external organizations' tasks,
                # verify they have read permissions
                user_ext_orgs = (db_query(UserPermission.organization_id).distinct().filter(
                    UserPermission.user_id == session_agent.user_id, UserPermission.organization_id
                    != session_agent.organization_id, UserPermission.resource_type == ResourceType.TASK,
                    UserPermission.action == ResourceAction.READ, UserPermission.allow.is_(True)).all())
                assert ext_orgs.issubset({x[0] for x in user_ext_orgs})

            # Load resources
            tasks = []
            for x in db_objects:
                try:
                    task = Task.get(agent=session_agent,
                                    db_object_or_id=x,
                                    check_parents=False,
                                    check_permissions=False)
                    if task not in tasks:
                        tasks.append(task)
                except PermissionDeniedError:
                    continue

            # Return JSONs
            return jsonify(dump(tasks))

        def _filter_accessible_tasks_objects(user: UserDB,
                                             permissions_subquery,
                                             return_inaccessible_tasks=False) -> tuple:
            """
            Filters accessible and optionally inaccessible tasks for a user based on task permissions.

            Args:
                user (UserDB): The user whose task permissions are being evaluated.
                permissions_subquery: Subquery containing permission data to filter the tasks.
                return_inaccessible_tasks (bool): Whether to return inaccessible tasks as well.

            Returns:
                tuple: A tuple containing the list of accessible tasks and inaccessible tasks (if requested).
            """
            all_tasks_filter = permissions_subquery.c.resource_uuid == NULL_UUID
            single_task_filter = permissions_subquery.c.resource_uuid != NULL_UUID
            allowed_filter = permissions_subquery.c.allow.is_(True)
            denied_filter = permissions_subquery.c.allow.is_(False)

            def q_union(q1, q2):
                if q1 is not None and q2 is not None:
                    return q1.union(q2)
                elif q1 is not None:
                    return q1
                else:
                    return q2

            def _gen_perms(allow_or_denied_filter, uuids_to_exclude=None):
                # Two scenarios: user permissions or role permissions
                if hasattr(permissions_subquery.c, 'organization_id'):
                    # User permissions
                    gen_perms = (db_query(permissions_subquery).filter(all_tasks_filter).filter(
                        allow_or_denied_filter).subquery())
                    q = (db_query(
                        TaskDB.uuid.label('uuid')).filter(TaskDB.organization_id == gen_perms.c.organization_id))

                elif db_query(permissions_subquery).filter(all_tasks_filter, allow_or_denied_filter).count() > 0:
                    # Role permissions
                    q = db_query(TaskDB.uuid.label('uuid')).filter(TaskDB.organization_id == user.organization_id)
                else:
                    return

                # Exclude UUIDs if specified
                if uuids_to_exclude is not None:
                    q = q.filter(TaskDB.uuid.notin_(uuids_to_exclude))

                return q

            # Filter explicitly denied tasks (denied resource permissions)
            rsrc_denied_uuids = (db_query(
                permissions_subquery.c.resource_uuid.label('uuid')).filter(single_task_filter).filter(denied_filter))

            # Filter explicitly allowed tasks (granted resource permissions)
            rsrc_allowed_uuids = (db_query(permissions_subquery.c.resource_uuid.label('uuid')).filter(
                TaskDB.uuid.notin_(rsrc_denied_uuids)).filter(single_task_filter).filter(allowed_filter))

            # Filter implicitly denied tasks (denied generic permissions)
            gen_denied_uuids = _gen_perms(allow_or_denied_filter=denied_filter, uuids_to_exclude=rsrc_allowed_uuids)

            # Filter implicitly allowed tasks (granted generic permissions)
            gen_allowed_uuids = _gen_perms(allow_or_denied_filter=allowed_filter, uuids_to_exclude=gen_denied_uuids)

            # Get accessible tasks
            allowed_task_uuids = q_union(gen_allowed_uuids, rsrc_allowed_uuids)
            # TODO: getting WARNING due to cartesian product in the line below
            allowed_tasks = TaskDB.query().filter(TaskDB.uuid.in_(allowed_task_uuids)).all()

            # Get inaccessible tasks
            if return_inaccessible_tasks:
                denied_task_uuids = q_union(gen_denied_uuids, rsrc_denied_uuids)
                denied_tasks = TaskDB.query().filter(TaskDB.uuid.in_(denied_task_uuids)).all()
            else:
                denied_tasks = None

            return allowed_tasks, denied_tasks

        session_agent = agent_from_token()
        org_tasks_query = TaskDB.query().filter_by(organization_id=session_agent.organization_id)

        # Apps (clients) can access all the tasks owned (administered) by their own organization
        # (as long as the corresponding scope is included in the API key)
        if isinstance(session_agent, ClientDB):
            if 'tasks.read' not in g.token['scope'].split(' '):
                return error_response(code=HTTP_BAD_REQUEST_STATUS_CODE, message='Invalid token')
            return _return_tasks(session_agent=session_agent, db_objects=org_tasks_query.all())

        # Subquery for getting user roles' permissions on tasks
        role_task_perms = (RolePermission.query().join(
            user_roles, user_roles.c.role_id == RolePermission.role_id).filter(
                user_roles.c.user_id == session_agent.user_id).filter(
                    RolePermission.resource_type == ResourceType.TASK).filter(
                        RolePermission.action == ResourceAction.READ).subquery())

        # Subquery for getting user permissions on tasks
        user_task_perms = (UserPermission.query().filter(UserPermission.resource_type == ResourceType.TASK).filter(
            UserPermission.action == ResourceAction.READ).filter(
                UserPermission.user_id == session_agent.user_id).subquery())

        # Filter by user permissions
        user_accessible, user_inaccessible = _filter_accessible_tasks_objects(user=session_agent,
                                                                              permissions_subquery=user_task_perms,
                                                                              return_inaccessible_tasks=True)
        user_accessible = set(user_accessible)

        # Filter by role permissions
        session_user_roles = get_user_roles(user=session_agent)
        if ADMIN_ROLE in session_user_roles or MAINTAINER_ROLE in session_user_roles:
            role_accessible = org_tasks_query.all()
        else:
            role_accessible, _ = _filter_accessible_tasks_objects(user=session_agent,
                                                                  permissions_subquery=role_task_perms)
            role_accessible = {t for t in role_accessible if t not in user_inaccessible}

        # Merge user-accessible and role-accessible tasks
        return _return_tasks(session_agent=session_agent, db_objects=user_accessible.union(role_accessible))

    @doc(tags=[SWAGGER_TAG_TASKS])
    @use_kwargs(TaskPOSTRequest, location='json')
    @marshal_with(TaskResponse)
    def post(self, **kwargs) -> Response:
        """
        Creates a new task.

        Args:
            kwargs: The task creation data sent in the request body.

        Returns:
            Response: The response containing the created task details.

        Raises:
            PermissionDeniedError: If the session user does not have the required roles.
        """
        return process_post_or_put_request(agent=agent_from_token(), resource_or_model=Task, json=kwargs)


class TaskView(_TaskView):

    @doc(tags=[SWAGGER_TAG_TASKS])
    @marshal_with(TaskResponse)
    def get(self, task_id: str, resources: List[Resource]) -> Response:
        """
        Retrieves the details of a task.

        Args:
            task_id (str): The ID of the task to retrieve.
            resources (List[Resource]): List of resources associated with the task.

        Returns:
            Response: The response containing the task details.
        """
        return process_get_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_TASKS])
    @use_kwargs(TaskPUTRequest, location='json')
    @marshal_with(TaskResponse)
    def put(self, task_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Updates the details of a task.

        Args:
            task_id (str): The ID of the task to update.
            resources (List[Resource]): List of resources associated with the task.
            kwargs: The task update data sent in the request body.

        Returns:
            Response: The response containing the updated task details.
        """
        return process_post_or_put_request(agent=agent_from_token(), resource_or_model=resources[-1], json=kwargs)

    @doc(tags=[SWAGGER_TAG_TASKS])
    def delete(self, task_id: str, resources: List[Resource]) -> Response:
        """
        Deletes a specific task.

        Args:
            task_id (str): The ID of the task to delete.
            resources (List[Resource]): List of resources associated with the task.

        Returns:
            Response: The response confirming the deletion of the task.
        """
        return process_delete_request(resource=resources[-1])


class TaskSchemaView(_TaskView):

    @doc(tags=[SWAGGER_TAG_TASKS])
    @marshal_with(TaskSchemaResponse)
    def get(self, task_id: str, resources: List[Resource]) -> Response:
        """
        Retrieves the schema definition of a task.

        Args:
            task_id (str): The ID of the task whose schema is being requested.
            resources (List[Resource]): List of resources associated with the task.

        Returns:
            Response: The response containing the task schema.
        """
        task = resources[-1]
        return jsonify(task.dump_task_schema())


class TaskSettingsView(_TaskView):

    @doc(tags=[SWAGGER_TAG_TASKS])
    @marshal_with(TaskSettingsSchema)
    def get(self, task_id: str, resources: List[Resource]) -> Response:
        """
        Retrieves the settings of a task.

        Args:
            task_id (str): The ID of the task whose settings are being requested.
            resources (List[Resource]): List of resources associated with the task.

        Returns:
            Response: The response containing the task's settings.
        """
        task = resources[-1]
        services = Service.filter_by_task(task_id=task.db_object().task_id)
        assert len(services) == len(ServiceType)
        service_settings = {service.type_.name.lower(): service.to_dict()['settings'] for service in services}
        return jsonify(TaskSettingsSchema().dump({'services': service_settings}))

    @doc(tags=[SWAGGER_TAG_TASKS])
    @use_kwargs(TaskSettingsSchema, location='json')
    @marshal_with(TaskSettingsSchema)
    def put(self, task_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Updates the settings of a task.

        Args:
            task_id (str): The ID of the task whose settings are being updated.
            resources (List[Resource]): List of resources associated with the task.
            kwargs: The new settings data sent in the request body.

        Returns:
            Response: The response containing the updated task settings.
        """
        task = resources[-1]
        service_settings: dict = kwargs['services']
        for service_name, settings in service_settings.items():
            service_type = ServiceType[service_name.upper()]
            service: Service = Service.filter_by_task_and_type(task_id=task.db_object().task_id, type_=service_type)
            old_settings: dict = deepcopy(service.settings)
            service.set_settings(settings)
            save_to_db(service)
            self._update_services_status(service=service, old_settings=old_settings, settings=settings)
        response = jsonify(TaskSettingsSchema().dump(kwargs))
        response.status_code = HTTP_PUT_STATUS_CODE
        return response

    @staticmethod
    def _update_services_status(service: Service, old_settings: dict, settings: dict):
        """
        Updates the status of services based on the new settings.

        This method checks if the service's enabled status has changed and updates the service status accordingly.

        Args:
            service (Service): The service whose status is being updated.
            old_settings (dict): The old service settings.
            settings (dict): The new service settings.
        """
        is_enabled: bool = settings.get('enabled', False)
        was_enabled: bool = old_settings.get('enabled', False)
        status_changed: bool = is_enabled != was_enabled

        status_map: dict = {
            ServiceType.INFERENCE: {
                'waiting_status': inference_waiting_status,
                'stopped_status': inference_stopped_status
            },
            ServiceType.CONTINUAL_LEARNING: {
                'waiting_status': cl_waiting_status,
                'stopped_status': cl_stopped_status
            },
            ServiceType.ACTIVE_LEARNING: {
                'waiting_status': al_waiting_status,
                'stopped_status': al_stopped_status
            },
            ServiceType.MONITORING: {
                'waiting_status': monitoring_waiting_status,
                'stopped_status': monitoring_stopped_status
            },
            ServiceType.TESTING: {
                'waiting_status': testing_waiting_status,
                'stopped_status': testing_stopped_status
            }
        }
        current_service_status: dict = status_map[service.type_]

        if is_enabled and status_changed:
            service.set_status(status=Status(template=current_service_status['waiting_status']))
        elif status_changed:
            service.set_status(status=Status(template=current_service_status['stopped_status']))


class TaskStatusView(_TaskView):

    @use_kwargs(StatusRequestSchema, location='json')
    @marshal_with(StatusResponseSchema)
    @allowed_clients(client_ids=[KNOWN_CLIENT_UUIDS['web']],
                     service_types=ServiceType,
                     error_code=HTTP_NOT_FOUND_STATUS_CODE)
    def put(self, task_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Updates the status of a task. Only available to the official Web App and AI services.

        Args:
            task_id (str): The ID of the task whose status is being updated.
            resources (List[Resource]): List of resources associated with the task.
            kwargs: The new status data sent in the request body.

        Returns:
            Response: The response containing the updated status information.
        Raises:
            InvalidDataError: If the status code is invalid or not found.
        """
        task = resources[-1]
        try:
            new_status = Status.from_dict(kwargs)
            task.db_object().set_status(new_status)
        except AttributeError:
            raise InvalidDataError(f'Status not found: "{kwargs["code"]}"')
        except ValueError as e:
            err_msg = str(e)
            if err_msg == 'Invalid status code':
                raise InvalidDataError(err_msg)
            else:
                raise e
        return jsonify(new_status.to_dict(include_state=True, expand_status=True))


class TaskQuotaUsageView(_TaskView):

    @use_kwargs(TaskQuotaUsageRequest, location='json')
    @marshal_with(TaskQuotaUsageResponse)
    @allowed_clients(service_types=ServiceType, error_code=HTTP_NOT_FOUND_STATUS_CODE)
    def post(self, task_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Updates the quota usage for a task. Only available to AI services.

        Args:
            task_id (str): The ID of the task whose quota usage is being updated.
            resources (List[Resource]): List of resources associated with the task.
            kwargs: The new quota usage data sent in the request body.

        Returns:
            Response: The response containing the updated quota usage and limit information.

        Raises:
            InvalidDataError: If the provided quota values are invalid (e.g., negative hours).
        """
        # Add quota usage to current accumulated quota usage
        task = resources[-1]
        if 'cpu_hours' in kwargs:
            if kwargs['cpu_hours'] < 0:
                raise InvalidDataError('Field "cpu_hours" must be greater than or equal to 0.')
            task.update_quota_usage(name='cpu', delta=kwargs['cpu_hours'])
        if 'gpu_hours' in kwargs:
            if kwargs['gpu_hours'] < 0:
                raise InvalidDataError('Field "gpu_hours" must be greater than or equal to 0.')
            task.update_quota_usage(name='gpu', delta=kwargs['gpu_hours'])
        # Return current accumulated quota usage and quota limit
        res_json = {
            'usage': {
                'cpu_hours': task.db_object().num_cpu_hours,
                'gpu_hours': task.db_object().num_gpu_hours
            },
            'limit': {
                'cpu_hours': task.db_object().max_cpu_hours,
                'gpu_hours': task.db_object().max_gpu_hours
            }
        }
        response = jsonify(TaskQuotaUsageResponse().dump(res_json))
        response.status_code = HTTP_POST_STATUS_CODE
        return response


class InputElementsView(_InputElementView):

    @doc(tags=[SWAGGER_TAG_TASKS])
    @marshal_with(ElementResponse(many=True))
    def get(self, task_id: str, resources: List[Resource]) -> Response:
        """
        Retrieves the input elements of a task.

        Args:
            task_id (str): The ID of the task whose input elements are being requested.
            resources (List[Resource]): List of resources associated with the task.

        Returns:
            Response: The response containing the task's input elements.
        """
        task = resources[-1]
        return jsonify(dump(task.input_elements()))

    @doc(tags=[SWAGGER_TAG_TASKS])
    @use_kwargs(ElementRequest, location='json')
    @marshal_with(ElementResponse)
    def post(self, task_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Creates a new input element for a task.

        Args:
            task_id (str): The ID of the task for which the input element is being created.
            resources (List[Resource]): List of resources associated with the task.
            kwargs: The input element creation data sent in the request body.

        Returns:
            Response: The response containing the created input element.
        """
        return process_post_or_put_request(agent=resources[-1].agent(),
                                           resource_or_model=InputElement,
                                           parents=resources,
                                           json=kwargs)


class InputElementView(_InputElementView):

    @doc(tags=[SWAGGER_TAG_TASKS])
    def delete(self, task_id: str, input_id: str, resources: List[Resource]) -> Response:
        """
        Deletes an input element.

        Args:
            task_id (str): The ID of the task from which the input element is being deleted.
            input_id (str): The ID of the input element to delete.
            resources (List[Resource]): List of resources associated with the task.

        Returns:
            Response: The response confirming the deletion of the input element.
        """
        return process_delete_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_TASKS])
    @marshal_with(ElementResponse)
    def get(self, task_id: str, input_id: str, resources: List[Resource]) -> Response:
        """
        Retrieves the details of an input element.

        Args:
            task_id (str): The ID of the task.
            input_id (str): The ID of the input element to retrieve.
            resources (List[Resource]): List of resources associated with the task.

        Returns:
            Response: The response containing the input element details.
        """
        return process_get_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_TASKS])
    @use_kwargs(ElementRequest, location='json')
    @marshal_with(ElementResponse)
    def put(self, task_id: str, input_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Updates the details of an input element.

        This method processes the PUT request to update an existing input element associated with the task.
        The input element data is updated based on the data provided in the request body (`kwargs`).

        Args:
            task_id (str): The ID of the task.
            input_id (str): The ID of the input element to update.
            resources (List[Resource]): List of resources associated with the task.
            kwargs: The input element update data sent in the request body.

        Returns:
            Response: The response containing the updated input element details.
        """
        resource = resources[-1]
        return process_post_or_put_request(agent=resource.agent(), resource_or_model=resource, json=kwargs)


class OutputElementsView(_OutputElementView):

    @doc(tags=[SWAGGER_TAG_TASKS])
    @marshal_with(ElementResponse(many=True))
    def get(self, task_id: str, resources: List[Resource]) -> Response:
        """
        Retrieves the output elements of a task.

        Args:
            task_id (str): The ID of the task whose output elements are being requested.
            resources (List[Resource]): List of resources associated with the task.

        Returns:
            Response: The response containing the task's output elements.
        """
        task = resources[-1]
        return jsonify(dump(task.output_elements()))

    @doc(tags=[SWAGGER_TAG_TASKS])
    @use_kwargs(ElementRequest, location='json')
    @marshal_with(ElementResponse)
    def post(self, task_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Creates a new output element for a task.

        Args:
            task_id (str): The ID of the task for which the output element is being created.
            resources (List[Resource]): List of resources associated with the task.
            kwargs: The output element creation data sent in the request body.

        Returns:
            Response: The response containing the created output element.
        """
        return process_post_or_put_request(agent=resources[-1].agent(),
                                           resource_or_model=OutputElement,
                                           parents=resources,
                                           json=kwargs)


class OutputElementView(_OutputElementView):

    @doc(tags=[SWAGGER_TAG_TASKS])
    def delete(self, task_id: str, output_id: str, resources: List[Resource]) -> Response:
        """
        Deletes an output element.

        Args:
            task_id (str): The ID of the task from which the output element is being deleted.
            output_id (str): The ID of the output element to delete.
            resources (List[Resource]): List of resources associated with the task.

        Returns:
            Response: The response confirming the deletion of the output element.
        """
        return process_delete_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_TASKS])
    @marshal_with(ElementResponse)
    def get(self, task_id: str, output_id: str, resources: List[Resource]) -> Response:
        """
        Retrieves the details of an output element.

        Args:
            task_id (str): The ID of the task.
            output_id (str): The ID of the output element to retrieve.
            resources (List[Resource]): List of resources associated with the task.

        Returns:
            Response: The response containing the output element details.
        """
        return process_get_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_TASKS])
    @use_kwargs(ElementRequest, location='json')
    @marshal_with(ElementResponse)
    def put(self, task_id: str, output_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Updates the details of an output element.

        Args:
            task_id (str): The ID of the task.
            output_id (str): The ID of the output element to update.
            resources (List[Resource]): List of resources associated with the task.
            kwargs: The output element update data sent in the request body.

        Returns:
            Response: The response containing the updated output element details.
        """
        resource = resources[-1]
        return process_post_or_put_request(agent=resource.agent(), resource_or_model=resource, json=kwargs)


class MetadataElementsView(_MetadataElementView):

    @doc(tags=[SWAGGER_TAG_TASKS])
    @marshal_with(ElementResponse(many=True))
    def get(self, task_id: str, resources: List[Resource]) -> Response:
        """
        Retrieves the metadata elements of a task.
        
        Args:
            task_id (str): The ID of the task whose metadata elements are being requested.
            resources (List[Resource]): List of resources associated with the task.

        Returns:
            Response: The response containing the task's metadata elements.
        """
        task = resources[-1]
        return jsonify(dump(task.metadata_elements()))

    @doc(tags=[SWAGGER_TAG_TASKS])
    @use_kwargs(ElementRequest, location='json')
    @marshal_with(ElementResponse)
    def post(self, task_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Creates a new metadata element.

        Args:
            task_id (str): The ID of the task for which the metadata element is being created.
            resources (List[Resource]): List of resources associated with the task.
            kwargs: The metadata element creation data sent in the request body.

        Returns:
            Response: The response containing the created metadata element.
        """
        return process_post_or_put_request(agent=resources[-1].agent(),
                                           resource_or_model=MetadataElement,
                                           parents=resources,
                                           json=kwargs)


class MetadataElementView(_MetadataElementView):

    @doc(tags=[SWAGGER_TAG_TASKS])
    def delete(self, task_id: str, metadata_id: str, resources: List[Resource]) -> Response:
        """
        Deletes a metadata element.

        Args:
            task_id (str): The ID of the task from which the metadata element is being deleted.
            metadata_id (str): The ID of the metadata element to delete.
            resources (List[Resource]): List of resources associated with the task.

        Returns:
            Response: The response confirming the deletion of the metadata element.
        """
        return process_delete_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_TASKS])
    @marshal_with(ElementResponse)
    def get(self, task_id: str, metadata_id: str, resources: List[Resource]) -> Response:
        """
        Retrieves the details of a metadata element.

        Args:
            task_id (str): The ID of the task.
            metadata_id (str): The ID of the metadata element to retrieve.
            resources (List[Resource]): List of resources associated with the task.

        Returns:
            Response: The response containing the metadata element details.
        """
        return process_get_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_TASKS])
    @use_kwargs(ElementRequest, location='json')
    @marshal_with(ElementResponse)
    def put(self, task_id: str, metadata_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Updates the details of a metadata element.

        Args:
            task_id (str): The ID of the task.
            metadata_id (str): The ID of the metadata element to update.
            resources (List[Resource]): List of resources associated with the task.
            kwargs: The metadata element update data sent in the request body.

        Returns:
            Response: The response containing the updated metadata element details.
        """
        resource = resources[-1]
        return process_post_or_put_request(agent=resource.agent(), resource_or_model=resource, json=kwargs)


class InputCategoriesView(_InputCategoryView):

    @doc(tags=[SWAGGER_TAG_TASKS])
    @use_kwargs(_category_url_params, location='query')
    @marshal_with(CategoryResponse(many=True))
    def get(self, task_id: str, input_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Retrieves the categories of an input element.

        Args:
            task_id (str): The ID of the task whose input categories are being requested.
            input_id (str): The ID of the input element whose categories are being fetched.
            resources (List[Resource]): List of resources associated with the input element.
            kwargs: The query parameters for pagination.

        Returns:
            Response: The response containing the input categories.
        """
        res_json = get_page_resources(query=resources[-1].db_object().categories,
                                      page_number=kwargs['page'],
                                      per_page=kwargs['per_page'],
                                      total_count=kwargs['total_count'],
                                      resource_type=InputCategory,
                                      parents=[resources[-1]])
        return jsonify(res_json)

    @doc(tags=[SWAGGER_TAG_TASKS])
    @use_kwargs(CategoryRequest, location='json')
    @marshal_with(CategoryResponse)
    def post(self, task_id: str, input_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Creates a new input category.

        Args:
            task_id (str): The ID of the task for which the input category is being created.
            input_id (str): The ID of the input element for which the category is being created.
            resources (List[Resource]): List of resources associated with the input element.
            kwargs: The input category creation data sent in the request body.

        Returns:
            Response: The response containing the created input category.
        """
        return process_post_or_put_request(agent=resources[-1].agent(),
                                           resource_or_model=InputCategory,
                                           parents=resources,
                                           json=kwargs)


class InputCategoryView(_InputCategoryView):

    @doc(tags=[SWAGGER_TAG_TASKS])
    def delete(self, task_id: str, input_id: str, category_id: str, resources: List[Resource]) -> Response:
        """
        Deletes an input category.

        Args:
            task_id (str): The ID of the task from which the input category is being deleted.
            input_id (str): The ID of the input element associated with the category.
            category_id (str): The ID of the category to delete.
            resources (List[Resource]): List of resources associated with the input element.

        Returns:
            Response: The response confirming the deletion of the input category.
        """
        return process_delete_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_TASKS])
    @marshal_with(CategoryResponse)
    def get(self, task_id: str, input_id: str, category_id: str, resources: List[Resource]) -> Response:
        """
        Retrieves the details of an input category.

        Args:
            task_id (str): The ID of the task.
            input_id (str): The ID of the input element.
            category_id (str): The ID of the input category to retrieve.
            resources (List[Resource]): List of resources associated with the input element.

        Returns:
            Response: The response containing the input category details.
        """
        return process_get_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_TASKS])
    @use_kwargs(CategoryRequest, location='json')
    @marshal_with(CategoryResponse)
    def put(self, task_id: str, input_id: str, category_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Updates the details of an input category.

        Args:
            task_id (str): The ID of the task.
            input_id (str): The ID of the input element.
            category_id (str): The ID of the input category to update.
            resources (List[Resource]): List of resources associated with the input element.
            kwargs: The input category update data sent in the request body.

        Returns:
            Response: The response containing the updated input category details.
        """
        resource = resources[-1]
        return process_post_or_put_request(agent=resource.agent(), resource_or_model=resource, json=kwargs)


class OutputCategoriesView(_OutputCategoryView):

    @doc(tags=[SWAGGER_TAG_TASKS])
    @use_kwargs(_category_url_params, location='query')
    @marshal_with(CategoryResponse(many=True))
    def get(self, task_id: str, output_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Retrieves the output categories of an output element.

        Args:
            task_id (str): The ID of the task whose output categories are being requested.
            output_id (str): The ID of the output element whose categories are being fetched.
            resources (List[Resource]): List of resources associated with the output element.
            kwargs: The query parameters for pagination.

        Returns:
            Response: The response containing the output categories.
        """
        res_json = get_page_resources(query=resources[-1].db_object().categories,
                                      page_number=kwargs['page'],
                                      per_page=kwargs['per_page'],
                                      total_count=kwargs['total_count'],
                                      resource_type=OutputCategory,
                                      parents=[resources[-1]])
        return jsonify(res_json)

    @doc(tags=[SWAGGER_TAG_TASKS])
    @use_kwargs(CategoryRequest, location='json')
    @marshal_with(CategoryResponse)
    def post(self, task_id: str, output_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Creates a new output category for a specific output element identified by the given output ID.

        This method processes the POST request to create a new output category for the output element.
        The category data is provided in the request body (`kwargs`).

        Args:
            task_id (str): The ID of the task for which the output category is being created.
            output_id (str): The ID of the output element for which the category is being created.
            resources (List[Resource]): List of resources associated with the output element.
            kwargs: The output category creation data sent in the request body.

        Returns:
            Response: The response containing the created output category.
        """
        return process_post_or_put_request(agent=resources[-1].agent(),
                                           resource_or_model=OutputCategory,
                                           parents=resources,
                                           json=kwargs)


class OutputCategoryView(_OutputCategoryView):

    @doc(tags=[SWAGGER_TAG_TASKS])
    def delete(self, task_id: str, output_id: str, category_id: str, resources: List[Resource]) -> Response:
        """
        Deletes an output category.

        Args:
            task_id (str): The ID of the task from which the output category is being deleted.
            output_id (str): The ID of the output element associated with the category.
            category_id (str): The ID of the category to delete.
            resources (List[Resource]): List of resources associated with the output element.

        Returns:
            Response: The response confirming the deletion of the output category.
        """
        return process_delete_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_TASKS])
    @marshal_with(CategoryResponse)
    def get(self, task_id: str, output_id: str, category_id: str, resources: List[Resource]) -> Response:
        """
        Retrieves the details of an output category.

        Args:
            task_id (str): The ID of the task.
            output_id (str): The ID of the output element.
            category_id (str): The ID of the output category to retrieve.
            resources (List[Resource]): List of resources associated with the output element.

        Returns:
            Response: The response containing the output category details.
        """
        return process_get_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_TASKS])
    @use_kwargs(CategoryRequest, location='json')
    @marshal_with(CategoryResponse)
    def put(self, task_id: str, output_id: str, category_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Updates the details of an output category.

        Args:
            task_id (str): The ID of the task.
            output_id (str): The ID of the output element.
            category_id (str): The ID of the output category to update.
            resources (List[Resource]): List of resources associated with the output element.
            kwargs: The output category update data sent in the request body.

        Returns:
            Response: The response containing the updated output category details.
        """
        resource = resources[-1]
        return process_post_or_put_request(agent=resource.agent(), resource_or_model=resource, json=kwargs)


class MetadataCategoriesView(_MetadataCategoryView):

    @doc(tags=[SWAGGER_TAG_TASKS])
    @use_kwargs(_category_url_params, location='query')
    @marshal_with(CategoryResponse(many=True))
    def get(self, task_id: str, metadata_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Retrieves the categories of a metadata element.

        Args:
            task_id (str): The ID of the task whose metadata categories are being requested.
            metadata_id (str): The ID of the metadata element whose categories are being fetched.
            resources (List[Resource]): List of resources associated with the metadata element.
            kwargs: The query parameters for pagination.

        Returns:
            Response: The response containing the metadata categories.
        """
        res_json = get_page_resources(query=resources[-1].db_object().categories,
                                      page_number=kwargs['page'],
                                      per_page=kwargs['per_page'],
                                      total_count=kwargs['total_count'],
                                      resource_type=MetadataCategory,
                                      parents=[resources[-1]])
        return jsonify(res_json)

    @doc(tags=[SWAGGER_TAG_TASKS])
    @use_kwargs(CategoryRequest, location='json')
    @marshal_with(CategoryResponse)
    def post(self, task_id: str, metadata_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Creates a new metadata category.

        Args:
            task_id (str): The ID of the task for which the metadata category is being created.
            metadata_id (str): The ID of the metadata element for which the category is being created.
            resources (List[Resource]): List of resources associated with the metadata element.
            kwargs: The metadata category creation data sent in the request body.

        Returns:
            Response: The response containing the created metadata category.
        """
        return process_post_or_put_request(agent=resources[-1].agent(),
                                           resource_or_model=MetadataCategory,
                                           parents=resources,
                                           json=kwargs)


class MetadataCategoryView(_MetadataCategoryView):

    @doc(tags=[SWAGGER_TAG_TASKS])
    def delete(self, task_id: str, metadata_id: str, category_id: str, resources: List[Resource]) -> Response:
        """
        Deletes a metadata category.

        Args:
            task_id (str): The ID of the task from which the metadata category is being deleted.
            metadata_id (str): The ID of the metadata element associated with the category.
            category_id (str): The ID of the category to delete.
            resources (List[Resource]): List of resources associated with the metadata element.

        Returns:
            Response: The response confirming the deletion of the metadata category.
        """
        return process_delete_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_TASKS])
    @marshal_with(CategoryResponse)
    def get(self, task_id: str, metadata_id: str, category_id: str, resources: List[Resource]) -> Response:
        """
        Retrieves the details of a metadata category.

        Args:
            task_id (str): The ID of the task.
            metadata_id (str): The ID of the metadata element.
            category_id (str): The ID of the metadata category to retrieve.
            resources (List[Resource]): List of resources associated with the metadata element.

        Returns:
            Response: The response containing the metadata category details.
        """
        return process_get_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_TASKS])
    @use_kwargs(CategoryRequest, location='json')
    @marshal_with(CategoryResponse)
    def put(self, task_id: str, metadata_id: str, category_id: str, resources: List[Resource], **kwargs) -> Response:
        """
        Updates the details of a metadata category.

        Args:
            task_id (str): The ID of the task.
            metadata_id (str): The ID of the metadata element.
            category_id (str): The ID of the metadata category to update.
            resources (List[Resource]): List of resources associated with the metadata element.
            kwargs: The metadata category update data sent in the request body.

        Returns:
            Response: The response containing the updated metadata category details.
        """
        resource = resources[-1]
        return process_post_or_put_request(agent=resource.agent(), resource_or_model=resource, json=kwargs)
