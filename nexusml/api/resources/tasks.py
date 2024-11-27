from abc import ABC
from abc import abstractmethod
import os
from typing import Iterable, List, Type, Union

from sqlalchemy import text as sql_text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.exc import OperationalError

from nexusml.api.endpoints import ENDPOINT_INPUT_CATEGORY
from nexusml.api.endpoints import ENDPOINT_INPUT_ELEMENT
from nexusml.api.endpoints import ENDPOINT_METADATA_CATEGORY
from nexusml.api.endpoints import ENDPOINT_METADATA_ELEMENT
from nexusml.api.endpoints import ENDPOINT_OUTPUT_CATEGORY
from nexusml.api.endpoints import ENDPOINT_OUTPUT_ELEMENT
from nexusml.api.endpoints import ENDPOINT_TASK
from nexusml.api.resources.base import dump
from nexusml.api.resources.base import DuplicateResourceError
from nexusml.api.resources.base import InvalidDataError
from nexusml.api.resources.base import Resource
from nexusml.api.resources.base import ResourceError
from nexusml.api.resources.base import ResourceNotFoundError
from nexusml.api.resources.base import UnprocessableRequestError
from nexusml.api.resources.base import users_permissions
from nexusml.api.resources.files import TaskFile as File
from nexusml.api.resources.organizations import Organization
from nexusml.api.resources.utils import check_quota_usage as check_quota_usage_
from nexusml.api.schemas.base import ResourceResponseSchema
from nexusml.api.schemas.tasks import CategoryRequest
from nexusml.api.schemas.tasks import CategoryResponse
from nexusml.api.schemas.tasks import ElementRequest
from nexusml.api.schemas.tasks import ElementResponse
from nexusml.api.schemas.tasks import TaskPOSTRequest
from nexusml.api.schemas.tasks import TaskResponse
from nexusml.api.schemas.tasks import TaskSchemaResponse
from nexusml.database.ai import PredictionDB
from nexusml.database.base import Entity
from nexusml.database.core import db_commit
from nexusml.database.core import db_execute
from nexusml.database.core import db_rollback
from nexusml.database.core import delete_from_db
from nexusml.database.core import save_to_db
from nexusml.database.examples import ExampleDB
from nexusml.database.files import TaskFileDB as FileDB
from nexusml.database.organizations import Agent
from nexusml.database.organizations import UserDB
from nexusml.database.services import Service
from nexusml.database.subscriptions import quotas
from nexusml.database.tasks import CategoryDB
from nexusml.database.tasks import ElementDB
from nexusml.database.tasks import TaskDB
from nexusml.database.tasks import TaskER
from nexusml.engine.buffers import ALBuffer
from nexusml.engine.buffers import ALBufferIO
from nexusml.engine.buffers import MonBuffer
from nexusml.engine.buffers import MonBufferIO
from nexusml.engine.schema.base import Schema as TaskSchema
from nexusml.engine.schema.templates import get_task_schema_template
from nexusml.enums import ElementType
from nexusml.enums import ElementValueType
from nexusml.enums import NotificationSource
from nexusml.enums import ResourceType
from nexusml.enums import TaskFileUse
from nexusml.enums import TaskTemplate
from nexusml.enums import TaskType
from nexusml.env import ENV_SUPPORT_EMAIL
from nexusml.statuses import Status
from nexusml.statuses import task_unknown_error_status


class _TaskSchemaResource(Resource):
    """
    Base resource class for task schema-related operations.

    This class provides a foundation for resources related to task schemas, ensuring
    consistency across resources that are part of the task schema. It inherits from the
    general Resource class and defines behavior specific to schema elements.
    """

    @classmethod
    def touch_parent(cls) -> bool:
        """
        Indicates that creating/updating/deleting a resource affects the parent task.

        Steps:
            1. Resource data is removed from the cache.
            2. The parent task's modification datetime is updated.

        Returns:
            bool: Always returns True to indicate the parent task was affected.
        """
        return True


class Element(_TaskSchemaResource, ABC):
    """
    Abstract base class for elements in a task schema.

    Elements represent parts of a task's schema (e.g., input, output, metadata elements)
    and provide methods to manipulate these elements. This class handles loading and
    dumping element data, as well as managing their interactions with parent tasks.
    """

    @classmethod
    @abstractmethod
    def element_type(cls) -> ElementType:
        """
        Abstract method that returns the specific type of the element.

        This must be implemented by subclasses to specify the element type.
        """
        raise NotImplementedError()

    @classmethod
    def db_model(cls):
        """
        Returns the database model associated with the element.

        Returns:
            ElementDB: The database model for this element type.
        """
        return ElementDB

    @classmethod
    def load_schema(cls):
        """
        Returns the schema used for loading data into the element.

        Returns:
            ElementRequest: Schema used for loading element data.
        """
        return ElementRequest

    @classmethod
    def dump_schema(cls):
        """
        Returns the schema used for dumping data from the element.

        Returns:
            ElementResponse: Schema used for dumping element data.
        """
        return ElementResponse

    def delete(self, notify_to: Iterable[UserDB] = None):
        """
        Deletes the current element and updates associated predictions.

        The method removes the element from the parent task, updates the examples'
        training flag, and registers the removed element in the predictions.

        Steps:
            1. Update removed element predictions.
            2. Call the parent delete method.
            3. Update the examples' training flag for the parent task.

        Args:
            notify_to (Iterable[UserDB], optional): List of users to notify about the deletion.
        """
        task = self.parents()[-1]
        # Update predictions to keep track of removed elements' values
        # TODO: What if the deletion fails?
        #       How to rollback the changes made to the `removed_elements` column of all predictions?
        self._update_removed_elements_predictions()
        # Delete element
        super().delete(notify_to=notify_to)
        # Update examples' training flag
        ExampleDB.set_examples_training_flag(task=task.db_object(), trained=False)

    @classmethod
    def get(cls,
            agent: Agent,
            db_object_or_id: Union[Entity, str],
            parents: list = None,
            cache: bool = False,
            check_permissions: bool = True,
            check_parents: bool = True,
            remove_notifications: bool = False):
        """
        Fetches an element resource from the database.

        Retrieves the resource corresponding to the given `db_object_or_id`, checking permissions
        and verifying the element type against the class's defined element type. Raises a
        ResourceNotFoundError if the element type does not match.

        Args:
            agent (Agent): The agent requesting the resource.
            db_object_or_id (Union[Entity, str]): The database object or identifier.
            parents (list, optional): The parent resources of the element.
            cache (bool, optional): Whether to use cached data.
            check_permissions (bool, optional): Whether to check permissions.
            check_parents (bool, optional): Whether to check parent resources.
            remove_notifications (bool, optional): Whether to remove notifications.

        Returns:
            Element: The element resource fetched from the database.

        Raises:
            ResourceNotFoundError: If the element type does not match the expected type.
        """
        resource = super().get(agent=agent,
                               db_object_or_id=db_object_or_id,
                               parents=parents,
                               cache=cache,
                               check_permissions=check_permissions,
                               check_parents=check_parents,
                               remove_notifications=remove_notifications)

        if resource.db_object().element_type != cls.element_type():
            raise ResourceNotFoundError(resource_id=(db_object_or_id if isinstance(db_object_or_id, str) else None))

        return resource

    @classmethod
    def post(cls,
             agent: Agent,
             data: dict,
             parents: list = None,
             check_permissions: bool = True,
             check_parents: bool = True,
             notify_to: Iterable[UserDB] = None):
        """
        Creates a new element resource in the database.

        Adds the element to the parent task and sets the training flag for the related
        examples to `False`. It checks permissions and parents before saving the new
        element.

        Args:
            agent (Agent): The agent creating the resource.
            data (dict): The data used to create the element.
            parents (list, optional): The parent resources of the element.
            check_permissions (bool, optional): Whether to check permissions.
            check_parents (bool, optional): Whether to check parent resources.
            notify_to (Iterable[UserDB], optional): List of users to notify about the creation.

        Returns:
            Element: The newly created element resource.
        """
        element = super().post(agent=agent,
                               data=data,
                               parents=parents,
                               check_permissions=check_permissions,
                               check_parents=check_parents,
                               notify_to=notify_to)
        ExampleDB.set_examples_training_flag(task=element.parents()[-1].db_object(), trained=False)

        return element

    def _set_data(self, data: dict, notify_to: Iterable[UserDB] = None):
        """
        Sets data for the current element, creating or updating the element in the database.

        This method manages the provided data, setting parents' primary key if applicable,
        and ensures that any necessary columns in the database are updated accordingly.
        In case of an error, the previous data is restored, and appropriate exceptions
        are raised for duplicate entries or invalid data.

        Note: we don't use super's method because `ElementDB`'s columns don't match request schema,
              which causes some data to be discarded.

        Args:
            data (dict): The data to set on the element.
            notify_to (Iterable[UserDB], optional): List of users to notify about the update.

        Raises:
            DuplicateResourceError: If a duplicate resource is detected based on the element name.
            InvalidDataError: If invalid data is provided.
        """
        if self.db_object() is not None:
            old_data = {c: getattr(self.db_object(), c) for c in self.db_model().columns()}
        else:
            old_data = None
        try:
            provided_data = {c: v for c, v in data.items() if c in self.db_model().columns()}
            provided_data['element_type'] = self.element_type()
            # Set parents' primary key
            if self.parents():
                parent_db_object = self.parents()[-1].db_object()
                for pk in parent_db_object.primary_key_columns():
                    provided_data[pk] = getattr(parent_db_object, pk)
            # Create or update database object
            if self.db_object() is None:
                self._db_object = self.db_model()(**provided_data)
            else:
                for column, value in provided_data.items():
                    setattr(self.db_object(), column, value)
            save_to_db(self._db_object)
        except Exception as e:
            db_rollback()
            if old_data:
                # Restore previous data (if the database object already existed)
                for column, value in old_data.items():
                    setattr(self.db_object(), column, value)
                save_to_db(self.db_object())
            duplicate_rsrc_msg = (f'Name "{data["name"]}" is already in use by another element. Names must be unique'
                                  f'across the inputs, outputs, and metadata elements of a task')
            if isinstance(e, IntegrityError):
                if e.orig.args[0] == 1062:
                    raise DuplicateResourceError(duplicate_rsrc_msg)
                else:
                    raise InvalidDataError()
            if '(pymysql.err.IntegrityError) (1062, "Duplicate entry' in str(e):
                # TODO: why is this exception not being raised as an `IntegrityError`?
                raise DuplicateResourceError(duplicate_rsrc_msg)
            raise e

    def _update_removed_elements_predictions(self):
        """
        Registers the values predicted for current element in the `PredictionDB.removed_elements` database column.

        The appended JSON object includes the element's name and all predicted values in the following format:

        {
            "element": "<element_name>",
            "values": [<value1>, <value2>, ...]
        }
        """

        # Get element info
        element_id = self.db_object().element_id
        element_name = self.db_object().name
        value_type = self.db_object().value_type

        # Get element-value database model
        pred_value_table = PredictionDB.value_type_models()[value_type].__tablename__

        # Use raw SQL
        raw_sql = sql_text(f"""
            UPDATE predictions p
            INNER JOIN (
                SELECT
                    pv.prediction_id,
                    JSON_ARRAYAGG(
                        JSON_EXTRACT(
                            JSON_OBJECT(
                                'value', pv.value
                            ),
                            '$.value'
                        )
                    ) AS values_json
                FROM {pred_value_table} pv
                WHERE pv.element_id = :element_id
                GROUP BY pv.prediction_id
                ORDER BY pv.index
            ) subq ON p.prediction_id = subq.prediction_id
            SET p.removed_elements = JSON_ARRAY_APPEND(
                p.removed_elements,
                '$',
                JSON_OBJECT(
                    'element', :element_name,
                    'values', subq.values_json
                )
            )
        """)

        # Execute the update
        db_execute(raw_sql, {'element_id': element_id, 'element_name': element_name})
        db_commit()


class InputElement(Element):
    """
    Represents an input element in the task schema.

    Input elements are used to define input data for tasks. This class handles the specific
    logic for managing input elements, including their type and collections.
    """

    @classmethod
    def element_type(cls) -> ElementType:
        """
        Returns the element type for input elements.

        Returns:
            ElementType: The type indicating this is an input element.
        """
        return ElementType.INPUT

    @classmethod
    def collections(cls) -> dict:
        """
        Defines collections of related resources for input elements.

        Returns:
            dict: A dictionary of related collections for input elements.
        """
        return {'categories': InputCategory}

    @classmethod
    def location(cls) -> str:
        """
        Returns the API endpoint location for input elements.

        Returns:
            str: The API endpoint for input elements.
        """
        return ENDPOINT_INPUT_ELEMENT


class OutputElement(Element):
    """
    Represents an output element in the task schema.

    Output elements define the output data produced by tasks. This class includes additional
    logic for handling output elements, such as ensuring the value types are valid.
    """

    ALLOWED_VALUE_TYPES = [
        ElementValueType.INTEGER, ElementValueType.FLOAT, ElementValueType.CATEGORY, ElementValueType.SHAPE,
        ElementValueType.SLICE
    ]

    @classmethod
    def element_type(cls) -> ElementType:
        """
        Returns the element type for output elements.

        Returns:
            ElementType: The type indicating this is an output element.
        """
        return ElementType.OUTPUT

    @classmethod
    def collections(cls) -> dict:
        """
        Defines collections of related resources for output elements.

        Returns:
            dict: A dictionary of related collections for output elements.
        """
        return {'categories': OutputCategory}

    @classmethod
    def location(cls) -> str:
        """
        Returns the API endpoint location for output elements.

        Returns:
            str: The API endpoint for output elements.
        """
        return ENDPOINT_OUTPUT_ELEMENT

    def _set_data(self, data: dict, notify_to: Iterable[UserDB] = None):
        """
        Sets data for an output element, ensuring the value type is valid.

        Validates that the provided value type for the output element is within the allowed
        types before setting the data. Calls the parent class method for data processing.

        Args:
            data (dict): The data to set for the output element.
            notify_to (Iterable[UserDB], optional): List of users to notify about the update.

        Raises:
            InvalidDataError: If the value type provided is not allowed.
        """
        if data['value_type'] not in self.ALLOWED_VALUE_TYPES:
            raise InvalidDataError(f'Invalid value type for an output element: {data["value_type"].name.lower()}')
        super()._set_data(data=data, notify_to=notify_to)


class MetadataElement(Element):
    """
    Represents a metadata element in the task schema.

    Metadata elements define additional information or metadata related to tasks.
    This class includes the logic for managing metadata elements, including their type and collections.
    """

    @classmethod
    def element_type(cls) -> ElementType:
        """
        Returns the element type for metadata elements.

        Returns:
            ElementType: The type indicating this is a metadata element.
        """
        return ElementType.METADATA

    @classmethod
    def collections(cls) -> dict:
        """
        Defines collections of related resources for metadata elements.

        Returns:
            dict: A dictionary of related collections for metadata elements.
        """
        return {'categories': MetadataCategory}

    @classmethod
    def location(cls) -> str:
        """
        Returns the API endpoint location for metadata elements.

        Returns:
            str: The API endpoint for metadata elements.
        """
        return ENDPOINT_METADATA_ELEMENT


class Category(_TaskSchemaResource, ABC):
    """
    Abstract base class for categories in a task schema.

    Categories group elements within a task schema, such as inputs or outputs, into distinct
    categories. This class handles the logic for loading and dumping category data and
    interacting with their parent elements.
    """

    @classmethod
    @abstractmethod
    def element_type(cls) -> ElementType:
        """
        Abstract method that returns the specific type of element the category applies to.

        This must be implemented by subclasses to specify the category's element type.
        """
        raise NotImplementedError()

    @classmethod
    def db_model(cls):
        """
        Returns the database model associated with the category.

        Returns:
            CategoryDB: The database model for this category.
        """
        return CategoryDB

    @classmethod
    def load_schema(cls):
        """
        Returns the schema used for loading data into the category.

        Returns:
            CategoryRequest: Schema used for loading category data.
        """
        return CategoryRequest

    @classmethod
    def dump_schema(cls):
        """
        Returns the schema used for dumping data from the category.

        Returns:
            CategoryResponse: Schema used for dumping category data.
        """
        return CategoryResponse

    @classmethod
    def get(cls,
            agent: Agent,
            db_object_or_id: Union[Entity, str],
            parents: list = None,
            cache: bool = False,
            check_permissions: bool = True,
            check_parents: bool = True,
            remove_notifications: bool = False):
        """
        Fetches a category resource from the database.

        Retrieves the resource corresponding to the given `db_object_or_id`, checking
        permissions and verifying that the parent element's type matches the category's
        element type. Raises a ResourceNotFoundError if the element type does not match.

        Args:
            agent (Agent): The agent requesting the resource.
            db_object_or_id (Union[Entity, str]): The database object or identifier.
            parents (list, optional): The parent resources of the category.
            cache (bool, optional): Whether to use cached data.
            check_permissions (bool, optional): Whether to check permissions.
            check_parents (bool, optional): Whether to check parent resources.
            remove_notifications (bool, optional): Whether to remove notifications.

        Returns:
            Category: The category resource fetched from the database.

        Raises:
            ResourceNotFoundError: If the parent element's type does not match the expected type.
        """
        resource = super().get(agent=agent,
                               db_object_or_id=db_object_or_id,
                               parents=parents,
                               cache=cache,
                               check_permissions=check_permissions,
                               check_parents=check_parents,
                               remove_notifications=remove_notifications)

        direct_parent = resource.parents()[-1] if resource.parents() else None
        if not (isinstance(direct_parent, Element) and direct_parent.db_object().element_type == cls.element_type()):
            raise ResourceNotFoundError(resource_id=(db_object_or_id if isinstance(db_object_or_id, str) else None))

        return resource


class InputCategory(Category):
    """
    Represents an input category in the task schema.

    Input categories group input elements within a task schema.
    """

    @classmethod
    def element_type(cls) -> ElementType:
        """
        Returns the element type for input categories.

        Returns:
            ElementType: The type indicating this is an input category.
        """
        return ElementType.INPUT

    @classmethod
    def location(cls) -> str:
        """
        Returns the API endpoint location for input categories.

        Returns:
            str: The API endpoint for input categories.
        """
        return ENDPOINT_INPUT_CATEGORY


class OutputCategory(Category):
    """
    Represents an output category in the task schema.

    Output categories group output elements within a task schema.
    """

    @classmethod
    def element_type(cls) -> ElementType:
        """
        Returns the element type for output categories.

        Returns:
            ElementType: The type indicating this is an output category.
        """
        return ElementType.OUTPUT

    @classmethod
    def location(cls) -> str:
        """
        Returns the API endpoint location for output categories.

        Returns:
            str: The API endpoint for output categories.
        """
        return ENDPOINT_OUTPUT_CATEGORY


class MetadataCategory(Category):
    """
    Represents a metadata category in the task schema.

    Metadata categories group metadata elements within a task schema.
    """

    @classmethod
    def element_type(cls) -> ElementType:
        """
        Returns the element type for metadata categories.

        Returns:
            ElementType: The type indicating this is a metadata category.
        """
        return ElementType.METADATA

    @classmethod
    def location(cls) -> str:
        """
        Returns the API endpoint location for metadata categories.

        Returns:
            str: The API endpoint for metadata categories.
        """
        return ENDPOINT_METADATA_CATEGORY


class Task(Resource):
    """
    Represents a task resource within the system.

    Tasks define a set of inputs, outputs, and metadata elements that describe a workflow.
    This class provides methods for creating, updating, and managing tasks and their schema,
    along with interacting with associated quotas, notifications, and services.
    """

    @classmethod
    def db_model(cls):
        """
        Returns the database model associated with the task.

        Returns:
            TaskDB: The database model for tasks.
        """
        return TaskDB

    @classmethod
    def load_schema(cls):
        """
        Returns the schema used for loading task data.

        Returns:
            TaskPOSTRequest: Schema used for loading task data.
        """
        return TaskPOSTRequest

    @classmethod
    def dump_schema(cls):
        """
        Returns the schema used for dumping task data.

        Returns:
            TaskResponse: Schema used for dumping task data.
        """
        return TaskResponse

    @classmethod
    def location(cls) -> str:
        """
        Returns the API endpoint location for tasks.

        Returns:
            str: The API endpoint for tasks.
        """
        return ENDPOINT_TASK

    @classmethod
    def permission_resource_type(cls):
        """
        Returns the resource type for permission checking on tasks.

        Returns:
            ResourceType: The resource type for task permissions.
        """
        return ResourceType.TASK

    @classmethod
    def notification_source_type(cls):
        """
        Returns the notification source type for tasks.

        Returns:
            NotificationSource: The notification source type for task notifications.
        """
        return NotificationSource.TASK

    @classmethod
    def post(cls,
             agent: Agent,
             data: dict,
             parents: list = None,
             check_permissions: bool = True,
             check_parents: bool = True,
             notify_to: Iterable[UserDB] = None):
        """
        Creates a new task in the database and updates related quotas.

        Checks the organization's quota usage for tasks and updates it accordingly.
        The task schema can optionally be initialized from a template. If a schema
        template is provided, the task type cannot be explicitly specified.

        Steps:
        1. Check quota usage for tasks.
        2. Create the task and update the organization's quota.
        3. Initialize the task, applying a schema template if specified.

        Args:
            agent (Agent): The agent creating the task.
            data (dict): The data used to create the task.
            parents (list, optional): The parent resources of the task.
            check_permissions (bool, optional): Whether to check permissions.
            check_parents (bool, optional): Whether to check parent resources.
            notify_to (Iterable[UserDB], optional): List of users to notify about the task creation.

        Returns:
            Task: The newly created task resource.

        Raises:
            UnprocessableRequestError: If the task type is specified when using a schema template.
        """
        # Check quota usage
        org = Organization.get(agent=agent, db_object_or_id=agent.organization, check_permissions=False)
        org.check_quota_usage(name='tasks', description='Maximum number of tasks', delta=1)
        # Create task
        data['organization_id'] = agent.organization_id
        task = super().post(agent=agent,
                            data=data,
                            parents=parents,
                            check_permissions=check_permissions,
                            check_parents=check_parents,
                            notify_to=notify_to)
        # Update quota usage
        org.update_quota_usage(name='tasks', delta=1)
        # Initialize task
        task.db_object().init_task()
        # Use task schema template if specified
        schema_template = data.get('template')
        if schema_template:
            if 'type' in data:
                raise UnprocessableRequestError('Task type cannot be specified when using a template')
            task.set_schema_from_template(task_template=schema_template, check_permissions=False)
            cls._clear_buffers(task)
        return task

    @classmethod
    def _clear_buffers(cls, task: 'Task'):
        """
        Clears buffers associated with the task.

        Buffers store task-related data, and this method ensures they are cleared
        when the task is initialized or reinitialized.

        Args:
            task (Task): The task whose buffers will be cleared.
        """
        al_buffer: ALBuffer = ALBuffer(buffer_io=ALBufferIO(task=task.db_object()))
        al_buffer.clear()

        mon_buffer = MonBuffer(buffer_io=MonBufferIO(task=task.db_object()))
        mon_buffer.clear()

    def put(self, data: dict, notify_to: Iterable[UserDB] = None):
        """
        Updates the task resource with new data.

        If a schema template is included in the request, it raises an error since
        templates can only be specified at creation time.

        Args:
            data (dict): The new data for the task.
            notify_to (Iterable[UserDB], optional): List of users to notify about the update.

        Raises:
            UnprocessableRequestError: If a template is specified in the update data.
        """
        if 'template' in data:
            raise UnprocessableRequestError('Templates can only be specified at creation time')
        super().put(data=data, notify_to=notify_to)

    def delete(self, notify_to: Iterable[UserDB] = None):
        """
        Deletes the task resource and updates related quotas.

        This method also deletes associated services and clients, and updates
        the organization's quota usage for tasks, examples, and space.

        Steps:
        1. Fetch task-related data (organization, services, examples, space usage).
        2. Delete the task and associated services.
        3. Update the organization's quota for tasks, examples, and space.

        Args:
            notify_to (Iterable[UserDB], optional): List of users to notify about the deletion.
        """
        # Get task data from database to use them after deleting the task
        org_db_object = self.db_object().organization

        services = Service.query().filter_by(task_id=self.db_object().task_id).all()
        services_clients = [service.client for service in services]

        num_examples = self.db_object().num_examples
        task_space_usage = self.db_object().space_usage

        # Delete task
        super().delete(notify_to=notify_to)

        # Delete services' clients
        delete_from_db(services_clients)

        # Update organization quota usage
        org = Organization.get(agent=self.agent(), db_object_or_id=org_db_object, check_permissions=False)

        org.update_quota_usage(name='tasks', delta=-1)
        org.update_quota_usage(name='examples', delta=-num_examples)
        org.update_quota_usage(name='space', delta=-task_space_usage)

    def dump(self,
             serialize=True,
             expand_associations=False,
             reference_parents=False,
             update_sync_state: bool = True) -> Union[ResourceResponseSchema, dict]:
        """
        Dumps the task data, including its icon, if present.
        WARNING: `icon` field is always expanded, even if `expand_associations=False`.

        If the task has an associated icon, it is always expanded, even if
        `expand_associations` is set to False.

        Args:
            serialize (bool, optional): Whether to serialize the output.
            expand_associations (bool, optional): Whether to expand associations in the output.
            reference_parents (bool, optional): Whether to reference parent resources.
            update_sync_state (bool, optional): Whether to update the sync state.

        Returns:
            Union[ResourceResponseSchema, dict]: The dumped task data.
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
                                 parents=[self],
                                 check_permissions=False,
                                 check_parents=False)
            dumped_data['icon'] = icon_file.dump(serialize=False)
        # Return dumped data
        return self.dump_data(dumped_data, serialize=serialize)

    def _set_data(self, data: dict, notify_to: Iterable[UserDB] = None):
        # Set status (if provided)
        status = data.pop('status', None)
        if status:
            self.db_object().set_status(Status.from_dict(status))
        # Get icon file
        icon_file_id = data.get('icon', None)
        if icon_file_id:
            icon_file = File.get(agent=self.agent(),
                                 db_object_or_id=icon_file_id,
                                 parents=[self],
                                 check_permissions=False,
                                 check_parents=False)
            if icon_file.db_object().use_for != TaskFileUse.PICTURE:
                raise InvalidDataError('Invalid task icon')
            data['icon'] = icon_file.db_object().file_id
        # Set resource data
        super()._set_data(data=data, notify_to=notify_to)

    def users_with_access(self) -> List[UserDB]:
        """
        Returns a list of users with access to the task.

        Uses the organization and task permissions to determine which users
        have access to the task.

        Returns:
            List[UserDB]: List of users with access to the task.
        """
        users_perms_ = users_permissions(organization=self.db_object().organization,
                                         resource_type=ResourceType.TASK,
                                         resource_uuid=self.uuid(),
                                         allow=True)
        return list(users_perms_.keys())

    def check_quota_usage(self, name: str, description: str = None, cache: bool = False, delta: Union[int, float] = 0):
        """
        Checks both the organization's and the task's quota usage to ensure limits are respected.

        This method first checks the organization's quota for a specific resource (identified by `name`) and then
        checks the task's own quota to ensure it remains within permissible limits. The task's quota usage may also
        be virtually adjusted using the `delta` parameter to simulate increases or decreases. If caching is enabled,
        the cached quota values are used; otherwise, the values are refreshed from the database.

        Steps:
        1. Retrieve the organization's quota usage and check if it exceeds the limit.
        2. Check if the task's cached data is available. If not, refresh the data from the database.
        3. Validate the task's quota usage by comparing its usage (plus the `delta` adjustment) to the allowed limit.

        Args: name (str): The name of the quota to check. description (str, optional): A description of the quota,
        used for logging or error messages. cache (bool, optional): If True, uses cached data for quota checks.
        Otherwise, it reloads data from the database. delta (Union[int, float], optional): A value to simulate an
        increase or decrease in quota usage for validation purposes.

        Raises:
            ResourceError: If the quota is exceeded, or if an error occurs while refreshing the task data or
            checking usage.
        """
        quota = quotas[name]
        pass  # TODO: the error message doesn't indicate whether the error
        #             refers to Organization's quota or Task's quota
        # Check Organization's quota limit
        org = Organization.get(agent=self.agent(),
                               db_object_or_id=self.db_object().organization,
                               check_permissions=False)
        org.check_quota_usage(name=name, description=description, cache=cache, delta=delta)
        # Check Task's quota limit
        if not cache and self.cached():
            self.refresh()
        check_quota_usage_(name=name,
                           usage=(getattr(self.db_object(), quota['usage']) + delta),
                           limit=getattr(self.db_object(), quota['limit']),
                           description=description)

    def update_quota_usage(self, name: str, delta: Union[int, float]):
        """
        Updates both the organization's and the task's quota usage.

        This method updates the quota usage for both the organization and the task by incrementing or decrementing
        the usage with the provided `delta` value. After updating, the task's data is refreshed to reflect the latest
        quota usage. If an operational error occurs during the update (e.g., database failure), a resource error is
        raised, and the task's status is set to indicate an unknown error.

        Steps:
        1. Update the organization's quota usage.
        2. Update the task's quota usage by applying the `delta`.
        3. If an error occurs during the update, an appropriate error message is logged and the task's status is
        updated.
        4. Refresh the task's data to reflect the latest quota usage.

        Args:
            name (str): The name of the quota being updated.
            delta (Union[int, float]): The amount to increment or decrement the quota usage.

        Raises:
            ResourceError: If there is an operational error while updating the task's quota, indicating a problem
            with the quota.
        """
        # Update Organization's quota usage
        org = Organization.get(agent=self.agent(),
                               db_object_or_id=self.db_object().organization,
                               check_permissions=False)
        org.update_quota_usage(name=name, delta=delta)
        # Update Task's quota usage
        try:
            self.db_object().update_numeric_value(column=quotas[name]['usage'], delta=delta)
        except OperationalError:
            ERR_MSG = f'There seems to be a problem with your quota. Please, contact {os.environ[ENV_SUPPORT_EMAIL]}'
            status = Status(template=task_unknown_error_status)
            status.details = {'message': ERR_MSG}
            self.db_object().set_status(status=status)
            raise ResourceError(ERR_MSG)
        # Refresh resource data
        self.refresh()

    def _get_elements(self, element_type: Type[Element]) -> List[Element]:
        """
        Retrieves all elements of the specified type for the current task.

        Fetches the input, output, or metadata elements associated with the task based on
        the provided element type. The method dynamically identifies the correct getter
        method for the specified element type and returns the elements.

        Args: element_type (Type[Element]): The type of element to retrieve (e.g., InputElement, OutputElement,
        MetadataElement).

        Returns:
            List[Element]: A list of elements of the specified type.
        """
        elements = list()
        getters = {
            InputElement: 'input_elements',
            OutputElement: 'output_elements',
            MetadataElement: 'metadata_elements',
        }
        for db_object in getattr(self.db_object(), getters[element_type])():
            element = element_type.get(agent=self.agent(),
                                       db_object_or_id=db_object,
                                       parents=(self.parents() + [self]),
                                       check_parents=False)
            elements.append(element)
        return elements

    def input_elements(self) -> List[InputElement]:
        """
        Returns all input elements associated with the task.

        Uses the `_get_elements` method to fetch input elements.

        Returns:
            List[InputElement]: A list of input elements for the task.
        """
        return self._get_elements(element_type=InputElement)

    def output_elements(self) -> List[OutputElement]:
        """
        Returns all output elements associated with the task.

        Uses the `_get_elements` method to fetch output elements.

        Returns:
            List[OutputElement]: A list of output elements for the task.
        """
        return self._get_elements(element_type=OutputElement)

    def metadata_elements(self) -> List[MetadataElement]:
        """
        Returns all metadata elements associated with the task.

        Uses the `_get_elements` method to fetch metadata elements.

        Returns:
            List[MetadataElement]: A list of metadata elements for the task.
        """
        return self._get_elements(element_type=MetadataElement)

    def _dump_elements(self) -> dict:
        return {
            'inputs': dump(resources=self.input_elements(), serialize=False, expand_associations=True),
            'outputs': dump(resources=self.output_elements(), serialize=False, expand_associations=True),
            'metadata': dump(resources=self.metadata_elements(), serialize=False, expand_associations=True)
        }

    def _infer_task_type(self, inputs: List[dict], outputs: List[dict]) -> Union[TaskType, List[TaskType]]:
        task_schema = TaskSchema(inputs=[InputElement.dump_data(data=x) for x in inputs],
                                 outputs=[OutputElement.dump_data(data=x) for x in outputs],
                                 task_type=self.db_object().type_)

        return task_schema.task_type

    def type_(self) -> Union[TaskType, List[TaskType]]:
        """
        Returns the task type.

        Returns:
            Union[TaskType, List[TaskType]]: The task type. If a list is returned, it indicates that no exact task type
                                             could be inferred from the task schema, and the list contains all the
                                             possible types the task could be.
        """
        elements = self._dump_elements()
        return self._infer_task_type(inputs=elements['inputs'], outputs=elements['outputs'])

    def dump_task_schema(self) -> dict:
        """
        Dumps the full task schema, including inputs, outputs, metadata, and inferred task type.

        The method extracts all elements (input, output, metadata) and constructs a schema
        dictionary containing these elements. It also infers and includes the task type.

        Returns:
            dict: A dictionary representing the full task schema, including the inferred task type.
        """
        # Dump elements
        task_schema_dict = self._dump_elements()

        # Update the dumped task schema with the inferred task type
        task_schema_dict['task_type'] = self._infer_task_type(inputs=task_schema_dict['inputs'],
                                                              outputs=task_schema_dict['outputs'])

        # Serialize and return the dumped task schema
        return TaskSchemaResponse().dump(task_schema_dict)

    def set_schema_from_template(self, task_template: TaskTemplate, check_permissions: bool = True):
        """
        Sets the task schema based on the provided template.

        Creates input, output, and metadata elements as defined by the template and associates
        them with the task. This method helps initialize the task schema by using predefined
        templates for specific task types.

        Args:
            task_template (TaskTemplate): The task template to base the schema on.
            check_permissions (bool, optional): Whether to check permissions when creating elements.

        Steps:
        1. Load the template schema.
        2. Create and associate input, output, and metadata elements based on the template.
        """

        def _create_resource(resource_model: Type[Resource], data: dict):
            data = resource_model.load_schema()().load(data)
            resource_model.post(agent=self.agent(),
                                data=data,
                                parents=[self],
                                check_permissions=check_permissions,
                                check_parents=check_permissions)

        schema_dict = get_task_schema_template(task_template=task_template)

        for input_element in schema_dict['inputs']:
            _create_resource(resource_model=InputElement, data=input_element)
        for output_element in schema_dict['outputs']:
            _create_resource(resource_model=OutputElement, data=output_element)
        for metadata_element in schema_dict.get('metadata', []):
            _create_resource(resource_model=MetadataElement, data=metadata_element)


class TaskResource(Resource):
    """
    Abstract base class representing task-related resources.

    This class serves as a base for all resources that are directly related to a task,
    providing common methods and structure for task resource management.
    """

    @classmethod
    @abstractmethod
    def db_model(cls) -> Type[TaskER]:
        """
        Abstract method to return the database model associated with the task resource.

        This method must be implemented by subclasses to define the specific task resource model.
        """
        pass
