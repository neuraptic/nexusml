from abc import abstractmethod
import copy
from datetime import datetime
from datetime import timedelta
import functools
import sys
from typing import Dict, Iterable, List, Optional, Type, Union

from sqlalchemy import bindparam
from sqlalchemy import func as sql_func
from sqlalchemy.exc import DatabaseError
from sqlalchemy.exc import StatementError

from nexusml.api.endpoints import ENDPOINT_EXAMPLE
from nexusml.api.endpoints import ENDPOINT_EXAMPLE_COMMENTS
from nexusml.api.endpoints import ENDPOINT_EXAMPLE_SHAPE
from nexusml.api.endpoints import ENDPOINT_EXAMPLE_SLICE
from nexusml.api.resources.base import InvalidDataError
from nexusml.api.resources.base import QuotaError
from nexusml.api.resources.base import Resource
from nexusml.api.resources.organizations import Organization
from nexusml.api.resources.tags import Tag
from nexusml.api.resources.tasks import InputElement
from nexusml.api.resources.tasks import OutputElement
from nexusml.api.resources.tasks import Task
from nexusml.api.resources.tasks import TaskResource
from nexusml.api.resources.utils import delete_example_or_prediction
from nexusml.api.resources.utils import dump_element_values
from nexusml.api.resources.utils import get_preloaded_db_object
from nexusml.api.resources.utils import merge_element_values_from_db_collections
from nexusml.api.resources.utils import preload_task_categories
from nexusml.api.resources.utils import preload_task_db_objects
from nexusml.api.resources.utils import split_element_values_into_db_collections
from nexusml.api.resources.utils import validate_element_values
from nexusml.api.schemas.base import ResourceResponseSchema
from nexusml.api.schemas.examples import CommentRequest
from nexusml.api.schemas.examples import CommentResponse
from nexusml.api.schemas.examples import ExampleRequest
from nexusml.api.schemas.examples import ExampleResponse
from nexusml.api.schemas.examples import ShapeRequest
from nexusml.api.schemas.examples import ShapeResponse
from nexusml.api.schemas.examples import SliceRequest
from nexusml.api.schemas.examples import SliceResponse
from nexusml.database.core import db
from nexusml.database.core import db_commit
from nexusml.database.core import db_execute
from nexusml.database.core import db_query
from nexusml.database.core import delete_from_db
from nexusml.database.core import save_to_db
from nexusml.database.examples import CommentDB
from nexusml.database.examples import ex_tags as ex_tags_table
from nexusml.database.examples import ExampleDB
from nexusml.database.examples import ExampleER
from nexusml.database.examples import ExValue
from nexusml.database.examples import ShapeCategory
from nexusml.database.examples import ShapeDB
from nexusml.database.examples import ShapeFloat
from nexusml.database.examples import SliceCategory
from nexusml.database.examples import SliceDB
from nexusml.database.examples import SliceFloat
from nexusml.database.notifications import AggregatedNotificationDB
from nexusml.database.organizations import Agent
from nexusml.database.organizations import UserDB
from nexusml.database.tags import TagDB
from nexusml.database.tasks import ElementDB
from nexusml.enums import DBRelationshipType
from nexusml.enums import ElementMultiValue
from nexusml.enums import ElementValueType
from nexusml.enums import LabelingStatus
from nexusml.enums import NotificationEvent
from nexusml.enums import NotificationSource
from nexusml.enums import ResourceAction
from nexusml.enums import ResourceCollectionOperation
from nexusml.enums import ResourceType


def _update_parent_example_activity_datetime(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Call wrapped function
        result = func(*args, **kwargs)

        # Get example resource
        if args and isinstance(args[0], ExampleResource):
            example_resource = args[0]
        elif isinstance(result, ExampleResource):
            example_resource = result
        else:
            return result

        # Update parent example's last activity datetime
        parent_example = example_resource.parents()[-1] if example_resource.parents() else None
        if isinstance(parent_example, Example):
            parent_example.db_object().activity_at = datetime.utcnow()
            parent_example.persist()

        # Return wrapped function's result
        return result

    return wrapper


def _update_quota_usage(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        def _check_space_quota_limit(task: Task, new_db_object: Union[ExampleDB, ExampleER],
                                     old_db_object: Union[ExampleDB, ExampleER]):
            """
            Check space quota limit (only if not deleting the resource). If space quota limit was exceeded:
                - If creating the resource, remove it.
                - If updating the resource, revert changes.
            """
            if func.__name__ == 'delete':
                return
            try:
                task.check_quota_usage(name='space', cache=True, delta=1)
            except QuotaError as qe:
                if func.__name__ == 'post':
                    delete_from_db(new_db_object)
                else:
                    db.session.merge(old_db_object)
                    db.session.commit()
                raise qe

        # Sanity check
        assert func.__name__ in ['post', 'put', 'delete']
        assert (isinstance(args[0], (Example, ExampleResource)) or args[0] == Example or
                issubclass(args[0], ExampleResource))

        # Get task and parent organization
        try:
            task = kwargs['parents'][0] if func.__name__ == 'post' else args[0].parents()[0]
            org = Organization.get(agent=task.agent(),
                                   db_object_or_id=task.db_object().organization,
                                   check_permissions=False)
        except Exception:
            task = None
            org = None
        assert isinstance(task, Task)
        assert isinstance(org, Organization)

        # Get example and child resource (if they exist)
        if isinstance(args[0], Example):
            example = args[0]
            resource = example
        elif isinstance(args[0], ExampleResource):
            example = args[0].parents()[-1]
            resource = args[0]
        else:
            example = kwargs['parents'][-1] if args[0] != Example else None
            resource = None
        assert isinstance(example, Example) or example is None

        # If creating an example, check the maximum number of examples
        creating_example = (example is None and func.__name__ == 'post')
        if creating_example:
            task.check_quota_usage(name='examples', cache=True, delta=1, description='Maximum number of examples')

        # Get example's current database entry and relationships, just in case we need to reverse changes to the example
        if func.__name__ == 'put':
            example_db_object = example.db_object()
            # Note: force relationship loading
            for relationship in example_db_object.relationships():
                getattr(example_db_object, relationship)
            old_example_db_object = copy.deepcopy(example_db_object)
        else:
            old_example_db_object = None

        # Get example's current size for later size comparisons
        # TODO: Calling `example.size(refresh=True)` leads to negative quota usage in database.
        #       Apparently, the example size registered in the database doesn't match the actual size.
        #       Figure out what's going on.
        old_example_size = example.size() if example is not None else None

        # Run decorated function
        func_result = func(*args, **kwargs)

        # Get example and child resource (if just created)
        if func.__name__ == 'post':
            resource = func_result
            example = resource if isinstance(resource, Example) else resource.parents()[-1]
            assert isinstance(example, Example)

        # Check space quota limit
        _check_space_quota_limit(task=task, new_db_object=example.db_object(), old_db_object=old_example_db_object)

        # Update space usage as follows:
        #   - If decorated function is `put()` or is being called for an example's child resource,
        #     subtract previous example size and add current.
        #   - If decorated function is `post()` and is being called for an example, add current example size.
        #   - If decorated function is `delete()` and is being called for an example, subtract current example size.
        if func.__name__ == 'put' or resource != example:
            new_example_size = example.size(refresh=True)
            size_increased = new_example_size > old_example_size
            if new_example_size != old_example_size:
                task.update_quota_usage(name='space', delta=-old_example_size)
                task.update_quota_usage(name='space', delta=new_example_size)
        else:
            size_increased = func.__name__ == 'post'
            if size_increased:
                task.update_quota_usage(name='space', delta=example.size(refresh=True))
            else:
                assert func.__name__ == 'delete'
                task.update_quota_usage(name='space', delta=-old_example_size)

        # Check space quota limit again (only if the example size was increased),
        # just in case space usage was updated by other request simultaneously.
        if size_increased:
            _check_space_quota_limit(task=task, new_db_object=example.db_object(), old_db_object=old_example_db_object)

        # If creating or deleting an example, update number of examples
        if creating_example:
            task.update_quota_usage(name='examples', delta=1)
        elif func.__name__ == 'delete' and resource == example:
            task.update_quota_usage(name='examples', delta=-1)

        # Return decorated function's result
        return func_result

    return wrapper


class ExampleResource(Resource):

    @classmethod
    @abstractmethod
    def db_model(cls) -> Type[ExampleER]:
        pass

    @classmethod
    @_update_parent_example_activity_datetime
    @_update_quota_usage
    def post(cls,
             agent: Agent,
             data: dict,
             parents: list = None,
             check_permissions: bool = True,
             check_parents: bool = True,
             notify_to: Iterable[UserDB] = None):
        return super().post(agent=agent,
                            data=data,
                            parents=parents,
                            check_permissions=check_permissions,
                            check_parents=check_parents,
                            notify_to=notify_to)

    @_update_parent_example_activity_datetime
    @_update_quota_usage
    def put(self, data: dict, notify_to: Iterable[UserDB] = None):
        return super().put(data=data, notify_to=notify_to)

    @_update_parent_example_activity_datetime
    @_update_quota_usage
    def delete(self, notify_to: Iterable[UserDB] = None):
        return super().delete(notify_to=notify_to)


class Comment(ExampleResource):

    @classmethod
    def db_model(cls):
        return CommentDB

    @classmethod
    def load_schema(cls):
        return CommentRequest

    @classmethod
    def dump_schema(cls):
        return CommentResponse

    @classmethod
    def location(cls) -> str:
        # WARNING: there is no endpoint for accessing an individual comment
        return ENDPOINT_EXAMPLE_COMMENTS + '/<comment_id>'

    @classmethod
    def notification_source_type(cls) -> NotificationSource:
        return NotificationSource.COMMENT

    def url(self) -> str:
        # Since there is no endpoint for accessing an individual comment,
        # remove comment ID from URL (will end with `/comments`)
        comments_url = super().url()
        return '/'.join(comments_url.split('/')[:-1])


class _ElementValueSubset(ExampleResource):
    """ Class implementing shapes and slices' common functionality. """

    @classmethod
    @abstractmethod
    def db_model(cls) -> Type[Union[ShapeDB, SliceDB]]:
        pass

    @classmethod
    @abstractmethod
    def load_schema(cls) -> Type[Union[ShapeRequest, SliceRequest]]:
        pass

    @classmethod
    @abstractmethod
    def dump_schema(cls) -> Type[Union[ShapeResponse, SliceResponse]]:
        pass

    @classmethod
    def touch_parent(cls) -> bool:
        # Touch parent example to track changes in shapes/slices in parent example's version history
        return True

    def delete(self, notify_to: Iterable[UserDB] = None):
        example = self.parents()[-1]
        # Delete shape/slice
        super().delete(notify_to=notify_to)
        # Update parent example's training flag
        example.db_object().trained = False
        save_to_db(example.db_object())

    def dump(self,
             serialize=True,
             expand_associations=False,
             reference_parents=True,
             update_sync_state: bool = True) -> Union[ResourceResponseSchema, dict]:
        """
        WARNING: `reference_parents` is always `True`, as parent element must always be referenced in shapes/slices.
        """
        return super().dump(serialize=serialize,
                            expand_associations=expand_associations,
                            reference_parents=True,
                            update_sync_state=update_sync_state)

    @classmethod
    def dump_data(cls,
                  data: dict,
                  serialize=True,
                  preloaded_elements: dict = None) -> Union[ResourceResponseSchema, dict]:
        # Merge output values
        merge_element_values_from_db_collections(data=data,
                                                 db_model=cls.db_model(),
                                                 preloaded_elements=preloaded_elements)
        # Return dumped data
        return super().dump_data(data, serialize=serialize)

    def _set_data(self, data: dict, notify_to: Iterable[UserDB] = None):
        # Note: we load resources instead of database objects to use cache and avoid hitting the database.

        task = self.parents()[-2]
        example = self.parents()[-1]

        # Verify related input element
        input_element = InputElement.get(agent=self.agent(),
                                         db_object_or_id=data['element'],
                                         parents=[task],
                                         check_permissions=False,
                                         check_parents=False)

        if self.db_model() == ShapeDB:
            visual_file_type = [ElementValueType.IMAGE_FILE, ElementValueType.VIDEO_FILE]
            if input_element.db_object().value_type not in visual_file_type:
                raise InvalidDataError('Related element must contain an image/video file')
        else:
            assert self.db_model() == SliceDB
            elem_db_obj = input_element.db_object()
            valid_value_types = [ElementValueType.INTEGER, ElementValueType.FLOAT]
            valid_multi_values = [ElementMultiValue.ORDERED, ElementMultiValue.TIME_SERIES]
            if not (elem_db_obj.value_type in valid_value_types and elem_db_obj.multi_value in valid_multi_values):
                raise InvalidDataError('Related element must contain a sequence of numbers')

        # Verify output element value types
        supported_value_types = [
            ElementValueType.INTEGER, ElementValueType.FLOAT, ElementValueType.TEXT, ElementValueType.CATEGORY
        ]
        for value in data.get('outputs', []):
            output_element = OutputElement.get(agent=task.agent(),
                                               db_object_or_id=value['element'],
                                               parents=[task],
                                               check_permissions=False,
                                               check_parents=False)
            if output_element.db_object().value_type not in supported_value_types:
                subsets_name = 'Shapes' if self.db_model() == ShapeDB else 'Slices'
                raise InvalidDataError(f"{subsets_name}' output values must be categories, numbers, or texts")

        # Verify slice's start/end indices
        if self.db_model() == SliceDB:
            elem_db_obj = input_element.db_object()
            elem_value_db_model = ExampleDB.value_type_models()[elem_db_obj.value_type]
            assert issubclass(elem_value_db_model, ExValue)
            min_max_index = db_query(
                sql_func.min(elem_value_db_model.index).label('min_index'),
                sql_func.max(elem_value_db_model.index).label('max_index')).filter(
                    elem_value_db_model.example_id == example.db_object().example_id,
                    elem_value_db_model.element_id == elem_db_obj.element_id).one()
            if data['start_index'] < min_max_index.min_index or data['end_index'] > min_max_index.max_index:
                raise InvalidDataError('Slice indices out of bounds')

        # Split output values into their corresponding database collections
        if 'outputs' in data:
            preloaded_elements = preload_task_db_objects(task=task.db_object(), db_model=ElementDB)
            preloaded_categories = preload_task_categories(task=task.db_object())

            check_required = example.db_object().labeling_status == LabelingStatus.LABELED

            validate_element_values(data=data,
                                    collection='outputs',
                                    preloaded_elements=preloaded_elements,
                                    allowed_value_types=self.db_model().value_type_models().keys(),
                                    check_required=check_required)

            split_element_values_into_db_collections(data=data,
                                                     collection='outputs',
                                                     db_model=self.db_model(),
                                                     task=task.db_object(),
                                                     preloaded_elements=preloaded_elements,
                                                     preloaded_categories=preloaded_categories)

        # Set resource data
        super()._set_data(data=data, notify_to=notify_to)

        # Propagate shape/slice's output values to parent example
        pass  # TODO: check whether the user has output propagation enabled in their settings
        self._propagate_outputs()

        # Update parent example's training flag
        example.db_object().trained = False
        save_to_db(example.db_object())

    def _propagate_outputs(self):
        """
        Propagates the output values of the specified shape or slice to the parent example.

        Notes on output value propagation criteria:
            - If an output element doesn't allow multiple values, the value of the parent example
              takes precedence over that of the shape/slice.
            - Otherwise, the values of the parent example and the shape/slice will be concatenated.
        """

        def _propagate_output_collection(parent_example: ExampleDB, output_collection: str, ex_value_collection: str):
            outputs = getattr(self.db_object(), output_collection)
            all_example_outputs = getattr(parent_example, ex_value_collection)
            for output in outputs:
                example_outputs = [x for x in all_example_outputs if x.element_id == output.element_id]
                if example_outputs:
                    element = ElementDB.get(element_id=output.element_id)
                    if element.multi_value is not None:
                        idx = max(x.index for x in example_outputs)
                    else:
                        # if the example already has a value for this output element, keep it and ignore shape/slice's
                        continue
                else:
                    idx = 0
                ex_value_model = getattr(ExampleDB, ex_value_collection).property.entity.entity
                example_output = ex_value_model(example_id=parent_example.example_id,
                                                element_id=output.element_id,
                                                index=(idx + 1),
                                                value=output.value)
                all_example_outputs.append(example_output)

        parent_example = self.parents()[-1].db_object()

        child_floats = {ShapeDB: 'shape_floats', SliceDB: 'slice_floats'}
        child_categories = {ShapeDB: 'shape_categories', SliceDB: 'slice_categories'}

        _propagate_output_collection(parent_example=parent_example,
                                     output_collection=child_floats[self.db_model()],
                                     ex_value_collection='ex_floats')

        _propagate_output_collection(parent_example=parent_example,
                                     output_collection=child_categories[self.db_model()],
                                     ex_value_collection='ex_categories')

        save_to_db(parent_example)


class Shape(_ElementValueSubset):

    @classmethod
    def db_model(cls):
        return ShapeDB

    @classmethod
    def load_schema(cls):
        return ShapeRequest

    @classmethod
    def dump_schema(cls):
        return ShapeResponse

    @classmethod
    def location(cls) -> str:
        return ENDPOINT_EXAMPLE_SHAPE

    @classmethod
    def associations(cls) -> dict:
        return {'shape_floats': ShapeFloat, 'shape_categories': ShapeCategory}


class Slice(_ElementValueSubset):

    @classmethod
    def db_model(cls):
        return SliceDB

    @classmethod
    def load_schema(cls):
        return SliceRequest

    @classmethod
    def dump_schema(cls):
        return SliceResponse

    @classmethod
    def location(cls) -> str:
        return ENDPOINT_EXAMPLE_SLICE

    @classmethod
    def associations(cls) -> dict:
        return {'slice_floats': SliceFloat, 'slice_categories': SliceCategory}


class Example(TaskResource):

    @classmethod
    def db_model(cls):
        return ExampleDB

    @classmethod
    def load_schema(cls):
        return ExampleRequest

    @classmethod
    def dump_schema(cls):
        return ExampleResponse

    @classmethod
    def location(cls) -> str:
        return ENDPOINT_EXAMPLE

    @classmethod
    def collections(cls) -> dict:
        return {'comments': Comment, 'shapes': Shape, 'slices': Slice}

    @classmethod
    def associations(cls) -> dict:
        assoc = {'tags': Tag}
        for value_collection in ExampleDB.value_collections():
            assoc[value_collection] = getattr(ExampleDB, value_collection).mapper.class_
        return assoc

    @classmethod
    def permission_resource_type(cls) -> ResourceType:
        return ResourceType.EXAMPLE

    @classmethod
    def notification_source_type(cls) -> NotificationSource:
        return NotificationSource.EXAMPLE

    @classmethod
    def post(cls,
             agent: Agent,
             data: dict,
             parents: list = None,
             check_permissions: bool = True,
             check_parents: bool = True,
             notify_to: Iterable[UserDB] = None):
        raise NotImplementedError('Individual example upload not supported. Use `post_batch()` instead')

    @_update_quota_usage
    def put(self, data: dict, notify_to: Iterable[UserDB] = None):
        # Get current labeling status if not provided
        if 'labeling_status' not in data:
            data['labeling_status'] = self.db_object().labeling_status
        # Update example
        super().put(data=data, notify_to=notify_to)
        # Update training flag
        self.db_object().trained = False
        self.persist()

    @_update_quota_usage
    def delete(self, notify_to: Iterable[UserDB] = None):
        delete_example_or_prediction(example_or_prediction=self, notify_to=notify_to)

    @classmethod
    def dump_data(cls,
                  data: dict,
                  serialize: bool = True,
                  preloaded_elements: dict = None) -> Union[ResourceResponseSchema, dict]:
        # Merge element types
        merge_element_values_from_db_collections(data=data, db_model=ExampleDB, preloaded_elements=preloaded_elements)
        # Return dumped data
        return super().dump_data(data, serialize=serialize)

    def _set_data(self, data: dict, notify_to: Iterable[UserDB] = None):
        task = self.parents()[0]
        assert isinstance(task, Task)

        preloaded_elements = preload_task_db_objects(task=task.db_object(), db_model=ElementDB)
        preloaded_categories = preload_task_categories(task=task.db_object())

        check_required = data.get('labeling_status') == LabelingStatus.LABELED

        validate_element_values(data=data,
                                collection='values',
                                preloaded_elements=preloaded_elements,
                                check_required=check_required)

        split_element_values_into_db_collections(data=data,
                                                 collection='values',
                                                 db_model=ExampleDB,
                                                 task=task.db_object(),
                                                 preloaded_elements=preloaded_elements,
                                                 preloaded_categories=preloaded_categories)

        super()._set_data(data=data, notify_to=notify_to)

    def _update_mutable_state(self, datetime_: datetime):
        super()._update_mutable_state(datetime_=datetime_)
        self.db_object().activity_at = datetime_

    def _update_collection(self,
                           collection_name: str,
                           operation: ResourceCollectionOperation,
                           resources: List = None,
                           persist: bool = True,
                           notify_to: Iterable[UserDB] = None):

        super()._update_collection(collection_name=collection_name,
                                   operation=operation,
                                   resources=resources,
                                   persist=persist,
                                   notify_to=notify_to)

        # Update last activity datetime
        self.db_object().activity_at = datetime.utcnow()
        self.persist()

    @classmethod
    def _size(cls, data_or_db_obj: Union[dict, ExampleDB]):

        def _getattr(obj, attr, default=None) -> object:
            if isinstance(obj, dict):
                return obj.get(attr, default)
            else:
                return getattr(obj, attr, default)

        def _element_values_size(data_or_db_obj_, type_name) -> int:
            return sum(sys.getsizeof(_getattr(x, 'value', 0)) for x in _getattr(data_or_db_obj_, type_name, []))

        size = sum(_element_values_size(data_or_db_obj, x) for x in cls.db_model().value_collections())

        size += len(_getattr(data_or_db_obj, 'tags', [])) * 8  # Assume example and tag primary keys are 4 bytes each

        size += sum(sys.getsizeof(x.message) for x in _getattr(data_or_db_obj, 'comments', []))

        for shape in _getattr(data_or_db_obj, 'shapes', []):
            # Shape definition
            size += sys.getsizeof(_getattr(shape, 'polygon')) if _getattr(shape, 'polygon') else 0
            size += sys.getsizeof(_getattr(shape, 'path')) if _getattr(shape, 'polygon') else 0
            size += sys.getsizeof(_getattr(shape, 'pixels')) if _getattr(shape, 'polygon') else 0
            # Shape values
            size += sum(_element_values_size(shape, x) for x in Shape.db_model().value_collections())

        for slice_ in _getattr(data_or_db_obj, 'slices', []):
            # Slice definition
            size += 2 * 4  # start/end indices, 4 bytes each
            # Slice values
            size += sum(_element_values_size(slice_, x) for x in Slice.db_model().value_collections())

        return size

    def size(self, refresh=False) -> int:
        """
        Returns the total size of the example considering:
            - Element values
            - Tags
            - Comments
            - Shapes
            - Slices

        WARNING:
            - The size is only computed if the `size` database field is `None` or if `refresh=True`.
              In the former case, the computed value will be registered in the `size` database field.
            - The returned size doesn't match the actual size of the example in the database.

        Args:
            refresh (bool): force current size computation, updating the size registered in the database

        Returns:
            int: size of the example in bytes
        """

        if not refresh and self.db_object().size is not None:
            return self.db_object().size

        # Get the data associated with the example
        relationships = self.db_object().relationship_types()
        children = relationships[DBRelationshipType.CHILD]
        assoc = relationships[DBRelationshipType.ASSOCIATION_OBJECT]

        data = {x: getattr(self.db_object(), x) for x in children + assoc}

        # Recalculate actual size and update registered size if necessary
        old_size = self.db_object().size
        new_size = self._size(data_or_db_obj=data)

        if new_size != old_size:
            self.db_object().size = new_size
            save_to_db(self.db_object())

        return self.db_object().size

    @classmethod
    def post_batch(cls, data: List[dict], task: Task) -> list:
        """
        Creates multiple examples at once.

        Args:
            data: list of JSONs containing examples' data
            task: parent task

        Returns:
            list: created examples (instances of this class)
        """

        # TODO: Decompose this function into smaller functions.

        def _delete_db_objs(db_objs):
            for db_obj in db_objs:
                try:
                    delete_from_db(db_obj)
                except Exception:
                    pass

        # Double check user permissions.
        # Note: Views calling this function should already have validated token scopes and user permissions.
        if task.user() is not None:
            Example.check_permissions(organization=task.db_object().organization,
                                      action=ResourceAction.CREATE,
                                      user=task.user(),
                                      check_parents=False)  # parent permissions already check when loading parent task

        # Check maximum number of examples quota
        task.check_quota_usage(name='examples', delta=len(data), description='Maximum number of examples')

        # Uncache direct parent's data if it is touched by the resource
        if cls.touch_parent():
            task.uncache()

        # Preload task elements, categories, and tags
        preloaded_elements = preload_task_db_objects(task=task.db_object(), db_model=ElementDB)
        preloaded_categories = preload_task_categories(task=task.db_object())
        preloaded_tags = preload_task_db_objects(task=task.db_object(), db_model=TagDB)

        # Validate and prepare provided data
        for ex_data in data:
            check_required = ex_data.get('labeling_status') == LabelingStatus.LABELED

            validate_element_values(data=ex_data,
                                    collection='values',
                                    preloaded_elements=preloaded_elements,
                                    check_required=check_required)

            split_element_values_into_db_collections(data=ex_data,
                                                     collection='values',
                                                     db_model=ExampleDB,
                                                     task=task.db_object(),
                                                     preloaded_elements=preloaded_elements,
                                                     preloaded_categories=preloaded_categories)

        # Check space quota
        total_size = sum(cls._size(data_or_db_obj=x) for x in data)
        task.check_quota_usage(name='space', delta=total_size)

        # Create examples
        examples = []
        examples_db_objects = []
        examples_tags: List[Optional[List[int]]] = []

        for ex_data in data:

            # Initialize example
            example = cls()
            example._db_object = cls.db_model()()

            # Set agent
            if isinstance(task.agent(), UserDB):
                example._user = task.agent()
            else:
                example._client = task.agent()

            # Set states
            utcnow = datetime.utcnow()
            if utcnow.microsecond >= 500000:
                utcnow += timedelta(seconds=1)
            utcnow = utcnow.replace(microsecond=0)

            example._set_immutable_state(datetime_=utcnow)
            example._update_mutable_state(datetime_=utcnow)

            # Set parents
            example._set_parents([task])

            # Set example attributes
            example.db_object().task_id = task.db_object().task_id
            for attr, value in ex_data.items():
                if attr not in cls.db_model().columns():
                    continue
                setattr(example.db_object(), attr, value)

            # Set size
            example.db_object().size = cls._size(data_or_db_obj=ex_data)

            # Add tags.
            # Note: We cannot keep track of the example's primary key because it hasn't been generated yet.
            # TODO: This is not working. Tags are not saved when calling `save_to_db()` below. Why?
            # example.db_object().tags = [
            #     get_preloaded_db_object(id_=tag_id, preloaded_db_objects=preloaded_tags)
            #     for tag_id in ex_data.get('tags', [])
            # ]
            example_tags = [
                get_preloaded_db_object(id_=tag_id, preloaded_db_objects=preloaded_tags).tag_id
                for tag_id in ex_data.get('tags', [])
            ]
            examples_tags.append(example_tags or None)

            # Add example
            examples.append(example)
            examples_db_objects.append(example.db_object())

        try:
            save_to_db(examples_db_objects)
        except (DatabaseError, StatementError):
            _delete_db_objs(examples_db_objects)
            raise InvalidDataError()

        # Create element-value association objects
        ex_values_db_objs = []

        for ex_db_object, ex_data in zip(examples_db_objects, data):

            element_values = []
            for ex_values_collection in cls.db_model().value_collections():
                element_values += ex_data.get(ex_values_collection, [])

            for element_value in element_values:
                element = get_preloaded_db_object(id_=element_value['element'], preloaded_db_objects=preloaded_elements)
                ex_value_db_model = cls.db_model().value_type_models()[element.value_type]
                ex_value_db_obj = ex_value_db_model(example_id=ex_db_object.example_id,
                                                    element_id=element.element_id,
                                                    index=element_value.get('index', 1),
                                                    value=element_value['value'])
                ex_values_db_objs.append(ex_value_db_obj)

        try:
            save_to_db(ex_values_db_objs)
        except (DatabaseError, StatementError):
            _delete_db_objs(ex_values_db_objs)
            _delete_db_objs(examples_db_objects)
            raise InvalidDataError()

        # Save tags
        # TODO: Use `ExampleDB.tags` relationship collection instead of "ex_tags" database table.
        #       This way, `save_to_db()` function will save the tags automatically when saving examples.
        ex_tag_rows = []
        for ex_db_obj, ex_tags in zip(examples_db_objects, examples_tags):
            if not ex_tags:
                continue
            ex_tag_rows += [{'example_id': ex_db_obj.example_id, 'tag_id': tag_id} for tag_id in ex_tags]
        if ex_tag_rows:
            ex_tag_insert = ex_tags_table.insert().values(example_id=bindparam('example_id'),
                                                          tag_id=bindparam('tag_id'))
            db_execute(ex_tag_insert, ex_tag_rows)
            db_commit()

        # Update quotas
        task.update_quota_usage(name='examples', delta=len(data))
        task.update_quota_usage(name='space', delta=total_size)

        # Notify users
        # Note: increase the value at SQL-level instead of Python-level to avoid race conditions
        for user in task.users_with_access():
            (AggregatedNotificationDB.query().filter_by(
                task_id=task.db_object().task_id,
                recipient=user.user_id,
                source_type=cls.notification_source_type(),
                event=NotificationEvent.CREATION).update(
                    {'count': AggregatedNotificationDB.count + len(examples_db_objects)}))

        db_commit()

        return examples

    @classmethod
    def dump_batch(cls, examples: list, task: Task, serialize: bool = True) -> List[dict]:
        """
        Dumps multiple examples at once.

        WARNING: use this function instead of `resources.base.dump()` to avoid performance issues.

        Args:
            examples (list): examples to be dumped.
                             They must be instances of either `ExampleDB` or `Example` (this class).
            task (Task): parent task.
            serialize (bool): see `resources.base.Resource.dump_data()`

        Returns:
            list: list of JSON objects.
        """

        # TODO: we can further optimize this function by building an SQL query with JOINs,
        #       instead of accessing each association object collection in each example
        #       (which builds a subquery for each collection in each example).
        #       See https://docs.sqlalchemy.org/en/13/orm/loading_relationships.html#select-in-loading

        def _dump_shapes_or_slices(db_objects: List[Union[ShapeDB, SliceDB]], element_names: Dict[int, str],
                                   category_names: Dict[int, str]):

            def _dump_collection(db_object: Union[ShapeDB, SliceDB], collection_name: str) -> List[dict]:
                dumped_collection = []

                for assoc_obj in getattr(db_object, collection_name):
                    dumped_assoc_obj = dict()
                    # Get related entities' IDs
                    dumped_assoc_obj['element'] = element_names[assoc_obj.element_id]
                    if collection_name in ['shape_categories', 'slice_categories']:
                        dumped_assoc_obj['category'] = category_names[assoc_obj.value]
                    # Get additional fields
                    for additional_field in assoc_obj.columns():
                        dumped_assoc_obj[additional_field] = getattr(assoc_obj, additional_field)
                    # Add dumped association object
                    dumped_collection.append(dumped_assoc_obj)

                return dumped_collection

            db_model = type(db_objects[0])
            assert all(isinstance(x, db_model) for x in db_objects)

            if db_model == ShapeDB:
                rsrc_model = Shape
                field_name = 'shapes'
                float_collection = 'shape_floats'
                category_collection = 'shape_categories'
            else:
                rsrc_model = Slice
                field_name = 'slices'
                float_collection = 'slice_floats'
                category_collection = 'slice_categories'

            example_dict[field_name] = []

            for db_obj in db_objects:
                obj_dict = db_obj.to_dict()
                obj_dict['id'] = obj_dict.pop('public_id')
                obj_dict['element'] = element_names[db_obj.element_id]
                obj_dict[float_collection] = _dump_collection(db_object=db_obj, collection_name=float_collection)
                obj_dict[category_collection] = _dump_collection(db_object=db_obj, collection_name=category_collection)

                obj_json = rsrc_model.dump_data(data=obj_dict, serialize=False, preloaded_elements=preloaded_elements)

                example_dict[field_name].append(obj_json)

        assert cls.db_model() == ExampleDB

        preloaded_elements = preload_task_db_objects(task=task.db_object(), db_model=ElementDB)
        element_names = {x.element_id: x.name for x in preloaded_elements['name'].values()}

        preloaded_categories = preload_task_categories(task=task.db_object())
        category_names = {x.category_id: x.name for x in preloaded_categories['name'].values()}

        preloaded_tags = preload_task_db_objects(task=task.db_object(), db_model=TagDB)
        tag_names = {x.tag_id: x.name for x in preloaded_tags['name'].values()}

        dumped_examples = []

        for example in examples:

            # Sanity check
            if isinstance(example, cls):
                assert len(example.parents()) == 1
                assert example.parents()[0].db_object() == task.db_object()
                example = example.db_object()
            else:
                assert isinstance(example, ExampleDB)
            assert example.task_id == task.db_object().task_id

            # Example attributes
            example_dict = example.to_dict()
            example_dict['id'] = example_dict.pop('public_id')

            # Element values
            example_dict.update(dump_element_values(db_object=example, element_names=element_names))

            # Tags
            example_tags = db_query(ex_tags_table).filter_by(example_id=example.example_id).all()
            example_dict['tags'] = [tag_names[example_tag.tag_id] for example_tag in example_tags]

            # Shapes and slices
            if example.shapes:
                _dump_shapes_or_slices(db_objects=example.shapes,
                                       element_names=element_names,
                                       category_names=category_names)
            if example.slices:
                _dump_shapes_or_slices(db_objects=example.slices,
                                       element_names=element_names,
                                       category_names=category_names)

            # Dump example
            dumped_example = Example.dump_data(data=example_dict,
                                               serialize=serialize,
                                               preloaded_elements=preloaded_elements)
            dumped_examples.append(dumped_example)

        return dumped_examples
