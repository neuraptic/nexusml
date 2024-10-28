import copy
from datetime import datetime
from datetime import timedelta
import sys
from typing import Iterable, List, Optional, TYPE_CHECKING, Union

from sqlalchemy.exc import DatabaseError
from sqlalchemy.exc import StatementError

from nexusml.api.endpoints import ENDPOINT_AI_MODEL
from nexusml.api.endpoints import ENDPOINT_AI_PREDICTION_LOG
from nexusml.api.resources.base import DuplicateResourceError
from nexusml.api.resources.base import ImmutableResourceError
from nexusml.api.resources.base import InvalidDataError
from nexusml.api.resources.base import PermissionDeniedError
from nexusml.api.resources.base import QuotaError
from nexusml.api.resources.base import ResourceNotFoundError
from nexusml.api.resources.files import TaskFile as File
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
from nexusml.api.schemas.ai import AIModelRequest
from nexusml.api.schemas.ai import AIModelResponse
from nexusml.api.schemas.ai import PredictionLogRequest
from nexusml.api.schemas.ai import PredictionLogResponse
from nexusml.api.schemas.ai import PredictionLogSchema
from nexusml.api.schemas.base import ResourceResponseSchema
from nexusml.database.ai import AIModelDB
from nexusml.database.ai import PredictionDB
from nexusml.database.ai import PredScores
from nexusml.database.core import db_commit
from nexusml.database.core import db_rollback
from nexusml.database.core import delete_from_db
from nexusml.database.core import save_to_db
from nexusml.database.files import TaskFileDB as FileDB
from nexusml.database.organizations import Agent
from nexusml.database.organizations import UserDB
from nexusml.database.services import Service
from nexusml.database.tasks import ElementDB
from nexusml.enums import AIEnvironment
from nexusml.enums import ElementType
from nexusml.enums import ElementValueType
from nexusml.enums import NotificationSource
from nexusml.enums import PredictionState
from nexusml.enums import ResourceType
from nexusml.enums import ServiceType
from nexusml.enums import TaskFileUse

if TYPE_CHECKING:
    from nexusml.api.resources.base import Resource


class AIModel(TaskResource):
    """
    Represents an AI model.

    Methods:
        db_model: Returns the database model for AIModel.
        load_schema: Returns the schema for loading AIModel data.
        dump_schema: Returns the schema for dumping AIModel data.
        location: Returns the endpoint location for AIModel.
        permission_resource_type: Returns the resource type for permission checking.
        notification_source_type: Returns the notification source type.
        post: Handles the creation of a new AI model, ensuring data is properly set and permissions are checked.
        delete: Handles the deletion of an AI model, updating the associated task and deleting related files.
        dump: Dumps the AI model data.
    """

    @classmethod
    def db_model(cls):
        return AIModelDB

    @classmethod
    def load_schema(cls):
        return AIModelRequest

    @classmethod
    def dump_schema(cls):
        return AIModelResponse

    @classmethod
    def location(cls) -> str:
        return ENDPOINT_AI_MODEL

    @classmethod
    def permission_resource_type(cls) -> ResourceType:
        return ResourceType.AI_MODEL

    @classmethod
    def notification_source_type(cls) -> NotificationSource:
        return NotificationSource.AI_MODEL

    @classmethod
    def post(cls,
             agent: Agent,
             data: dict,
             parents: list = None,
             check_permissions: bool = True,
             check_parents: bool = True,
             notify_to: Iterable[UserDB] = None) -> 'Resource':
        """
        Handles the creation of a new AI model.

        Steps:
        1. Verify parents and task.
        2. Set the task schema in data.
        3. Call the superclass post method with the provided arguments.

        Args:
            agent (Agent): The agent making the request.
            data (dict): The data for the new AI model.
            parents (list): List of parent resources.
            check_permissions (bool): Whether to check permissions.
            check_parents (bool): Whether to check parent resources.
            notify_to (Iterable[UserDB]): Users to notify.

        Returns:
            Resource: Response from the superclass post method.
        """
        assert parents
        task = parents[0]
        assert isinstance(task, Task)
        data['task_schema'] = task.dump_task_schema()
        return super().post(agent=agent,
                            data=data,
                            parents=parents,
                            check_permissions=check_permissions,
                            check_parents=check_parents,
                            notify_to=notify_to)

    def delete(self, notify_to: Iterable[UserDB] = None):
        """
        Handles the deletion of an AI model.

        Important note: `DELETE /<model_id>` endpoint is not enabled (see `views.ai.AIModelView`).

        Steps:
        1. Call the superclass delete method.
        2. Unset production/testing model in the associated task if applicable.
        3. Delete the associated file.

        Args:
            notify_to (Iterable[UserDB]): Users to notify.
                        We implement this method just in case we decide to enable this endpoint in the future.
        """
        super().delete(notify_to=notify_to)

        # Unset production/testing model.
        # Note: This wouldn't be necessary if we were able to add a Foreign Key
        #       constraint on `tasks.prod_model_id` and `tasks.test_model_id`.
        assert self.parents()
        task = self.parents()[0]
        assert isinstance(task, Task)

        task_affected = False

        if task.db_object().prod_model_id == self.db_object().model_id:
            task.db_object().prod_model_id = None
            task_affected = True
        if task.db_object().test_model_id == self.db_object().model_id:
            task.db_object().test_model_id = None
            task_affected = True

        if task_affected:
            task.persist()

        # Delete associated file
        # Note: Files cannot be shared by multiple AI models (they are unique in each task)
        file = File.get(agent=self.agent(),
                        db_object_or_id=self.db_object().file,
                        parents=self.parents(),
                        check_permissions=False,
                        check_parents=False)
        file.delete(notify_to=notify_to)

    def dump(self,
             serialize=True,
             expand_associations=False,
             reference_parents=False,
             update_sync_state: bool = True) -> Union[ResourceResponseSchema, dict]:
        """
        Dumps the AI model data.

        WARNING: `file` is always expanded, even if `expand_associations=False`.

        Args:
            serialize (bool): Whether to serialize the data.
            expand_associations (bool): Whether to expand associations.
            reference_parents (bool): Whether to reference parents.
            update_sync_state (bool): Whether to update the sync state.

        Returns:
            Union[ResourceResponseSchema, dict]: Dumped data.
        """

        # Dump resource data
        dumped_data = super().dump(serialize=False,
                                   expand_associations=expand_associations,
                                   reference_parents=reference_parents,
                                   update_sync_state=update_sync_state)

        # Get file
        parent_task = self.parents()[0]
        assert isinstance(parent_task, Task)
        file_db_object = FileDB.get(file_id=self.db_object().file_id)
        file = File.get(agent=parent_task.agent(), db_object_or_id=file_db_object, parents=[parent_task])
        dumped_data['file'] = file.dump(serialize=False,
                                        expand_associations=expand_associations,
                                        reference_parents=reference_parents)

        # Return dumped data
        return self.dump_data(dumped_data, serialize=serialize)

    def _set_data(self, data: dict, notify_to: Iterable[UserDB] = None):
        """
        Sets the data for the AI model.

        Args:
            data (dict): The data for the AI model.
            notify_to (Iterable[UserDB]): Users to notify.

        Raises:
            InvalidDataError: If the provided data is not valid.
        """
        # Get associated file
        parent_task = self.parents()[0]
        assert isinstance(parent_task, Task)
        model_file = FileDB.get_from_id(id_value=data.pop('file'), parent=parent_task.db_object())
        if model_file.use_for != TaskFileUse.AI_MODEL:
            raise InvalidDataError(f'File "{model_file.filename}" is not an AI model file')
        data['file_id'] = model_file.file_id

        # Set resource data
        try:
            super()._set_data(data=data, notify_to=notify_to)
        except DuplicateResourceError:
            raise DuplicateResourceError('Model file is already in use by another AI model')


class PredictionLog(TaskResource):
    """
    Represents prediction logs.

    Methods:
        db_model: Returns the database model for PredictionLog.
        load_schema: Returns the schema for loading PredictionLog data.
        dump_schema: Returns the schema for dumping PredictionLog data.
        location: Returns the endpoint location for PredictionLog.
        associations: Returns associations for PredictionLog.
        permission_resource_type: Returns the resource type for permission checking.
        post: Raises NotImplementedError as individual log upload is not supported.
        put: Raises ImmutableResourceError as PredictionLog resources cannot be modified.
        delete: Handles the deletion of a prediction log and updates quota usage.
        dump_data: Dumps the prediction log data, merging element types and loading AI model.
        _set_data: Raises NotImplementedError as individual log upload is not supported.
        _merge_prediction_data_collections: Merges prediction data collections into a single list.
        _save_invalid_prediction: Saves invalid prediction data in the database.
        _size: Calculates the size of the prediction data.
        size: Returns the total size of the prediction.
        post_batch: Handles the batch upload of prediction logs.
        dump_batch: Dumps multiple prediction logs at once.
    """

    _FORBIDDEN_UPLOAD_MSG = 'Endpoint available only to the Inference Service and the Testing Service'
    _UNSUPPORTED_INDIVIDUAL_UPLOAD_MSG = 'Individual log upload is not supported. Use "post_batch()" instead'

    @classmethod
    def db_model(cls):
        return PredictionDB

    @classmethod
    def load_schema(cls):
        return PredictionLogRequest

    @classmethod
    def dump_schema(cls):
        return PredictionLogResponse

    @classmethod
    def location(cls) -> str:
        return ENDPOINT_AI_PREDICTION_LOG

    @classmethod
    def associations(cls) -> dict:
        """
        Returns associations.

        Returns:
            dict: Associations mapping value collections to database classes.
        """
        _associations = dict()

        for value_collection in PredictionDB.value_collections():
            _associations[value_collection] = getattr(PredictionDB, value_collection).mapper.class_

        return _associations

    @classmethod
    def permission_resource_type(cls) -> ResourceType:
        return ResourceType.PREDICTION

    @classmethod
    def post(cls,
             agent: Agent,
             data: dict,
             parents: list = None,
             check_permissions: bool = True,
             check_parents: bool = True,
             notify_to: Iterable[UserDB] = None):
        """
        Raises NotImplementedError as individual log upload is not supported.

        Args:
            agent (Agent): The agent making the request.
            data (dict): The data for the prediction log.
            parents (list): List of parent resources.
            check_permissions (bool): Whether to check permissions.
            check_parents (bool): Whether to check parent resources.
            notify_to (Iterable[UserDB]): Users to notify.

        Raises:
            NotImplementedError: Individual log upload is not supported.
        """
        raise NotImplementedError(cls._UNSUPPORTED_INDIVIDUAL_UPLOAD_MSG)

    def put(self, data: dict, notify_to: Iterable[UserDB] = None):
        """
        Raises ImmutableResourceError as PredictionLog resources cannot be modified.

        Args:
            data (dict): The data for the prediction log.
            notify_to (Iterable[UserDB]): Users to notify.

        Raises:
            ImmutableResourceError: PredictionLog resources cannot be modified.
        """
        raise ImmutableResourceError()

    def delete(self, notify_to: Iterable[UserDB] = None):
        """
        Handles the deletion of a prediction log.

        Steps:
        1. Get the size of the prediction log.
        2. Delete the prediction log.
        3. Update quota usage.

        Args:
            notify_to (Iterable[UserDB]): Users to notify.
        """
        # Get the info to be used after deleting the database object
        size = self.size()

        # Delete prediction log
        delete_example_or_prediction(example_or_prediction=self, notify_to=notify_to)

        # Update quota usage.
        # Note: the counter used for registering the total number of predictions is automatically restarted every month.
        task = self.parents()[-1]
        assert isinstance(task, Task)

        task.update_quota_usage(name='space', delta=-size)

    @classmethod
    def dump_data(cls,
                  data: dict,
                  serialize: bool = True,
                  preloaded_elements: dict = None) -> Union[ResourceResponseSchema, dict]:
        """
        Dumps the prediction log data.

        Steps:
        1. Merge element types into data.
        2. Load the AI model based on the model ID in data.
        3. Call the superclass dump_data method with the provided arguments.

        Args:
            data (dict): The data to dump.
            serialize (bool): Whether to serialize the data.
            preloaded_elements (dict): Preloaded elements for the task.

        Returns:
            Union[ResourceResponseSchema, dict]: Dumped data.

        Raises:
            ResourceNotFoundError: If no AI model is found.
        """
        # Merge element types
        merge_element_values_from_db_collections(data=data,
                                                 db_model=PredictionDB,
                                                 preloaded_elements=preloaded_elements)

        # Load AI model
        # Note: we don't pass `reference_parents=True` to `dump()` because we don't want to load parent task
        try:
            model_id = data['model_id']
            data['ai_model'] = AIModelDB.get(model_id=model_id).public_id if model_id is not None else None
        except Exception:
            raise ResourceNotFoundError('No AI model found')

        # Return dumped data
        return super().dump_data(data, serialize=serialize)

    def _set_data(self, data: dict, notify_to: Iterable[UserDB] = None):
        """
        Raises NotImplementedError as individual log upload is not supported.

        Args:
            data (dict): The data for the prediction log.
            notify_to (Iterable[UserDB]): Users to notify.

        Raises:
            NotImplementedError: Individual log upload is not supported.
        """
        raise NotImplementedError(self._UNSUPPORTED_INDIVIDUAL_UPLOAD_MSG)

    @staticmethod
    def _merge_prediction_data_collections(prediction_data: dict):
        """
        Merges prediction data collections (inputs, outputs, and metadata) into a single list.

        Args:
            prediction_data (dict): The prediction data to merge.
        """
        prediction_data['values'] = (
            prediction_data['inputs']  # Inputs are always required
            + (prediction_data.pop('outputs', []) or [])  # Avoid `None`
            + (prediction_data.pop('metadata', []) or [])  # Avoid `None`
        )

    @staticmethod
    def _save_invalid_prediction(request_data: dict, prediction_db_object: PredictionDB):
        """
        Saves request data in the `invalid_data` column (of JSON type) and clears element values.

        Steps:
        1. Clear element values.
        2. Save request data in the invalid_data column.
        3. Update state and size.
        4. Save to database.

        Args:
            request_data (dict): The request data to save.
            prediction_db_object (PredictionDB): The prediction database object.
        """
        # Clear element values
        for value_collection in prediction_db_object.value_collections():
            getattr(prediction_db_object, value_collection).clear()
        # Save request data
        dumped_request_data = PredictionLogSchema().dump(request_data)
        elem_values = {k: v for k, v in dumped_request_data.items() if k in ['inputs', 'outputs', 'metadata']}
        prediction_db_object.invalid_data = elem_values
        # Update state and size
        prediction_db_object.state = PredictionState.FAILED
        prediction_db_object.size = len(str(prediction_db_object.invalid_data))
        # Save to database
        save_to_db(prediction_db_object)

    @classmethod
    def _size(cls, data_or_db_obj: Union[dict, PredictionDB]):
        """
        Calculates the size of the prediction data.

        Steps:
        1. Calculate the size of element values in data_or_db_obj.
        2. Add the size of shapes in data_or_db_obj.
        3. Return the total size.

        Args:
            data_or_db_obj (Union[dict, PredictionDB]): The data or database object.

        Returns:
            int: The size of the prediction data.
        """

        def _getattr(obj, attr, default=None) -> object:
            if isinstance(obj, dict):
                return obj.get(attr, default)
            else:
                return getattr(obj, attr, default)

        def _element_values_size(data_or_db_obj_, type_name) -> int:
            return sum(sys.getsizeof(_getattr(x, 'value', 0)) for x in _getattr(data_or_db_obj_, type_name, []))

        if isinstance(data_or_db_obj, dict) and 'values' in data_or_db_obj:
            size = len(str(data_or_db_obj['values']))
        else:
            size = sum(_element_values_size(data_or_db_obj, x) for x in cls.db_model().value_collections())

        size += len(str(_getattr(data_or_db_obj, 'shapes', '')))

        return size

    def size(self, refresh=False) -> int:
        """
        Returns the total size of the prediction.

        WARNING:
            - The size is only computed if the `size` database field is `None` or if `refresh=True`.
              In the former case, the computed value will be registered in the `size` database field.
            - The returned size doesn't match the actual size of the prediction in the database.

        Args:
            refresh (bool): Force current size computation, updating the size registered in the database.

        Returns:
            int: Size of the prediction in bytes.
        """

        # TODO: Can we reuse the code from `nexusml.resources.examples.Example`?

        if not refresh and self.db_object().size is not None:
            return self.db_object().size

        # Get prediction data
        data = {x: getattr(self.db_object(), x) for x in self.db_object().value_collections()}

        # Recalculate actual size and update registered size if necessary
        old_size = self.db_object().size
        new_size = self._size(data_or_db_obj=data)

        if new_size != old_size:
            self.db_object().size = new_size
            save_to_db(self.db_object())

        return self.db_object().size

    @classmethod
    def post_batch(cls, data: List[dict], task: Task, environment: AIEnvironment) -> list:
        """
        Handles the batch upload of prediction logs.

        WARNING: check permissions before calling this function.

        Steps:
        1. Verify the request was made by the Inference Service or the Testing Service.
        2. Preload task elements and categories.
        3. Initialize predictions.
        4. Validate and prepare prediction data.
        5. Create resource and database objects for predictions.
        6. Save predictions to the database.
        7. Create and save element-value association objects.
        8. Update state in complete predictions.
        9. Update quota usage and apply FIFO if necessary.

        Note: When uploading a batch of predictions, checking integrity (value types etc.) is not a priority.
              Contrary to examples, the priority in this case is to upload as many predictions as possible
              to keep track of as much information as possible.

        TODO: Validate predictions individually instead of using an all-or-none approach.
              This way, we can save valid predictions and reject (fill `invalid_data` of) only the invalid ones.

        Args:
            data (List[dict]): List of dictionaries following PredictionLogRequest schema.
            task (Task): Parent task.
            environment (AIEnvironment): Environment at which the prediction was made.

        Returns:
            list: Uploaded predictions (instances of this class).

        Raises:
            PermissionDeniedError: If the request was not made by the appropriate service.
            InvalidDataError: If there is an error saving predictions to the database.
        """

        def _quota_and_fifo(task: Task, new_predictions: List[PredictionLog]) -> List[PredictionLog]:
            """ Updates quota usage and applies FIFO (if needed). """
            task.update_quota_usage(name='predictions', delta=len(new_predictions))
            task.update_quota_usage(name='space', delta=sum(x.size() for x in new_predictions))

            return _prediction_logs_fifo(task=task, track_predictions=new_predictions)

        def _save_invalid_predictions(predictions: List[PredictionLog]) -> List[PredictionLog]:
            """ Saves invalid predictions to the database and returns those that could be saved. """
            saved_predictions = []

            # Save predictions
            for prediction, pred_data in zip(predictions, data):
                pred_db_obj = prediction.db_object()
                try:
                    cls._save_invalid_prediction(request_data=pred_data, prediction_db_object=pred_db_obj)
                    saved_predictions.append(prediction)
                except Exception:
                    db_rollback()
                    pass

            # Update quota usage and apply FIFO (if needed)
            return _quota_and_fifo(task=task, new_predictions=saved_predictions)

        # Double check that the request was made by the Inference Service or the Testing Service.
        # Note: Views calling this function should already have verified
        #       that the request was made by either of the services.
        if task.client() is None or task.user() is not None:
            raise PermissionDeniedError(cls._FORBIDDEN_UPLOAD_MSG)

        inference_service = Service.filter_by_task_and_type(task_id=task.db_object().task_id,
                                                            type_=ServiceType.INFERENCE)

        testing_service = Service.filter_by_task_and_type(task_id=task.db_object().task_id, type_=ServiceType.TESTING)

        if environment == AIEnvironment.PRODUCTION and task.client().client_id != inference_service.client_id:
            raise PermissionDeniedError(cls._FORBIDDEN_UPLOAD_MSG)
        elif environment == AIEnvironment.TESTING and task.client().client_id != testing_service.client_id:
            raise PermissionDeniedError(cls._FORBIDDEN_UPLOAD_MSG)

        # Keep original data
        # Note: we make a deep copy because inner JSONs are also modified (IDs are converted into primary keys, etc.)
        predictions_data = [copy.deepcopy(x) for x in data]

        # Preload task elements and categories
        preloaded_elements = preload_task_db_objects(task=task.db_object(), db_model=ElementDB)
        preloaded_categories = preload_task_categories(task=task.db_object())

        # Cache AI models' IDs
        ai_models = dict()

        # Initialize predictions
        predictions = []
        predictions_db_objects = []
        valid_data = True

        for pred_data in predictions_data:
            # Validate and prepare data
            try:
                # Merge "inputs", "outputs", and "metadata" fields into "values" field
                cls._merge_prediction_data_collections(pred_data)

                # Validate element values.
                # Note: There might be pending predictions without predicted values.
                validate_element_values(data=pred_data,
                                        collection='values',
                                        preloaded_elements=preloaded_elements,
                                        excluded_required=[ElementType.OUTPUT])

                # Validate target values.
                # Note: Target values are optional.
                validate_element_values(data=pred_data,
                                        collection='targets',
                                        preloaded_elements=preloaded_elements,
                                        check_required=False)

                # Mark target values
                target_values = pred_data.get('targets') or []  # Avoid `None`
                for element_value in target_values:
                    element_value['is_target'] = True

                # Group element-value JSONs by type
                kwargs = {
                    'data': pred_data,
                    'db_model': PredictionDB,
                    'task': task.db_object(),
                    'preloaded_elements': preloaded_elements,
                    'preloaded_categories': preloaded_categories
                }
                split_element_values_into_db_collections(collection='values', **kwargs)
                split_element_values_into_db_collections(collection='targets', **kwargs)
            except Exception:
                valid_data = False

            # Create resource object and database object
            prediction = cls()
            prediction._db_object = cls.db_model()()

            # Set agent
            if isinstance(task.agent(), UserDB):
                prediction._user = task.agent()
            else:
                prediction._client = task.agent()

            # Set immutable entity states
            utcnow = datetime.utcnow()
            if utcnow.microsecond >= 500000:
                utcnow += timedelta(seconds=1)
            utcnow = utcnow.replace(microsecond=0)

            prediction._set_immutable_state(datetime_=utcnow)

            # Set parents
            prediction._set_parents([task])

            # Set task ID
            prediction.db_object().task_id = task.db_object().task_id

            # Set AI model
            try:
                ai_model_id = pred_data.pop('ai_model')

                if ai_model_id not in ai_models:
                    ai_model = AIModelDB.get_from_id(id_value=ai_model_id, parent=task.db_object())
                    assert ai_model.task_id == task.db_object().task_id
                    ai_models[ai_model_id] = ai_model.model_id

                prediction.db_object().model_id = ai_models[ai_model_id]
            except Exception:
                prediction.db_object().model_id = None

            # Set environment
            prediction.db_object().environment = environment

            # Set prediction attributes
            for attr, value in pred_data.items():
                if attr == 'state' or attr in cls.db_model().value_collections():
                    continue
                setattr(prediction.db_object(), attr, value)

            # Set state and size
            if pred_data['state'] == PredictionState.COMPLETE:
                # Temporarily set state to "pending" until all element-values have been saved to database correctly.
                state = PredictionState.PENDING
            else:
                state = pred_data['state']

            prediction.db_object().state = state
            prediction.db_object().size = cls._size(data_or_db_obj=pred_data)

            # Add prediction
            predictions.append(prediction)
            predictions_db_objects.append(prediction.db_object())

        # Save predictions to database
        try:
            save_to_db(predictions_db_objects)
        except (DatabaseError, StatementError):
            raise InvalidDataError()

        # Save element-value association objects
        pred_values_db_objs = []

        try:
            if not valid_data:
                raise InvalidDataError()

            # Create database objects
            for pred_db_obj, pred_data in zip(predictions_db_objects, predictions_data):
                # Get element-value JSONs
                element_values = []
                for pred_values_collection in cls.db_model().value_collections():
                    element_values += pred_data.get(pred_values_collection, [])

                # Create and save element-value database objects
                for element_value in element_values:
                    element = get_preloaded_db_object(id_=element_value['element'],
                                                      preloaded_db_objects=preloaded_elements)

                    is_pred_score = (element.element_type == ElementType.OUTPUT and
                                     element.value_type == ElementValueType.CATEGORY)

                    if is_pred_score:
                        pred_value_db_model = PredScores
                    else:
                        pred_value_db_model = cls.db_model().value_type_models()[element.value_type]

                    pred_value_db_obj = pred_value_db_model(prediction_id=pred_db_obj.prediction_id,
                                                            element_id=element.element_id,
                                                            is_target=element_value.get('is_target', False),
                                                            index=element_value.get('index', 1),
                                                            value=element_value['value'])

                    pred_values_db_objs.append(pred_value_db_obj)

            # Save objects to database
            save_to_db(pred_values_db_objs)
        except Exception:
            db_rollback()
            return _save_invalid_predictions(predictions)

        # Update state in complete predictions
        # (Until all element-values are saved to database correctly, the state is set to "pending")
        states_updated = False

        for pred_db_obj, pred_data in zip(predictions_db_objects, predictions_data):
            if pred_data['state'] == PredictionState.COMPLETE:
                states_updated = True
                pred_db_obj.state = PredictionState.COMPLETE

        if states_updated:
            db_commit()

        # Update quota usage and apply FIFO (if needed)
        return _quota_and_fifo(task=task, new_predictions=predictions)

    @classmethod
    def dump_batch(cls, predictions: list, task: Task, serialize: bool = True) -> List[dict]:
        """
        Dumps multiple prediction logs at once. This method is optimized for performance and avoids
        accessing each association object collection in each prediction.

        WARNING: Use this function instead of `resources.base.dump()` to avoid performance issues.

        Steps:
        1. Preload task elements.
        2. Initialize dictionaries for element names and AI models.
        3. Iterate over predictions, dumping data and element values.
        4. Return the dumped predictions.

        Args:
            predictions (list): Predictions to be dumped. They must be instances of either PredictionDB or this class.
            task (Task): Parent task.
            serialize (bool): Whether to serialize the data.

        Returns:
            list: List of JSON objects.
        """

        # TODO: we can further optimize this function by building an SQL query with JOINs,
        #       instead of accessing each association object collection in each prediction
        #       (which builds a subquery for each collection in each prediction).
        #       See https://docs.sqlalchemy.org/en/13/orm/loading_relationships.html#select-in-loading

        assert cls.db_model() == PredictionDB

        preloaded_elements = preload_task_db_objects(task=task.db_object(), db_model=ElementDB)
        element_names = {x.element_id: x.name for x in preloaded_elements['name'].values()}

        ai_models = dict()

        dumped_predictions = []

        for prediction in predictions:

            # Sanity check
            if isinstance(prediction, cls):
                assert len(prediction.parents()) == 1
                assert prediction.parents()[0].db_object() == task.db_object()
                prediction = prediction.db_object()
            else:
                assert isinstance(prediction, PredictionDB)
            assert prediction.task_id == task.db_object().task_id

            # Prediction attributes
            prediction_dict = prediction.to_dict()
            prediction_dict['id'] = prediction_dict.pop('public_id')

            # Element values
            prediction_dict.update(dump_element_values(db_object=prediction, element_names=element_names))

            # AI model
            if prediction.model_id is not None and prediction.model_id not in ai_models:
                ai_model = AIModelDB.get(model_id=prediction.model_id)
                if ai_model is not None:
                    ai_models[prediction.model_id] = ai_model.public_id

            prediction_dict['ai_model'] = ai_models.get(prediction.model_id)

            # Dump prediction
            dumped_prediction = PredictionLog.dump_data(data=prediction_dict,
                                                        serialize=serialize,
                                                        preloaded_elements=preloaded_elements)
            dumped_predictions.append(dumped_prediction)

        return dumped_predictions


def _prediction_logs_fifo(task: Task, track_predictions: List[PredictionLog] = None) -> Optional[List[PredictionLog]]:
    """
    Checks the limit of predictions and the space quota. If exceeded, the oldest predictions are deleted.

    TODO: 1) This method wouldn't work under race conditions.
          2) Checking and updating quota each time a prediction is deleted is very slow.
             Optimize the selection of the predictions to be deleted.

    Steps:
    1. Check quota usage.
    2. If quota exceeded, delete the oldest prediction.
    3. Update quota usage.
    4. Repeat until quota usage is within limits.
    5. Return the tracked predictions that were not deleted.

    Args:
        task (Task): Parent task
        track_predictions (List[PredictionLog]): if specified, the function will return the predictions given by
                                                 `track_predictions` that remain after applying FIFO.

    Returns:
        Optional[List[PredictionLog]]: if `track_predictions` is specified, the function will return the predictions
                                       given by `track_predictions` that remain after applying FIFO.
    """
    remaining_tracked_predictions = [x.db_object().uuid for x in (track_predictions or [])]

    first_check = True
    fifo_done = False

    while not fifo_done:
        try:
            task.check_quota_usage(name='space', cache=first_check)
            fifo_done = True
        except QuotaError:
            # Get the oldest prediction
            oldest_prediction = (PredictionDB.query().filter_by(task_id=task.db_object().task_id).order_by(
                PredictionDB.created_at).first())
            if oldest_prediction is None:
                break

            # Delete prediction
            delete_from_db(oldest_prediction)
            if oldest_prediction.uuid in remaining_tracked_predictions:
                remaining_tracked_predictions.remove(oldest_prediction.uuid)

            # Update quota.
            # Note: the counter used for registering the total number
            #       of predictions is automatically restarted every month.
            task.update_quota_usage(name='space', delta=-oldest_prediction.size)
        finally:
            first_check = False

    # Return the tracked predictions that were not deleted
    if track_predictions:
        return [x for x in track_predictions if x.db_object().uuid in remaining_tracked_predictions]
