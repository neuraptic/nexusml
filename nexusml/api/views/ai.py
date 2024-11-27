import copy
import json
import time
from typing import Dict, List

from celery import shared_task
from flask import jsonify
from flask import Response
from flask_apispec import doc
from flask_apispec import marshal_with
from flask_apispec import use_kwargs
from marshmallow import fields
from marshmallow import validate
from sqlalchemy import or_ as sql_or

from nexusml.api.ext import redis_buffer
from nexusml.api.jobs.event_jobs import run_mon_service
from nexusml.api.jobs.event_jobs import train
from nexusml.api.resources.ai import AIModel
from nexusml.api.resources.ai import PredictionLog
from nexusml.api.resources.base import dump
from nexusml.api.resources.base import PermissionDeniedError
from nexusml.api.resources.base import Resource
from nexusml.api.resources.base import ResourceNotFoundError
from nexusml.api.resources.base import UnprocessableRequestError
from nexusml.api.resources.files import TaskFile as File
from nexusml.api.resources.tasks import Task
from nexusml.api.resources.utils import get_preloaded_db_object
from nexusml.api.resources.utils import preload_task_db_objects
from nexusml.api.resources.utils import validate_element_values
from nexusml.api.schemas.ai import AIModelRequest
from nexusml.api.schemas.ai import AIModelResponse
from nexusml.api.schemas.ai import DeploymentRequest
from nexusml.api.schemas.ai import DeploymentResponse
from nexusml.api.schemas.ai import InferenceRequest
from nexusml.api.schemas.ai import InferenceResponse
from nexusml.api.schemas.ai import PredictionLoggingRequest
from nexusml.api.schemas.ai import PredictionLogPage
from nexusml.api.schemas.ai import PredictionLogResponse
from nexusml.api.schemas.ai import PredictionLogSchema
from nexusml.api.schemas.ai import TestRequest
from nexusml.api.schemas.ai import TrainingRequest
from nexusml.api.utils import config
from nexusml.api.utils import get_engine_type
from nexusml.api.views.base import create_view
from nexusml.api.views.core import agent_from_token
from nexusml.api.views.core import error_response
from nexusml.api.views.core import process_delete_request
from nexusml.api.views.core import process_get_request
from nexusml.api.views.core import process_post_or_put_request
from nexusml.api.views.utils import get_examples_or_predictions
from nexusml.api.views.utils import paging_url_params
from nexusml.constants import HTTP_BAD_REQUEST_STATUS_CODE
from nexusml.constants import REDIS_PREDICTION_LOG_BUFFER_KEY
from nexusml.constants import SWAGGER_TAG_AI
from nexusml.database.ai import AIModelDB
from nexusml.database.ai import PredictionDB
from nexusml.database.ai import PredScores
from nexusml.database.core import db_commit
from nexusml.database.core import db_query
from nexusml.database.core import save_to_db
from nexusml.database.organizations import ClientDB
from nexusml.database.organizations import UserDB
from nexusml.database.services import Service
from nexusml.database.tasks import ElementDB
from nexusml.database.tasks import TaskDB
from nexusml.engine.buffers import ALBuffer
from nexusml.engine.buffers import ALBufferIO
from nexusml.engine.buffers import MonBuffer
from nexusml.engine.buffers import MonBufferIO
from nexusml.engine.workers import get_engine
from nexusml.enums import AIEnvironment
from nexusml.enums import ElementType
from nexusml.enums import ElementValueType
from nexusml.enums import PredictionState
from nexusml.enums import ResourceAction
from nexusml.enums import ServiceType
from nexusml.enums import TaskType
from nexusml.statuses import CL_DEPLOYING_STATUS_CODE
from nexusml.statuses import CL_STOPPED_STATUS_CODE
from nexusml.statuses import CL_WAITING_STATUS_CODE
from nexusml.statuses import inference_processing_status
from nexusml.statuses import INFERENCE_STOPPED_STATUS_CODE
from nexusml.statuses import inference_waiting_status
from nexusml.statuses import Status
from nexusml.statuses import TASK_ACTIVE_STATUS_CODE
from nexusml.statuses import testing_processing_status
from nexusml.statuses import TESTING_STOPPED_STATUS_CODE
from nexusml.statuses import testing_waiting_status
from nexusml.utils import FILE_TYPES

################
# Define views #
################

_AIView = create_view(resource_types=[Task])
_AIModelView = create_view(resource_types=[Task, AIModel])
_PredictionLogView = create_view(resource_types=[Task, PredictionLog])

##########################
# Training and Inference #
##########################


class TrainingView(_AIView):

    @doc(tags=[SWAGGER_TAG_AI])
    @use_kwargs(TrainingRequest, location='json')
    def post(self, task_id: str, resources: List[Resource], **kwargs):
        """
        Handles the training process for an AI model. It verifies that the Continual Learning Service
        is running and that no previous training is still in progress. It also checks the quota usage
        for CPU and GPU before triggering the training.

        Steps:
        1. Ensure the Continual Learning Service is running and not stopped.
        2. Verify that the previous training session is finished.
        3. Check the task's CPU and GPU quota usage.
        4. Retrieve the latest AI model, or validate the task schema if no model exists.
        5. Run the training process for the AI model.

        Args:
            task_id (str): The task ID for which training is requested.
            resources (List[Resource]): The list of resources, with the task being the last resource.
            **kwargs: Additional request payload containing training data.

        Returns:
            Response: HTTP 202 response indicating the request is accepted.
        """
        task = resources[-1]

        cl_service = Service.filter_by_task_and_type(task_id=task.db_object().task_id,
                                                     type_=ServiceType.CONTINUAL_LEARNING)

        # Verify Continual Learning Service is running
        if cl_service.status['code'] == CL_STOPPED_STATUS_CODE:
            raise UnprocessableRequestError('Continual Learning Service is not running')

        # Verify previous training finished
        if cl_service.status['code'] != CL_WAITING_STATUS_CODE:
            raise UnprocessableRequestError('Previous training has not finished yet')

        # Check CPU/GPU quota
        task.check_quota_usage(name='cpu', description='Maximum CPU hours', delta=.01)
        task.check_quota_usage(name='gpu', description='Maximum GPU hours', delta=.01)

        # Get the latest AI model
        latest_model = (db_query(AIModelDB).filter_by(task_id=task.db_object().task_id).order_by(
            AIModelDB.created_at.desc()).first())

        # If there is no trained model, verify that the task type is known
        if latest_model is None:
            if task.type_() == TaskType.UNKNOWN:
                raise UnprocessableRequestError('Unknown task type. Please, revise task schema')

        # Run training
        train.delay(task_uuid=task.uuid(), model_uuid=(str(latest_model.uuid) if latest_model is not None else None))
        return Response(status=202)


class InferenceView(_AIView):

    @doc(tags=[SWAGGER_TAG_AI])
    @use_kwargs(InferenceRequest, location='json')
    @marshal_with(InferenceResponse)
    def post(self, task_id: str, resources: List[Resource], **kwargs):
        """
        Handles inference requests for AI models. It verifies that Inference Service is available,
        checks quota usage for the number of predictions, ensures the inference service is running,
        and loads the AI model to run predictions on the provided data. Finally, it logs the predictions
        and returns the results.

        WARNING: quota usage is updated by `save_buffered_prediction_logs()`.

        Steps:
        1. Verify the Inference Service is running and not stopped.
        2. Validate the task's quota for predictions.
        3. Load the production AI model and check for integrity.
        4. Preload task elements and validate the incoming batch of data.
        5. Run inference using the AI model and log predictions.
        6. Return the predictions in the response.

        Args:
            task_id (str): The task ID for which inference is requested.
            resources (List[Resource]): The list of resources, with the task being the last resource.
            **kwargs: Additional request payload containing inference data.

        Returns:
            Response: The response containing the prediction results and AI model UUID.
        """
        task = resources[-1]
        assert isinstance(task, Task)

        # Verify Inference Service is running
        # TODO: cache inference service to speed up its retrieval
        inference_service = Service.filter_by_task_and_type(task_id=task.db_object().task_id,
                                                            type_=ServiceType.INFERENCE)
        if inference_service.status['code'] == INFERENCE_STOPPED_STATUS_CODE:
            raise UnprocessableRequestError('Inference Service is not running')

        # Check predictions quota limit
        task.check_quota_usage(name='predictions',
                               description='Maximum number of predictions',
                               delta=len(kwargs['batch']))

        # Load production model
        ai_model = AIModelDB.get(model_id=task.db_object().prod_model_id)
        if ai_model is None:
            raise ResourceNotFoundError('No production AI model found')

        # Preload task elements
        preloaded_elements = preload_task_db_objects(task=task.db_object(), db_model=ElementDB)

        # Validate request data
        for observation in kwargs['batch']:
            validate_element_values(data=observation,
                                    collection='values',
                                    preloaded_elements=preloaded_elements,
                                    excluded_required=[ElementType.OUTPUT, ElementType.METADATA])

        # Prepare request data for engine
        data = _prepare_engine_request_data(original_request_data=InferenceRequest().dump(kwargs),
                                            task=task,
                                            preloaded_elements=preloaded_elements)

        # Get engine
        engine = get_engine(engine_type=get_engine_type(), task_uuid=task.uuid())

        # Run inference
        inference_service.set_status(status=Status(template=inference_processing_status))
        predictions = engine.predict(environment=AIEnvironment.PRODUCTION, data=data['batch'])
        inference_service.set_status(status=Status(template=inference_waiting_status))

        # Log predictions
        _log_predictions(observations=kwargs['batch'],
                         task_id=task.db_object().task_id,
                         client_id=inference_service.client_id,
                         preloaded_elements=preloaded_elements,
                         ai_model_uuid=ai_model.uuid,
                         predictions=predictions)

        # Return predictions
        dumped_predictions = InferenceResponse().dump({'predictions': predictions, 'ai_model': ai_model.uuid})

        return jsonify(dumped_predictions)


##############
# AI Testing #
##############


@shared_task
def trigger_pending_test_predictions(task_id: int, max_attempts: int = 6):
    """
    Triggers pending test predictions.

    Args:
        task_id (int): The task ID for which predictions are triggered.
        max_attempts (int, optional): Maximum number of attempts to find pending predictions. Defaults to 6.
    """
    # Load testing service and task
    testing_service = Service.filter_by_task_and_type(task_id=task_id, type_=ServiceType.TESTING)
    task = Task.get(agent=testing_service.client, db_object_or_id=TaskDB.get(task_id=task_id))

    # Load testing model
    ai_model = AIModelDB.get(model_id=task.db_object().test_model_id)
    if ai_model is None:
        return

    # Preload task elements
    preloaded_elements = preload_task_db_objects(task=task.db_object(), db_model=ElementDB)

    # Load pending test predictions
    pending_predictions = []
    attempt = 0

    while not pending_predictions and attempt < max_attempts:
        pending_predictions = (PredictionDB.query().filter_by(task_id=task_id,
                                                              environment=AIEnvironment.TESTING,
                                                              state=PredictionState.PENDING).all())
        time.sleep(2**attempt)  # Wait for (2 ** max_attempts - 1) seconds maximum
        attempt += 1

    if attempt > max_attempts:
        return

    # Update predictions' state
    for x in pending_predictions:
        x.state = PredictionState.IN_PROGRESS
    db_commit()

    # Prepare request data for engine
    request_data = {
        'batch': [{
            'values': x.get('inputs', []) + x.get('metadata', [])
        } for x in PredictionLog.dump_batch(predictions=pending_predictions, task=task)]
    }

    data = _prepare_engine_request_data(original_request_data=request_data,
                                        task=task,
                                        preloaded_elements=preloaded_elements)

    # Get engine
    engine = get_engine(engine_type=get_engine_type(), task_uuid=task.uuid())

    # Make predictions
    testing_service.set_status(status=Status(template=testing_processing_status))
    predicted_values_jsons = engine.predict(environment=AIEnvironment.TESTING, data=data['batch'])
    testing_service.set_status(status=Status(template=testing_waiting_status))

    # Save predicted values and update predictions' state
    predicted_value_db_objs = []

    for prediction, predicted_values_json in zip(pending_predictions, predicted_values_jsons):
        try:
            # Set output values
            for element_values in predicted_values_json['outputs']:
                # Get element
                element: ElementDB = get_preloaded_db_object(id_=element_values['element'],
                                                             preloaded_db_objects=preloaded_elements)
                if element.element_type != ElementType.OUTPUT:
                    continue  # This should never happen

                # Save output value
                predicted_values = element_values['value']
                if not isinstance(predicted_values, list):
                    predicted_values = [predicted_values]
                for idx, predicted_value in enumerate(predicted_values):
                    is_pred_score = (element.element_type == ElementType.OUTPUT and
                                     element.value_type == ElementValueType.CATEGORY)
                    if is_pred_score:
                        output_value_db_model = PredScores
                    else:
                        output_value_db_model = PredictionDB.value_type_models()[element.value_type]
                    predicted_value_db_obj = output_value_db_model(prediction_id=prediction.prediction_id,
                                                                   element_id=element.element_id,
                                                                   index=(idx + 1),
                                                                   value=predicted_value)
                    predicted_value_db_objs.append(predicted_value_db_obj)

            # Update prediction state
            prediction.state = PredictionState.COMPLETE
        except Exception:
            prediction.state = PredictionState.FAILED

    save_to_db(predicted_value_db_objs)
    db_commit()  # Commit updates in predictions' state


class TestingView(_AIView):

    @doc(tags=[SWAGGER_TAG_AI])
    @use_kwargs(TestRequest, location='json')
    def post(self, task_id: str, resources: List[Resource], **kwargs):
        """
        Handles AI model testing requests. It validates checks prediction quota limits, and triggers a Celery task
        to process pending test predictions asynchronously.

        WARNING: quota usage is updated by `save_buffered_prediction_logs()`.

        Steps:
        1. Verify that the testing service is running.
        2. Validate the task's quota for predictions.
        3. Log pending predictions in the buffer.
        4. Trigger a Celery task to process pending predictions.

        Args:
            task_id (str): The task ID for which testing is requested.
            resources (List[Resource]): The list of resources, with the task being the last resource.
            **kwargs: Additional request payload containing testing data.

        Returns:
            Response: HTTP 202 response indicating the request is accepted and processing will continue asynchronously.
        """
        agent = agent_from_token()

        task = resources[-1]
        assert isinstance(task, Task)

        # Verify Testing Service is running
        testing_service = Service.filter_by_task_and_type(task_id=task.db_object().task_id, type_=ServiceType.TESTING)
        if testing_service.status['code'] == TESTING_STOPPED_STATUS_CODE:
            raise UnprocessableRequestError('Testing Service is not running')

        # Check predictions quota limit
        task.check_quota_usage(name='predictions',
                               description='Maximum number of predictions',
                               delta=len(kwargs['batch']))

        # Load the AI model chosen for tests
        testing_ai_model_db_obj = AIModelDB.get(model_id=task.db_object().test_model_id)
        if testing_ai_model_db_obj is None:
            raise ResourceNotFoundError('No testing AI model found')

        testing_ai_model = AIModel.get(agent=agent, db_object_or_id=testing_ai_model_db_obj, parents=[task])

        # Log pending predictions
        _log_predictions(observations=kwargs['batch'],
                         task_id=task.db_object().task_id,
                         client_id=testing_service.client_id,
                         preloaded_elements=preload_task_db_objects(task=task.db_object(), db_model=ElementDB),
                         ai_model_uuid=testing_ai_model.uuid(),
                         flush=True)

        # Make predictions in a Celery task
        trigger_pending_test_predictions.delay(task_id=task.db_object().task_id)

        return Response(status=202)


#######################
# AI Model Management #
#######################


@shared_task
def deploy_ai_model(model_id: str, task_id: int, environment: str, client_id: int, user_id: int = None):
    """
    Deploys the specified AI model in the given environment.

    Steps:
    1. Load the task and ensure it is in an active state.
    2. Load AI model metadata.
    3. Update the parent task's reference to the deployed AI model.
    4. Remove Monitoring Service templates if deploying to production.

    Args:
        model_id (str): The ID of the AI model to deploy.
        task_id (int): The ID of the task to which the model belongs.
        environment (str): The environment in which to deploy the model (production or testing).
        client_id (int): The ID of the client making the request.
        user_id (int, optional): The ID of the user making the request (if applicable).
    """

    environment = AIEnvironment[environment.upper()]
    agent = UserDB.get(user_id=user_id) if user_id else ClientDB.get(client_id=client_id)

    # Load task
    task = Task.get(agent=agent, db_object_or_id=TaskDB.get(task_id=task_id))

    # Verify task status
    assert task.db_object().status['code'] == TASK_ACTIVE_STATUS_CODE

    # Load AI model metadata
    ai_model = AIModel.get(agent=agent, db_object_or_id=model_id, parents=[task])

    # Get engine
    engine = get_engine(engine_type=get_engine_type(), task_uuid=task.uuid())

    # Deploy model
    engine.deploy_model(model_uuid=ai_model.uuid(), environment=environment)

    # Update deployed AI model reference in parent task
    setattr(task.db_object(), _env_model_id_col[environment], ai_model.db_object().model_id)
    task.persist()

    # Remove Monitoring Service templates (only if deploying to production)
    if environment == AIEnvironment.PRODUCTION:
        service = Service.filter_by_task_and_type(task_id=task.db_object().task_id, type_=ServiceType.MONITORING)
        service.data = dict()
        save_to_db(service)


class AIModelsView(_AIModelView):

    @doc(tags=[SWAGGER_TAG_AI])
    @marshal_with(AIModelResponse(many=True))
    def get(self, task_id: str, resources: List[Resource]):
        """
        Retrieves a list of all AI models associated with a given task.

        Steps:
        1. Verify the agent has permission to read the AI models.
        2. Query the database for all AI models associated with the task.
        3. Return a JSON response containing the list of AI models.

        Args:
            task_id (str): The task ID to retrieve AI models for.
            resources (List[Resource]): The list of resources, with the task being the last resource.

        Returns:
            Response: The response with the list of AI models.
        """
        agent = agent_from_token()

        task = resources[-1]
        assert isinstance(task, Task)

        # Check permissions
        if isinstance(agent, UserDB):
            AIModel.check_permissions(organization=task.db_object().organization,
                                      action=ResourceAction.READ,
                                      user=agent)

        # Get database objects
        db_objects = AIModel.db_model().filter_by_task(task.db_object().task_id)

        # Get resources
        resources = [
            AIModel.get(agent=agent, db_object_or_id=x, parents=[task], check_parents=False) for x in db_objects
        ]

        return jsonify(dump(resources))

    @doc(tags=[SWAGGER_TAG_AI])
    @use_kwargs(AIModelRequest, location='json')
    @marshal_with(AIModelResponse)
    def post(self, task_id: str, resources: List[Resource], **kwargs):
        """
        Creates a new AI model associated with a given task.

        Steps:
        1. Validate the request payload and task.
        2. Process the creation of the new AI model.
        3. Return a JSON response with the created AI model details.

        Args:
            task_id (str): The task ID to associate the AI model with.
            resources (List[Resource]): The list of resources, with the task being the last resource.
            **kwargs: Request payload containing the AI model details.

        Returns:
            Response: The response with the created AI model.
        """
        return process_post_or_put_request(agent=resources[-1].agent(),
                                           resource_or_model=AIModel,
                                           parents=resources,
                                           json=kwargs)


class AIModelView(_AIModelView):

    @doc(tags=[SWAGGER_TAG_AI])
    @marshal_with(AIModelResponse)
    def get(self, task_id: str, model_id: str, resources: List[Resource]):
        """
        Retrieves details of a specific AI model associated with a task.

        Steps:
        1. Validate the task and AI model.
        2. Return a JSON response containing the details of the requested AI model.

        Args:
            task_id (str): The task ID associated with the AI model.
            model_id (str): The ID of the AI model to retrieve.
            resources (List[Resource]): The list of resources, with the task being the last resource.

        Returns:
            Response: The response with the AI model details.
        """
        return process_get_request(resource=resources[-1])


class DeploymentView(_AIModelView):
    """
    Handles deployment requests for AI models to either production or testing environments.
    """

    _allowed_envs = [x.name.lower() for x in AIEnvironment]
    _allowed_envs_str = ' | '.join([f'"{x}"' for x in _allowed_envs])
    _url_params = {
        'environment':
            fields.String(required=True,
                          validate=validate.OneOf(_allowed_envs),
                          description=f'Environment to deploy the AI model to: '
                          f'{_allowed_envs_str}')
    }

    @doc(tags=[SWAGGER_TAG_AI])
    @use_kwargs(_url_params, location='query')
    @marshal_with(DeploymentResponse)
    def get(self, task_id: str, resources: List[Resource], **kwargs):
        """
        Retrieves deployment details for an AI model in a specified environment (production or testing).

        Steps:
        1. Load the AI model metadata for the specified environment.
        2. Build and return a response with the AI model's deployment details.

        Args:
            task_id (str): The task ID associated with the AI model.
            resources (List[Resource]): The list of resources, with the task being the last resource.
            **kwargs: Query parameters including the environment.

        Returns:
            Response: The response with the deployment details of the AI model.
        """
        task = resources[-1]
        environment = AIEnvironment[kwargs['environment'].upper()]

        # Load AI model metadata
        ai_model_db_obj = AIModelDB.get(model_id=getattr(task.db_object(), _env_model_id_col[environment]))
        if ai_model_db_obj is None:
            raise ResourceNotFoundError('No deployed AI model found')

        ai_model = AIModel.get(agent=agent_from_token(), db_object_or_id=ai_model_db_obj, parents=[task])

        # Build response
        response_data = {'environment': environment, **ai_model.dump(serialize=False)}
        response_json = DeploymentResponse().dump(response_data)

        response = jsonify(response_json)
        response.headers['Location'] = ai_model.url()

        return response

    @doc(tags=[SWAGGER_TAG_AI])
    @use_kwargs(DeploymentRequest, location='json')
    def post(self, task_id: str, resources: List[Resource], **kwargs):
        """
        Deploys an AI model to the specified environment (production or testing).

        Steps:
        1. Validate the current task and continual learning service status.
        2. Check if the task is active and allows deployment.
        3. Trigger a Celery task to deploy the AI model to the requested environment.

        Args:
            task_id (str): The task ID associated with the AI model.
            resources (List[Resource]): The list of resources, with the task being the last resource.
            **kwargs: Request payload containing the environment and AI model details.

        Returns:
            Response: HTTP 202 response indicating the deployment request is accepted.
        """
        task: Task = resources[-1]
        task_db_obj: TaskDB = task.db_object()

        # Check if the current task and CL status allows for a new deployment
        cl_service: Service = Service.filter_by_task_and_type(task_id=task_db_obj.task_id,
                                                              type_=ServiceType.CONTINUAL_LEARNING)
        if cl_service:
            cl_service_status_code: str = cl_service.status['code']
            if cl_service_status_code == CL_DEPLOYING_STATUS_CODE:
                raise UnprocessableRequestError('A previous deployment is still in progress. '
                                                'Please try again in a few minutes')

        task_status_code: str = task.db_object().status['code']
        if task_status_code != TASK_ACTIVE_STATUS_CODE:
            raise UnprocessableRequestError('Deployments are not available in non-active tasks. '
                                            'Check current task status')

        # Get user/client ID
        agent = agent_from_token()

        if isinstance(agent, UserDB):
            client_id = None
            user_id = agent.user_id
        else:
            client_id = agent.client_id
            user_id = None

        # Run Celery task
        deploy_ai_model.delay(model_id=kwargs['ai_model'],
                              task_id=task.db_object().task_id,
                              environment=kwargs['environment'].name.lower(),
                              client_id=client_id,
                              user_id=user_id)

        return Response(status=202)


###################
# Prediction Logs #
###################


def _write_prediction_to_buffers(task: TaskDB, predictions_list: List[PredictionLog]) -> None:
    """
    Writes predictions to both the Active Learning (AL) and Monitoring Service buffers.
    These buffers hold prediction logs until they are processed by the respective services.

    Steps:
    1. Write the predictions to the Active Learning Service buffer.
    2. Write the predictions to the Monitoring Service buffer.
    3. Trigger the Monitoring Service asynchronously to process the buffer.

    Args:
        task (TaskDB): The task for which the predictions were made.
        predictions_list (List[PredictionLog]): List of prediction logs to be written to buffers.
    """
    # Write prediction to Active Learning Service buffer
    al_buffer: ALBuffer = ALBuffer(buffer_io=ALBufferIO(task=task))
    al_buffer.write(items=predictions_list)

    # Write prediction to Monitoring Service buffer
    mon_buffer: MonBuffer = MonBuffer(buffer_io=MonBufferIO(task=task))
    mon_buffer.write(items=predictions_list)

    # Trigger Monitoring Service to process the buffer
    run_mon_service.delay(task_id=task.task_id)


@shared_task
def save_buffered_prediction_logs():
    """Saves buffered prediction logs from Redis to the database. The logs are grouped by task and client,
    and each group's logs are saved at once.

    The buffered data must be a JSON with the following fields:
        - `prediction`: Dictionary (JSON) with the prediction to log.
                        The JSON schema must be the same as `nexusml.schemas.ai.PredictionLoggingRequest`
        - `task_id`: Integer with the task ID (database surrogate key).
        - `client_id`: Integer with the client ID (database surrogate key).

    Steps:
    1. Retrieve all buffered prediction logs from Redis.
    2. Group logs by task and client for efficient database writing.
    3. Save logs to the database for each task and client group.
    4. Write complete prediction logs to Active Learning and Monitoring buffers.

    Returns:
        None
    """
    # Check if the buffer is empty
    if redis_buffer.llen(REDIS_PREDICTION_LOG_BUFFER_KEY) == 0:
        return

    # Retrieve all buffered data and reset the buffer
    buffered_data = redis_buffer.lrange(REDIS_PREDICTION_LOG_BUFFER_KEY, 0, -1)
    redis_buffer.ltrim(REDIS_PREDICTION_LOG_BUFFER_KEY, len(buffered_data), -1)
    all_logs = [json.loads(data.decode()) for data in buffered_data]

    # Group logs by task and client to save all related logs at once
    grouped_logs = dict()

    for log in all_logs:
        # Get group
        group = (log['task_id'], log['client_id'])
        if group not in grouped_logs:
            # Initialize group
            grouped_logs[group] = {'predictions': []}
        # Add prediction to group
        grouped_logs[group]['predictions'].append(log['prediction'])

    # Get the clients of the Inference Service and the Testing Service
    # of each task to determine the environment of predictions.
    inference_services = dict()
    testing_services = dict()

    for task_id in [group[0] for group in grouped_logs]:
        inference_service = Service.filter_by_task_and_type(task_id=task_id, type_=ServiceType.INFERENCE)
        testing_service = Service.filter_by_task_and_type(task_id=task_id, type_=ServiceType.TESTING)
        inference_services[task_id] = inference_service
        testing_services[task_id] = testing_service

    # Save logs to database
    for group, logs in grouped_logs.items():
        # Load task and client from database
        task_db_obj = TaskDB.get(task_id=group[0])
        client = ClientDB.get(client_id=group[1])

        # Get the environment related to each service client
        client_envs = {
            inference_services[task_db_obj.task_id].client_id: AIEnvironment.PRODUCTION,
            testing_services[task_db_obj.task_id].client_id: AIEnvironment.TESTING
        }

        # Get logs' data
        logs = PredictionLoggingRequest().load(logs)

        # Save logs
        task: Task = Task.get(agent=client, db_object_or_id=task_db_obj)
        new_predictions_logs: list[PredictionLog] = PredictionLog.post_batch(data=logs['predictions'],
                                                                             task=task,
                                                                             environment=client_envs[client.client_id])

        # Write complete prediction logs to buffers
        complete_predictions: list = [
            prediction_log for prediction_log in new_predictions_logs
            if prediction_log.db_object().state == PredictionState.COMPLETE
        ]
        _write_prediction_to_buffers(task=task.db_object(), predictions_list=complete_predictions)


class PredictionLogsView(_PredictionLogView):
    """
    View to handle retrieval of prediction logs for a task, with filtering and pagination options.
    """

    _order_by_fields = ['created_at']
    _url_params = {
        **paging_url_params(collection_name='predictions'), 'order_by':
            fields.String(description='Parameter to order by. Default: "created_at"'),
        'order':
            fields.String(description='"asc" (ascending) or "desc" (descending). Default: "desc"'),
        'created_at':
            fields.String(description='Predictions registered at the given datetime'),
        'created_at[min]':
            fields.String(description='Predictions registered after the given datetime (inclusive)'),
        'created_at[max]':
            fields.String(description='Predictions registered before the given datetime (inclusive)'),
        'environment':
            fields.String(description=('"production" for predictions made in production '
                                       'or "testing" for predictions made in testing environment')),
        'ai_model':
            fields.String(description='ID of the AI model which made the predictions')
    }

    @doc(tags=[SWAGGER_TAG_AI],
         description='To represent AND and OR operators within the value of a query parameter, use "," for AND and "|" '
         'for OR. For example: '
         '\n\n- `<field>=<value_1>,<value_2>` (AND)'
         '\n- `<field>=<value_1>|<value_2>` (OR)'
         '\n\nFor datetimes, use ISO 8601 format (YYYY-MM-DDTHH:MM:SS), e.g.: '
         '`created_at=2021-04-28T16:24:03`'
         '\n\nIn addition to the predefined query parameters, input/output/metadata elements can also be '
         'filtered following the format `<element_name>=<value>`.')
    @use_kwargs(_url_params, location='query')
    @marshal_with(PredictionLogPage)
    def get(self, task_id: str, resources: List[Resource], **kwargs):
        """
        Retrieves prediction logs for a given task with optional filters such as date range, environment,
        and AI model. The response is paginated and ordered based on the provided query parameters.

        Steps:
        1. Validate user permissions for accessing the prediction logs.
        2. Apply filters and retrieve logs based on the query parameters.
        3. Return a paginated response containing the filtered prediction logs.

        Args:
            task_id (str): The task ID to retrieve prediction logs for.
            resources (List[Resource]): The list of resources, with the task being the last resource.
            **kwargs: Query parameters for filtering the prediction logs.

        Returns:
            Response: The response with filtered and paginated prediction logs.
        """
        agent = agent_from_token()

        task = resources[-1]
        assert isinstance(task, Task)

        # Check user permissions (token scopes are already validated)
        if isinstance(agent, UserDB):
            PredictionLog.check_permissions(organization=task.db_object().organization,
                                            action=ResourceAction.READ,
                                            user=agent,
                                            check_parents=False)  # Parent permissions already checked when loading task

        # Set filters
        filters = []
        db_models = []

        if 'environment' in kwargs:
            env = kwargs['environment']
            if ',' in env:
                return error_response(code=HTTP_BAD_REQUEST_STATUS_CODE,
                                      message='Invalid query. A prediction can be made in only one environment')
            elif '|' in env:
                # Note: currently we have only 2 environments. This code enables adding more environments in the future
                filtered_envs = set(x.strip().lower() for x in env.split('|'))
                if any(x not in AIEnvironment for x in filtered_envs):
                    return error_response(code=HTTP_BAD_REQUEST_STATUS_CODE,
                                          message=f'Invalid query. Invalid environment: "{env}"')
                filtered_envs = [AIEnvironment[x.upper()] for x in filtered_envs]
                filters.append(PredictionDB.environment.in_(filtered_envs))
            else:
                filters.append(PredictionDB.environment == AIEnvironment[env.strip().upper()])

        if 'ai_model' in kwargs:
            ai_model = kwargs['ai_model']
            if ',' in ai_model:
                return error_response(code=HTTP_BAD_REQUEST_STATUS_CODE,
                                      message='Invalid query. A prediction can be made by only one AI model')
            elif '|' in ai_model:
                ai_models = [x.strip().lower() for x in ai_model.split('|')]
                filters.append(sql_or(AIModelDB.uuid.in_(ai_models), AIModelDB.public_id.in_(ai_models)))
            else:
                ai_model = ai_model.strip().lower()
                filters.append(sql_or(AIModelDB.uuid == ai_model, AIModelDB.public_id == ai_model))
            db_models.append(AIModelDB)

        # Get predictions
        # TODO: Replace `page_db_objects` with `_` to remove the pylint disable warning.
        # pylint: disable-next=unused-variable
        page_resources, page_db_objects = get_examples_or_predictions(agent=agent,
                                                                      task=task,
                                                                      resource_type=PredictionLog,
                                                                      predefined_query_params=kwargs,
                                                                      supported_order_by=self._order_by_fields,
                                                                      extra_filters=(filters, db_models))

        # Build response
        return jsonify(page_resources)


class PredictionLogView(_PredictionLogView):

    @doc(tags=[SWAGGER_TAG_AI], description='Note: Logs of predictions made in production cannot be deleted.')
    def delete(self, task_id: str, prediction_id: str, resources: List[Resource]):
        """
        Deletes a specific prediction log for a task, but only if the log was made in the testing environment.
        Logs of predictions made in production cannot be deleted.

        Steps:
        1. Verify that the prediction log was made in the testing environment.
        2. Delete the prediction log and return a success response.

        Args:
            task_id (str): The task ID associated with the prediction log.
            prediction_id (str): The ID of the prediction log to delete.
            resources (List[Resource]): The list of resources, with the task being the last resource.

        Returns:
            Response: Success response on deletion.
        """
        prediction_log = resources[-1]
        if prediction_log.db_object().environment == AIEnvironment.PRODUCTION:
            raise UnprocessableRequestError('Logs of predictions made in production cannot be deleted')

        return process_delete_request(resource=prediction_log)

    @doc(tags=[SWAGGER_TAG_AI])
    @marshal_with(PredictionLogResponse)
    def get(self, task_id: str, prediction_id: str, resources: List[Resource], **kwargs):
        return process_get_request(resource=resources[-1])


class PredictionLoggingView(_PredictionLogView):

    @doc(tags=[SWAGGER_TAG_AI])
    @use_kwargs(PredictionLoggingRequest, location='json')
    def post(self, task_id: str, resources: List[Resource], **kwargs):
        """
        Logs predictions for a task made by either the Inference Service or Testing Service.
        The predictions are added to a buffer for asynchronous saving and processing.

        Steps:
        1. Verify the request was made by the client (Inference or Testing Service).
        2. Verify the AI model exists for each prediction and add the prediction logs to the buffer.
        3. Log the predictions asynchronously via the buffer.

        Args:
            task_id (str): The task ID associated with the predictions.
            resources (List[Resource]): The list of resources, with the task being the last resource.
            **kwargs: Request payload containing the predictions to log.

        Returns:
            Response: HTTP 202 response indicating the predictions are accepted for logging.
        """
        # Verify the request was made by a client
        agent = agent_from_token()
        if not isinstance(agent, ClientDB):
            raise PermissionDeniedError()

        # Verify the request was made by either the Inference Service or the Testing Service
        task = resources[-1]
        assert isinstance(task, Task)

        inference_service = Service.filter_by_task_and_type(task_id=task.db_object().task_id,
                                                            type_=ServiceType.INFERENCE)

        testing_service = Service.filter_by_task_and_type(task_id=task.db_object().task_id, type_=ServiceType.TESTING)

        if agent.client_id not in (inference_service.client_id, testing_service.client_id):
            raise PermissionDeniedError()

        # Add prediction logs to buffer
        specified_ai_models = set()
        for prediction in kwargs['predictions']:
            # Verify that the specified AI model exists
            ai_model_id = prediction['ai_model']
            if ai_model_id not in specified_ai_models:
                AIModel.get(agent=task.agent(),
                            db_object_or_id=ai_model_id,
                            parents=(task.parents() + [task]),
                            check_permissions=False,
                            check_parents=False)
                specified_ai_models.add(ai_model_id)
            # Add prediction log to buffer
            _add_prediction_log_to_buffer(prediction=prediction,
                                          task_id=task.db_object().task_id,
                                          client_id=agent.client_id)

        return Response(status=202)


##################################
# Helper variables and functions #
##################################

_env_model_id_col = {AIEnvironment.PRODUCTION: 'prod_model_id', AIEnvironment.TESTING: 'test_model_id'}


def _prepare_engine_request_data(original_request_data: dict, task: Task,
                                 preloaded_elements: Dict[str, Dict[str, ElementDB]]) -> dict:
    """
    Prepares request data to be sent to the engine for inference or testing purposes.
    This involves copying the original data and injecting additional metadata, such as file
    information for elements of type `_file`.

    Steps:
    1. Copy the original request data to ensure no modifications are made to the original.
    2. For each observation in the data, inject file information for elements that require it.
    3. Disable service status updates by setting a flag in the request data.

    Args:
        original_request_data (dict): The original request data to be sent to the engine.
        task (Task): The task associated with the request.
        preloaded_elements (Dict[str, Dict[str, ElementDB]]): Preloaded task elements to be used for validation.

    Returns:
        dict: The prepared request data with file information injected where needed.
    """

    # Note: We copy the whole dictionary because we're injecting files' JSONs.
    data = copy.deepcopy(original_request_data)

    # Inject files' info (download URL, filename, etc.)
    for observation in original_request_data['batch']:
        for element_value in observation['values']:
            # Skip non-file-type values
            element = get_preloaded_db_object(id_=element_value['element'], preloaded_db_objects=preloaded_elements)
            if element.value_type not in FILE_TYPES:
                continue
            # Dump file info
            file = File.get(agent=task.agent(), db_object_or_id=element_value['value'], parents=[task])
            element_value['value'] = file.dump()

    return data


def _add_prediction_log_to_buffer(prediction: dict, task_id: int, client_id: int, flush: bool = False):
    """
    Adds a prediction log to the Redis buffer for later processing. If the buffer reaches a
    configured size, it triggers a background task to save the buffered logs.

    Steps:
    1. Prepare the prediction log data for buffering, including task and client information.
    2. Add the prediction log data to the Redis buffer.
    3. If the buffer size exceeds the configured limit, trigger the saving task.

    Args:
        prediction (dict): The prediction log data to add to the buffer.
        task_id (int): The ID of the task associated with the prediction.
        client_id (int): The ID of the client making the request.
        flush (bool, optional): Whether to flush the buffer immediately by triggering the save task.
    """
    # Prepare data to pass to the Celery task
    buffer_data = json.dumps({
        'prediction': {
            'ai_model': prediction.pop('ai_model'),
            **PredictionLogSchema().dump(prediction)
        },
        'task_id': task_id,
        'client_id': client_id,
    })
    # Add predictions to buffer
    redis_buffer.lpush(REDIS_PREDICTION_LOG_BUFFER_KEY, buffer_data)
    if flush or redis_buffer.llen(REDIS_PREDICTION_LOG_BUFFER_KEY) >= config.get('jobs')['log_buffer_size']:
        save_buffered_prediction_logs.delay()


def _log_predictions(observations: List[dict],
                     task_id: int,
                     client_id: int,
                     preloaded_elements: Dict[str, Dict[str, ElementDB]],
                     ai_model_uuid: str,
                     predictions: List[dict] = None,
                     flush: bool = False):
    """
    Logs a batch of predictions for a task, separating input values, metadata, and targets.
    The logs are added to the Redis buffer for later processing.

    Steps:
    1. Ensure predictions are provided, or initialize an empty list.
    2. For each observation, separate input values from metadata.
    3. Add the prediction log to the Redis buffer for each observation.

    Args:
        observations (List[dict]): The list of observations to log.
        task_id (int): The ID of the task associated with the predictions.
        client_id (int): The ID of the client making the request.
        preloaded_elements (Dict[str, Dict[str, ElementDB]]): Preloaded task elements for validation.
        ai_model_uuid (str): The UUID of the AI model used for predictions.
        predictions (List[dict], optional): The list of prediction outputs. Defaults to None.
        flush (bool, optional): Whether to flush the buffer immediately by triggering the save task.

    Returns:
        None
    """
    # Verify existing predictions or init predictions
    if predictions:
        assert len(predictions) == len(observations)
    else:
        predictions = [None] * len(observations)

    for observation, prediction in zip(observations, predictions):
        # Separate input values and metadata values
        inputs = []
        metadata = []

        for element_value in observation['values']:
            element = get_preloaded_db_object(id_=element_value['element'], preloaded_db_objects=preloaded_elements)
            if element.element_type == ElementType.INPUT:
                inputs.append(element_value)
            else:
                metadata.append(element_value)

        # Get outputs and state
        if prediction is None:
            outputs = None
            state = PredictionState.PENDING
        else:
            outputs = prediction['outputs']
            state = PredictionState.COMPLETE

        # Set prediction log
        prediction_log = {
            'ai_model': ai_model_uuid,
            'state': state,
            'inputs': inputs,
            'outputs': outputs,
            'metadata': metadata,
            'targets': observation.get('targets')
        }

        # Add prediction log to buffer
        _add_prediction_log_to_buffer(prediction=prediction_log, task_id=task_id, client_id=client_id, flush=flush)
