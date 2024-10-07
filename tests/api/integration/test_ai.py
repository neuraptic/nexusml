from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
import copy
from datetime import timedelta
import time
from typing import Dict, List, Tuple

import pytest
from sqlalchemy.orm import make_transient

from nexusml.api.endpoints import ENDPOINT_AI_DEPLOYMENT
from nexusml.api.endpoints import ENDPOINT_AI_INFERENCE
from nexusml.api.endpoints import ENDPOINT_AI_MODEL
from nexusml.api.endpoints import ENDPOINT_AI_MODELS
from nexusml.api.endpoints import ENDPOINT_AI_PREDICTION_LOG
from nexusml.api.endpoints import ENDPOINT_AI_PREDICTION_LOGS
from nexusml.api.endpoints import ENDPOINT_AI_TESTING
from nexusml.api.endpoints import ENDPOINT_AI_TRAINING
from nexusml.api.external.ext import cache
from nexusml.api.external.ext import redis_buffer
from nexusml.api.resources.ai import AIModel
from nexusml.api.resources.ai import PredictionLog
from nexusml.api.resources.tasks import Task
from nexusml.api.schemas.ai import PredictionLogResponse
from nexusml.api.views.ai import save_buffered_prediction_logs
from nexusml.constants import DATETIME_FORMAT
from nexusml.constants import HTTP_BAD_REQUEST_STATUS_CODE
from nexusml.constants import HTTP_GET_STATUS_CODE
from nexusml.constants import HTTP_METHOD_NOT_ALLOWED_STATUS_CODE
from nexusml.constants import HTTP_POST_STATUS_CODE
from nexusml.constants import HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
from nexusml.constants import REDIS_PREDICTION_LOG_BUFFER_KEY
from nexusml.database.ai import AIModelDB
from nexusml.database.ai import PredictionDB
from nexusml.database.ai import PredScores
from nexusml.database.core import delete_from_db
from nexusml.database.core import empty_table
from nexusml.database.core import save_to_db
from nexusml.database.files import TaskFileDB as FileDB
from nexusml.database.services import Service
from nexusml.database.subscriptions import get_active_subscription
from nexusml.database.subscriptions import quotas
from nexusml.database.tasks import CategoryDB
from nexusml.database.tasks import ElementDB
from nexusml.enums import AIEnvironment
from nexusml.enums import ElementType
from nexusml.enums import ElementValueType
from nexusml.enums import FileStorageBackend
from nexusml.enums import ServiceType
from nexusml.enums import TaskFileUse
from nexusml.statuses import cl_stopped_status
from nexusml.statuses import cl_training_status
from nexusml.statuses import cl_waiting_status
from nexusml.statuses import inference_stopped_status
from nexusml.statuses import inference_waiting_status
from nexusml.statuses import Status
from nexusml.statuses import testing_stopped_status
from nexusml.statuses import testing_waiting_status
from tests.api.conftest import restore_db
from tests.api.constants import CLIENT_MAX_THREADS
from tests.api.integration.conftest import Backend
from tests.api.integration.conftest import MockClient
from tests.api.integration.utils import get_endpoint
from tests.api.integration.utils import mock_element_values_json
from tests.api.integration.utils import mock_prediction_log_batch_request
from tests.api.integration.utils import mock_prediction_log_json
from tests.api.integration.utils import verify_example_or_prediction_deletion
from tests.api.integration.utils import verify_resource_request
from tests.api.integration.utils import verify_response_examples_or_prediction_logs
from tests.api.integration.utils import verify_response_json
from tests.api.integration.utils import verify_response_jsons
from tests.api.integration.utils import verify_wrong_element_values
from tests.api.utils import db_commit_and_expire
from tests.api.utils import get_json_from_db_object
from tests.api.utils import load_default_resource
from tests.api.utils import set_quota_limit
from tests.api.utils import set_quota_usage
from tests.api.utils import verify_quota_usage

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestTraining:

    def test_post(self, client: MockClient, mock_s3):
        # TODO: raising exception "botocore.exceptions.ClientError: An error occurred (AuthFailure) when calling the
        #                          RunInstances operation: AWS was not able to validate the provided access credentials"
        return

        task = load_default_resource(resource_type=Task)
        cl_service = Service.filter_by_task_and_type(task_id=task.db_object().task_id,
                                                     type_=ServiceType.CONTINUAL_LEARNING)
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_AI_TRAINING, resource=task)
        ai_models = AIModelDB.filter_by_task(task_id=task.db_object().task_id)
        subscription_plan = get_active_subscription(organization_id=task.db_object().organization_id).plan
        setattr(subscription_plan, quotas['gpu']['limit'], 120.0)
        setattr(subscription_plan, quotas['cpu']['limit'], 240.0)
        setattr(task.db_object(), quotas['gpu']['limit'], 120.0)
        setattr(task.db_object(), quotas['cpu']['limit'], 240.0)
        save_to_db(subscription_plan)
        task.persist()
        ####################
        # Initial training #
        ####################
        delete_from_db(ai_models)
        set_quota_usage(db_object=task.db_object(), quota='cpu', usage=0)
        set_quota_usage(db_object=task.db_object(), quota='gpu', usage=0)
        # response = client.send_request(method='POST', url=endpoint_url)
        pass  # TODO
        ##############
        # Retraining #
        ##############
        # Set incremental creation datetime in AI models
        for idx, ai_model in enumerate(ai_models):
            make_transient(ai_model)
            ai_model.created_at = ai_model.created_at - timedelta(seconds=10 * (idx + 1))
        # Set the maximum creation datetime for the middle AI model,
        # so we can ensure that the latest AI model is the one being retrained
        ai_models[int(len(ai_models) / 2)].created_at = max(x.created_at for x in ai_models) + timedelta(seconds=10)
        save_to_db(ai_models)
        # Make request
        # response = client.send_request(method='POST', url=endpoint_url)
        pass  # TODO
        #############################################################
        # Try to train when the CL Service is stopped (not allowed) #
        #############################################################
        cl_service.set_status(status=Status(template=cl_stopped_status))
        response = client.send_request(method='POST', url=endpoint_url)
        assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
        assert response.json()['error']['message'] == 'Continual Learning Service is not running'
        ##############################################################################
        # Try to train when the previous training has not finished yet (not allowed) #
        ##############################################################################
        cl_service.set_status(status=Status(template=cl_training_status))
        response = client.send_request(method='POST', url=endpoint_url)
        assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
        assert response.json()['error']['message'] == 'Previous training has not finished yet'
        ###############################################################################
        # Try to train when the task type is unknown (not allowed) #
        ###############################################################################
        pass  # TODO: update the schema of `task` to have an unknown type
        cl_service.set_status(status=Status(template=cl_waiting_status))
        response = client.send_request(method='POST', url=endpoint_url)
        assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
        assert response.json()['error']['message'] == 'Unknown task type. Please, revise task schema'


class TestInference:

    def test_post(self, client: MockClient, mock_s3):
        task = load_default_resource(resource_type=Task)

        inference_service = Service.filter_by_task_and_type(task_id=task.db_object().task_id,
                                                            type_=ServiceType.INFERENCE)

        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_AI_INFERENCE, resource=task)
        request_json = {'batch': mock_inference_request_json(task_id=task.db_object().task_id)}
        ####################
        # Valid prediction #
        ####################
        inference_service.set_status(status=Status(template=inference_waiting_status))
        # TODO: Uncomment the line below
        # response = client.send_request(method='POST', url=endpoint_url, json=request_json)
        pass  # TODO
        ################################################################################
        # Try to make a prediction when the Inference Service is stopped (not allowed) #
        ################################################################################
        inference_service.set_status(status=Status(template=inference_stopped_status))
        response = client.send_request(method='POST', url=endpoint_url, json=request_json)
        assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
        assert response.json()['error']['message'] == 'Inference Service is not running'
        ####################
        # Test quota check #
        ####################
        inference_service.set_status(status=Status(template=inference_waiting_status))

        _test_num_predictions_quota(client=client, task=task, endpoint_url=endpoint_url)


class TestTesting:

    def test_post(self, client: MockClient, mock_s3):
        task = load_default_resource(resource_type=Task)

        testing_service = Service.filter_by_task_and_type(task_id=task.db_object().task_id, type_=ServiceType.TESTING)

        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_AI_TESTING, resource=task)
        #################
        # Valid request #
        #################

        # Enable Testing Service
        testing_service.set_status(status=Status(template=testing_waiting_status))

        # Prepare request data
        pred_data = mock_prediction_log_json(task_id=task.db_object().task_id)

        pred_data['values'] = pred_data.pop('inputs') + pred_data.pop('metadata')
        pred_data.pop('outputs')
        pred_data.pop('ai_model')
        pred_data.pop('state')

        request_json = {'batch': [pred_data]}

        # Make request
        # TODO: Uncomment the snippet below when `nexusml.views.ai.process_test_request()` is mocked
        # response = client.send_request(method='POST', url=endpoint_url, json=request_json)
        # assert response.status_code == 202
        # pass  # TODO
        ##############################################################################
        # Try to make a prediction when the Testing Service is stopped (not allowed) #
        ##############################################################################
        testing_service.set_status(status=Status(template=testing_stopped_status))
        response = client.send_request(method='POST', url=endpoint_url, json=request_json)
        assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
        assert response.json()['error']['message'] == 'Testing Service is not running'
        ####################
        # Test quota check #
        ####################
        testing_service.set_status(status=Status(template=testing_waiting_status))

        _test_num_predictions_quota(client=client, task=task, endpoint_url=endpoint_url)

    def test_process_test_request(self, mock_s3):
        pass  # TODO


class TestAIModels:

    def test_delete(self, client: MockClient, mock_s3):
        _verify_not_allowed(client=client, method='DELETE')

    def test_get(self, client: MockClient, mock_s3):
        # Regular download
        parent_task = load_default_resource(resource_type=Task)
        expected_jsons = []
        for ai_model in AIModelDB.filter_by_task(task_id=parent_task.db_object().task_id):
            expected_jsons.append(_get_ai_model_json(ai_model))
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_AI_MODELS, resource=parent_task)
        response = client.send_request(method='GET', url=endpoint_url)
        pass  # TODO: check task schema and model file URL
        verify_response_jsons(actual_jsons=response.json(),
                              expected_jsons=expected_jsons,
                              optional_fields={'task_schema', 'format', 'download_url', 'upload_url'})

    def test_post(self, client: MockClient, mock_s3):
        parent_task = load_default_resource(resource_type=Task)
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_AI_MODELS, resource=parent_task)

        # Create new files for the new AI models
        new_model_files = [
            FileDB(task_id=1,
                   filename=f'ai_model_file_{idx}',
                   size=200 * idx,
                   use_for=TaskFileUse.AI_MODEL,
                   created_by_user=1) for idx in range(1, 4)
        ]
        save_to_db(new_model_files)
        #################
        # VALID REQUEST #
        #################
        valid_request_json = {
            'file': new_model_files[0].public_id,
            'version': '0.1.0-alpha2',
            'training_time': 3.4,
            'training_device': 'gpu',
            'extra_metadata': {
                'architecture': 'ResNet',
                'optimizer': 'adam',
                'dropout': True,
            }
        }
        response = client.send_request(method='POST', url=endpoint_url, json=valid_request_json)
        assert response.status_code == HTTP_POST_STATUS_CODE
        res_json = response.json()
        db_commit_and_expire()
        expected_ai_model = AIModelDB.get_from_uuid(res_json['uuid'])
        _verify_ai_model_response(response_json=res_json, expected_ai_model=expected_ai_model)


class TestAIModel:

    def test_delete(self, client: MockClient, mock_s3):
        ai_model = load_default_resource(resource_type=AIModel, parents_types=[Task])
        _verify_not_allowed(client=client, method='DELETE', ai_model=ai_model)

    def test_get(self, client: MockClient, mock_s3):
        # Regular download
        parent_task = load_default_resource(resource_type=Task)
        ai_model = load_default_resource(resource_type=AIModel, parents_types=[Task])
        assert ai_model.db_object().task_id == parent_task.db_object().task_id
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_AI_MODEL, resource=ai_model)
        response = client.send_request(method='GET', url=endpoint_url)
        _verify_ai_model_response(response_json=response.json(), expected_ai_model=ai_model.db_object())

    def test_put(self, client: MockClient, mock_s3):
        ai_model = load_default_resource(resource_type=AIModel, parents_types=[Task])
        _verify_not_allowed(client=client, method='PUT', ai_model=ai_model)


class _TestDeployment:
    _env_model_id_col = {
        AIEnvironment.PRODUCTION: 'prod_model_id',
        AIEnvironment.TESTING: 'test_model_id',
    }

    def __init__(self, environment: AIEnvironment):
        self._environment = environment
        self._model_id_col = self._env_model_id_col[environment]

    def test_get(self, client: MockClient):
        task = load_default_resource(resource_type=Task)
        ai_model = AIModelDB.get(model_id=getattr(task.db_object(), self._model_id_col))
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_AI_DEPLOYMENT, resource=task)
        request_url = endpoint_url + '?environment=' + self._environment.name.lower()

        response = client.send_request(method='GET', url=request_url)
        _verify_ai_model_response(response_json=response.json(),
                                  expected_ai_model=ai_model,
                                  additional_values={
                                      'environment': self._environment.name.lower(),
                                  })

    def test_post(self, client: MockClient):
        # Prepare request data
        task = load_default_resource(resource_type=Task)

        old_model = AIModelDB.get(model_id=getattr(task.db_object(), self._model_id_col))
        new_model = AIModelDB.get(model_id=3)
        assert new_model.model_id != old_model.model_id

        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_AI_DEPLOYMENT, resource=task)
        request_json = {'ai_model': new_model.public_id, 'environment': self._environment.name.lower()}

        # Make request and verify response and database
        # TODO: uncomment the snippet below when `nexusml.views.ai.deploy_ai_model()` is mocked
        # response = client.send_request(method='POST', url=endpoint_url, json=request_json)
        # assert response.status_code == 202
        # db_commit_and_expire()
        # assert getattr(task.db_object(), self._model_id_col) == new_model.model_id


class TestProductionDeployment:

    def test_get(self, client: MockClient, mock_s3):
        _tester = _TestDeployment(environment=AIEnvironment.PRODUCTION)
        _tester.test_get(client=client)

    def test_post(self, client: MockClient, mock_s3):
        _tester = _TestDeployment(environment=AIEnvironment.PRODUCTION)
        _tester.test_post(client=client)


class TestTestingDeployment:

    def test_get(self, client: MockClient, mock_s3):
        _tester = _TestDeployment(environment=AIEnvironment.TESTING)
        _tester.test_get(client=client)

    def test_post(self, client: MockClient, mock_s3):
        _tester = _TestDeployment(environment=AIEnvironment.TESTING)
        _tester.test_post(client=client)


class TestPredictionLogs:

    def test_delete(self, client: MockClient):
        parent_task = load_default_resource(resource_type=Task)
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_AI_PREDICTION_LOGS, resource=parent_task)
        response = client.send_request(method='DELETE', url=endpoint_url)
        assert response.status_code == HTTP_METHOD_NOT_ALLOWED_STATUS_CODE

    def test_get(self, client: MockClient):

        def _element_value_filter(element_id: int, predictions: List[PredictionDB], filtered_values: Dict[int, object],
                                  unfiltered_value: object):

            filtered_predictions = [x for x in predictions if x.prediction_id in filtered_values]

            # Set database
            # TODO: how to handle regex?
            element = ElementDB.get(element_id=element_id)
            categories = CategoryDB.query().filter_by(element_id=element.element_id).all()

            element_value_model = PredictionDB.value_type_models()[element.value_type]
            if element.element_type == ElementType.OUTPUT:
                element_value_model = PredScores

            prediction_values = element_value_model.query().filter_by(element_id=element_id).all()

            for prediction_value in prediction_values:
                # If it's a categorical value:
                # - In case of inputs/metadata, convert the provided category name into a primary key.
                # - In case of outputs, create a category-score JSON.
                if element.value_type == ElementValueType.CATEGORY:
                    # Get category to assign
                    if prediction_value.prediction_id in filtered_values:
                        filtered_cats = {
                            x: CategoryDB.get_from_id(id_value=x, parent=element) for x in filtered_values.values()
                        }
                        assigned_cat = filtered_cats[filtered_values[prediction_value.prediction_id]]
                    else:
                        assigned_cat = CategoryDB.get_from_id(id_value=unfiltered_value, parent=element)
                    # Assign category
                    if element.element_type == ElementType.OUTPUT:
                        prediction_value.value = {
                            'category': assigned_cat.name,
                            'scores': {
                                x.name: (1.0 if x.category_id == assigned_cat.category_id else 0.0) for x in categories
                            }
                        }
                    else:
                        prediction_value.value = assigned_cat.category_id
                # Otherwise, use the provided value as is
                else:
                    prediction_value.value = filtered_values.get(prediction_value.prediction_id, unfiltered_value)

            save_to_db(prediction_values)

            # Make request and verify response
            url_query = f'?{element.name}={"|".join({str(x) for x in filtered_values.values()})}'
            response = client.send_request(method='GET', url=endpoint_url + url_query)
            assert response.status_code == HTTP_GET_STATUS_CODE
            verify_response_examples_or_prediction_logs(actual_jsons=response.json()['data'],
                                                        expected_db_objects=filtered_predictions)

        parent_task = load_default_resource(resource_type=Task)
        predictions = PredictionDB.filter_by_task(task_id=parent_task.db_object().task_id)
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_AI_PREDICTION_LOGS, resource=parent_task)
        #######################
        # GET ALL PREDICTIONS #
        #######################
        response = client.send_request(method='GET', url=endpoint_url)
        assert response.status_code == HTTP_GET_STATUS_CODE
        verify_response_examples_or_prediction_logs(actual_jsons=response.json()['data'],
                                                    expected_db_objects=predictions)
        ############################
        # FILTER BY ELEMENT VALUES #
        ############################
        # `input_1=true`
        _element_value_filter(element_id=1,
                              predictions=predictions,
                              filtered_values={
                                  1: True,
                                  3: True,
                                  4: True
                              },
                              unfiltered_value=False)

        # `input_2=1|4|12`
        _element_value_filter(element_id=2,
                              predictions=predictions,
                              filtered_values={
                                  2: 1,
                                  3: 4,
                                  5: 12
                              },
                              unfiltered_value=10)

        # `input_3=3.54`
        _element_value_filter(element_id=3, predictions=predictions, filtered_values={5: 3.54}, unfiltered_value=0.68)

        # `input_4=return_this_prediction|return_this_too`
        _element_value_filter(element_id=4,
                              predictions=predictions,
                              filtered_values={
                                  4: 'return_this_prediction',
                                  8: 'return_this_too'
                              },
                              unfiltered_value='do_not_return_this_prediction')

        # `input_6=Input Category 1|Input Category 3`
        _element_value_filter(element_id=6,
                              predictions=predictions,
                              filtered_values={
                                  1: 'Input Category 2',
                                  2: 'Input Category 3',
                                  5: 'Input Category 3'
                              },
                              unfiltered_value='Input Category 1')

        # `output_3=Output Category 1|Output Category 3`
        _element_value_filter(element_id=15,
                              predictions=predictions,
                              filtered_values={
                                  1: 'Output Category 3',
                                  3: 'Output Category 1',
                                  6: 'Output Category 1'
                              },
                              unfiltered_value='Output Category 2')

        # TODO: test metadata elements
        pass
        ###############################
        # FILTER BY CREATION DATETIME #
        ###############################
        # Set database
        for idx, prediction in enumerate(predictions):
            prediction.created_at += timedelta(days=30 * idx)
        save_to_db(predictions)

        # Try different filters
        prediction_datetimes = [x.created_at.strftime(DATETIME_FORMAT) for x in predictions]

        # Exact datetime
        url_query = 'created_at=' + prediction_datetimes[2]
        response = client.send_request(method='GET', url=endpoint_url + '?' + url_query)
        verify_response_examples_or_prediction_logs(actual_jsons=response.json()['data'],
                                                    expected_db_objects=[predictions[2]])

        # After a datetime
        url_query = 'created_at[min]=' + prediction_datetimes[2]
        response = client.send_request(method='GET', url=endpoint_url + '?' + url_query)
        verify_response_examples_or_prediction_logs(actual_jsons=response.json()['data'],
                                                    expected_db_objects=predictions[2:])

        # Before a datetime
        url_query = 'created_at[max]=' + prediction_datetimes[2]
        response = client.send_request(method='GET', url=endpoint_url + '?' + url_query)
        verify_response_examples_or_prediction_logs(actual_jsons=response.json()['data'],
                                                    expected_db_objects=predictions[:3])

        # Datetime interval
        url_query = 'created_at[min]=' + prediction_datetimes[1] + '&created_at[max]=' + prediction_datetimes[3]
        response = client.send_request(method='GET', url=endpoint_url + '?' + url_query)
        verify_response_examples_or_prediction_logs(actual_jsons=response.json()['data'],
                                                    expected_db_objects=predictions[1:4])
        #########################
        # FILTER BY ENVIRONMENT #
        #########################
        # Set database
        for prediction in predictions:
            prediction.environment = AIEnvironment.PRODUCTION
        predictions[1].environment = AIEnvironment.TESTING
        predictions[3].environment = AIEnvironment.TESTING
        save_to_db(predictions)

        # Filter by environment
        response = client.send_request(method='GET', url=endpoint_url + '?environment=testing')
        verify_response_examples_or_prediction_logs(actual_jsons=response.json()['data'],
                                                    expected_db_objects=[predictions[1], predictions[3]])

        # A prediction can be made only at one environment
        response = client.send_request(method='GET', url=endpoint_url + '?environment=production,testing')
        assert response.status_code == HTTP_BAD_REQUEST_STATUS_CODE
        assert response.json()['error']['message'] == 'Invalid query. A prediction can be made in only one environment'
        ######################
        # FILTER BY AI MODEL #
        ######################
        pass  # TODO
        ##########################
        # APPLY MULTIPLE FILTERS #
        ##########################
        pass  # TODO
        ############
        # ORDERING #
        ############
        # Ascending order
        asc_response = client.send_request(method='GET', url=endpoint_url + '?order=asc')
        assert asc_response.status_code == HTTP_GET_STATUS_CODE
        pass  # TODO: check order

        # Descending order
        desc_response = client.send_request(method='GET', url=endpoint_url + '?order=desc')
        assert desc_response.status_code == HTTP_GET_STATUS_CODE
        pass  # TODO: check order

        # Compare orders
        asc_uuids = [x['uuid'] for x in asc_response.json()['data']]
        desc_uuids = [x['uuid'] for x in desc_response.json()['data']]
        assert list(reversed(asc_uuids)) == desc_uuids
        ##########
        # PAGING #
        ##########
        # TODO: create a function based on `tests.integration.test_files.TestFiles.test_get()`
        #       in `tests.integration.utils`
        pass


class TestPredictionLog:

    @pytest.mark.parametrize('file_storage_backend', [FileStorageBackend.LOCAL, FileStorageBackend.S3])
    def test_delete(self, client: MockClient, mock_s3, file_storage_backend: FileStorageBackend, mock_client_id: str,
                    session_user_id: str, session_user_auth0_id: str):
        # Valid request (testing prediction)
        verify_example_or_prediction_deletion(client=client,
                                              resource_type=PredictionLog,
                                              file_storage_backend=file_storage_backend)

        # Invalid request (prediction made in production)
        restore_db(mock_client_id=mock_client_id,
                   session_user_id=session_user_id,
                   session_user_auth0_id=session_user_auth0_id)

        prediction_log = load_default_resource(resource_type=PredictionLog, parents_types=[Task])
        prediction_log.db_object().environment = AIEnvironment.PRODUCTION

        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_AI_PREDICTION_LOG, resource=prediction_log)

        response = client.send_request(method='DELETE', url=endpoint_url)
        assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
        assert response.json()['error']['message'] == 'Logs of predictions made in production cannot be deleted'

        db_commit_and_expire()
        assert PredictionDB.get(prediction_id=prediction_log.db_object().prediction_id) is not None

    def test_get(self, client: MockClient, mock_s3):
        prediction = load_default_resource(resource_type=PredictionLog, parents_types=[Task])
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_AI_PREDICTION_LOG, resource=prediction)
        response = client.send_request(method='GET', url=endpoint_url)
        assert response.status_code == HTTP_GET_STATUS_CODE
        verify_response_examples_or_prediction_logs(actual_jsons=[response.json()],
                                                    expected_db_objects=[prediction.db_object()])

    def test_put(self, client: MockClient, mock_s3):
        """ Verify PUT is not allowed. """
        prediction = load_default_resource(resource_type=PredictionLog, parents_types=[Task])
        request_json = mock_prediction_log_json(task_id=1)
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_AI_PREDICTION_LOG, resource=prediction)
        response = client.send_request(method='PUT', url=endpoint_url, json=request_json)
        assert response.status_code == HTTP_METHOD_NOT_ALLOWED_STATUS_CODE


class TestPredictionLogging:

    def test_post(self, backend: Backend, client: MockClient, mock_client_id: str, session_user_id: str,
                  session_user_auth0_id: str):
        """
        Since prediction logging is asynchronous (Celery task), we cannot check responses as we do for examples.
        Instead, we test `nexusml.resources.ai.PredictionLog.post_batch()` function.
        """

        def _set_request() -> Tuple[Task, dict, dict]:
            # Restore database
            restore_db(mock_client_id=mock_client_id,
                       session_user_id=session_user_id,
                       session_user_auth0_id=session_user_auth0_id)
            empty_table(PredictionDB)
            # Get task
            task: Task = load_default_resource(resource_type=Task)
            # Mock request data
            single_json = {'predictions': [mock_prediction_log_json(task_id=1)]}
            batch_json = {'predictions': [mock_prediction_log_json(task_id=1)] * 5}

            return task, single_json, batch_json

        def _send_request(task: Task = None, request_json: dict = None) -> List[dict]:
            if task is None or request_json is None:
                task_, single_json, _ = _set_request()
                if task is None:
                    task = task_
                if request_json is None:
                    request_json = single_json

            return mock_prediction_log_batch_request(task=task, request_json=request_json)

        def _verify_quota_limit_not_exceeded(task: Task, quota: str, limit: int):
            db_commit_and_expire()
            subscription = get_active_subscription(organization_id=task.db_object().organization_id)
            assert getattr(task.db_object(), quotas[quota]['usage']) <= limit
            assert getattr(subscription, quotas[quota]['usage']) <= limit

        def _test_space_quota_in_parallel(num_requests: int):
            """ Send prediction logs in parallel to test race conditions. """

            def _request_thread(task: Task, request_json: dict) -> List[dict]:
                # Note: we need to enter app context again because this function is run by another thread
                with backend.app.app_context():
                    cache.clear()  # force `resources.tasks.Task.check_quota_usage()` to use current database value
                    return _send_request(task=task, request_json=request_json)

            task, single_json, _ = _set_request()

            limit = task.db_object().space_usage + len(str(single_json)) * 3

            set_quota_limit(db_object=task.db_object(), quota='space', limit=limit)

            # Send requests in different threads.
            # Note: using multiple workers results in a `ConnectionAbortedError`
            futures = []
            with ThreadPoolExecutor(max_workers=CLIENT_MAX_THREADS) as executor:
                for i in range(num_requests):
                    futures.append(executor.submit(_request_thread, task, single_json))
                wait(futures)
                time.sleep(0.05)  # TODO: we shouldn't need this

            # Get results from threads
            res_jsons = []

            for future in futures:
                prediction_logs = future.result()
                res_jsons.append(prediction_logs)

            # Verify responses
            for res_json in res_jsons:
                schema_errors = PredictionLogResponse(many=True).validate(res_json)
                assert not schema_errors
                assert not any(x.get('invalid_data') for x in res_json)

            # Verify quota
            _verify_quota_limit_not_exceeded(task=task, quota='space', limit=limit)

        #######################
        # Test regular upload #
        #######################
        res_jsons = _send_request()
        db_commit_and_expire()
        expected_predictions = [PredictionDB.get_from_uuid(x['uuid']) for x in res_jsons]
        assert not any(x.get('invalid_data') for x in res_jsons)
        verify_response_examples_or_prediction_logs(actual_jsons=res_jsons, expected_db_objects=expected_predictions)
        #######################################
        # Try to provide wrong element values #
        #######################################
        _, single_json, _ = _set_request()
        verify_wrong_element_values(client=client,
                                    resource_type=PredictionLog,
                                    request_method='POST',
                                    resource_endpoint=ENDPOINT_AI_PREDICTION_LOGS,
                                    mock_example_or_prediction_jsons=single_json)
        ##################################################
        # Try to exceed space quota limit (batch upload) #
        ##################################################
        task, single_json, _ = _set_request()
        previous_usage = task.db_object().space_usage
        set_quota_limit(db_object=task.db_object(), quota='space', limit=previous_usage)
        cache.clear()  # force `resources.tasks.Task.check_quota_usage()` to use current database value
        _send_request(task=task, request_json=single_json)
        _verify_quota_limit_not_exceeded(task=task, quota='space', limit=previous_usage)
        pass  # TODO: Verify FIFO
        #########################################################
        # Try to exceed space quota limit (concurrent requests) #
        #########################################################
        _test_space_quota_in_parallel(num_requests=5)
        pass  # TODO: Verify FIFO

    def test_add_prediction_logs_to_buffer(self):
        pass  # TODO: How to mock Redis?

    def test_save_buffered_prediction_logs(self):
        return  # TODO: How to mock Redis?

        # Prepare data to pass to the Celery task
        prediction_log_1 = {
            'prediction': {
                'field_11': 'value_11',
                'field_12': 2.14,
                'field_13': False,
            },
            'task_id': 1,
            'client_id': 1  # TODO: replace with a real client
        }

        prediction_log_2 = {
            'prediction': {
                'field_21': 'value_21',
                'field_22': 6.6,
                'field_23': True
            },
            'task_id': 1,
            'client_id': 1  # TODO: replace with a real client
        }

        prediction_log_3 = {
            'prediction': {
                'field_31': 'value_31',
                'field_32': 0.59,
                'field_33': False
            },
            'task_id': 3,
            'client_id': 1  # TODO: replace with a real client
        }

        # Add prediction logs to buffer
        redis_buffer.lpush(REDIS_PREDICTION_LOG_BUFFER_KEY, prediction_log_1)
        redis_buffer.lpush(REDIS_PREDICTION_LOG_BUFFER_KEY, prediction_log_2)
        redis_buffer.lpush(REDIS_PREDICTION_LOG_BUFFER_KEY, prediction_log_3)

        # Process buffer
        save_buffered_prediction_logs()

        # Check buffer and database
        pass  # TODO

        # Test buffer limit
        pass  # TODO


def mock_inference_request_json(task_id: int, num_observations: int = 4) -> List[dict]:
    mock_element_values = mock_element_values_json(task_id=task_id, include_outputs=False, include_metadata=False)
    batch = []
    for idx in range(1, num_observations + 1):
        observation = copy.deepcopy(mock_element_values)
        for element_value in observation['values']:
            if isinstance(element_value['value'], (int, float)):
                element_value['value'] += idx
            elif isinstance(element_value['value'], str):
                element_value['value'] += (f'_{idx}')
            elif isinstance(element_value['value'], bool):
                element_value['value'] = idx % 2 == 0
        batch.append(observation)
    return batch


def _verify_ai_model_response(response_json: dict, expected_ai_model: AIModelDB, additional_values: dict = None):
    expected_json = _get_ai_model_json(ai_model=expected_ai_model)
    expected_json.update(additional_values or dict())
    pass  # TODO: check task schema and model file URL
    verify_response_json(actual_json=response_json,
                         expected_json=expected_json,
                         optional_fields={'task_schema', 'format', 'download_url', 'upload_url'})


def _verify_not_allowed(client: MockClient, method: str, ai_model: AIModel = None):
    if ai_model is None:
        parent_task = load_default_resource(resource_type=Task)
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_AI_MODELS, resource=parent_task)
        response = client.send_request(method=method, url=endpoint_url)
        assert response.status_code == HTTP_METHOD_NOT_ALLOWED_STATUS_CODE
    else:
        if method.lower() == 'put':
            request_json = get_json_from_db_object(db_object=ai_model.db_object())
        else:
            request_json = None
        verify_resource_request(client=client,
                                method=method,
                                endpoint=ENDPOINT_AI_MODEL,
                                resource=ai_model,
                                request_json=request_json,
                                expected_status_code=HTTP_METHOD_NOT_ALLOWED_STATUS_CODE)


def _get_ai_model_json(ai_model: AIModelDB) -> dict:
    ai_model_json = get_json_from_db_object(db_object=ai_model)

    file = get_json_from_db_object(db_object=FileDB.get(file_id=ai_model.file_id))

    ai_model_json['file'] = file
    ai_model_json['version'] = ai_model_json.pop('version')

    ai_model_json.pop('training_time')
    ai_model_json.pop('training_device')

    return ai_model_json


def _test_num_predictions_quota(client: MockClient, task: Task, endpoint_url: str):
    LIMIT = 3

    err_msg = f'Maximum number of predictions ({LIMIT}) exceeded'

    single_json = {'batch': [mock_element_values_json(task_id=1)]}
    batch_json = {'batch': [mock_element_values_json(task_id=1)] * 5}
    ##################################################################
    # Try to exceed the maximum number of predictions (batch upload) #
    ##################################################################
    empty_table(PredictionDB)

    set_quota_usage(db_object=task.db_object(), quota='predictions', usage=0)
    set_quota_limit(db_object=task.db_object(), quota='predictions', limit=LIMIT)

    cache.clear()  # force `resources.tasks.Task.check_quota_usage()` to use current database value

    response = client.send_request(method='POST', url=endpoint_url, json=batch_json)
    assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
    assert response.json()['error']['message'] == err_msg

    verify_quota_usage(db_object=task.db_object(), quota='predictions', expected_usage=0)
    #########################################################################
    # Try to exceed the maximum number of predictions (concurrent requests) #
    #########################################################################
    # TODO: Uncomment the snippet below
    # verify_quota_error(client=client,
    #                    endpoint_url=endpoint_url,
    #                    request_json=single_json,
    #                    num_requests=(LIMIT + 2),
    #                    err_msg=err_msg)
    # verify_quota_usage(db_object=task.db_object(), quota='predictions', expected_usage=LIMIT)
