import copy
from datetime import datetime
from datetime import timedelta
import uuid

import pytest
from sqlalchemy import and_ as sql_and

from nexusml.api.ext import cache
from nexusml.api.resources.tasks import Task
from nexusml.api.views import services
from nexusml.constants import DATETIME_FORMAT
from nexusml.constants import ENDPOINT_AL_SERVICE
from nexusml.constants import ENDPOINT_AL_SERVICE_NOTIFICATIONS
from nexusml.constants import ENDPOINT_AL_SERVICE_STATUS
from nexusml.constants import ENDPOINT_CL_SERVICE
from nexusml.constants import ENDPOINT_CL_SERVICE_NOTIFICATIONS
from nexusml.constants import ENDPOINT_CL_SERVICE_STATUS
from nexusml.constants import ENDPOINT_INFERENCE_SERVICE
from nexusml.constants import ENDPOINT_INFERENCE_SERVICE_NOTIFICATIONS
from nexusml.constants import ENDPOINT_INFERENCE_SERVICE_STATUS
from nexusml.constants import ENDPOINT_MONITORING_SERVICE
from nexusml.constants import ENDPOINT_MONITORING_SERVICE_NOTIFICATIONS
from nexusml.constants import ENDPOINT_MONITORING_SERVICE_STATUS
from nexusml.constants import ENDPOINT_MONITORING_SERVICE_TEMPLATES
from nexusml.constants import ENDPOINT_SERVICES
from nexusml.constants import HTTP_BAD_REQUEST_STATUS_CODE
from nexusml.constants import HTTP_GET_STATUS_CODE
from nexusml.constants import HTTP_POST_STATUS_CODE
from nexusml.constants import HTTP_PUT_STATUS_CODE
from nexusml.constants import HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
from nexusml.database.ai import AIModelDB
from nexusml.database.core import save_to_db
from nexusml.database.examples import ExampleDB
from nexusml.database.services import Service
from nexusml.database.subscriptions import get_active_subscription
from nexusml.enums import ElementValueType
from nexusml.enums import LabelingStatus
from nexusml.enums import ServiceType
from nexusml.statuses import al_running_state
from nexusml.statuses import al_stopped_status
from nexusml.statuses import al_waiting_status
from nexusml.statuses import cl_running_state
from nexusml.statuses import cl_training_status
from nexusml.statuses import cl_waiting_status
from nexusml.statuses import inference_processing_status
from nexusml.statuses import inference_running_state
from nexusml.statuses import inference_stopped_status
from nexusml.statuses import monitoring_running_state
from nexusml.statuses import monitoring_stopped_status
from nexusml.statuses import monitoring_waiting_status
from nexusml.statuses import Status
from tests.api.integration.conftest import MockClient
from tests.api.integration.utils import get_endpoint
from tests.api.integration.utils import verify_response_json
from tests.api.utils import db_commit_and_expire
from tests.api.utils import load_default_resource

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_ORG_ID = 2


class TestServices:

    def test_get(self, client: MockClient):
        task = load_default_resource(resource_type=Task)
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_SERVICES, resource=task)
        response = client.send_request(method='GET', url=endpoint_url)
        assert response.status_code == HTTP_GET_STATUS_CODE
        res_json = response.json()
        _verify_service(task=task,
                        service_type=ServiceType.INFERENCE,
                        actual_service=res_json['inference'],
                        expected_service=services._inference_service)
        _verify_service(task=task,
                        service_type=ServiceType.CONTINUAL_LEARNING,
                        actual_service=res_json['continual_learning'],
                        expected_service=services._cl_service)
        _verify_service(task=task,
                        service_type=ServiceType.ACTIVE_LEARNING,
                        actual_service=res_json['active_learning'],
                        expected_service=services._al_service)
        _verify_service(task=task,
                        service_type=ServiceType.MONITORING,
                        actual_service=res_json['monitoring'],
                        expected_service=services._monitoring_service)


def _test_get_service(client: MockClient, service_type: ServiceType, parameterized_endpoint: str,
                      expected_service: dict):

    task = load_default_resource(resource_type=Task)
    endpoint_url = get_endpoint(parameterized_endpoint=parameterized_endpoint, resource=task)
    response = client.send_request(method='GET', url=endpoint_url)
    assert response.status_code == HTTP_GET_STATUS_CODE
    _verify_service(task=task,
                    service_type=service_type,
                    actual_service=response.json(),
                    expected_service=expected_service)


class TestInferenceService:

    def test_get(self, client: MockClient):
        _test_get_service(client=client,
                          service_type=ServiceType.INFERENCE,
                          parameterized_endpoint=ENDPOINT_INFERENCE_SERVICE,
                          expected_service=services._inference_service)


class TestCLService:

    def test_get(self, client: MockClient):
        _test_get_service(client=client,
                          service_type=ServiceType.CONTINUAL_LEARNING,
                          parameterized_endpoint=ENDPOINT_CL_SERVICE,
                          expected_service=services._cl_service)


class TestALService:

    def test_get(self, client: MockClient):
        _test_get_service(client=client,
                          service_type=ServiceType.ACTIVE_LEARNING,
                          parameterized_endpoint=ENDPOINT_AL_SERVICE,
                          expected_service=services._al_service)


class TestMonitoringService:

    def test_get(self, client: MockClient):
        _test_get_service(client=client,
                          service_type=ServiceType.MONITORING,
                          parameterized_endpoint=ENDPOINT_MONITORING_SERVICE,
                          expected_service=services._monitoring_service)


def _test_put_service_status(client: MockClient, service_type: ServiceType):
    task = load_default_resource(resource_type=Task)
    service = Service.filter_by_task_and_type(task_id=task.db_object().task_id, type_=service_type)

    # Mock service client
    client.update_client_id(client_id=service.client.uuid, token_type='api_key')

    # Set endpoint
    endpoints = {
        ServiceType.INFERENCE: ENDPOINT_INFERENCE_SERVICE_STATUS,
        ServiceType.CONTINUAL_LEARNING: ENDPOINT_CL_SERVICE_STATUS,
        ServiceType.ACTIVE_LEARNING: ENDPOINT_AL_SERVICE_STATUS,
        ServiceType.MONITORING: ENDPOINT_MONITORING_SERVICE_STATUS,
    }
    endpoint_url = get_endpoint(parameterized_endpoint=endpoints[service_type], resource=task)

    # Define statuses
    init_statuses = {
        ServiceType.INFERENCE: inference_stopped_status,
        ServiceType.CONTINUAL_LEARNING: cl_waiting_status,
        ServiceType.ACTIVE_LEARNING: al_stopped_status,
        ServiceType.MONITORING: monitoring_stopped_status
    }
    new_states = {
        ServiceType.INFERENCE: inference_running_state,
        ServiceType.CONTINUAL_LEARNING: cl_running_state,
        ServiceType.ACTIVE_LEARNING: al_running_state,
        ServiceType.MONITORING: monitoring_running_state
    }
    new_statuses = {
        ServiceType.INFERENCE: inference_processing_status,
        ServiceType.CONTINUAL_LEARNING: cl_training_status,
        ServiceType.ACTIVE_LEARNING: al_waiting_status,
        ServiceType.MONITORING: monitoring_waiting_status
    }

    # Valid request
    new_status = {'code': new_statuses[service_type].code}
    response = client.send_request(method='PUT', url=endpoint_url, json=new_status)
    assert response.status_code == HTTP_PUT_STATUS_CODE
    res_json = response.json()
    assert res_json['state_code'] == new_states[service_type].code
    assert res_json['status_code'] == new_statuses[service_type].code
    assert res_json['name'] == new_statuses[service_type].name
    assert res_json['display_name'] == new_statuses[service_type].display_name
    assert res_json['description'] == new_statuses[service_type].description
    assert isinstance(datetime.strptime(res_json['started_at'], DATETIME_FORMAT), datetime)
    assert res_json['updated_at'] is None
    assert res_json['ended_at'] is None
    assert res_json['details'] is None
    assert res_json['prev_status'] == init_statuses[service_type].code
    db_commit_and_expire()
    assert service.status['code'] == new_statuses[service_type].code
    assert isinstance(datetime.strptime(service.status['started_at'], DATETIME_FORMAT), datetime)
    assert service.status['updated_at'] is None
    assert service.status['ended_at'] is None
    assert service.status['details'] is None
    assert service.status['prev_status'] == init_statuses[service_type].code

    # Invalid status: non-existing
    new_status = {'code': 'invalid_status'}
    response = client.send_request(method='PUT', url=endpoint_url, json=new_status)
    assert response.status_code == HTTP_BAD_REQUEST_STATUS_CODE
    assert response.json()['error']['message'] == 'Status not found: "invalid_status"'

    # Invalid status: invalid group/prefix
    new_status = {'code': '00000'}
    response = client.send_request(method='PUT', url=endpoint_url, json=new_status)
    assert response.status_code == HTTP_BAD_REQUEST_STATUS_CODE
    assert response.json()['error']['message'] == 'Invalid status code'

    # Try to update service status in the Free Plan
    service.set_status(status=Status(template=init_statuses[service_type]))

    subscription = get_active_subscription(organization_id=task.db_object().organization_id)
    prev_plan_id = subscription.plan_id
    subscription.plan_id = 1
    save_to_db(subscription)

    cache.clear()

    new_status = {'code': new_statuses[service_type].code}

    response = client.send_request(method='PUT', url=endpoint_url, json=new_status)
    assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
    assert response.json()['error']['message'] == 'Services are disabled in the Free Plan'
    db_commit_and_expire()
    assert service.status['code'] == init_statuses[service_type].code

    subscription.plan_id = prev_plan_id
    save_to_db(subscription)

    cache.clear()


class TestInferenceServiceStatus:

    def test_put(self, client: MockClient):
        _test_put_service_status(client=client, service_type=ServiceType.INFERENCE)


class TestCLServiceStatus:

    def test_put(self, client: MockClient):
        _test_put_service_status(client=client, service_type=ServiceType.CONTINUAL_LEARNING)
        ###########################
        # Test status transitions #
        ###########################
        task = load_default_resource(resource_type=Task)
        service = Service.filter_by_task_and_type(task_id=task.db_object().task_id,
                                                  type_=ServiceType.CONTINUAL_LEARNING)
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_CL_SERVICE_STATUS, resource=task)

        # Mock service client
        client.update_client_id(client_id=service.client.uuid, token_type='api_key')

        # Start training session
        past_dt = datetime.utcnow() - timedelta(seconds=30)
        ExampleDB.query().filter_by(task_id=task.db_object().task_id).update({'trained': False, 'activity_at': past_dt})
        db_commit_and_expire()
        assert all(not x.trained for x in ExampleDB.query().filter_by(task_id=task.db_object().task_id).all())
        service.set_status(status=Status(template=cl_waiting_status))
        new_status = {'code': cl_training_status.code}
        response = client.send_request(method='PUT', url=endpoint_url, json=new_status)
        assert response.status_code == HTTP_PUT_STATUS_CODE
        db_commit_and_expire()
        assert all(not x.trained for x in ExampleDB.query().filter_by(task_id=task.db_object().task_id).all())

        # Update training progress
        new_status = {'code': cl_training_status.code, 'details': {'progress': 25}}
        response = client.send_request(method='PUT', url=endpoint_url, json=new_status)
        assert response.status_code == HTTP_PUT_STATUS_CODE
        db_commit_and_expire()
        assert all(not x.trained for x in ExampleDB.query().filter_by(task_id=task.db_object().task_id).all())

        # Finish training session
        new_status = {'code': cl_waiting_status.code}
        response = client.send_request(method='PUT', url=endpoint_url, json=new_status)
        assert response.status_code == HTTP_PUT_STATUS_CODE
        db_commit_and_expire()
        trained_examples = (ExampleDB.query().filter_by(task_id=task.db_object().task_id,
                                                        labeling_status=LabelingStatus.LABELED).all())
        untrained_examples = (ExampleDB.query().filter(
            sql_and(ExampleDB.task_id == task.db_object().task_id, ExampleDB.labeling_status
                    != LabelingStatus.LABELED)).all())
        assert all(x.trained for x in trained_examples)
        assert all(not x.trained for x in untrained_examples)


class TestALServiceStatus:

    def test_put(self, client: MockClient):
        _test_put_service_status(client=client, service_type=ServiceType.ACTIVE_LEARNING)


class TestMonitoringServiceStatus:

    def test_put(self, client: MockClient):
        _test_put_service_status(client=client, service_type=ServiceType.MONITORING)


def _test_service_notification(client: MockClient, service_type: ServiceType):
    task = load_default_resource(resource_type=Task)
    service = Service.filter_by_task_and_type(task_id=task.db_object().task_id, type_=service_type)

    # Mock service client
    client.update_client_id(client_id=service.client.uuid, token_type='api_key')

    # Set endpoint
    endpoints = {
        ServiceType.INFERENCE: ENDPOINT_INFERENCE_SERVICE_NOTIFICATIONS,
        ServiceType.CONTINUAL_LEARNING: ENDPOINT_CL_SERVICE_NOTIFICATIONS,
        ServiceType.ACTIVE_LEARNING: ENDPOINT_AL_SERVICE_NOTIFICATIONS,
        ServiceType.MONITORING: ENDPOINT_MONITORING_SERVICE_NOTIFICATIONS,
    }
    endpoint_url = get_endpoint(parameterized_endpoint=endpoints[service_type], resource=task)

    # Make request
    notification = {'message': f'Notification from "{service_type}"'}
    response = client.send_request(method='POST', url=endpoint_url, json=notification)
    assert response.status_code == HTTP_POST_STATUS_CODE
    assert response.json() == notification


class TestInferenceServiceNotifications:

    def test_post(self, client: MockClient):
        pass
        # _test_service_notification(client=client, service_type=ServiceType.inference)


class TestCLServiceNotifications:

    def test_post(self, client: MockClient):
        pass
        #  _test_service_notification(client=client, service_type=ServiceType.continual_learning)


class TestALServiceNotifications:

    def test_post(self, client: MockClient):
        pass
        #  _test_service_notification(client=client, service_type=ServiceType.active_learning)


class TestMonitoringServiceNotifications:

    def test_post(self, client: MockClient):
        pass
        #  _test_service_notification(client=client, service_type=ServiceType.monitoring)


class TestMonitoringServiceTemplates:

    @staticmethod
    def _mock_outputs_template(task: Task) -> dict:
        _mock_cat_uuids = [str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())]
        numerical_output = [x for x in task.db_object().output_elements() if x.value_type == ElementValueType.FLOAT][0]
        categorical_output = [
            x for x in task.db_object().output_elements() if x.value_type == ElementValueType.CATEGORY
        ][0]
        return {
            'ai_model':
                str(AIModelDB.filter_by_task(task_id=task.db_object().task_id)[0].uuid),
            'outputs': [{
                'element': str(numerical_output.uuid),
                'template': {
                    'mean': 0.469,
                    'std': 0.230
                }
            }, {
                'element':
                    str(categorical_output.uuid),
                'template': [{
                    'category':
                        _mock_cat_uuids[0],
                    'template': [
                        {
                            'category': _mock_cat_uuids[0],
                            'mean': 0.333333333333333
                        },
                        {
                            'category': _mock_cat_uuids[1],
                            'mean': 0.333333333333333
                        },
                        {
                            'category': _mock_cat_uuids[2],
                            'mean': 0.333333333333333
                        },
                    ]
                }, {
                    'category':
                        _mock_cat_uuids[1],
                    'template': [
                        {
                            'category': _mock_cat_uuids[0],
                            'mean': 0.050
                        },
                        {
                            'category': _mock_cat_uuids[1],
                            'mean': 0.900
                        },
                        {
                            'category': _mock_cat_uuids[2],
                            'mean': 0.050
                        },
                    ]
                }, {
                    'category':
                        _mock_cat_uuids[2],
                    'template': [
                        {
                            'category': _mock_cat_uuids[0],
                            'mean': 0.100
                        },
                        {
                            'category': _mock_cat_uuids[1],
                            'mean': 0.300
                        },
                        {
                            'category': _mock_cat_uuids[2],
                            'mean': 0.600
                        },
                    ]
                }]
            }]
        }

    def test_get(self, client: MockClient):
        # Set database
        task = load_default_resource(resource_type=Task)
        service = Service.filter_by_task_and_type(task_id=task.db_object().task_id, type_=ServiceType.MONITORING)
        service.data = self._mock_outputs_template(task=task)
        save_to_db(service)
        # Make request and verify response
        client.update_client_id(client_id=service.client.uuid, token_type='api_key')
        endpoint = get_endpoint(parameterized_endpoint=ENDPOINT_MONITORING_SERVICE_TEMPLATES, resource=task)
        response = client.send_request(method='GET', url=endpoint)
        assert response.status_code == HTTP_GET_STATUS_CODE
        assert response.json() == service.data

    def test_put(self, client: MockClient):
        # Prepare environment
        task = load_default_resource(resource_type=Task)
        mock_outputs_templates = self._mock_outputs_template(task=task)
        mon_service = Service.filter_by_task_and_type(task_id=task.db_object().task_id, type_=ServiceType.MONITORING)
        cl_service = Service.filter_by_task_and_type(task_id=task.db_object().task_id,
                                                     type_=ServiceType.CONTINUAL_LEARNING)
        mon_service.data = mock_outputs_templates
        save_to_db(mon_service)
        client.update_client_id(client_id=cl_service.client.uuid, token_type='api_key')
        endpoint = get_endpoint(parameterized_endpoint=ENDPOINT_MONITORING_SERVICE_TEMPLATES, resource=task)
        ##################
        # Valid template #
        ##################
        # Prepare new template
        valid_template = copy.deepcopy(mock_outputs_templates)
        assert valid_template == mock_outputs_templates
        new_out_templ = valid_template['outputs']
        new_out_templ[0]['template'] = {'mean': 200.39, 'std': 4.6}
        cat_means = new_out_templ[1]['template'][1]['template']
        cat_means[0]['mean'] = 0.01
        cat_means[1]['mean'] = 0.87
        cat_means[2]['mean'] = 0.12
        assert valid_template != mock_outputs_templates
        # Make request and verify response and database
        response = client.send_request(method='PUT', url=endpoint, json=valid_template)
        assert response.status_code == HTTP_PUT_STATUS_CODE
        assert response.json() == valid_template
        db_commit_and_expire()
        assert mon_service.data == valid_template
        ####################
        # Invalid template #
        ####################
        pass  # TODO


def _verify_service(task: Task, service_type: ServiceType, actual_service: dict, expected_service: dict):
    assert set(actual_service.keys()) == {'name', 'display_name', 'description', 'status'}
    service = Service.filter_by_task_and_type(task_id=task.db_object().task_id, type_=service_type)
    db_service_status = Status.from_dict(service.status).to_dict(include_state=True, expand_status=True)
    assert set(db_service_status.keys()) == {
        'state_code', 'status_code', 'name', 'display_name', 'description', 'started_at', 'updated_at', 'ended_at',
        'details', 'prev_status'
    }
    expected_service['status'] = db_service_status
    verify_response_json(actual_json=actual_service, expected_json=expected_service)
