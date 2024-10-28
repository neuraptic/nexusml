import copy
from datetime import datetime
import os
from typing import List, Type, Union

import pytest
import requests

from nexusml.api.endpoints import ENDPOINT_INPUT_CATEGORY
from nexusml.api.endpoints import ENDPOINT_INPUT_ELEMENT
from nexusml.api.endpoints import ENDPOINT_METADATA_CATEGORY
from nexusml.api.endpoints import ENDPOINT_METADATA_ELEMENT
from nexusml.api.endpoints import ENDPOINT_OUTPUT_CATEGORY
from nexusml.api.endpoints import ENDPOINT_OUTPUT_ELEMENT
from nexusml.api.endpoints import ENDPOINT_TASK
from nexusml.api.endpoints import ENDPOINT_TASK_QUOTA_USAGE
from nexusml.api.endpoints import ENDPOINT_TASK_SCHEMA
from nexusml.api.endpoints import ENDPOINT_TASK_SETTINGS
from nexusml.api.endpoints import ENDPOINT_TASK_STATUS
from nexusml.api.endpoints import ENDPOINT_TASKS
from nexusml.api.ext import cache
from nexusml.api.resources.tasks import InputCategory
from nexusml.api.resources.tasks import InputElement
from nexusml.api.resources.tasks import MetadataCategory
from nexusml.api.resources.tasks import MetadataElement
from nexusml.api.resources.tasks import OutputCategory
from nexusml.api.resources.tasks import OutputElement
from nexusml.api.resources.tasks import Task
from nexusml.constants import DATETIME_FORMAT
from nexusml.constants import HTTP_BAD_REQUEST_STATUS_CODE
from nexusml.constants import HTTP_CONFLICT_STATUS_CODE
from nexusml.constants import HTTP_DELETE_STATUS_CODE
from nexusml.constants import HTTP_METHOD_NOT_ALLOWED_STATUS_CODE
from nexusml.constants import HTTP_POST_STATUS_CODE
from nexusml.constants import HTTP_PUT_STATUS_CODE
from nexusml.database.ai import AIModelDB
from nexusml.database.core import db
from nexusml.database.core import save_to_db
from nexusml.database.examples import ExampleDB
from nexusml.database.files import TaskFileDB as FileDB
from nexusml.database.organizations import ClientDB
from nexusml.database.organizations import UserDB
from nexusml.database.permissions import UserPermission
from nexusml.database.services import Service
from nexusml.database.subscriptions import get_active_subscription
from nexusml.database.tasks import CategoryDB
from nexusml.database.tasks import ElementDB
from nexusml.database.tasks import TaskDB
from nexusml.enums import ElementValueType
from nexusml.enums import FileType
from nexusml.enums import ResourceAction
from nexusml.enums import ResourceType
from nexusml.enums import ServiceType
from nexusml.enums import TaskFileUse
from nexusml.enums import TaskTemplate
from nexusml.env import ENV_WEB_CLIENT_ID
from nexusml.statuses import inference_data_error_status
from nexusml.statuses import task_created_status
from nexusml.statuses import task_inactive_state
from nexusml.statuses import task_paused_status
from tests.api.constants import TEST_CONFIG
from tests.api.integration.conftest import MockClient
from tests.api.integration.utils import get_endpoint
from tests.api.integration.utils import verify_out_of_sync
from tests.api.integration.utils import verify_resource_request
from tests.api.integration.utils import verify_response_json
from tests.api.integration.utils import verify_response_json_with_picture
from tests.api.integration.utils import verify_response_jsons
from tests.api.utils import db_commit_and_expire
from tests.api.utils import get_json_from_db_object
from tests.api.utils import load_default_resource

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_ElementModel = Union[InputElement, OutputElement, MetadataElement]
_CategoryModel = Union[InputCategory, OutputCategory, MetadataCategory]

_ORG_ID = 2  # 1 is reserved for the main organization


class TestTasks:

    def test_delete(self, client: MockClient):
        response = client.send_request(method='DELETE', url=get_endpoint(parameterized_endpoint=ENDPOINT_TASKS))
        assert response.status_code == HTTP_METHOD_NOT_ALLOWED_STATUS_CODE

    def test_get(self, client: MockClient):

        def _set_user_permissions(user: UserDB, permissions: List[UserPermission]):
            cache.clear()
            user.permissions.delete()
            save_to_db(user)
            for perm in permissions:
                db.session.merge(perm)
            db_commit_and_expire()

        ##################################
        # Try different user permissions #
        ##################################
        session_user = UserDB.get(user_id=1)
        session_user.roles.clear()
        save_to_db(session_user)

        # 1. User has permissions to access every task of the organization (granted generic permission)
        user_perms_1 = [
            UserPermission(organization_id=session_user.organization_id,
                           user_id=session_user.user_id,
                           resource_type=ResourceType.TASK,
                           action=ResourceAction.READ,
                           allow=True)
        ]
        _set_user_permissions(user=session_user, permissions=user_perms_1)
        org_tasks = TaskDB.query().filter_by(organization_id=session_user.organization_id).all()
        assert len(org_tasks) > 1
        expected_tasks = [Task.get(agent=session_user, db_object_or_id=task).dump() for task in org_tasks]
        response = client.send_request(method='GET', url=get_endpoint(parameterized_endpoint=ENDPOINT_TASKS))
        verify_response_jsons(actual_jsons=response.json(), expected_jsons=expected_tasks)

        # 2. User has no permissions to access any task of the organization (denied generic permission)
        user_perms_2 = [
            UserPermission(organization_id=session_user.organization_id,
                           user_id=session_user.user_id,
                           resource_type=ResourceType.TASK,
                           action=ResourceAction.READ,
                           allow=False)
        ]
        _set_user_permissions(user=session_user, permissions=user_perms_2)
        response = client.send_request(method='GET', url=get_endpoint(parameterized_endpoint=ENDPOINT_TASKS))
        verify_response_jsons(actual_jsons=response.json(), expected_jsons=[])

        # 3. User has the permissions in 2. plus access to a task of the organization (granted resource permission)
        task = org_tasks[-1]
        task_perm = [
            UserPermission(organization_id=session_user.organization_id,
                           user_id=session_user.user_id,
                           resource_uuid=task.uuid,
                           resource_type=ResourceType.TASK,
                           action=ResourceAction.READ,
                           allow=True)
        ]
        user_perms_3 = user_perms_2 + task_perm
        _set_user_permissions(user=session_user, permissions=user_perms_3)
        expected_tasks = [Task.get(agent=session_user, db_object_or_id=task).dump()]
        response = client.send_request(method='GET', url=get_endpoint(parameterized_endpoint=ENDPOINT_TASKS))
        verify_response_jsons(actual_jsons=response.json(), expected_jsons=expected_tasks)

        # 4. User has permissions in 1. plus all the tasks of an external organization
        org_tasks = TaskDB.query().filter_by(organization_id=session_user.organization_id).all()
        assert len(org_tasks) > 1
        ext_org_id = 3
        assert session_user.organization_id != ext_org_id
        TaskDB.query().filter_by(organization_id=ext_org_id).delete()
        db_commit_and_expire()
        ext_tasks = [
            TaskDB(organization_id=ext_org_id,
                   name=f'External organization mock task {idx}',
                   description=f'External organization mock task description {idx}',
                   synced_by_users=[1]) for idx in range(1, 7)
        ]
        save_to_db(ext_tasks)
        ext_org_perms = [
            UserPermission(organization_id=ext_org_id,
                           user_id=session_user.user_id,
                           resource_type=ResourceType.TASK,
                           action=ResourceAction.READ,
                           allow=True)
        ]
        user_perms_4 = user_perms_1 + ext_org_perms
        _set_user_permissions(user=session_user, permissions=user_perms_4)
        expected_tasks = [Task.get(agent=session_user, db_object_or_id=task).dump() for task in org_tasks + ext_tasks]
        response = client.send_request(method='GET', url=get_endpoint(parameterized_endpoint=ENDPOINT_TASKS))
        verify_response_jsons(actual_jsons=response.json(), expected_jsons=expected_tasks)

        # 5. User has permissions in 2. plus all the tasks of an external organization
        user_perms_5 = user_perms_2 + ext_org_perms
        _set_user_permissions(user=session_user, permissions=user_perms_5)
        expected_tasks = [Task.get(agent=session_user, db_object_or_id=task).dump() for task in ext_tasks]
        response = client.send_request(method='GET', url=get_endpoint(parameterized_endpoint=ENDPOINT_TASKS))
        verify_response_jsons(actual_jsons=response.json(), expected_jsons=expected_tasks)

        # 6. User has the permissions in 1. plus three tasks of an external organization
        ext_tasks_perms = [
            UserPermission(organization_id=ext_org_id,
                           user_id=session_user.user_id,
                           resource_uuid=ext_task.uuid,
                           resource_type=ResourceType.TASK,
                           action=ResourceAction.READ,
                           allow=True) for ext_task in ext_tasks[2:5]
        ]
        user_perms_6 = user_perms_1 + ext_tasks_perms
        _set_user_permissions(user=session_user, permissions=user_perms_6)
        expected_tasks = [
            Task.get(agent=session_user, db_object_or_id=task).dump() for task in (org_tasks + ext_tasks[2:5])
        ]
        response = client.send_request(method='GET', url=get_endpoint(parameterized_endpoint=ENDPOINT_TASKS))
        verify_response_jsons(actual_jsons=response.json(), expected_jsons=expected_tasks)

        # 7. User has the permissions in 2. plus three tasks of an external organization
        user_perms_7 = user_perms_2 + ext_tasks_perms
        _set_user_permissions(user=session_user, permissions=user_perms_7)
        expected_tasks = [Task.get(agent=session_user, db_object_or_id=task).dump() for task in ext_tasks[2:5]]
        response = client.send_request(method='GET', url=get_endpoint(parameterized_endpoint=ENDPOINT_TASKS))
        verify_response_jsons(actual_jsons=response.json(), expected_jsons=expected_tasks)
        ##################################
        # Try different role permissions #
        ##################################
        pass  # TODO
        ##############################
        # Get only out of sync tasks #
        ##############################
        pass  # TODO

    def test_post(self, client: MockClient):
        """
        Create a task from scratch
        """
        req_json = {'name': 'New task name', 'description': 'New task description', 'icon': None}
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_TASKS)
        response = client.send_request(method='POST', url=endpoint_url, json=req_json)
        assert response.status_code == HTTP_POST_STATUS_CODE
        res_json = response.json()
        assert res_json['name'] == req_json['name']
        assert res_json['description'] == req_json['description']
        assert res_json['icon'] == req_json['icon']
        # Verify that service clients were automatically created
        db_commit_and_expire()
        task = TaskDB.get_from_uuid(res_json['uuid'])
        assert len(Service.filter_by_task(task_id=task.task_id)) == len(ServiceType)
        #################################
        # Create a task from a template #
        #################################
        req_json = {
            'name': 'New task from template',
            'description': 'New task description',
            'template': TaskTemplate.OBJECT_DETECTION.name.lower()
        }
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_TASKS)
        response = client.send_request(method='POST', url=endpoint_url, json=req_json)
        assert response.status_code == HTTP_POST_STATUS_CODE
        res_json = response.json()
        db_commit_and_expire()
        task = TaskDB.get_from_uuid(res_json['uuid'])

        # Check inputs
        inputs = task.input_elements()
        assert len(inputs) == 1

        input_img = inputs[0]
        assert input_img.name == 'image'
        assert input_img.value_type == ElementValueType.IMAGE_FILE
        assert not input_img.nullable
        assert input_img.required

        # Check outputs
        outputs = task.output_elements()
        assert len(outputs) == 2

        output_bbox = outputs[0]
        assert output_bbox.name == 'bounding_box'
        assert output_bbox.value_type == ElementValueType.SHAPE
        assert not output_bbox.nullable
        assert output_bbox.required

        output_class = outputs[1]
        assert output_class.name == 'class'
        assert output_class.value_type == ElementValueType.CATEGORY
        assert not output_class.nullable
        assert not output_class.required


class TestTask:

    def test_get(self, client: MockClient, mock_s3):
        task = load_default_resource(resource_type=Task)
        verify_resource_request(client=client,
                                method='GET',
                                endpoint=ENDPOINT_TASK,
                                resource=task,
                                expected_jsons=[task.dump()])

    def test_put(self, client: MockClient, mock_s3):
        task = load_default_resource(resource_type=Task)
        json_data = {'name': 'Modified name', 'description': 'Modified description', 'icon': None}

        # Basic update
        icon_file = FileDB(task_id=task.db_object().task_id,
                           filename='task_icon_1',
                           use_for=TaskFileUse.PICTURE,
                           type_=FileType.IMAGE,
                           size=215)
        save_to_db(icon_file)
        request_data_1 = dict(json_data)
        request_data_1['icon'] = icon_file.public_id
        endpoint = get_endpoint(parameterized_endpoint=ENDPOINT_TASK, resource=task)
        response = client.send_request(method='PUT', url=endpoint, json=request_data_1)
        assert response.status_code == HTTP_PUT_STATUS_CODE
        res_json = {k: v for k, v in response.json().items() if k in request_data_1}
        verify_response_json_with_picture(request_json=request_data_1,
                                          response_json=res_json,
                                          picture_db_object=icon_file,
                                          picture_field='icon')

        # Try to provide an invalid image file for task icon
        pass  # TODO

        # Try to modify fields that are restricted in this endpoint
        request_data_4 = {
            'name': 'Modified name 2',
            'description': 'Modified description 2',
            'icon': None,
            'inputs': [{
                'name': 'new_input',
                'display_name': 'New Input',
                'type': 'float'
            }]
        }

        verify_resource_request(client=client,
                                method='PUT',
                                endpoint=ENDPOINT_TASK,
                                resource=task,
                                request_json=request_data_4,
                                expected_status_code=HTTP_BAD_REQUEST_STATUS_CODE)

        # Force an out-of-sync state
        request_data_5 = dict(json_data)
        verify_out_of_sync(client=client, endpoint=ENDPOINT_TASK, resource=task, request_json=request_data_5)

    def test_delete(self, client: MockClient, mock_s3):
        task = load_default_resource(resource_type=Task)

        task_id = task.db_object().task_id
        org_id = task.db_object().organization_id

        services = Service.query().filter_by(task_id=task_id).all()
        services_client_ids = [service.client_id for service in services]

        subscription = get_active_subscription(organization_id=org_id)

        org_num_tasks = subscription.num_tasks
        org_num_examples = subscription.num_examples
        org_space_usage = subscription.space_usage
        task_num_examples = task.db_object().num_examples
        task_space_usage = task.db_object().space_usage

        verify_resource_request(client=client,
                                method='DELETE',
                                endpoint=ENDPOINT_TASK,
                                resource=task,
                                expected_status_code=HTTP_DELETE_STATUS_CODE)

        db_commit_and_expire()

        # Verify that task files were deleted
        assert len(FileDB.filter_by_task(task_id=task_id)) == 0

        # Verify that AI models were deleted
        assert len(AIModelDB.filter_by_task(task_id=task_id)) == 0

        # Verify that all schema elements were deleted
        assert len(ElementDB.filter_by_task(task_id=task_id)) == 0

        # Verify that services and associated clients were deleted
        assert len(Service.filter_by_task(task_id=task_id)) == 0
        assert all(ClientDB.get(client_id=x) is None for x in services_client_ids)

        # Verify that examples were deleted
        assert len(ExampleDB.filter_by_task(task_id=task_id)) == 0

        # Get org subscription and verify that the number of tasks, examples and usage was updated
        new_subscription = get_active_subscription(organization_id=org_id)
        assert new_subscription.num_tasks == org_num_tasks - 1
        assert new_subscription.num_examples == org_num_examples - task_num_examples
        assert new_subscription.space_usage == org_space_usage - task_space_usage


class TestTaskSchema:

    def test_get(self, client: MockClient):
        task = TaskDB.get(task_id=1)
        # Set expected response
        expected_inputs = []
        for input_element in task.input_elements():
            expected_input = _get_element_json_from_db_object(db_object=input_element)
            expected_inputs.append(expected_input)

        expected_outputs = []
        for output_element in task.output_elements():
            expected_output = _get_element_json_from_db_object(db_object=output_element)
            expected_outputs.append(expected_output)

        expected_metadata = []
        for metadata_element in task.metadata_elements():
            expected_metadata_element = _get_element_json_from_db_object(db_object=metadata_element)
            expected_metadata.append(expected_metadata_element)

        expected_schema = {
            'inputs': expected_inputs,
            'outputs': expected_outputs,
            'metadata': expected_metadata,
            # TODO: Specify "task_type" field
        }

        # Make request and verify response
        task = load_default_resource(resource_type=Task)
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_TASK_SCHEMA, resource=task)
        response = client.send_request(method='GET', url=endpoint_url)
        verify_response_json(actual_json=response.json(), expected_json=expected_schema, optional_fields={'task_type'})


class TestTaskSettings:

    def _set_task(self) -> Task:
        # Set database
        task = load_default_resource(resource_type=Task)
        services = Service.filter_by_task(task_id=task.db_object().task_id)
        defaults = {
            ServiceType.INFERENCE: TEST_CONFIG['engine']['services']['inference'],
            ServiceType.CONTINUAL_LEARNING: TEST_CONFIG['engine']['services']['continual_learning'],
            ServiceType.ACTIVE_LEARNING: TEST_CONFIG['engine']['services']['active_learning'],
            ServiceType.MONITORING: TEST_CONFIG['engine']['services']['monitoring'],
            ServiceType.TESTING: TEST_CONFIG['engine']['services']['testing']
        }
        for service in services:
            service.set_settings(defaults[service.type_])
        save_to_db(services)
        # Verify database was set correctly
        db_commit_and_expire()
        for service in services:
            assert service.to_dict()['settings'] == defaults[service.type_]
        return task

    def test_get(self, client: MockClient):
        task = self._set_task()
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_TASK_SETTINGS, resource=task)
        response = client.send_request(method='GET', url=endpoint_url)
        expected_json = {
            'services': {
                'inference': TEST_CONFIG['engine']['services']['inference'],
                'continual_learning': TEST_CONFIG['engine']['services']['continual_learning'],
                'active_learning': TEST_CONFIG['engine']['services']['active_learning'],
                'monitoring': TEST_CONFIG['engine']['services']['monitoring']
            }
        }
        verify_response_json(actual_json=response.json(), expected_json=expected_json)

    def test_put(self, client: MockClient):
        task = self._set_task()
        # Make request and verify response
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_TASK_SETTINGS, resource=task)
        new_settings = {
            'services': {
                'inference': {
                    'enabled': False,
                },
                'continual_learning': {
                    'enabled': False,
                    'min_days': 14.0,
                    'max_days': 30.0,
                    'min_sample': 0.2,
                    'cpu_hard_limit': 600.0,
                    'gpu_hard_limit': 500.0,
                    'max_cpu_quota': 300.0,
                    'max_gpu_quota': 250.0,
                    'min_cpu_quota': 150.0,
                    'min_gpu_quota': 125.0,
                },
                'active_learning': {
                    'enabled': False,
                    'query_interval': 14,
                    'max_examples_per_query': 200,
                },
                'monitoring': {
                    'enabled': False,
                    'refresh_interval': 15,
                    'ood_predictions': {
                        'min_sample': 10,
                        'sensitivity': 0.5,
                        'smoothing': 0.5,
                    }
                }
            }
        }
        response = client.send_request(method='PUT', url=endpoint_url, json=new_settings)
        verify_response_json(actual_json=response.json(), expected_json=new_settings)
        # Verify database
        db_commit_and_expire()
        task_id = task.db_object().task_id
        inference_service = Service.filter_by_task_and_type(task_id=task_id, type_=ServiceType.INFERENCE)
        cl_service = Service.filter_by_task_and_type(task_id=task_id, type_=ServiceType.CONTINUAL_LEARNING)
        al_service = Service.filter_by_task_and_type(task_id=task_id, type_=ServiceType.ACTIVE_LEARNING)
        monitoring_service = Service.filter_by_task_and_type(task_id=task_id, type_=ServiceType.MONITORING)
        assert inference_service.to_dict()['settings'] == new_settings['services']['inference']
        assert cl_service.to_dict()['settings'] == new_settings['services']['continual_learning']
        assert al_service.to_dict()['settings'] == new_settings['services']['active_learning']
        assert monitoring_service.to_dict()['settings'] == new_settings['services']['monitoring']


class TestTaskStatus:

    @pytest.mark.parametrize('custom_client', [(os.environ[ENV_WEB_CLIENT_ID],)], indirect=True)
    def test_put(self, custom_client: MockClient):
        # Valid request
        task = load_default_resource(resource_type=Task)
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_TASK_STATUS, resource=task)
        new_status = {'code': task_paused_status.code}
        response = custom_client.send_request(method='PUT', url=endpoint_url, json=new_status)
        assert response.status_code == HTTP_PUT_STATUS_CODE
        res_json = response.json()
        assert res_json['state_code'] == task_inactive_state.code
        assert res_json['status_code'] == task_paused_status.code
        assert res_json['name'] == task_paused_status.name
        assert res_json['display_name'] == task_paused_status.display_name
        assert res_json['description'] == task_paused_status.description
        assert isinstance(datetime.strptime(res_json['started_at'], DATETIME_FORMAT), datetime)
        assert res_json['updated_at'] is None
        assert res_json['ended_at'] is None
        assert res_json['details'] is None
        assert res_json['prev_status'] == task_created_status.code
        db_commit_and_expire()
        status = task.db_object().status
        assert status['code'] == task_paused_status.code
        assert isinstance(datetime.strptime(status['started_at'], DATETIME_FORMAT), datetime)
        assert status['updated_at'] is None
        assert status['ended_at'] is None
        assert status['details'] is None
        assert status['prev_status'] == task_created_status.code

        # Invalid status: non-existing
        new_status = {'code': 'invalid_status'}
        response = custom_client.send_request(method='PUT', url=endpoint_url, json=new_status)
        assert response.status_code == HTTP_BAD_REQUEST_STATUS_CODE
        assert response.json()['error']['message'] == 'Status not found: "invalid_status"'

        # Invalid status: invalid group/prefix
        new_status = {'code': inference_data_error_status.code}
        response = custom_client.send_request(method='PUT', url=endpoint_url, json=new_status)
        assert response.status_code == HTTP_BAD_REQUEST_STATUS_CODE
        assert response.json()['error']['message'] == 'Invalid status code'


class TestTaskQuotaUsage:

    def test_post(self, client: MockClient):
        task = load_default_resource(resource_type=Task)
        cl_service = Service.filter_by_task_and_type(task_id=task.db_object().task_id,
                                                     type_=ServiceType.CONTINUAL_LEARNING)

        # Mock service client
        client.update_client_id(client_id=cl_service.client.uuid, token_type='api_key')

        # Set database
        subscription = get_active_subscription(organization_id=_ORG_ID)
        subscription.num_cpu_hours = 20
        subscription.num_gpu_hours = 30
        subscription.max_cpu_hours = 240
        subscription.max_gpu_hours = 120
        task.db_object().num_cpu_hours = 5
        task.db_object().num_gpu_hours = 7
        task.db_object().max_cpu_hours = 30
        task.db_object().max_gpu_hours = 15
        save_to_db([subscription, task.db_object()])

        # Make request
        cpu_hours_to_add = 3.71842
        gpu_hours_to_add = 5.203
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_TASK_QUOTA_USAGE, resource=task)
        usage_json = {
            'cpu_hours': cpu_hours_to_add,
            'gpu_hours': gpu_hours_to_add,
        }
        response = client.send_request(method='POST', url=endpoint_url, json=usage_json)

        # Verify response
        assert response.status_code == HTTP_POST_STATUS_CODE
        rounded_cpu_hours_to_add = float(f'{cpu_hours_to_add:.2f}')
        rounded_gpu_hours_to_add = float(f'{gpu_hours_to_add:.2f}')
        assert response.json() == {
            'usage': {
                'cpu_hours': 5 + rounded_cpu_hours_to_add,
                'gpu_hours': 7 + rounded_gpu_hours_to_add,
            },
            'limit': {
                'cpu_hours': 30.,
                'gpu_hours': 15.
            }
        }

        # Verify database
        db_commit_and_expire()
        assert float(task.db_object().num_cpu_hours) == 5 + rounded_cpu_hours_to_add
        assert float(task.db_object().num_gpu_hours) == 7 + rounded_gpu_hours_to_add
        assert float(task.db_object().max_cpu_hours) == 30
        assert float(task.db_object().max_gpu_hours) == 15
        assert float(subscription.num_cpu_hours) == 20 + rounded_cpu_hours_to_add
        assert float(subscription.num_gpu_hours) == 30 + rounded_gpu_hours_to_add

        # Make request with negative values
        usage_json = {'gpu_hours': -10}
        response = client.send_request(method='POST', url=endpoint_url, json=usage_json)
        assert response.status_code == HTTP_BAD_REQUEST_STATUS_CODE

        usage_json = {'cpu_hours': -10}
        response = client.send_request(method='POST', url=endpoint_url, json=usage_json)
        assert response.status_code == HTTP_BAD_REQUEST_STATUS_CODE


class TestInputElements:

    def test_delete(self, client: MockClient):
        _test_elements_method(client=client,
                              method='DELETE',
                              resource_type=InputElement,
                              expected_status_code=HTTP_METHOD_NOT_ALLOWED_STATUS_CODE)

    def test_get(self, client: MockClient):
        task = TaskDB.get(task_id=1)
        elements_jsons = [
            InputElement.get(agent=UserDB(user_id=1), db_object_or_id=x).dump() for x in task.input_elements()
        ]
        _test_elements_method(client=client, method='GET', resource_type=InputElement, expected_jsons=elements_jsons)

    def test_post(self, client: MockClient):
        # Regular request
        new_elem_json = {'name': 'new_input', 'display_name': 'New Input', 'type': 'float'}
        _test_elements_method(client=client, method='POST', resource_type=InputElement, request_json=new_elem_json)
        # Try to create a new element with the same name (not allowed)
        _test_element_name_uniqueness(client=client,
                                      method='POST',
                                      element_model=InputElement,
                                      request_json=new_elem_json)


class TestInputElement:

    def test_delete(self, client: MockClient):
        _test_element_method(client=client, method='DELETE', resource_type=InputElement)

    def test_get(self, client: MockClient):
        _test_element_method(client=client, method='GET', resource_type=InputElement)

    def test_put(self, client: MockClient):
        updated_element = {'name': 'modified_input', 'display_name': 'Modified Input', 'type': 'integer'}
        _test_element_method(client=client, method='PUT', resource_type=InputElement, request_json=updated_element)
        _test_element_name_uniqueness(client=client,
                                      method='PUT',
                                      element_model=InputElement,
                                      request_json=updated_element)


class TestOutputElements:

    def test_delete(self, client: MockClient):
        _test_elements_method(client=client,
                              method='DELETE',
                              resource_type=OutputElement,
                              expected_status_code=HTTP_METHOD_NOT_ALLOWED_STATUS_CODE)

    def test_get(self, client: MockClient):
        task = TaskDB.get(task_id=1)
        elements_jsons = [
            OutputElement.get(agent=UserDB(user_id=1), db_object_or_id=x).dump() for x in task.output_elements()
        ]
        _test_elements_method(client=client, method='GET', resource_type=OutputElement, expected_jsons=elements_jsons)

    def test_post(self, client: MockClient):
        # Regular request
        new_element = {'name': 'new_output', 'display_name': 'New Output', 'type': 'float'}
        _test_elements_method(client=client, method='POST', resource_type=OutputElement, request_json=new_element)
        _test_element_name_uniqueness(client=client,
                                      method='POST',
                                      element_model=OutputElement,
                                      request_json=new_element)

        # Verify not allowed types
        for value_type in set(ElementValueType) - set(OutputElement.ALLOWED_VALUE_TYPES):
            val_type_str = value_type.name.lower()
            new_element = {'name': f'new_output_{val_type_str}', 'display_name': 'New Output', 'type': val_type_str}
            response = _test_elements_method(client=client,
                                             method='POST',
                                             resource_type=OutputElement,
                                             request_json=new_element,
                                             expected_status_code=HTTP_BAD_REQUEST_STATUS_CODE)
            assert response.json()['error']['message'] == f'Invalid value type for an output element: {val_type_str}'


class TestOutputElement:

    def test_delete(self, client: MockClient):
        _test_element_method(client=client, method='DELETE', resource_type=OutputElement)

    def test_get(self, client: MockClient):
        _test_element_method(client=client, method='GET', resource_type=OutputElement)

    def test_put(self, client: MockClient):
        # Regular request
        updated_element = {'name': 'modified_output', 'display_name': 'Modified Output', 'type': 'integer'}
        _test_element_method(client=client, method='PUT', resource_type=OutputElement, request_json=updated_element)
        _test_element_name_uniqueness(client=client,
                                      method='PUT',
                                      element_model=OutputElement,
                                      request_json=updated_element)

        # Verify not allowed types
        for value_type in set(ElementValueType) - set(OutputElement.ALLOWED_VALUE_TYPES):
            val_type_str = value_type.name.lower()
            updated_element = {
                'name': f'modified_output_{val_type_str}',
                'display_name': 'Modified Output',
                'type': val_type_str
            }
            response = _test_elements_method(client=client,
                                             method='POST',
                                             resource_type=OutputElement,
                                             request_json=updated_element,
                                             expected_status_code=HTTP_BAD_REQUEST_STATUS_CODE)
            assert response.json()['error']['message'] == f'Invalid value type for an output element: {val_type_str}'


class TestMetadataElements:

    def test_delete(self, client: MockClient):
        _test_elements_method(client=client,
                              method='DELETE',
                              resource_type=MetadataElement,
                              expected_status_code=HTTP_METHOD_NOT_ALLOWED_STATUS_CODE)

    def test_get(self, client: MockClient):
        task = TaskDB.get(task_id=1)
        elements_jsons = [
            MetadataElement.get(agent=UserDB(user_id=1), db_object_or_id=x).dump() for x in task.metadata_elements()
        ]
        _test_elements_method(client=client, method='GET', resource_type=MetadataElement, expected_jsons=elements_jsons)

    def test_post(self, client: MockClient):
        new_element = {'name': 'new_metadata', 'display_name': 'New Metadata', 'type': 'float'}
        _test_elements_method(client=client, method='POST', resource_type=MetadataElement, request_json=new_element)
        _test_element_name_uniqueness(client=client,
                                      method='POST',
                                      element_model=MetadataElement,
                                      request_json=new_element)


class TestMetadataElement:

    def test_delete(self, client: MockClient):
        _test_element_method(client=client, method='DELETE', resource_type=MetadataElement)

    def test_get(self, client: MockClient):
        _test_element_method(client=client, method='GET', resource_type=MetadataElement)

    def test_put(self, client: MockClient):
        updated_element = {'name': 'modified_metadata', 'display_name': 'Modified Metadata', 'type': 'integer'}
        _test_element_method(client=client, method='PUT', resource_type=MetadataElement, request_json=updated_element)
        _test_element_name_uniqueness(client=client,
                                      method='PUT',
                                      element_model=MetadataElement,
                                      request_json=updated_element)


class TestInputCategories:

    def test_delete(self, client: MockClient):
        _test_categories_method(client=client,
                                method='DELETE',
                                category_model=InputCategory,
                                expected_status_code=HTTP_METHOD_NOT_ALLOWED_STATUS_CODE)

    def test_get(self, client: MockClient):
        element = _get_default_categorical_element(element_model=InputElement)
        elem_pks = {pk_col: getattr(element.db_object(), pk_col) for pk_col in element.db_model().primary_key_columns()}
        category_jsons = [
            _get_category_json_from_db_object(db_object=x)
            for x in InputCategory.db_model().query().filter_by(**elem_pks).all()
        ]
        _test_categories_method(client=client,
                                method='GET',
                                category_model=InputCategory,
                                expected_jsons=category_jsons)

    def test_post(self, client: MockClient):
        new_category = {
            'name': 'new_input_category',
            'display_name': 'New Input Category',
            'description': 'New input description',
            'color': '#FF0000'
        }
        _test_categories_method(client=client, method='POST', category_model=InputCategory, request_json=new_category)


class TestInputCategory:

    def test_delete(self, client: MockClient):
        _test_category_method(client=client, method='DELETE', category_model=InputCategory)

    def test_get(self, client: MockClient):
        _test_category_method(client=client, method='GET', category_model=InputCategory)

    def test_put(self, client: MockClient):
        json_data = {
            'name': 'modified_input_category',
            'display_name': 'Modified Input Category',
            'description': 'Modified input description',
            'color': '#0000FF'
        }
        _test_category_method(client=client, method='PUT', category_model=InputCategory, request_json=json_data)


class TestOutputCategories:

    def test_delete(self, client: MockClient):
        _test_categories_method(client=client,
                                method='DELETE',
                                category_model=OutputCategory,
                                expected_status_code=HTTP_METHOD_NOT_ALLOWED_STATUS_CODE)

    def test_get(self, client: MockClient):
        element = _get_default_categorical_element(element_model=OutputElement)
        elem_pks = {pk_col: getattr(element.db_object(), pk_col) for pk_col in element.db_model().primary_key_columns()}
        category_jsons = [
            _get_category_json_from_db_object(db_object=x)
            for x in OutputCategory.db_model().query().filter_by(**elem_pks).all()
        ]
        _test_categories_method(client=client,
                                method='GET',
                                category_model=OutputCategory,
                                expected_jsons=category_jsons)

    def test_post(self, client: MockClient):
        new_category = {
            'name': 'new_output_category',
            'display_name': 'New Output Category',
            'description': 'New output description',
            'color': '#FF0000'
        }
        _test_categories_method(client=client, method='POST', category_model=OutputCategory, request_json=new_category)


class TestOutputCategory:

    def test_delete(self, client: MockClient):
        _test_category_method(client=client, method='DELETE', category_model=OutputCategory)

    def test_get(self, client: MockClient):
        _test_category_method(client=client, method='GET', category_model=OutputCategory)

    def test_put(self, client: MockClient):
        json_data = {
            'name': 'modified_output_category',
            'display_name': 'Modified Output Category',
            'description': 'Modified output description',
            'color': '#0000FF'
        }
        _test_category_method(client=client, method='PUT', category_model=OutputCategory, request_json=json_data)


class TestMetadataCategories:

    def test_delete(self, client: MockClient):
        _test_categories_method(client=client,
                                method='DELETE',
                                category_model=MetadataCategory,
                                expected_status_code=HTTP_METHOD_NOT_ALLOWED_STATUS_CODE)

    def test_get(self, client: MockClient):
        element = _get_default_categorical_element(element_model=MetadataElement)
        elem_pks = {pk_col: getattr(element.db_object(), pk_col) for pk_col in element.db_model().primary_key_columns()}
        category_jsons = [
            _get_category_json_from_db_object(db_object=x)
            for x in MetadataCategory.db_model().query().filter_by(**elem_pks).all()
        ]
        _test_categories_method(client=client,
                                method='GET',
                                category_model=MetadataCategory,
                                expected_jsons=category_jsons)

    def test_post(self, client: MockClient):
        new_category = {
            'name': 'new_metadata_category',
            'display_name': 'New Metadata Category',
            'description': 'New metadata description',
            'color': '#FF0000'
        }
        _test_categories_method(client=client,
                                method='POST',
                                category_model=MetadataCategory,
                                request_json=new_category)


class TestMetadataCategory:

    def test_delete(self, client: MockClient):
        _test_category_method(client=client, method='DELETE', category_model=MetadataCategory)

    def test_get(self, client: MockClient):
        _test_category_method(client=client, method='GET', category_model=MetadataCategory)

    def test_put(self, client: MockClient):
        json_data = {
            'name': 'modified_metadata_category',
            'display_name': 'Modified Metadata Category',
            'description': 'Modified metadata description',
            'color': '#0000FF'
        }
        _test_category_method(client=client, method='PUT', category_model=MetadataCategory, request_json=json_data)


def _test_elements_method(client: MockClient,
                          method: str,
                          resource_type: Type[_ElementModel],
                          request_json: dict = None,
                          expected_jsons: list = None,
                          expected_status_code: int = None) -> requests.Response:

    method = method.strip().lower()

    collections = {InputElement: 'inputs', OutputElement: 'outputs', MetadataElement: 'metadata'}

    parent_task = load_default_resource(resource_type=Task)

    if request_json and not expected_jsons:
        expected_jsons = [request_json]

    update_training_status = (resource_type in [InputElement, OutputElement] and method == 'post' and
                              expected_status_code in [HTTP_POST_STATUS_CODE, None])
    if update_training_status:
        ExampleDB.set_examples_training_flag(task=parent_task.db_object(), trained=True)

    response = verify_resource_request(client=client,
                                       method=method,
                                       endpoint=ENDPOINT_TASK_SCHEMA,
                                       resource=parent_task,
                                       collection=(collections[resource_type], resource_type),
                                       request_json=request_json,
                                       expected_jsons=expected_jsons,
                                       expected_status_code=expected_status_code)

    if update_training_status:
        db_commit_and_expire()
        assert all(not x.trained for x in ExampleDB.filter_by_task(task_id=parent_task.db_object().task_id))

    return response


def _test_element_method(client: MockClient,
                         method: str,
                         resource_type: Type[_ElementModel],
                         request_json: dict = None,
                         expected_jsons: list = None,
                         expected_status_code: int = None) -> requests.Response:

    method = method.strip().lower()

    # Get endpoint
    endpoints = {
        InputElement: ENDPOINT_INPUT_ELEMENT,
        OutputElement: ENDPOINT_OUTPUT_ELEMENT,
        MetadataElement: ENDPOINT_METADATA_ELEMENT,
    }

    parent_task = load_default_resource(resource_type=Task)
    element_collections = {
        InputElement: 'input_elements',
        OutputElement: 'output_elements',
        MetadataElement: 'metadata_elements',
    }
    resource = getattr(parent_task, element_collections[resource_type])()[0]

    # Set examples' initial training status
    update_training_status = (resource_type in [InputElement, OutputElement] and method == 'delete' and
                              expected_status_code in [HTTP_DELETE_STATUS_CODE, None])

    if update_training_status:
        ExampleDB.set_examples_training_flag(task=parent_task.db_object(), trained=True)

    # Make request and verify response
    if not expected_jsons:
        if method == 'delete':
            expected_jsons = None
        elif request_json:
            expected_jsons = [request_json]
        else:
            expected_jsons = [resource.dump()]

    response = verify_resource_request(client=client,
                                       method=method,
                                       endpoint=endpoints[resource_type],
                                       resource=resource,
                                       request_json=request_json,
                                       expected_jsons=expected_jsons,
                                       expected_status_code=expected_status_code)

    # Verify training status was updated correctly
    if update_training_status:
        db_commit_and_expire()
        assert all(not x.trained for x in ExampleDB.filter_by_task(task_id=parent_task.db_object().task_id))

    # Verify prediction database entries' `removed_elements` column was updated correctly
    if method == 'delete':
        pass  # TODO

    return response


def _test_element_name_uniqueness(client: MockClient, method: str, element_model: Type[_ElementModel],
                                  request_json: dict):

    def _test_name(name: str):
        element_json = copy.deepcopy(request_json)
        element_json['name'] = name
        request_call = _test_elements_method if method == 'post' else _test_element_method
        response = request_call(client=client,
                                method=method,
                                resource_type=element_model,
                                request_json=element_json,
                                expected_status_code=HTTP_CONFLICT_STATUS_CODE)
        response_text = response.text.lower()
        assert 'already in use' in response_text or 'already exists' in response_text

    method = method.strip().lower()
    assert method in ['post', 'put']
    # Try to assign a name already in use by another input element
    if element_model != InputElement or method == 'post':
        _test_name('input_1')
    # Try to assign a name already in use by another output element
    if element_model != OutputElement or method == 'post':
        _test_name('output_1')
    # Try to assign a name already in use by another metadata element
    if element_model != MetadataElement or method == 'post':
        _test_name('metadata_1')


def _test_categories_method(client: MockClient,
                            method: str,
                            category_model: Type[_CategoryModel],
                            request_json: dict = None,
                            expected_jsons: list = None,
                            expected_status_code: int = None):
    element_models = {InputCategory: InputElement, OutputCategory: OutputElement, MetadataCategory: MetadataElement}

    element = _get_default_categorical_element(element_model=element_models[category_model])

    endpoints = {
        InputCategory: ENDPOINT_INPUT_ELEMENT,
        OutputCategory: ENDPOINT_OUTPUT_ELEMENT,
        MetadataCategory: ENDPOINT_METADATA_ELEMENT
    }

    if request_json and not expected_jsons:
        expected_jsons = [request_json]

    verify_resource_request(client=client,
                            method=method,
                            endpoint=endpoints[category_model],
                            resource=element,
                            collection=('categories', category_model),
                            request_json=request_json,
                            expected_jsons=expected_jsons,
                            expected_status_code=expected_status_code)

    # TODO: test pagination
    if method.strip().lower() == 'get':
        pass


def _test_category_method(client: MockClient,
                          method: str,
                          category_model: Type[_CategoryModel],
                          request_json: dict = None):

    method = method.strip().lower()

    element_models = {InputCategory: InputElement, OutputCategory: OutputElement, MetadataCategory: MetadataElement}

    element = _get_default_categorical_element(element_model=element_models[category_model])

    category = category_model.get(agent=element.user(),
                                  db_object_or_id=element.db_object().categories.first(),
                                  parents=element.parents() + [element])

    endpoints = {
        InputCategory: ENDPOINT_INPUT_CATEGORY,
        OutputCategory: ENDPOINT_OUTPUT_CATEGORY,
        MetadataCategory: ENDPOINT_METADATA_CATEGORY
    }

    if method == 'delete':
        expected_jsons = None
    elif request_json:
        expected_jsons = [request_json]
    else:
        expected_jsons = [category.dump()]

    verify_resource_request(client=client,
                            method=method,
                            endpoint=endpoints[category_model],
                            resource=category,
                            request_json=request_json,
                            expected_jsons=expected_jsons)


def _get_default_categorical_element(element_model: Type[_ElementModel]) -> _ElementModel:
    categorical_elements = [
        x for x in ElementDB.filter_by_task(task_id=1)
        if x.element_type == element_model.element_type() and x.value_type == ElementValueType.CATEGORY
    ]
    element_db_object = categorical_elements[0]
    task = Task.get(agent=UserDB.get(user_id=1), db_object_or_id=TaskDB.get(task_id=element_db_object.task_id))
    element = element_model.get(agent=task.user(), db_object_or_id=element_db_object, parents=[task])
    return element


def _get_element_json_from_db_object(db_object: ElementDB):
    element_json = get_json_from_db_object(db_object=db_object, ignore_columns=['element_type'])
    element_json['type'] = element_json.pop('value_type')
    return element_json


def _get_category_json_from_db_object(db_object: CategoryDB):
    category_json = get_json_from_db_object(db_object=db_object)
    if 'color' in category_json and category_json['color']:
        category_json['color'] = '#' + category_json['color']
    return category_json
