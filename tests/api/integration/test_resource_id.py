from typing import Callable
import urllib.parse

import pytest

from nexusml.api.endpoints import ENDPOINT_AI_MODEL
from nexusml.api.endpoints import ENDPOINT_AI_PREDICTION_LOG
from nexusml.api.endpoints import ENDPOINT_EXAMPLE
from nexusml.api.endpoints import ENDPOINT_EXAMPLE_SHAPE
from nexusml.api.endpoints import ENDPOINT_EXAMPLE_SLICE
from nexusml.api.endpoints import ENDPOINT_INPUT_ELEMENT
from nexusml.api.endpoints import ENDPOINT_METADATA_ELEMENT
from nexusml.api.endpoints import ENDPOINT_OUTPUT_ELEMENT
from nexusml.api.endpoints import ENDPOINT_TAG
from nexusml.api.endpoints import ENDPOINT_TASK
from nexusml.api.endpoints import ENDPOINT_TASK_FILE
from nexusml.api.external.ext import cache
from nexusml.api.resources.ai import AIModel
from nexusml.api.resources.ai import PredictionLog
from nexusml.api.resources.examples import Example
from nexusml.api.resources.examples import Shape
from nexusml.api.resources.examples import Slice
from nexusml.api.resources.files import TaskFile
from nexusml.api.resources.tags import Tag
from nexusml.api.resources.tasks import InputElement
from nexusml.api.resources.tasks import MetadataElement
from nexusml.api.resources.tasks import OutputElement
from nexusml.api.resources.tasks import Task
from nexusml.database.organizations import MutableEntity
from nexusml.enums import AIEnvironment
from tests.api.conftest import restore_db
from tests.api.integration.conftest import MockClient
from tests.api.integration.utils import get_endpoint
from tests.api.integration.utils import mock_element_values_json
from tests.api.integration.utils import mock_prediction_log_json
from tests.api.integration.utils import mock_shape_or_slice_json
from tests.api.utils import db_commit_and_expire
from tests.api.utils import load_default_resource

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestResourceID:

    @staticmethod
    def _resources() -> list:
        return [
            {
                'type': Task,
                'parents': [],
                'endpoint': ENDPOINT_TASK,
                'data': {
                    'name': 'Test name',
                    'description': 'Test description',
                    'icon': None
                }
            },
            {
                'type': InputElement,
                'parents': [Task],
                'endpoint': ENDPOINT_INPUT_ELEMENT,
                'data': {
                    'name': 'test_input',
                    'display_name': 'Test Input',
                    'type': 'float'
                },
                'resource_loader': _load_input_element
            },
            {
                'type': OutputElement,
                'parents': [Task],
                'endpoint': ENDPOINT_OUTPUT_ELEMENT,
                'data': {
                    'name': 'test_output',
                    'display_name': 'Test Output',
                    'type': 'float'
                },
                'resource_loader': _load_output_element
            },
            {
                'type': MetadataElement,
                'parents': [Task],
                'endpoint': ENDPOINT_METADATA_ELEMENT,
                'data': {
                    'name': 'test_metadata',
                    'display_name': 'Test Metadata',
                    'type': 'float'
                },
                'resource_loader': _load_metadata_element
            },
            # TODO: input/output/metadata categories
            {
                'type': AIModel,
                'parents': [Task],
                'endpoint': ENDPOINT_AI_MODEL,
                'data': {
                    'name': 'test_ai_model',
                    'version': '1.0.0'
                }
            },
            {
                'type': TaskFile,
                'parents': [Task],
                'endpoint': ENDPOINT_TASK_FILE,
                'data': {
                    'filename': 'test_file',
                    'size': 48,
                    'use_for': 'input',
                    'type': 'image'
                }
            },
            {
                'type': Example,
                'parents': [Task],
                'endpoint': ENDPOINT_EXAMPLE,
                'data': mock_element_values_json(task_id=1)
            },
            {
                'type': PredictionLog,
                'parents': [Task],
                'endpoint': ENDPOINT_AI_PREDICTION_LOG,
                'data': mock_prediction_log_json(task_id=1),
                'resource_loader': _load_prediction_log
            },
            {
                'type': Shape,
                'parents': [Task, Example],
                'endpoint': ENDPOINT_EXAMPLE_SHAPE,
                'data': mock_shape_or_slice_json(type_='shape', task_id=1)
            },
            {
                'type': Slice,
                'parents': [Task, Example],
                'endpoint': ENDPOINT_EXAMPLE_SLICE,
                'data': mock_shape_or_slice_json(type_='slice', task_id=1)
            },
            {
                'type': Tag,
                'parents': [Task],
                'endpoint': ENDPOINT_TAG,
                'data': {
                    'name': 'Test tag name',
                    'description': 'Test tag description',
                    'color': '0082FF'
                }
            },
        ]

    @staticmethod
    def _verify_method(client: MockClient, method: str, mock_client_id: str, session_user_id: str,
                       session_user_auth0_id: str):

        def _verify_resource_url(resource_info: dict, use_uuids: bool):
            restore_db(mock_client_id=mock_client_id,
                       session_user_id=session_user_id,
                       session_user_auth0_id=session_user_auth0_id)
            if isinstance(resource_info.get('resource_loader'), Callable):
                resource = resource_info['resource_loader']()
            else:
                resource = load_default_resource(resource_type=resource_info['type'],
                                                 parents_types=resource_info['parents'])
            # Set database
            if method == 'put' and isinstance(resource.db_object(), MutableEntity):
                cache.clear()
                resource.db_object().synced_by_users = [1]  # avoid out-of-sync error
                resource.persist()
            # Prepare request
            req_url = get_endpoint(parameterized_endpoint=resource_info['endpoint'],
                                   resource=resource,
                                   use_uuids=use_uuids)
            payload = resource_info['data'] if method in ['post', 'put'] else None
            # Make request and check returned resource location
            response = client.send_request(method=method, url=req_url, json=payload)
            assert response.status_code in [200, 201, 204, 405, 501]
            if 'Location' in response.headers:
                actual_res_url = urllib.parse.unquote(response.headers['Location'])
                expected_res_url = get_endpoint(parameterized_endpoint=resource_info['endpoint'],
                                                resource=resource,
                                                use_uuids=True)
                actual_res_url_levels = actual_res_url.split('/')
                expected_res_url_levels = expected_res_url.split('/')
                # Check parent resource's URL
                actual_res_parent_url = '/'.join(actual_res_url_levels[:-1])
                expected_res_parent_url = '/'.join(expected_res_url_levels[:-1])
                assert actual_res_parent_url == expected_res_parent_url
                # Check resource ID
                actual_rsrc_id = actual_res_url_levels[-1]
                expected_rsrc_id = expected_res_url_levels[-1]
                assert actual_rsrc_id == expected_rsrc_id

        method = method.lower()

        for resource_info in TestResourceID._resources():
            # Access the resource using its UUID and its public ID
            _verify_resource_url(resource_info, use_uuids=True)
            _verify_resource_url(resource_info, use_uuids=False)
            pass  # TODO: access the resource using a unique identifier (e.g. element name, category name, tag, etc.)

    def test_delete(self, client: MockClient, mock_s3, mock_client_id: str, session_user_id: str,
                    session_user_auth0_id: str):
        self._verify_method(client=client,
                            method='DELETE',
                            mock_client_id=mock_client_id,
                            session_user_id=session_user_id,
                            session_user_auth0_id=session_user_auth0_id)

    def test_get(self, client: MockClient, mock_s3, mock_client_id: str, session_user_id: str,
                 session_user_auth0_id: str):
        self._verify_method(client=client,
                            method='GET',
                            mock_client_id=mock_client_id,
                            session_user_id=session_user_id,
                            session_user_auth0_id=session_user_auth0_id)

    def test_put(self, client: MockClient, mock_s3, mock_client_id: str, session_user_id: str,
                 session_user_auth0_id: str):
        self._verify_method(client=client,
                            method='PUT',
                            mock_client_id=mock_client_id,
                            session_user_id=session_user_id,
                            session_user_auth0_id=session_user_auth0_id)


def _load_input_element() -> InputElement:
    task = load_default_resource(resource_type=Task)
    return task.input_elements()[0]


def _load_output_element() -> OutputElement:
    task = load_default_resource(resource_type=Task)
    return task.output_elements()[0]


def _load_metadata_element() -> MetadataElement:
    task = load_default_resource(resource_type=Task)
    return task.metadata_elements()[0]


def _load_prediction_log() -> PredictionLog:
    prediction_log = load_default_resource(resource_type=PredictionLog, parents_types=[Task])
    prediction_log.db_object().environment = AIEnvironment.TESTING
    db_commit_and_expire()
    return prediction_log
