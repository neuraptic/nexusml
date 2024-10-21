""" Tests for endpoints exposing system information (config, feature flags, health, status, etc.). """
import os

import pytest
import requests

from nexusml.constants import ENDPOINT_SYS_CONFIG
from nexusml.constants import HTTP_GET_STATUS_CODE
from nexusml.constants import HTTP_NOT_FOUND_STATUS_CODE
from nexusml.env import ENV_WEB_CLIENT_ID
from tests.api.constants import TEST_CONFIG
from tests.api.integration.conftest import MockClient
from tests.api.integration.utils import get_endpoint


class TestAPIConfig:

    @pytest.mark.parametrize('custom_client', [(os.environ[ENV_WEB_CLIENT_ID], 'api_key')], indirect=True)
    def test_valid_request(self, custom_client: MockClient):
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_SYS_CONFIG)
        response = custom_client.send_request(method='GET', url=endpoint_url)
        assert response.status_code == HTTP_GET_STATUS_CODE
        assert response.json() == {
            'auth_enabled': TEST_CONFIG['general']['auth_enabled']
        }

    def test_invalid_request(self):
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_SYS_CONFIG)
        response = requests.get(endpoint_url)
        assert response.status_code == HTTP_NOT_FOUND_STATUS_CODE
