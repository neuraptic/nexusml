from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait

import pytest

from nexusml.api.resources.tasks import Task
from nexusml.api.views.core import limiter
from nexusml.constants import ENDPOINT_EXAMPLES
from nexusml.constants import ENDPOINT_TASK
from nexusml.constants import HTTP_GET_STATUS_CODE
from nexusml.constants import HTTP_PAYLOAD_TOO_LARGE_STATUS_CODE
from nexusml.constants import HTTP_TOO_MANY_REQUESTS_STATUS_CODE
from tests.api.constants import TEST_CONFIG
from tests.api.integration.conftest import Backend
from tests.api.integration.conftest import MockClient
from tests.api.integration.test_api_keys import generate_api_key
from tests.api.integration.utils import get_endpoint
from tests.api.integration.utils import send_request
from tests.api.utils import load_default_resource

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestMaxPayload:

    def test_max_payload(self, client: MockClient):
        task = load_default_resource(resource_type=Task)
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_EXAMPLES, resource=task)
        max_payload = TEST_CONFIG['limits']['requests']['max_payload']
        random_payload = {'empty_strings': [''] * max_payload}
        response = client.send_request(method='POST', url=endpoint_url, json=random_payload)
        assert response.status_code == HTTP_PAYLOAD_TOO_LARGE_STATUS_CODE


class TestAPIRateLimits:

    def test_user_api_rate_limits(self, backend: Backend, client: MockClient):
        _test_agent_api_rate_limits(backend=backend, client=client, agent_type='user')

    def test_client_api_rate_limits(self, backend: Backend, client: MockClient):
        _test_agent_api_rate_limits(backend=backend, client=client, agent_type='client')


def _test_agent_api_rate_limits(backend: Backend, client: MockClient, agent_type: str):

    def make_concurrent_requests(total_requests: int, concurrent_requests: int) -> int:

        def make_request():
            # Note: Since this function is called from another thread,
            #       the context pushed by `_backend_app_context()` is not available.
            #       The function `generate_api_key()` needs to query `ClientDB`,
            #       so we need to enter the app context again.
            with backend.app.app_context():
                if agent_type == 'client':
                    return send_request(method='GET', url=endpoint_url, api_key=generate_api_key())
                else:
                    return client.send_request(method='GET', url=endpoint_url)

        task = load_default_resource(resource_type=Task)
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_TASK, resource=task)

        limiter.reset()

        futures = []
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            for _ in range(total_requests):
                futures.append(executor.submit(make_request))
            wait(futures)

        accepted_requests = 0
        for future in futures:
            response = future.result()
            if response.status_code == HTTP_GET_STATUS_CODE:
                accepted_requests += 1
            else:
                assert response.status_code == HTTP_TOO_MANY_REQUESTS_STATUS_CODE

        return accepted_requests

    assert agent_type in ['user', 'client']

    MAX_REQUESTS_PER_SECOND = 3
    MAX_REQUESTS_PER_MINUTE = 10

    backend.update_app_config(param_path=['limits', 'requests', 'requests_per_second'], value=MAX_REQUESTS_PER_SECOND)
    backend.update_app_config(param_path=['limits', 'requests', 'requests_per_minute'], value=MAX_REQUESTS_PER_MINUTE)
    ###########################
    # Max requests per second #
    ###########################
    accepted_requests = make_concurrent_requests(total_requests=MAX_REQUESTS_PER_SECOND + 1,
                                                 concurrent_requests=MAX_REQUESTS_PER_SECOND + 1)
    assert accepted_requests == MAX_REQUESTS_PER_SECOND
    ###########################
    # Max requests per minute #
    ###########################
    accepted_requests = make_concurrent_requests(total_requests=(MAX_REQUESTS_PER_MINUTE + 1),
                                                 concurrent_requests=MAX_REQUESTS_PER_SECOND)
    assert accepted_requests == MAX_REQUESTS_PER_MINUTE
