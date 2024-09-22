import pytest

from nexusml.api.resources.tags import Tag
from nexusml.api.resources.tasks import Task
from nexusml.constants import ENDPOINT_TAG
from nexusml.constants import ENDPOINT_TAGS
from nexusml.constants import HTTP_GET_STATUS_CODE
from nexusml.constants import HTTP_METHOD_NOT_ALLOWED_STATUS_CODE
from nexusml.constants import HTTP_POST_STATUS_CODE
from nexusml.database.tags import TagDB
from tests.api.constants import BACKEND_JSON_FIELDS
from tests.api.integration.conftest import MockClient
from tests.api.integration.utils import get_endpoint
from tests.api.integration.utils import verify_resource_request
from tests.api.integration.utils import verify_response_json
from tests.api.integration.utils import verify_response_jsons
from tests.api.utils import get_json_from_db_object
from tests.api.utils import load_default_resource

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.tmp]  # TODO: remove "tmp" mark


class TestTags:

    def test_delete(self, client: MockClient):
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_TAGS,
                                    resource=load_default_resource(resource_type=Task))
        response = client.send_request(method='DELETE', url=endpoint_url)
        assert response.status_code == HTTP_METHOD_NOT_ALLOWED_STATUS_CODE

    def test_get(self, client: MockClient):
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_TAGS,
                                    resource=load_default_resource(resource_type=Task))
        response = client.send_request(method='GET', url=endpoint_url)
        assert response.status_code == HTTP_GET_STATUS_CODE
        expected_jsons = [_get_tag_json_from_db_object(db_object=tag) for tag in TagDB.filter_by_task(task_id=1)]
        verify_response_jsons(actual_jsons=response.json(), expected_jsons=expected_jsons)

    def test_post(self, client: MockClient):
        new_tag = {'name': 'New tag name', 'description': 'New tag description', 'color': '#2B72B5'}
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_TAGS,
                                    resource=load_default_resource(resource_type=Task))
        response = client.send_request(method='POST', url=endpoint_url, json=new_tag)
        assert response.status_code == HTTP_POST_STATUS_CODE
        verify_response_json(actual_json=response.json(), expected_json=new_tag, optional_fields=BACKEND_JSON_FIELDS)


class TestTag:

    @staticmethod
    def _test_tag_method(client: MockClient, method: str, request_json: dict = None):
        method = method.lower().strip()

        tag = load_default_resource(resource_type=Tag, parents_types=[Task])

        if method == 'delete':
            expected_jsons = None
        elif request_json:
            expected_json = dict(request_json)
            if not expected_json['color'].startswith('#'):
                expected_json['color'] = '#' + expected_json['color']
            expected_jsons = [expected_json]
        else:
            expected_jsons = [_get_tag_json_from_db_object(db_object=tag.db_object())]

        verify_resource_request(client=client,
                                method=method,
                                endpoint=ENDPOINT_TAG,
                                resource=tag,
                                request_json=request_json,
                                expected_jsons=expected_jsons)

        if request_json and 'color' in request_json and request_json['color']:
            db_color = tag.db_object().color
            assert len(db_color) == 6
            assert not db_color.startswith('#')

    def test_delete(self, client: MockClient):
        self._test_tag_method(client=client, method='DELETE')

    def test_get(self, client: MockClient):
        self._test_tag_method(client=client, method='GET')

    def test_put(self, client: MockClient):
        # Regular update without including "#" character in the color code
        json_data = {'name': 'Modified tag name', 'description': 'Modified tag description', 'color': '0082FF'}
        self._test_tag_method(client=client, method='PUT', request_json=json_data)
        # Regular update including "#" character in the color code
        json_data['color'] = '#B4C048'
        self._test_tag_method(client=client, method='PUT', request_json=json_data)
        # Provide an invalid color
        pass  # TODO


def _get_tag_json_from_db_object(db_object: TagDB):
    tag_json = get_json_from_db_object(db_object=db_object)
    if 'color' in tag_json and tag_json['color']:
        tag_json['color'] = '#' + tag_json['color']
    return tag_json
