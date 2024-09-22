from abc import abstractmethod
import copy
from datetime import timedelta
from typing import Dict, List, Type, Union

import pytest

from nexusml.api.ext import cache
from nexusml.api.resources.examples import Comment
from nexusml.api.resources.examples import Example
from nexusml.api.resources.examples import Shape
from nexusml.api.resources.examples import Slice
from nexusml.api.resources.tasks import Task
from nexusml.constants import DATETIME_FORMAT
from nexusml.constants import ENDPOINT_EXAMPLE
from nexusml.constants import ENDPOINT_EXAMPLE_SHAPE
from nexusml.constants import ENDPOINT_EXAMPLE_SLICE
from nexusml.constants import ENDPOINT_EXAMPLES
from nexusml.constants import HTTP_BAD_REQUEST_STATUS_CODE
from nexusml.constants import HTTP_GET_STATUS_CODE
from nexusml.constants import HTTP_METHOD_NOT_ALLOWED_STATUS_CODE
from nexusml.constants import HTTP_POST_STATUS_CODE
from nexusml.constants import HTTP_PUT_STATUS_CODE
from nexusml.constants import HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
from nexusml.database.core import empty_table
from nexusml.database.core import save_to_db
from nexusml.database.examples import ex_tags
from nexusml.database.examples import ExampleDB
from nexusml.database.examples import ExCategory
from nexusml.database.tasks import CategoryDB
from nexusml.database.tasks import ElementDB
from nexusml.enums import ElementValueType
from nexusml.enums import FileStorageBackend
from nexusml.enums import LabelingStatus
from tests.api.conftest import restore_db
from tests.api.integration.conftest import MockClient
from tests.api.integration.utils import get_endpoint
from tests.api.integration.utils import mock_element_values_json
from tests.api.integration.utils import mock_shape_or_slice_json
from tests.api.integration.utils import verify_example_or_prediction_deletion
from tests.api.integration.utils import verify_out_of_sync
from tests.api.integration.utils import verify_quota_error
from tests.api.integration.utils import verify_resource_request
from tests.api.integration.utils import verify_response_examples_or_prediction_logs
from tests.api.integration.utils import verify_wrong_element_values
from tests.api.utils import assert_same_element_values
from tests.api.utils import db_commit_and_expire
from tests.api.utils import get_json_from_db_object
from tests.api.utils import get_shape_or_slice_json_from_db_object
from tests.api.utils import load_default_resource
from tests.api.utils import set_quota_limit
from tests.api.utils import set_quota_usage
from tests.api.utils import verify_quota_usage

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestExamples:

    def test_delete(self, client: MockClient):
        parent_task = load_default_resource(resource_type=Task)
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_EXAMPLES, resource=parent_task)
        response = client.send_request(method='DELETE', url=endpoint_url)
        assert response.status_code == HTTP_METHOD_NOT_ALLOWED_STATUS_CODE

    def test_get(self, client: MockClient):

        def _element_value_filter(element_id: int, examples: List[ExampleDB], filtered_values: Dict[int, object],
                                  unfiltered_value: object):

            filtered_examples = [x for x in examples if x.example_id in filtered_values]

            # Set database
            # TODO: how to handle regex?
            element = ElementDB.get(element_id=element_id)
            element_value_model = ExampleDB.value_type_models()[element.value_type]
            example_values = element_value_model.query().filter_by(element_id=element_id).all()
            for example_value in example_values:
                # If it's a categorical value, convert the provided category name into a category ID
                if element_value_model == ExCategory:
                    filtered_cats = {
                        x: CategoryDB.get_from_id(id_value=x, parent=element).category_id
                        for x in filtered_values.values()
                    }
                    unfiltered_cat = CategoryDB.get_from_id(id_value=unfiltered_value, parent=element).category_id
                    if example_value.example_id in filtered_values:
                        example_value.value = filtered_cats[filtered_values[example_value.example_id]]
                    else:
                        example_value.value = unfiltered_cat
                # Otherwise, use the provided value as is
                else:
                    if example_value.example_id in filtered_values:
                        example_value.value = filtered_values[example_value.example_id]
                    else:
                        example_value.value = unfiltered_value
            save_to_db(example_values)

            # Make request and verify response
            url_query = f'?{element.name}={"|".join({str(x) for x in filtered_values.values()})}'
            response = client.send_request(method='GET', url=endpoint_url + url_query)
            assert response.status_code == HTTP_GET_STATUS_CODE
            verify_response_examples_or_prediction_logs(actual_jsons=response.json()['data'],
                                                        expected_db_objects=filtered_examples)

        parent_task = load_default_resource(resource_type=Task)
        examples = ExampleDB.filter_by_task(task_id=parent_task.db_object().task_id)
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_EXAMPLES, resource=parent_task)
        ####################
        # GET ALL EXAMPLES #
        ####################
        response = client.send_request(method='GET', url=endpoint_url)
        assert response.status_code == HTTP_GET_STATUS_CODE
        verify_response_examples_or_prediction_logs(actual_jsons=response.json()['data'], expected_db_objects=examples)
        ############################
        # FILTER BY ELEMENT VALUES #
        ############################
        # `input_1=true`
        _element_value_filter(element_id=1,
                              examples=examples,
                              filtered_values={
                                  1: True,
                                  3: True,
                                  4: True
                              },
                              unfiltered_value=False)
        # `input_2=1|4|12`
        _element_value_filter(element_id=2, examples=examples, filtered_values={2: 1, 3: 4, 5: 12}, unfiltered_value=10)

        # `input_3=3.54`
        _element_value_filter(element_id=3, examples=examples, filtered_values={5: 3.54}, unfiltered_value=0.68)

        # `input_4=return_this_example|return_this_too`
        _element_value_filter(element_id=4,
                              examples=examples,
                              filtered_values={
                                  4: 'return_this_example',
                                  8: 'return_this_too'
                              },
                              unfiltered_value='do_not_return_this_example')

        # `output_3=Output Category 1|Output Category 3`
        _element_value_filter(element_id=15,
                              examples=examples,
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
        for idx, example in enumerate(examples):
            example.created_at += timedelta(days=30 * idx)
            example.modified_at += timedelta(days=35 * idx)
            example.activity_at += timedelta(days=40 * idx)
        save_to_db(examples)

        # Filter by creation, modification, and activity datetimes
        for query_param in ['created_at', 'modified_at', 'activity_at']:
            example_datetimes = [getattr(x, query_param).strftime(DATETIME_FORMAT) for x in examples]
            # Exact datetime
            url_query = f'{query_param}={example_datetimes[2]}'
            response = client.send_request(method='GET', url=endpoint_url + '?' + url_query)
            expected_examples = [examples[2]]
            verify_response_examples_or_prediction_logs(actual_jsons=response.json()['data'],
                                                        expected_db_objects=expected_examples)

            # After a datetime
            url_query = f'{query_param}[min]={example_datetimes[2]}'
            response = client.send_request(method='GET', url=endpoint_url + '?' + url_query)
            verify_response_examples_or_prediction_logs(actual_jsons=response.json()['data'],
                                                        expected_db_objects=examples[2:])

            # Before a datetime
            url_query = f'{query_param}[max]={example_datetimes[2]}'
            response = client.send_request(method='GET', url=endpoint_url + '?' + url_query)
            verify_response_examples_or_prediction_logs(actual_jsons=response.json()['data'],
                                                        expected_db_objects=examples[:3])

            # Datetime interval
            url_query = ((f'{query_param}[min]={example_datetimes[1]}') +
                         (f'&{query_param}[max]={example_datetimes[3]}'))
            response = client.send_request(method='GET', url=endpoint_url + '?' + url_query)
            verify_response_examples_or_prediction_logs(actual_jsons=response.json()['data'],
                                                        expected_db_objects=examples[1:4])
        #############################
        # FILTER BY LABELING STATUS #
        #############################
        # Set database
        for example in examples:
            example.labeling_status = LabelingStatus.UNLABELED
        examples[1].labeling_status = LabelingStatus.REJECTED
        examples[3].labeling_status = LabelingStatus.PENDING_REVIEW
        save_to_db(examples)

        # Filter by labeling status
        response = client.send_request(method='GET', url=endpoint_url + '?labeling_status=rejected|pending_review')
        expected_examples = [examples[1], examples[3]]
        verify_response_examples_or_prediction_logs(actual_jsons=response.json()['data'],
                                                    expected_db_objects=expected_examples)

        # An example can only have one labeling status
        response = client.send_request(method='GET', url=endpoint_url + '?labeling_status=rejected,pending_review')
        assert response.status_code == HTTP_BAD_REQUEST_STATUS_CODE
        #################################
        # FILTER EXAMPLES WITH COMMENTS #
        #################################
        response = client.send_request(method='GET', url=endpoint_url + '?only_with_comments=true')
        expected_examples = [examples[0], examples[1], examples[3], examples[4]]
        verify_response_examples_or_prediction_logs(actual_jsons=response.json()['data'],
                                                    expected_db_objects=expected_examples)
        ##################
        # FILTER BY TAGS #
        ##################
        # `tag=Tag 3`
        response = client.send_request(method='GET', url=endpoint_url + '?tag=Tag 3')
        expected_examples = [examples[0], examples[1], examples[4]]
        verify_response_examples_or_prediction_logs(actual_jsons=response.json()['data'],
                                                    expected_db_objects=expected_examples)

        # `tag=Tag 5|Tag 6`
        response = client.send_request(method='GET', url=endpoint_url + '?tag=Tag 5|Tag 6')
        expected_examples = [examples[0], examples[2], examples[4], examples[5]]
        verify_response_examples_or_prediction_logs(actual_jsons=response.json()['data'],
                                                    expected_db_objects=expected_examples)

        # `tag=Tag 2,Tag 5`
        # NOTE: This test is commented out due to a known bug with tag filtering in the API,
        # where filtering by multiple tags simulating an "AND" operation does not work as expected.
        # TODO: Uncomment this test when the bug is fixed.
        # response = client.send_request(method='GET', url=endpoint_url + '?tag=Tag 2,Tag 5')
        # expected_examples = [examples[2]]
        # verify_response_examples_or_prediction_logs(actual_jsons=response.json()['data'],
        #                                             expected_db_objects=expected_examples)
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

    def test_post(self, client: MockClient):
        task = load_default_resource(resource_type=Task)
        single_json = {'batch': [mock_element_values_json(task_id=1)]}
        batch_json = {'batch': [mock_element_values_json(task_id=1)] * 5}
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_EXAMPLES, resource=task)
        #######################
        # Test regular upload #
        #######################
        empty_table(ExampleDB)
        response = client.send_request(method='POST', url=endpoint_url, json=batch_json)
        assert response.status_code == HTTP_POST_STATUS_CODE
        db_commit_and_expire()
        res_jsons = response.json()['batch']
        expected_examples = [ExampleDB.get_from_uuid(x['uuid']) for x in res_jsons]
        verify_response_examples_or_prediction_logs(actual_jsons=res_jsons, expected_db_objects=expected_examples)
        #######################################
        # Try to provide wrong element values #
        #######################################
        verify_wrong_element_values(client=client,
                                    resource_type=Example,
                                    request_method='POST',
                                    resource_endpoint=ENDPOINT_EXAMPLES,
                                    mock_example_or_prediction_jsons=single_json)
        ###############################################################
        # Try to exceed the maximum number of examples (batch upload) #
        ###############################################################
        restore_db()
        task = load_default_resource(resource_type=Task)
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_EXAMPLES, resource=task)
        max_examples = 3
        set_quota_usage(db_object=task.db_object(), quota='examples', usage=0)
        set_quota_limit(db_object=task.db_object(), quota='examples', limit=max_examples)
        cache.clear()  # force `resources.tasks.Task.check_quota_usage()` to use current database value
        response = client.send_request(method='POST', url=endpoint_url, json=batch_json)
        assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
        assert response.json()['error']['message'] == f'Maximum number of examples ({max_examples}) exceeded'
        verify_quota_usage(db_object=task.db_object(), quota='examples', expected_usage=0)
        ######################################################################
        # Try to exceed the maximum number of examples (concurrent requests) #
        ######################################################################
        restore_db()
        empty_table(ExampleDB)

        task = load_default_resource(resource_type=Task)

        max_examples = 3

        set_quota_usage(db_object=task.db_object(), quota='examples', usage=0)
        set_quota_limit(db_object=task.db_object(), quota='examples', limit=max_examples)

        verify_quota_error(client=client,
                           endpoint_url=get_endpoint(parameterized_endpoint=ENDPOINT_EXAMPLES, resource=task),
                           request_json=single_json,
                           num_requests=5,
                           err_msg=f'Maximum number of examples ({max_examples}) exceeded')

        verify_quota_usage(db_object=task.db_object(), quota='examples', expected_usage=max_examples)
        ##################################################
        # Try to exceed space quota limit (batch upload) #
        ##################################################
        restore_db()
        task = load_default_resource(resource_type=Task)
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_EXAMPLES, resource=task)
        set_quota_usage(db_object=task.db_object(), quota='space', usage=0)
        # TODO: set a known quota to ensure that only X examples are uploaded
        set_quota_limit(db_object=task.db_object(), quota='space', limit=3072)
        cache.clear()  # force `resources.tasks.Task.check_quota_usage()` to use current database value
        response = client.send_request(method='POST', url=endpoint_url, json=batch_json)
        assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
        assert response.json()['error']['message'] == 'Space quota limit (0.00 MB) exceeded'
        verify_quota_usage(db_object=task.db_object(), quota='space', expected_usage=0)
        #########################################################
        # Try to exceed space quota limit (concurrent requests) #
        #########################################################
        restore_db()

        task = load_default_resource(resource_type=Task)

        set_quota_usage(db_object=task.db_object(), quota='space', usage=0)
        # TODO: set a known quota to ensure that only X examples are uploaded
        set_quota_limit(db_object=task.db_object(), quota='space', limit=3072)

        verify_quota_error(client=client,
                           endpoint_url=get_endpoint(parameterized_endpoint=ENDPOINT_EXAMPLES, resource=task),
                           request_json=single_json,
                           num_requests=5,
                           err_msg='Space quota limit (0.00 MB) exceeded')

        pass  # TODO: check space


class TestExample:

    @pytest.mark.parametrize('file_storage_backend', [FileStorageBackend.LOCAL, FileStorageBackend.S3])
    def test_delete(self, client: MockClient, mock_s3, file_storage_backend: FileStorageBackend):
        verify_example_or_prediction_deletion(client=client,
                                              resource_type=Example,
                                              file_storage_backend=file_storage_backend)

    def test_get(self, client: MockClient, mock_s3):
        example = load_default_resource(resource_type=Example, parents_types=[Task])
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_EXAMPLE, resource=example)
        response = client.send_request(method='GET', url=endpoint_url)
        assert response.status_code == HTTP_GET_STATUS_CODE
        verify_response_examples_or_prediction_logs(actual_jsons=[response.json()],
                                                    expected_db_objects=[example.db_object()])

    def test_put(self, client: MockClient, mock_s3):
        # Get resources and endpoint
        example = load_default_resource(resource_type=Example, parents_types=[Task])
        request_json = mock_element_values_json(task_id=1)
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_EXAMPLE, resource=example)

        # Set database
        example.db_object().size = 30
        example.db_object().trained = True
        example.db_object().synced_by_users = [1]
        save_to_db(example.db_object())
        #######################
        # Test regular update #
        #######################
        response = client.send_request(method='PUT', url=endpoint_url, json=request_json)
        assert response.status_code == HTTP_PUT_STATUS_CODE
        verify_response_examples_or_prediction_logs(actual_jsons=[response.json()],
                                                    expected_db_objects=[example.db_object()])
        db_commit_and_expire()
        assert not example.db_object().trained
        ##################################################################################
        # Test reference-based element value assignments (files, categories, and shapes) #
        ##################################################################################
        pass  # TODO
        ################
        # Test tagging #
        ################
        empty_table(ex_tags)
        tagging_json = copy.deepcopy(request_json)
        tagging_json['tags'] = ['Tag 1', 'Tag 3']
        response = client.send_request(method='PUT', url=endpoint_url, json=tagging_json)
        assert response.status_code == HTTP_PUT_STATUS_CODE
        verify_response_examples_or_prediction_logs(actual_jsons=[response.json()],
                                                    expected_db_objects=[example.db_object()])
        #######################################
        # Try to provide wrong element values #
        #######################################
        verify_wrong_element_values(client=client,
                                    resource_type=Example,
                                    request_method='PUT',
                                    resource_endpoint=ENDPOINT_EXAMPLE,
                                    mock_example_or_prediction_jsons=request_json)
        ###################################
        # Try to exceed space quota limit #
        ###################################
        example = load_default_resource(resource_type=Example, parents_types=[Task])
        task = example.parents()[0]
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_EXAMPLE, resource=example)

        cache.clear()  # force `resources.tasks.Task.check_quota_usage()` to use current database value

        set_quota_limit(db_object=task.db_object(), quota='space', limit=10240)  # 10KB

        example.db_object().force_relationship_loading()

        old_example_db_object = copy.deepcopy(example.db_object())

        big_example_json = mock_element_values_json(task_id=task.db_object().task_id)
        for element_value in big_example_json['values']:
            element = ElementDB.get_from_id(id_value=element_value['element'], parent=task.db_object())
            if element.value_type != ElementValueType.TEXT:
                continue
            element_value['value'] = '-' * (task.db_object().space_limit + 1)

        response = client.send_request(method='PUT', url=endpoint_url, json=big_example_json)
        assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
        response_text = response.text.lower()
        assert 'exceeded' in response_text or 'reached' in response_text
        assert_same_element_values(db_object_1=old_example_db_object,
                                   db_object_2=ExampleDB.get(example_id=example.db_object().example_id))
        ##############################
        # Force an out-of-sync state #
        ##############################
        example.db_object().synced_by_users = []
        save_to_db(example.db_object())
        verify_out_of_sync(client=client, endpoint=ENDPOINT_EXAMPLE, resource=example, request_json=request_json)


class TestComments:

    @staticmethod
    def _test_comments_method(client: MockClient, method: str):
        method = method.strip().lower()
        example = load_default_resource(resource_type=Example, parents_types=[Task])

        if method == 'get':
            comment_db_objects = example.db_object().comments
            assert len(comment_db_objects) > 0
            request_json = None
            expected_jsons = [get_json_from_db_object(db_object=x) for x in comment_db_objects]
            expected_status_code = HTTP_GET_STATUS_CODE
        elif method == 'post':
            request_json = {'message': 'New example comment'}
            expected_jsons = [request_json]
            expected_status_code = HTTP_POST_STATUS_CODE
        else:
            request_json = None
            expected_jsons = None
            expected_status_code = HTTP_METHOD_NOT_ALLOWED_STATUS_CODE

        verify_resource_request(client=client,
                                method=method,
                                endpoint=ENDPOINT_EXAMPLE,
                                resource=example,
                                collection=('comments', Comment),
                                request_json=request_json,
                                expected_jsons=expected_jsons,
                                expected_status_code=expected_status_code,
                                check_resource_location=False)  # no endpoint for individual comment

    def test_delete(self, client: MockClient):
        self._test_comments_method(client=client, method='DELETE')

    def test_get(self, client: MockClient):
        self._test_comments_method(client=client, method='GET')

    def test_post(self, client: MockClient):
        self._test_comments_method(client=client, method='POST')


class _TestShapesOrSlices:

    @classmethod
    @abstractmethod
    def resource_type(cls) -> Type[Union[Shape, Slice]]:
        raise NotImplementedError()

    @classmethod
    def _test_method(cls, client: MockClient, method: str):
        method = method.strip().lower()

        resource_type_name = 'shape' if cls.resource_type() == Shape else 'slice'
        collection_name = 'shapes' if cls.resource_type() == Shape else 'slices'

        example = load_default_resource(resource_type=Example, parents_types=[Task])

        if method == 'get':
            db_objects = getattr(example.db_object(), collection_name)
            assert len(db_objects) > 0
            request_json = None
            expected_jsons = [get_shape_or_slice_json_from_db_object(db_object=x) for x in db_objects]
        elif method == 'post':
            request_json = mock_shape_or_slice_json(type_=resource_type_name, task_id=example.db_object().task_id)
            expected_jsons = [request_json]
        else:
            request_json = None
            expected_jsons = None

        if method != 'get':
            example.db_object().trained = True
            save_to_db(example.db_object())

        verify_resource_request(client=client,
                                method=method,
                                endpoint=ENDPOINT_EXAMPLE,
                                resource=example,
                                collection=(collection_name, cls.resource_type()),
                                request_json=request_json,
                                expected_jsons=expected_jsons)

        if method != 'get':
            db_commit_and_expire()
            assert not example.db_object().trained

    def test_delete(self, client: MockClient):
        self._test_method(client=client, method='DELETE')

    def test_get(self, client: MockClient):
        self._test_method(client=client, method='GET')

    def test_post(self, client: MockClient):
        # Regular upload
        self._test_method(client=client, method='POST')
        # Test output propagation
        pass  # TODO


class TestShapes(_TestShapesOrSlices):

    @classmethod
    @abstractmethod
    def resource_type(cls) -> Type[Shape]:
        return Shape


class TestSlices(_TestShapesOrSlices):

    @classmethod
    @abstractmethod
    def resource_type(cls) -> Type[Slice]:
        return Slice


class _TestShapeOrSlice:

    @classmethod
    @abstractmethod
    def resource_type(cls) -> Type[Union[Shape, Slice]]:
        raise NotImplementedError()

    @classmethod
    def _test_method(cls, client: MockClient, method: str):
        method = method.strip().lower()

        resource = load_default_resource(resource_type=cls.resource_type(), parents_types=[Task, Example])
        resource_type_name = 'shape' if cls.resource_type() == Shape else 'slice'
        endpoint = ENDPOINT_EXAMPLE_SHAPE if cls.resource_type() == Shape else ENDPOINT_EXAMPLE_SLICE

        example = resource.parents()[-1]

        if method == 'get':
            request_json = None
            expected_jsons = [get_shape_or_slice_json_from_db_object(db_object=resource.db_object())]
        elif method == 'put':
            request_json = mock_shape_or_slice_json(type_=resource_type_name,
                                                    task_id=resource.db_object().example.task_id)
            expected_jsons = [request_json]
        else:
            request_json = None
            expected_jsons = None

        if method != 'get':
            example.db_object().trained = True
            save_to_db(example.db_object())

        verify_resource_request(client=client,
                                method=method,
                                endpoint=endpoint,
                                resource=resource,
                                request_json=request_json,
                                expected_jsons=expected_jsons)

        if method != 'get':
            db_commit_and_expire()
            assert not example.db_object().trained

    def test_delete(self, client: MockClient):
        self._test_method(client=client, method='DELETE')

    def test_get(self, client: MockClient):
        self._test_method(client=client, method='GET')

    def test_put(self, client: MockClient):
        float_collection = {Shape: 'shape_floats', Slice: 'slice_floats'}
        category_collection = {Shape: 'shape_categories', Slice: 'slice_categories'}

        resource_type_name = 'shape' if self.resource_type() == Shape else 'slice'
        endpoint = ENDPOINT_EXAMPLE_SHAPE if self.resource_type() == Shape else ENDPOINT_EXAMPLE_SLICE

        # Regular upload
        self._test_method(client=client, method='PUT')

        # Remove outputs
        db_commit_and_expire()
        resource = load_default_resource(resource_type=self.resource_type(), parents_types=[Task, Example])
        assert getattr(resource.db_object(), float_collection[self.resource_type()])
        assert getattr(resource.db_object(), category_collection[self.resource_type()])
        request_json = mock_shape_or_slice_json(type_=resource_type_name, task_id=resource.db_object().example.task_id)
        request_json['outputs'] = []
        verify_resource_request(client=client,
                                method='PUT',
                                endpoint=endpoint,
                                resource=resource,
                                request_json=request_json,
                                expected_jsons=[request_json])
        db_commit_and_expire()
        assert not getattr(resource.db_object(), float_collection[self.resource_type()])
        assert not getattr(resource.db_object(), category_collection[self.resource_type()])

        # Test output propagation
        pass  # TODO


class TestShape(_TestShapeOrSlice):

    @classmethod
    @abstractmethod
    def resource_type(cls) -> Type[Shape]:
        return Shape


class TestSlice(_TestShapeOrSlice):

    @classmethod
    @abstractmethod
    def resource_type(cls) -> Type[Slice]:
        return Slice
