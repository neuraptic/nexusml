"""
This module contains integration tests for the files endpoints.

Note: For the moment, there is only one file use in organizations ("picture").
      Since the maximum picture size is smaller than the maximum file size for a single upload,
      multipart uploads never take place in organization files.
"""
# TODO: We haven't been able to make the local storage backend tests work.
#       Check `MockClient._set_file_data_for_test_client()`.
# TODO: Uncomment `@pytest.mark.parametrize` code snippets when the issue is resolved.

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from datetime import timedelta
import math
import os
import shutil
import tempfile
from typing import Iterable, List, Optional, Type, Union
import uuid

import boto3
from flask import Flask
from flask import Response
from flask_apispec import FlaskApiSpec
import numpy as np
from PIL import Image
import pytest
import requests

from nexusml.api import create_app
from nexusml.api.endpoints import ENDPOINT_ORG_FILE
from nexusml.api.endpoints import ENDPOINT_ORG_FILES
from nexusml.api.endpoints import ENDPOINT_ORG_LOCAL_FILE_STORE_DOWNLOAD
from nexusml.api.endpoints import ENDPOINT_ORG_LOCAL_FILE_STORE_MULTIPART_UPLOAD
from nexusml.api.endpoints import ENDPOINT_ORG_LOCAL_FILE_STORE_UPLOAD
from nexusml.api.endpoints import ENDPOINT_TASK_FILE
from nexusml.api.endpoints import ENDPOINT_TASK_FILE_PARTS
from nexusml.api.endpoints import ENDPOINT_TASK_FILE_PARTS_COMPLETION
from nexusml.api.endpoints import ENDPOINT_TASK_FILES
from nexusml.api.endpoints import ENDPOINT_TASK_LOCAL_FILE_STORE_DOWNLOAD
from nexusml.api.endpoints import ENDPOINT_TASK_LOCAL_FILE_STORE_MULTIPART_UPLOAD
from nexusml.api.endpoints import ENDPOINT_TASK_LOCAL_FILE_STORE_UPLOAD
from nexusml.api.ext import cache
from nexusml.api.resources.files import OrgFile
from nexusml.api.resources.files import TaskFile
from nexusml.api.resources.organizations import Organization
from nexusml.api.resources.tasks import Task
from nexusml.api.utils import get_local_file_storage_config
from nexusml.api.utils import NexusMLConfig
from nexusml.constants import DATETIME_FORMAT
from nexusml.constants import HTTP_GET_STATUS_CODE
from nexusml.constants import HTTP_METHOD_NOT_ALLOWED_STATUS_CODE
from nexusml.constants import HTTP_NOT_FOUND_STATUS_CODE
from nexusml.constants import HTTP_POST_STATUS_CODE
from nexusml.constants import HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
from nexusml.constants import PREFIX_THUMBNAILS
from nexusml.database.core import db
from nexusml.database.core import empty_table
from nexusml.database.core import save_to_db
from nexusml.database.files import OrgFileDB
from nexusml.database.files import OrgUpload
from nexusml.database.files import TaskFileDB
from nexusml.database.files import TaskUpload
from nexusml.database.subscriptions import get_active_subscription
from nexusml.enums import FileStorageBackend
from nexusml.enums import FileType
from nexusml.enums import OrgFileUse
from nexusml.enums import TaskFileUse
from nexusml.utils import get_s3_config
from tests.api.conftest import restore_db
from tests.api.constants import CLIENT_MAX_THREADS
from tests.api.constants import TEST_CONFIG
from tests.api.integration.conftest import FlaskClient
from tests.api.integration.conftest import MockClient
from tests.api.integration.utils import get_endpoint
from tests.api.integration.utils import verify_resource_request
from tests.api.integration.utils import verify_response_jsons
from tests.api.utils import db_commit_and_expire
from tests.api.utils import get_files_in_s3
from tests.api.utils import get_json_from_db_object
from tests.api.utils import load_default_resource
from tests.api.utils import set_quota_limit
from tests.api.utils import set_quota_usage
from tests.api.utils import verify_quota_usage

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_OrgOrTask = Union[Organization, Task]
_OrgOrTaskFile = Union[OrgFile, TaskFile]
_OrgOrTaskUpload = Union[OrgUpload, TaskUpload]

_FILE_OPTIONAL_FIELDS = {'download_url', 'format'}

###############################################
# FIXTURES FOR THE LOCAL FILE STORAGE BACKEND #
###############################################


@contextmanager
def _local_storage_backend(monkeypatch, create_root_path: bool = False) -> Flask:
    #########
    # SETUP #
    #########
    # Create a copy of the default test config values
    test_config = deepcopy(TEST_CONFIG)

    # Set the local storage backend
    test_config['storage']['files']['backend'] = FileStorageBackend.LOCAL.name.lower()

    # Create a new config object
    config = NexusMLConfig()

    # Monkeypatch the global config object `nexusml.api.utils.config`.
    # TODO: We should avoid using a global config object.
    # Note: We also monkeypatch the imported copies used by the following functions:
    #       - `nexusml.api.create_app()`
    monkeypatch.setattr('nexusml.api.utils.config', config)
    monkeypatch.setattr('nexusml.api.config', config)

    # Monkeypatch `nexusml.api.ext.docs`.
    # TODO: Is this a sign of a bad design?
    # Note: We also monkeypatch the imported copies used by the following functions:
    #       - `nexusml.api.create_app()`
    docs = FlaskApiSpec()
    monkeypatch.setattr('nexusml.api.ext.docs', docs)
    monkeypatch.setattr('nexusml.api.docs', docs)

    # Monkeypatch the `nexusml.api.utils.get_file_storage_backend()` function to ensure it uses the patched config.
    # Note: We also monkeypatch the imported copies used by the following functions:
    #       - `nexusml.api._add_docs()`
    #       - `nexusml.api.routes.register_files_endpoints()`
    def patched_get_file_storage_backend():
        return FileStorageBackend.LOCAL

    monkeypatch.setattr('nexusml.api.utils.get_file_storage_backend', patched_get_file_storage_backend)
    monkeypatch.setattr('nexusml.api.get_file_storage_backend', patched_get_file_storage_backend)
    monkeypatch.setattr('nexusml.api.routes.get_file_storage_backend', patched_get_file_storage_backend)

    # Create the Flask app and set test config
    app = create_app()
    config.set(test_config)

    # Set the custom client class
    app.test_client_class = FlaskClient

    # Clean the root path
    root_path = get_local_file_storage_config()['root_path']
    shutil.rmtree(root_path, ignore_errors=True)
    if create_root_path:
        os.makedirs(root_path)

    #############
    # OPERATION #
    #############
    try:
        yield app
    finally:
        ###########
        # CLEANUP #
        ###########
        # Delete the root path
        shutil.rmtree(root_path, ignore_errors=True)


def _create_local_store_test_client(monkeypatch, create_root_path: bool) -> MockClient:
    with _local_storage_backend(monkeypatch, create_root_path=create_root_path) as app:
        client = MockClient(app=app,
                            client_id=str(uuid.uuid4()),
                            session_user_id=str(uuid.uuid4()),
                            session_user_auth0_id=str(uuid.uuid4()))
        client._always_use_test_client = True
        yield client


# TODO: Uncomment when the issues with the local storage backend tests are resolved.
# @pytest.fixture()
# def local_store_download_test_client(monkeypatch) -> MockClient:
#     yield from _create_local_store_test_client(monkeypatch, create_root_path=True)


# TODO: Remove when the issues with the local storage backend tests are resolved.
@pytest.fixture()
def local_store_download_test_client(monkeypatch, client) -> MockClient:
    return client


# TODO: Uncomment when the issues with the local storage backend tests are resolved.
# @pytest.fixture()
# def local_store_upload_test_client(monkeypatch) -> MockClient:
#     yield from _create_local_store_test_client(monkeypatch, create_root_path=False)


# TODO: Remove when the issues with the local storage backend tests are resolved.
@pytest.fixture()
def local_store_upload_test_client(monkeypatch, client) -> MockClient:
    return client


###################
# TESTS FOR VIEWS #
###################


class TestFilesView:
    # @pytest.mark.parametrize(
    #     'parent_type, storage_backend',
    #     [
    #         (Organization, FileStorageBackend.LOCAL),
    #         (Organization, FileStorageBackend.S3),
    #         (Task, FileStorageBackend.LOCAL),
    #         (Task, FileStorageBackend.S3)
    #     ]
    # )
    @pytest.mark.parametrize('parent_type', [Organization, Task])
    def test_delete(
        self,
        mock_s3,
        client: MockClient,
        local_store_download_test_client: MockClient,
        parent_type: Type[_OrgOrTask],
        # storage_backend: FileStorageBackend
    ):
        storage_backend = FileStorageBackend.S3  # TODO: Remove this line when the local storage backend tests are fixed
        if storage_backend == FileStorageBackend.LOCAL:
            client = local_store_download_test_client

        parent: _OrgOrTask = load_default_resource(resource_type=parent_type)

        endpoint = ENDPOINT_TASK_FILES if parent_type == Task else ENDPOINT_ORG_FILES
        endpoint_url = get_endpoint(parameterized_endpoint=endpoint, resource=parent)

        response = client.send_request(method='DELETE', url=endpoint_url)
        assert response.status_code == HTTP_METHOD_NOT_ALLOWED_STATUS_CODE

    # @pytest.mark.parametrize(
    #     'parent_type, storage_backend',
    #     [
    #         (Organization, FileStorageBackend.LOCAL),
    #         (Organization, FileStorageBackend.S3),
    #         (Task, FileStorageBackend.LOCAL),
    #         (Task, FileStorageBackend.S3)
    #     ]
    # )
    @pytest.mark.parametrize('parent_type', [Organization, Task])
    def test_get(
        self,
        mock_s3,
        client: MockClient,
        local_store_download_test_client: MockClient,
        parent_type: Type[_OrgOrTask],
        # storage_backend: FileStorageBackend
    ):
        storage_backend = FileStorageBackend.S3  # TODO: Remove this line when the local storage backend tests are fixed
        if storage_backend == FileStorageBackend.LOCAL:
            client = local_store_download_test_client

        def _verify_request_response(endpoint_url: str,
                                     url_query: str = None,
                                     expected_jsons: Iterable[dict] = None) -> List[dict]:

            responses_jsons = []

            default_items_per_page = TEST_CONFIG['views']['default_items_per_page']
            page = 1
            res_jsons = [None] * default_items_per_page

            while len(res_jsons) == default_items_per_page:
                # Make request
                url_ = endpoint_url + f'?page={page}{(("&" + url_query) if url_query else "")}'
                response = client.send_request(method='GET', url=url_)
                # Get response JSONs
                assert response.status_code in [HTTP_GET_STATUS_CODE, HTTP_NOT_FOUND_STATUS_CODE]
                if response.status_code == HTTP_GET_STATUS_CODE:
                    res_jsons = response.json().get('data', [])
                else:
                    res_jsons = []  # empty page
                responses_jsons += res_jsons
                # Increment page
                page += 1

            if expected_jsons:
                verify_response_jsons(actual_jsons=responses_jsons,
                                      expected_jsons=expected_jsons,
                                      optional_fields=_FILE_OPTIONAL_FIELDS)

            return responses_jsons

        parent: _OrgOrTask = load_default_resource(resource_type=parent_type)
        file_type = OrgFile if parent_type == Organization else TaskFile

        endpoint = ENDPOINT_TASK_FILES if parent_type == Task else ENDPOINT_ORG_FILES
        endpoint_url = get_endpoint(parameterized_endpoint=endpoint, resource=parent)

        # Upload files
        _post_files_metadata_and_content(files_metadata=_get_files_metadata(parent),
                                         parent=parent,
                                         client=client,
                                         storage_backend=storage_backend)

        #################
        # GET ALL FILES #
        #################
        files = _get_files_metadata(parent=parent)
        expected_jsons = [_get_file_metadata_json(db_object=x) for x in files]
        response_jsons = _verify_request_response(endpoint_url=endpoint_url, expected_jsons=expected_jsons)
        for file_json in response_jsons:
            _verify_file(file_json=file_json,
                         file_type=file_type,
                         is_thumbnail=False,
                         parent=parent,
                         storage_backend=storage_backend)

        #############################
        # GET ALL FILES' THUMBNAILS #
        #############################
        response_jsons = _verify_request_response(endpoint_url=endpoint_url, url_query='thumbnail=true')
        for file_json in response_jsons:
            _verify_file(file_json=file_json,
                         file_type=file_type,
                         is_thumbnail=True,
                         parent=parent,
                         storage_backend=storage_backend)

        ###############################
        # FILTER BY CREATION DATETIME #
        ###############################
        query_param = 'created_at'

        # Set database
        files = _get_files_metadata(parent=parent)
        for idx, file in enumerate(files):
            file.created_at += timedelta(days=30 * idx)
        save_to_db(files)
        file_jsons = [_get_file_metadata_json(db_object=file) for file in files]

        # Exact datetime
        url_query = f'{query_param}={file_jsons[2][query_param]}'
        _verify_request_response(endpoint_url=endpoint_url, url_query=url_query, expected_jsons=[file_jsons[2]])

        # After a datetime
        url_query = f'{query_param}[min]={file_jsons[2][query_param]}'
        _verify_request_response(endpoint_url=endpoint_url, url_query=url_query, expected_jsons=file_jsons[2:])

        # Before a datetime
        url_query = f'{query_param}[max]={file_jsons[2][query_param]}'
        _verify_request_response(endpoint_url=endpoint_url, url_query=url_query, expected_jsons=file_jsons[:3])

        # Datetime interval
        url_query = (f'{query_param}[min]={file_jsons[1][query_param]}' +
                     f'&{query_param}[max]={file_jsons[3][query_param]}')
        _verify_request_response(endpoint_url=endpoint_url, url_query=url_query, expected_jsons=file_jsons[1:4])

        #######################################
        # FILTER BY USE (only for task files) #
        #######################################
        if file_type == TaskFile:
            input_files = (TaskFileDB.query().filter_by(task_id=parent.db_object().task_id,
                                                        use_for=TaskFileUse.INPUT).all())
            ai_model_files = (TaskFileDB.query().filter_by(task_id=parent.db_object().task_id,
                                                           use_for=TaskFileUse.AI_MODEL).all())

            expected_input_file_jsons = [_get_file_metadata_json(db_object=x) for x in input_files]
            expected_ai_model_file_jsons = [_get_file_metadata_json(db_object=x) for x in ai_model_files]

            _verify_request_response(endpoint_url=endpoint_url,
                                     url_query='use_for=input',
                                     expected_jsons=expected_input_file_jsons)
            _verify_request_response(endpoint_url=endpoint_url,
                                     url_query='use_for=ai_model',
                                     expected_jsons=expected_ai_model_file_jsons)

        ########################################
        # FILTER BY TYPE (only for task files) #
        ########################################
        if file_type == TaskFile:
            image_files = (TaskFileDB.query().filter_by(task_id=parent.db_object().task_id, type_=FileType.IMAGE).all())

            expected_image_file_jsons = [_get_file_metadata_json(db_object=x) for x in image_files]

            _verify_request_response(endpoint_url=endpoint_url,
                                     url_query='type=image',
                                     expected_jsons=expected_image_file_jsons)

        ############
        # ORDERING #
        ############
        # Ascending order
        asc_response_jsons = _verify_request_response(endpoint_url=endpoint_url, url_query='order=asc')
        pass  # TODO: check order

        # Descending order
        desc_response_jsons = _verify_request_response(endpoint_url=endpoint_url, url_query='order=desc')
        pass  # TODO: check order

        # Compare orders
        asc_file_uuids = [x['uuid'] for x in asc_response_jsons]
        desc_file_uuids = [x['uuid'] for x in desc_response_jsons]
        assert list(reversed(asc_file_uuids)) == desc_file_uuids

        ##########
        # PAGING #
        ##########
        NUM_PAGES = 3
        per_page = TEST_CONFIG['views']['default_items_per_page']

        # Upload new files to have multiple pages
        empty_table(file_type.db_model())

        files = []
        for file_id in range(1, per_page * NUM_PAGES + 1):
            filename = f'File {file_id}'
            parent_pk = ({
                'task_id': parent.db_object().task_id
            } if file_type == TaskFile else {
                'organization_id': parent.db_object().organization_id
            })
            use_for = (TaskFileUse.INPUT if file_type == TaskFile else OrgFileUse.PICTURE)
            file_object = file_type.db_model()(**parent_pk,
                                               file_id=file_id,
                                               filename=filename,
                                               size=40,
                                               use_for=use_for,
                                               type_=FileType.IMAGE)
            files.append(file_object)

        save_to_db(files)

        _post_files_metadata_and_content(files_metadata=files,
                                         parent=parent,
                                         client=client,
                                         storage_backend=storage_backend)

        # Load files from database again, since `_post_files()` removes previously loaded instances
        files = _get_files_metadata(parent=parent)

        # No query parameters
        response = client.send_request(method='GET', url=endpoint_url)
        res_json = response.json()
        assert len(res_json['data']) == per_page
        assert res_json['links']['previous'] is None
        assert res_json['links']['current'] == endpoint_url + '?page=1'
        assert res_json['links']['next'] == endpoint_url + '?page=2'
        assert 'total_count' not in res_json

        # Get the total number of files
        response = client.send_request(method='GET', url=endpoint_url + '?total_count=true')
        assert response.json()['total_count'] == len(files)

        # Request the 2nd page
        response = client.send_request(method='GET', url=endpoint_url + '?page=2')
        res_json = response.json()
        assert len(res_json['data']) == per_page
        assert res_json['links']['previous'] == endpoint_url + '?page=1'
        assert res_json['links']['current'] == endpoint_url + '?page=2'
        assert res_json['links']['next'] == endpoint_url + '?page=3'
        assert 'total_count' not in res_json

        # Request the last page
        response = client.send_request(method='GET', url=endpoint_url + f'?page={NUM_PAGES}')
        res_json = response.json()
        assert len(res_json['data']) == per_page
        assert res_json['links']['previous'] == endpoint_url + f'?page={NUM_PAGES - 1}'
        assert res_json['links']['current'] == endpoint_url + f'?page={NUM_PAGES}'
        assert res_json['links']['next'] is None

        # Request a custom number of files per page
        CUSTOM_PER_PAGE = 3
        response = client.send_request(method='GET', url=endpoint_url + f'?per_page={CUSTOM_PER_PAGE}')
        res_json = response.json()
        assert len(res_json['data']) == CUSTOM_PER_PAGE
        assert res_json['links']['previous'] is None
        assert res_json['links']['current'] == endpoint_url + f'?page=1&per_page={CUSTOM_PER_PAGE}'
        assert res_json['links']['next'] == endpoint_url + f'?page=2&per_page={CUSTOM_PER_PAGE}'
        assert 'total_count' not in res_json

        # Request the 4th page with a custom number of files per page
        response = client.send_request(method='GET', url=endpoint_url + f'?per_page={CUSTOM_PER_PAGE}&page=4')
        res_json = response.json()
        assert len(res_json['data']) == CUSTOM_PER_PAGE
        assert res_json['links']['previous'] == endpoint_url + f'?per_page={CUSTOM_PER_PAGE}&page=3'
        assert res_json['links']['current'] == endpoint_url + f'?per_page={CUSTOM_PER_PAGE}&page=4'
        assert res_json['links']['next'] == endpoint_url + f'?per_page={CUSTOM_PER_PAGE}&page=5'
        assert 'total_count' not in res_json

        # Request the 3rd page with a custom number of files per page and get the total number of files
        response = client.send_request(method='GET',
                                       url=endpoint_url + f'?page=3&total_count=true&per_page={CUSTOM_PER_PAGE}')
        res_json = response.json()
        assert len(res_json['data']) == CUSTOM_PER_PAGE
        assert res_json['links']['previous'] == endpoint_url + f'?page=2&per_page={CUSTOM_PER_PAGE}'
        assert res_json['links']['current'] == endpoint_url + f'?page=3&per_page={CUSTOM_PER_PAGE}'
        assert res_json['links']['next'] == endpoint_url + f'?page=4&per_page={CUSTOM_PER_PAGE}'
        assert res_json['total_count'] == len(files)

        # Check file ordering
        for idx, file in enumerate(files):
            file.created_at = datetime.utcnow() + timedelta(days=30 * idx)
        save_to_db(files)
        db_commit_and_expire()

        files.reverse()

        for page in range(1, NUM_PAGES + 1):
            page_files = files[(page - 1) * per_page:page * per_page]
            response = client.send_request(method='GET', url=endpoint_url + f'?page={page}')
            expected_jsons = [_get_file_metadata_json(f) for f in page_files]
            verify_response_jsons(actual_jsons=response.json()['data'],
                                  expected_jsons=expected_jsons,
                                  optional_fields=_FILE_OPTIONAL_FIELDS)

        # Check filtering
        num_filtered = CUSTOM_PER_PAGE * NUM_PAGES
        max_created_at = files[0].created_at
        for idx, file in enumerate(files[:num_filtered]):
            file.created_at = max_created_at - timedelta(days=idx + 1)
        for idx, file in enumerate(files[num_filtered:]):
            file.created_at = max_created_at + timedelta(days=idx + 1)
        save_to_db(files)
        max_created_at = max_created_at.strftime(DATETIME_FORMAT)

        for page in range(1, NUM_PAGES + 1):
            response = client.send_request(method='GET',
                                           url=(endpoint_url + f'?page={page}&per_page={CUSTOM_PER_PAGE}'
                                                f'&created_at[max]={max_created_at}'))
            res_json = response.json()
            assert len(res_json['data']) == CUSTOM_PER_PAGE
            assert 'total_count' not in res_json

        response = client.send_request(method='GET',
                                       url=(endpoint_url + f'?page={NUM_PAGES + 1}&per_page={CUSTOM_PER_PAGE}'
                                            f'&created_at[max]={max_created_at}'))
        assert response.status_code == HTTP_NOT_FOUND_STATUS_CODE

    # @pytest.mark.parametrize(
    #     'parent_type, storage_backend',
    #     [
    #         (Organization, FileStorageBackend.LOCAL),
    #         (Organization, FileStorageBackend.S3),
    #         (Task, FileStorageBackend.LOCAL),
    #         (Task, FileStorageBackend.S3)
    #     ]
    # )
    @pytest.mark.parametrize('parent_type', [Organization, Task])
    def test_post(
        self,
        mock_s3,
        client: MockClient,
        local_store_upload_test_client: MockClient,
        parent_type: Type[_OrgOrTask],
        # storage_backend: FileStorageBackend
    ):
        storage_backend = FileStorageBackend.S3  # TODO: Remove this line when the local storage backend tests are fixed
        if storage_backend == FileStorageBackend.LOCAL:
            client = local_store_upload_test_client

        def _request_thread(endpoint_url: str, request_json: dict) -> Response:
            cache.clear()  # force `resources.tasks.Task.check_quota_usage()` to use current database value
            return client.send_request(method='POST', url=endpoint_url, json=request_json)

        parent: _OrgOrTask = load_default_resource(resource_type=parent_type)
        file_type = OrgFile if parent_type == Organization else TaskFile

        files = _get_files_metadata(parent=parent)
        endpoint = ENDPOINT_TASK_FILES if parent_type == Task else ENDPOINT_ORG_FILES
        endpoint_url = get_endpoint(parameterized_endpoint=endpoint, resource=parent)

        ##################
        # REGULAR UPLOAD #
        ##################
        _post_files_metadata_and_content(files_metadata=files,
                                         parent=parent,
                                         client=client,
                                         storage_backend=storage_backend)

        #############################################################################################################
        # TRY TO EXCEED THE MAXIMUM FILE SIZE FOR A SINGLE UPLOAD                                                   #
        #############################################################################################################
        if file_type == TaskFile:
            max_upload_size = TEST_CONFIG['storage']['files']['s3']['max_upload_size']
            file_metadata = {'filename': 'large_file', 'size': max_upload_size + 1, 'use_for': 'ai_model', 'type': None}
            response = client.send_request(method='POST', url=endpoint_url, json=file_metadata)
            assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
            assert 'upload_url' not in response.json()

        ###################################
        # TRY TO EXCEED SPACE QUOTA LIMIT #
        ###################################
        _delete_all_files(parent=parent, storage_backend=storage_backend)

        FILES_TO_UPLOAD = 6
        FILE_SIZE = 40
        files_to_accept = round(FILES_TO_UPLOAD / 2)
        space_limit = (files_to_accept * FILE_SIZE) + 1

        set_quota_limit(db_object=parent.db_object(), quota='space', limit=space_limit)

        futures = []
        # Note: using multiple workers results in a `ConnectionAbortedError`
        with ThreadPoolExecutor(max_workers=CLIENT_MAX_THREADS) as executor:
            for i in range(FILES_TO_UPLOAD):
                file_metadata = {'filename': f'new_file_{i + 1}', 'size': FILE_SIZE}
                if file_type == TaskFile:
                    file_metadata['use_for'] = 'input'
                    file_metadata['type'] = 'image'
                futures.append(executor.submit(_request_thread, endpoint_url, file_metadata))
            wait(futures)

        num_uploaded = 0
        for future in futures:
            response = future.result()
            if response.status_code == HTTP_POST_STATUS_CODE:
                num_uploaded += 1
            else:
                assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
                response_text = response.json()['error']['message'].lower()
                assert 'exceeded' in response_text

        assert num_uploaded == files_to_accept
        assert num_uploaded * FILE_SIZE <= space_limit

        #################################################################
        # TRY TO EXCEED THE FILE SIZE LIMITED BY THE PRESIGNED POST URL #
        #################################################################
        # TODO: `moto` seems to ignore the `content-length-range` condition in presigned POST URLs,
        #       so we cannot test payload size limit for the moment
        # restore_db()
        # response = client.send_request(method='POST', url=endpoint_url, json={'filename': 's3_test_file', 'size': 40})
        # s3_post = response.json()['upload_url']
        # file_data = os.urandom(TEST_CONFIG['storage']['files']['s3']['max_payload'] + 1)  # random bytes
        # response = requests.post(s3_post['url'], data=s3_post['fields'], files={'file': file_data})
        # assert response.status_code == HTTP_PAYLOAD_TOO_LARGE_STATUS_CODE


class TestFileView:

    # @pytest.mark.parametrize(
    #     'file_type, storage_backend',
    #     [
    #         (OrgFile, FileStorageBackend.LOCAL),
    #         (OrgFile, FileStorageBackend.S3),
    #         (TaskFile, FileStorageBackend.LOCAL),
    #         (TaskFile, FileStorageBackend.S3)
    #     ]
    # )
    @pytest.mark.parametrize('file_type', [OrgFile, TaskFile])
    def test_delete(
        self,
        mock_s3,
        client: MockClient,
        file_type: Type[_OrgOrTaskFile],
        # storage_backend: FileStorageBackend
    ):
        storage_backend = FileStorageBackend.S3  # TODO: Remove this line when the local storage backend tests are fixed
        if storage_backend == FileStorageBackend.LOCAL:
            client._always_use_test_client = True

        def _delete_file(parent: _OrgOrTask, file: _OrgOrTaskFile):
            file_id = file.db_object().file_id
            file_uuid = file.uuid()
            file_size = file.db_object().size

            # Get space usage
            if isinstance(parent, Task):
                prev_space_usage = parent.db_object().space_usage
            else:
                subscription = get_active_subscription(organization_id=parent.db_object().organization_id)
                prev_space_usage = subscription.space_usage

            # Make request
            endpoint = ENDPOINT_TASK_FILE if isinstance(parent, Task) else ENDPOINT_ORG_FILE
            verify_resource_request(client=client, method='DELETE', endpoint=endpoint, resource=file)

            # Check file store
            if storage_backend == FileStorageBackend.S3:
                assert file_uuid not in get_files_in_s3(parent_resource=parent)
            else:
                assert not os.path.exists(file.path(thumbnail=False))
                assert not os.path.exists(file.path(thumbnail=True))

            # Check database
            db_commit_and_expire()
            assert file.db_model().get(file_id=file_id) is None
            set_quota_usage(db_object=parent.db_object(), quota='space', usage=prev_space_usage - file_size)

        parent_type = Organization if file_type == OrgFile else Task
        parent: _OrgOrTask = load_default_resource(resource_type=parent_type)

        # Upload files
        _post_files_metadata_and_content(files_metadata=_get_files_metadata(parent=parent),
                                         parent=parent,
                                         client=client,
                                         storage_backend=storage_backend)

        # Reload parent task and files (`_post_files()` removes loaded instances)
        parent: _OrgOrTask = load_default_resource(resource_type=parent_type)
        files = [
            file_type.get(agent=parent.user(), db_object_or_id=x, parents=[parent])
            for x in _get_files_metadata(parent=parent)
        ]

        # Get the directory containing all the files in the file store
        if storage_backend == FileStorageBackend.S3:
            files_dir = '/'.join(files[0].path(thumbnail=False).split('/')[:-1])
            thumbnails_dir = '/'.join(files[0].path(thumbnail=True).split('/')[:-1])
        else:
            files_dir = os.path.dirname(files[0].path(thumbnail=False))
            thumbnails_dir = os.path.dirname(files[0].path(thumbnail=True))

        # Delete each file individually
        for file in files:
            _delete_file(parent=parent, file=file)

        # Verify no files are left in the file store
        if storage_backend == FileStorageBackend.S3:
            assert not get_files_in_s3(parent_resource=parent)
        else:
            assert not os.listdir(files_dir)
            assert not os.listdir(thumbnails_dir)

        # Check database
        db_commit_and_expire()
        assert not _get_files_metadata(parent=parent)
        verify_quota_usage(db_object=parent.db_object(), quota='space', expected_usage=0)

    # @pytest.mark.parametrize(
    #     'file_type, storage_backend',
    #     [
    #         (OrgFile, FileStorageBackend.LOCAL),
    #         (OrgFile, FileStorageBackend.S3),
    #         (TaskFile, FileStorageBackend.LOCAL),
    #         (TaskFile, FileStorageBackend.S3)
    #     ]
    # )
    @pytest.mark.parametrize('file_type', [OrgFile, TaskFile])
    def test_get(
        self,
        mock_s3,
        client: MockClient,
        file_type: Type[_OrgOrTaskFile],
        # storage_backend: FileStorageBackend
    ):
        storage_backend = FileStorageBackend.S3  # TODO: Remove this line when the local storage backend tests are fixed
        if storage_backend == FileStorageBackend.LOCAL:
            client._always_use_test_client = True

        ###################
        # Prepare request #
        ###################
        parent_type = Organization if file_type == OrgFile else Task
        parent: _OrgOrTask = load_default_resource(resource_type=parent_type)
        file_object = _get_files_metadata(parent=parent)[0]

        _post_files_metadata_and_content(files_metadata=[file_object],
                                         parent=parent,
                                         client=client,
                                         storage_backend=storage_backend)

        parent: _OrgOrTask = load_default_resource(resource_type=parent_type)
        file = file_type.get(agent=parent.user(),
                             db_object_or_id=_get_files_metadata(parent=parent)[0],
                             parents=[parent])
        file_json = _get_file_metadata_json(db_object=file.db_object())
        file_json.pop('format')

        endpoint = ENDPOINT_TASK_FILE if parent_type == Task else ENDPOINT_ORG_FILE

        ############
        # Get file #
        ############
        response = verify_resource_request(client=client,
                                           method='GET',
                                           endpoint=endpoint,
                                           resource=file,
                                           expected_jsons=[file_json])

        _verify_file(file_json=response.json(),
                     file_type=file_type,
                     is_thumbnail=False,
                     parent=parent,
                     storage_backend=storage_backend)

        ######################
        # Get file thumbnail #
        ######################
        response = verify_resource_request(client=client,
                                           method='GET',
                                           endpoint=endpoint,
                                           resource=file,
                                           url_query='?thumbnail=true',
                                           expected_jsons=[file_json])

        _verify_file(file_json=response.json(),
                     file_type=file_type,
                     is_thumbnail=True,
                     parent=parent,
                     storage_backend=storage_backend)

        ##############################################################
        # Verify the thumbnail is automatically created when missing #
        ##############################################################

        file_path = file.path(thumbnail=False)
        thumbnail_path = file.path(thumbnail=True)

        s3_client = boto3.client('s3')
        s3_config = get_s3_config()

        # Delete the file and its thumbnail
        if storage_backend == FileStorageBackend.S3:
            s3_client.delete_object(Bucket=s3_config['bucket'], Key=file_path)
            s3_client.delete_object(Bucket=s3_config['bucket'], Key=thumbnail_path)
        else:
            os.remove(file_path)
            os.remove(thumbnail_path)

        # Verify the file doesn't exist
        if storage_backend == FileStorageBackend.S3:
            try:
                s3_client.get_object(Bucket=s3_config['bucket'], Key=file_path)
                assert False
            except Exception:
                pass
        else:
            assert not os.path.exists(file_path)

        # Verify the thumbnail doesn't exist
        if storage_backend == FileStorageBackend.S3:
            try:
                s3_client.get_object(Bucket=s3_config['bucket'], Key=thumbnail_path)
                assert False
            except Exception:
                pass
        else:
            assert not os.path.exists(thumbnail_path)

        # Create and upload a random image
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        file_data = np.random.rand(256, 256, 3) * 255
        Image.fromarray(file_data.astype('uint8')).save(tmp_file.name, 'JPEG')
        tmp_file.close()
        if storage_backend == FileStorageBackend.S3:
            s3_client.upload_file(tmp_file.name, s3_config['bucket'], file_path)
            os.remove(tmp_file.name)
        else:
            shutil.move(tmp_file.name, file_path)

        # Verify the thumbnail was automatically created
        response = verify_resource_request(client=client,
                                           method='GET',
                                           endpoint=endpoint,
                                           resource=file,
                                           url_query='?thumbnail=true',
                                           expected_jsons=[file_json])

        _verify_file(file_json=response.json(),
                     file_type=file_type,
                     is_thumbnail=True,
                     parent=parent,
                     storage_backend=storage_backend)

    # @pytest.mark.parametrize(
    #     'file_type, storage_backend',
    #     [
    #         (OrgFile, FileStorageBackend.LOCAL),
    #         (OrgFile, FileStorageBackend.S3),
    #         (TaskFile, FileStorageBackend.LOCAL),
    #         (TaskFile, FileStorageBackend.S3)
    #     ]
    # )
    @pytest.mark.parametrize('file_type', [OrgFile, TaskFile])
    def test_put(
        self,
        mock_s3,
        client: MockClient,
        file_type: Type[_OrgOrTaskFile],
        # storage_backend: FileStorageBackend
    ):
        storage_backend = FileStorageBackend.S3  # TODO: Remove this line when the local storage backend tests are fixed
        if storage_backend == FileStorageBackend.LOCAL:
            client._always_use_test_client = True

        parent_type = Organization if file_type == OrgFile else Task
        file: _OrgOrTaskFile = load_default_resource(resource_type=file_type, parents_types=[parent_type])

        request_json = {'filename': 'test_directory/test_file.jpg', 'size': 30}
        if file_type == TaskFile:
            request_json['use_for'] = 'input'
            request_json['type'] = 'image'

        verify_resource_request(client=client,
                                method='PUT',
                                endpoint=ENDPOINT_TASK_FILE if parent_type == Task else ENDPOINT_ORG_FILE,
                                resource=file,
                                request_json=request_json,
                                expected_status_code=HTTP_METHOD_NOT_ALLOWED_STATUS_CODE)


class TestMultipartUploadView:
    """
    Tests for the multipart upload views.

    Note: For the moment, there is only one file use in organizations ("picture").
          Since the maximum picture size is smaller than the maximum file size for a single upload,
          multipart uploads never take place in organization files.
    """

    # @pytest.mark.parametrize('storage_backend', [FileStorageBackend.LOCAL, FileStorageBackend.S3])
    def test_multipart_upload(
        self,
        mock_s3,
        client: MockClient,
        # storage_backend: FileStorageBackend,
        mock_client_id: str,
        session_user_id: str,
        session_user_auth0_id: str
    ):
        storage_backend = FileStorageBackend.S3  # TODO: Remove this line when the local storage backend tests are fixed
        if storage_backend == FileStorageBackend.LOCAL:
            client._always_use_test_client = True

        def _upload_part(file: TaskFile, part_number: int, part_size: int, expected_error_message: str = None) -> dict:
            # Get the upload URL from the backend
            endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_TASK_FILE_PARTS, resource=file)
            response = client.send_request(method='POST', url=endpoint_url, json={'part_number': part_number})
            if expected_error_message:
                assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
                assert response.json()['error']['message'] == expected_error_message
                return dict()
            assert response.status_code == 200

            # Upload file part to the storage backend
            part_data = os.urandom(part_size)
            if storage_backend == FileStorageBackend.S3:
                assert 's3.amazonaws.com' in response.headers['Location']
                response = requests.put(response.json()['upload_url'], data=part_data)
            else:
                response = client.post(response.json()['upload_url'], data=part_data)
            assert response.status_code == 200
            return {'etag': response.headers['ETag'], 'part_number': part_number}  # TODO: Is this ok?

        NUM_PARTS = 3

        ##########################
        # VALID MULTIPART UPLOAD #
        ##########################
        max_upload_size = TEST_CONFIG['storage']['files']['s3']['max_upload_size']
        task: Task = load_default_resource(resource_type=Task)

        # Increase space quota limit
        set_quota_limit(db_object=task.db_object(), quota='space', limit=max_upload_size * NUM_PARTS + 1024**2)

        # Set file metadata
        file_metadata = {'filename': 'multipart_file', 'size': max_upload_size * NUM_PARTS}
        file_metadata['use_for'] = 'ai_model'
        file_metadata['type'] = None

        # Upload file metadata
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_TASK_FILES, resource=task)
        response = client.send_request(method='POST', url=endpoint_url, json=file_metadata)
        assert response.status_code == HTTP_POST_STATUS_CODE
        db_commit_and_expire()
        file_db_object = TaskFileDB.get_from_uuid(response.json()['uuid'])
        file = Task.get(agent=task.agent(), db_object_or_id=file_db_object, parents=[task])

        # Upload parts
        uploaded_parts = []
        for part_number in range(1, NUM_PARTS + 1):
            uploaded_part = _upload_part(file=file, part_number=part_number, part_size=max_upload_size)
            uploaded_parts.append(uploaded_part)

        # Complete upload
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_TASK_FILE_PARTS_COMPLETION, resource=file)
        request_json = {'uploaded_parts': uploaded_parts}
        response = client.send_request(method='POST', url=endpoint_url, json=request_json)
        assert response.status_code == 204

        ####################################
        # TRY TO EXCEED DECLARED FILE SIZE #
        ####################################
        restore_db(mock_client_id=mock_client_id,
                   session_user_id=session_user_id,
                   session_user_auth0_id=session_user_auth0_id)
        cache.clear()

        file: TaskFile = load_default_resource(resource_type=TaskFile, parents_types=[Task])
        file.db_object().size = max_upload_size * NUM_PARTS
        file.persist()

        # Upload all the parts
        for part_number in range(1, NUM_PARTS + 1):
            _upload_part(file=file, part_number=part_number, part_size=max_upload_size)
        assert TaskUpload.query().filter_by(file_id=file.db_object().file_id).first() is not None

        # Try to upload an extra part
        _upload_part(file=file,
                     part_number=(NUM_PARTS + 1),
                     part_size=max_upload_size,
                     expected_error_message='Parts size exceeds declared file size. Upload aborted')
        db_commit_and_expire()
        # Verify the upload was aborted
        assert TaskUpload.query().filter_by(file_id=file.db_object().file_id).first() is None

        #########################################################
        # TRY TO COMPLETE AN UPLOAD WHEN SOME PARTS ARE MISSING #
        #########################################################
        restore_db(mock_client_id=mock_client_id,
                   session_user_id=session_user_id,
                   session_user_auth0_id=session_user_auth0_id)
        cache.clear()

        file: TaskFile = load_default_resource(resource_type=TaskFile, parents_types=[Task])
        file.db_object().size = max_upload_size * NUM_PARTS
        file.persist()

        # Upload some parts
        uploaded_parts = []
        for part_number in range(1, math.ceil(NUM_PARTS / 2) + 1):
            uploaded_part = _upload_part(file=file, part_number=part_number, part_size=max_upload_size)
            uploaded_parts.append(uploaded_part)

        # Try to complete the upload
        endpoint_url = get_endpoint(parameterized_endpoint=ENDPOINT_TASK_FILE_PARTS_COMPLETION, resource=file)
        request_json = {'uploaded_parts': uploaded_parts}
        response = client.send_request(method='POST', url=endpoint_url, json=request_json)
        assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
        assert response.json()['error']['message'] == 'File not uploaded completely. Some parts are missing'
        db_commit_and_expire()
        # Verify the upload was not aborted
        assert TaskUpload.query().filter_by(file_id=file.db_object().file_id).first() is not None


#######################################
# TESTS FOR THE LOCAL STORAGE BACKEND #
#######################################

# TODO: Uncomment when the issues with the local storage backend tests are resolved.
# class TestLocalStoreDownload:
#
#     @pytest.mark.parametrize('file_type', [OrgFile, TaskFile])
#     def test_valid_download(
#         self,
#         local_store_download_test_client: MockClient,
#         file_type: Type[_OrgOrTaskFile]
#     ):
#         client = local_store_download_test_client
#
#         # Get file metadata
#         file_metadata = _get_default_file_metadata(file_type=file_type)
#
#         # Write file content to local store
#         os.makedirs(os.path.dirname(file_metadata.path()), exist_ok=True)
#         with open(file_metadata.path(), 'wb') as file:
#             file.write(os.urandom(file_metadata.db_object().size))
#
#         # Make request
#         endpoint = _get_local_store_download_endpoint(file=file_metadata)
#         response = client.get(endpoint)
#         assert response.status_code == 200
#         assert response.data == open(file_metadata.path(), 'rb').read()
#
#     @pytest.mark.parametrize('file_type', [OrgFile, TaskFile])
#     def test_valid_thumbnail_download(
#         self,
#         local_store_download_test_client: MockClient,
#         file_type: Type[_OrgOrTaskFile]
#     ):
#         client = local_store_download_test_client
#         pass  # TODO
#
#     @pytest.mark.parametrize('parent_type', [Organization, Task])
#     def test_nonexistent_file_download(
#         self,
#         local_store_download_test_client: MockClient,
#         parent_type: Type[_OrgOrTask]
#     ):
#         client = local_store_download_test_client
#         file_metadata = _get_default_file_metadata(file_type=OrgFile if parent_type == Organization else TaskFile)
#         endpoint = _get_local_store_download_endpoint(file=file_metadata)
#         response = client.get(endpoint.replace(file_metadata.uuid(), 'nonexistent_file_id'))
#         assert response.status_code == HTTP_NOT_FOUND_STATUS_CODE
#         assert response.json['error']['message'] == 'Resource not found: "nonexistent_file_id"'
#
#
# class TestLocalStoreUpload:
#
#     @pytest.mark.parametrize('file_type', [OrgFile, TaskFile])
#     def test_valid_upload(
#         self,
#         local_store_upload_test_client: MockClient,
#         file_type: Type[_OrgOrTaskFile]
#     ):
#         client = local_store_upload_test_client
#         file_metadata = _get_default_file_metadata(file_type=file_type)
#         endpoint = _get_local_store_upload_endpoint(file=file_metadata)
#         response = _upload_file_content_to_local_store(client=client, endpoint=endpoint, file_metadata=file_metadata)
#         assert response.status_code == 200
#
#     @pytest.mark.parametrize('file_type', [OrgFile, TaskFile])
#     def test_no_file_provided(
#         self,
#         local_store_upload_test_client: MockClient,
#         file_type: Type[_OrgOrTaskFile]
#     ):
#         client = local_store_upload_test_client
#         parent_type = Organization if file_type == OrgFile else Task
#         file_metadata: _OrgOrTaskFile = load_default_resource(resource_type=file_type, parents_types=[parent_type])
#         endpoint = _get_local_store_upload_endpoint(file=file_metadata)
#         response = client.post(endpoint)
#         assert response.status_code == HTTP_BAD_REQUEST_STATUS_CODE
#         assert response.json()['error']['message'] == 'No file provided'
#
#     @pytest.mark.parametrize('file_type', [OrgFile, TaskFile])
#     def test_maximum_upload_size_exceeded(
#         self,
#         local_store_upload_test_client: MockClient,
#         file_type: Type[_OrgOrTaskFile]
#     ):
#         client = local_store_upload_test_client
#         file_metadata = _get_default_file_metadata(file_type=file_type)
#         max_upload_size = TEST_CONFIG['storage']['files']['local']['max_upload_size']
#         endpoint = _get_local_store_upload_endpoint(file=file_metadata)
#         response = _upload_file_content_to_local_store(client=client,
#                                                        endpoint=endpoint,
#                                                        file_metadata=file_metadata,
#                                                        content_length=max_upload_size + 1)
#         assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
#         assert response.json()['error']['message'].startswith(f'Maximum upload size exceeded: ')
#
#     @pytest.mark.parametrize('file_type', [OrgFile, TaskFile])
#     def test_declared_size_exceeded(
#         self,
#         local_store_upload_test_client: MockClient,
#         file_type: Type[_OrgOrTaskFile]
#     ):
#         client = local_store_upload_test_client
#         file_metadata = _get_default_file_metadata(file_type=file_type)
#         endpoint = _get_local_store_upload_endpoint(file=file_metadata)
#         response = _upload_file_content_to_local_store(client=client,
#                                                        endpoint=endpoint,
#                                                        file_metadata=file_metadata,
#                                                        content_length=file_metadata.db_object().size + 1)
#         assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
#         assert response.json()['error']['message'] == 'File size exceeds declared size'
#
#
# class TestLocalStoreMultipartUpload:
#     NUM_PARTS = 3
#
#     @pytest.fixture(autouse=True)
#     def delete_all_uploads(self):
#
#         def _delete_all_uploads():
#             empty_table(OrgUpload)
#             empty_table(TaskUpload)
#
#         _delete_all_uploads()
#         yield
#         _delete_all_uploads()
#
#     @pytest.fixture()
#     def org_file_metadata(self) -> OrgFile:
#         return _get_default_file_metadata(file_type=OrgFile)
#
#     @pytest.fixture()
#     def task_file_metadata(self) -> TaskFile:
#         return _get_default_file_metadata(file_type=TaskFile)
#
#     @pytest.fixture()
#     def org_upload_db_object(self, org_file_metadata) -> OrgUpload:
#         return _init_multipart_upload(file=org_file_metadata)
#
#     @pytest.fixture()
#     def task_upload_db_object(self, task_file_metadata) -> TaskUpload:
#         return _init_multipart_upload(file=task_file_metadata)
#
#     @classmethod
#     def _upload_part(
#         cls,
#         local_store_upload_test_client: MockClient,
#         file_metadata: _OrgOrTaskFile,
#         upload_db_object: _OrgOrTaskUpload,
#         part_number: int
#     ) -> 'TestResponse':  # TODO: Import `TestResponse`
#         client = local_store_upload_test_client
#         part_size = file_metadata.db_object().size // cls.NUM_PARTS
#         url_args = f'?part_number={part_number}'
#         endpoint = _get_local_store_multipart_upload_endpoint(upload=upload_db_object, file=file_metadata) + url_args
#         return client.put(endpoint, data=os.urandom(part_size))
#
#     @pytest.mark.parametrize(
#         'file_metadata,upload_db_object,part_number',
#         [
#             ('org_file_metadata', 'org_upload_db_object', 1),
#             ('task_file_metadata', 'task_upload_db_object', 1),
#             ('org_file_metadata', 'org_upload_db_object', 2),
#             ('task_file_metadata', 'task_upload_db_object', 2),
#             ('org_file_metadata', 'org_upload_db_object', 3),
#             ('task_file_metadata', 'task_upload_db_object', 3),
#         ],
#         indirect=['file_metadata', 'upload_db_object']
#     )
#     def test_valid_part_upload(
#         self,
#         local_store_upload_test_client: MockClient,
#         file_metadata: _OrgOrTaskFile,
#         upload_db_object: _OrgOrTaskUpload,
#         part_number: int
#     ):
#         response = self._upload_part(local_store_upload_test_client=local_store_upload_test_client,
#                                      file_metadata=file_metadata,
#                                      upload_db_object=upload_db_object,
#                                      part_number=part_number)
#         assert response.status_code == 200
#
#     @pytest.mark.parametrize(
#         'file_metadata,upload_db_object,part_number',
#         [
#             ('org_file_metadata', 'org_upload_db_object', 1),
#             ('task_file_metadata', 'task_upload_db_object', 1),
#             ('org_file_metadata', 'org_upload_db_object', 2),
#             ('task_file_metadata', 'task_upload_db_object', 2),
#             ('org_file_metadata', 'org_upload_db_object', 3),
#             ('task_file_metadata', 'task_upload_db_object', 3),
#             ('org_file_metadata', 'org_upload_db_object', 4),
#             ('task_file_metadata', 'task_upload_db_object', 4),
#         ],
#         indirect=['file_metadata', 'upload_db_object']
#     )
#     def test_total_file_size_exceeded(
#         self,
#         local_store_upload_test_client: MockClient,
#         file_metadata: _OrgOrTaskFile,
#         upload_db_object: _OrgOrTaskUpload,
#         part_number: int
#     ):
#         # Make the request
#         response = self._upload_part(local_store_upload_test_client=local_store_upload_test_client,
#                                      file_metadata=file_metadata,
#                                      upload_db_object=upload_db_object,
#                                      part_number=part_number)
#
#         # Check the response
#         if part_number <= self.NUM_PARTS:
#             assert response.status_code == 200
#         else:
#             assert response.status_code == HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE
#             assert response.json == 'Parts size exceeds declared file size. Upload aborted'
#
#     @pytest.mark.parametrize(
#         'file_metadata',
#         ['org_file_metadata', 'task_file_metadata'],
#         indirect=['file_metadata']
#     )
#     def test_nonexistent_upload(self, local_store_upload_test_client: MockClient, file_metadata: _OrgOrTaskFile):
#         client = local_store_upload_test_client
#
#         # Specify a nonexistent upload ID in the URL
#         url_args = f'/test_upload?part_number=1'
#         endpoint = _get_local_store_upload_endpoint(file=file_metadata) + url_args
#
#         # Make the request
#         response = client.put(endpoint, data=os.urandom(100))
#         assert response.status_code == HTTP_NOT_FOUND_STATUS_CODE
#         assert response.json()['error']['message'] == 'Upload not found'
#
#     def test_upload_wrong_parent(
#         self,
#         local_store_upload_test_client: MockClient,
#         org_file_metadata: OrgFile,
#         task_file_metadata: TaskFile
#     ):
#         client = local_store_upload_test_client
#         pass  # TODO

#########
# UTILS #
#########


def _get_files_metadata(parent: _OrgOrTask) -> List[Union[OrgFileDB, TaskFileDB]]:
    if isinstance(parent, Task):
        return TaskFileDB.filter_by_task(task_id=parent.db_object().task_id)
    else:
        return OrgFileDB.filter_by_organization(organization_id=parent.db_object().organization_id)


def _get_file_metadata(file_id: str, file_type: Type[_OrgOrTaskFile], parent: _OrgOrTask) -> _OrgOrTaskFile:
    if file_type == TaskFile:
        return file_type.get(agent=parent.agent(), db_object_or_id=TaskFileDB.get_from_id(file_id), parents=[parent])
    else:
        return file_type.get(agent=parent.agent(), db_object_or_id=OrgFileDB.get_from_id(file_id), parents=[parent])


def _get_default_file_metadata(file_type: Type[_OrgOrTaskFile]) -> _OrgOrTaskFile:
    parent_type = Task if file_type == TaskFile else Organization
    return load_default_resource(resource_type=file_type, parents_types=[parent_type])


def _get_file_metadata_json(db_object: Union[OrgFileDB, TaskFileDB]) -> dict:
    file_json = get_json_from_db_object(db_object=db_object)
    if isinstance(db_object, OrgFileDB):
        file_json.pop('use_for')
        file_json.pop('type')
    return file_json


def _delete_all_files(parent: _OrgOrTask, storage_backend: FileStorageBackend):
    # Get parent's prefix
    separator = '/' if storage_backend == FileStorageBackend.S3 else os.path.sep
    root_prefix = 'tasks' if isinstance(parent, Task) else 'organizations'
    parent_prefix = separator.join([root_prefix, parent.uuid()])

    # Delete files' metadata
    empty_table(TaskFileDB if isinstance(parent, Task) else OrgFileDB)
    parent._db_object = db.session.merge(parent.db_object())

    # Delete files' content
    if storage_backend == FileStorageBackend.S3:
        s3_client = boto3.client('s3')
        bucket = get_s3_config()['bucket']
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=parent_prefix)
        for obj in response.get('Contents', []):
            s3_client.delete_object(Bucket=bucket, Key=obj['Key'])
    else:
        parent_path = os.path.join(get_local_file_storage_config()['root_path'], parent_prefix)
        if os.path.isdir(parent_path):
            for file in os.listdir(parent_path):
                os.remove(os.path.join(parent_path, file))

    # Update quota
    set_quota_usage(db_object=parent.db_object(), quota='space', usage=0)


def _post_files_metadata_and_content(files_metadata: Iterable[Union[OrgFileDB, TaskFileDB]],
                                     parent: _OrgOrTask,
                                     client: MockClient,
                                     storage_backend: FileStorageBackend,
                                     clean_previous=True):
    if storage_backend == FileStorageBackend.LOCAL:
        client._always_use_test_client = True

    file_jsons = [_get_file_metadata_json(db_object=x) for x in files_metadata]
    for file_json in file_jsons:
        file_json.pop('id')
        file_json.pop('uuid')
        file_json.pop('format')
        file_json.pop('created_at', None)
        file_json.pop('created_by', None)

    # Clean previous states
    if clean_previous:
        _delete_all_files(parent=parent, storage_backend=storage_backend)

    # Get space usage
    subscription = get_active_subscription(organization_id=parent.db_object().organization_id)
    prev_space_usage = subscription.space_usage
    if isinstance(parent, Task):
        assert prev_space_usage == parent.db_object().space_usage

    # Make request
    res_data = client.upload_files(files=file_jsons, parent=parent, storage_backend=storage_backend)
    if clean_previous:
        assert len(res_data['uploaded_metadata']) == len(file_jsons)
        assert set(x['uuid'] for x in res_data['uploaded_metadata']) == set(res_data['uploaded_data'])
    else:
        assert not res_data['uploaded_metadata'] and not res_data['uploaded_data']
        return

    # Check database
    db_commit_and_expire()
    file_db_model = TaskFileDB if isinstance(parent, Task) else OrgFileDB
    assert all(file_db_model.get_from_uuid(x['uuid']) is not None for x in res_data['uploaded_metadata'])
    verify_quota_usage(db_object=parent.db_object(),
                       quota='space',
                       expected_usage=prev_space_usage + sum(f['size'] for f in file_jsons))

    # Check uploaded files
    for file_json in res_data['uploaded_metadata']:
        _verify_file(file_json=file_json,
                     file_type=TaskFile if isinstance(parent, Task) else OrgFile,
                     is_thumbnail=False,
                     parent=parent,
                     storage_backend=storage_backend,
                     verify_download_url=False,
                     verify_upload_url=True)


def _verify_file(
    file_json: dict,
    file_type: Type[_OrgOrTaskFile],
    is_thumbnail: bool,
    parent: _OrgOrTask,
    storage_backend: FileStorageBackend,
    verify_download_url: bool = True,
    verify_upload_url: bool = False,
):
    if bool(verify_download_url) == bool(verify_upload_url):
        raise ValueError('Exactly one of `verify_download_url` and `verify_upload_url` must be True')

    # Get file metadata
    file_uuid = file_json['uuid']
    file = _get_file_metadata(file_id=file_uuid, file_type=file_type, parent=parent)
    file_path = file.path(thumbnail=is_thumbnail)

    # Check download URL
    if verify_download_url:
        download_url = file_json['download_url']

        if storage_backend == FileStorageBackend.S3:
            assert 's3' in download_url  # TODO: Be more specific.
        else:
            endpoint = (ENDPOINT_ORG_LOCAL_FILE_STORE_DOWNLOAD
                        if isinstance(parent, Organization) else ENDPOINT_TASK_LOCAL_FILE_STORE_DOWNLOAD)
            expected_url = endpoint.replace('<organization_id>', parent.uuid()).replace('<file_id>', file.uuid())
            assert download_url == expected_url

        if is_thumbnail:
            url_splits = download_url[:download_url.index('?')].split('/')
            assert url_splits[-2] == PREFIX_THUMBNAILS.replace('/', '')
            assert url_splits[-1] == file_uuid

    # Check upload URL
    if verify_upload_url:
        if is_thumbnail:
            raise ValueError('Cannot verify upload URL for thumbnails')

        upload_url = file_json['upload_url']['url']
        upload_fields = file_json['upload_url']['fields']

        if storage_backend == FileStorageBackend.S3:
            assert 's3' in upload_url  # TODO: Be more specific.
        else:
            endpoint = (ENDPOINT_ORG_LOCAL_FILE_STORE_UPLOAD
                        if isinstance(parent, Organization) else ENDPOINT_TASK_LOCAL_FILE_STORE_UPLOAD)
            expected_url = endpoint.replace('<organization_id>', parent.uuid()).replace('<file_id>', file.uuid())
            assert upload_url == expected_url
            assert isinstance(upload_fields.get('token'), str)

    # Verify file content exists
    if storage_backend == FileStorageBackend.S3:
        s3_client = boto3.client('s3')
        s3_config = get_s3_config()
        res_s3 = s3_client.get_object(Bucket=s3_config['bucket'], Key=file_path)
        assert 'Body' in res_s3
        assert res_s3['ContentLength'] > 0  # TODO: Should it be equal to `file.db_object().size`?
    else:
        assert os.path.isfile(file_path)
        assert os.path.getsize(file_path) == file.db_object().size


def _get_local_store_download_endpoint(file: _OrgOrTaskFile) -> str:
    if isinstance(file, TaskFile):
        parameterized_endpoint = ENDPOINT_TASK_LOCAL_FILE_STORE_DOWNLOAD
        endpoint = parameterized_endpoint.replace('<task_id>', file.parents()[0].uuid())
    else:
        parameterized_endpoint = ENDPOINT_ORG_LOCAL_FILE_STORE_DOWNLOAD
        endpoint = parameterized_endpoint.replace('<organization_id>', file.parents()[0].uuid())
    return endpoint.replace('<file_id>', file.uuid())


def _get_local_store_upload_endpoint(file: _OrgOrTaskFile) -> str:
    parent = file.parents()[0]
    if isinstance(file, TaskFile):
        parameterized_endpoint = ENDPOINT_TASK_LOCAL_FILE_STORE_UPLOAD
        endpoint = parameterized_endpoint.replace('<task_id>', parent.uuid())
    else:
        parameterized_endpoint = ENDPOINT_ORG_LOCAL_FILE_STORE_UPLOAD
        endpoint = parameterized_endpoint.replace('<organization_id>', parent.uuid())
    return endpoint.replace('<file_id>', file.uuid())


def _get_local_store_multipart_upload_endpoint(upload: _OrgOrTaskUpload, file: _OrgOrTaskFile) -> str:
    parent = file.parents()[0]

    # Get parent upload endpoint
    if isinstance(parent, Task):
        upload_endpoint = ENDPOINT_TASK_LOCAL_FILE_STORE_MULTIPART_UPLOAD
    else:
        upload_endpoint = ENDPOINT_ORG_LOCAL_FILE_STORE_MULTIPART_UPLOAD
    upload_endpoint = get_endpoint(parameterized_endpoint=upload_endpoint, resource=parent)

    # Get partial upload endpoint
    return upload_endpoint.replace('<upload_id>', upload.upload_id)


def _init_multipart_upload(file: _OrgOrTaskFile) -> _OrgOrTaskUpload:
    upload_id = str(uuid.uuid4())
    upload_db_model = TaskUpload if isinstance(file, TaskFile) else OrgUpload
    upload_db_obj = upload_db_model(upload_id=upload_id, file_id=file.db_object().file_id)
    save_to_db(upload_db_obj)
    return upload_db_obj


def _upload_file_content_to_local_store(client: MockClient,
                                        endpoint: str,
                                        file_metadata: _OrgOrTaskFile,
                                        content_length: Optional[int] = None):
    part_content = {'file': (os.urandom(content_length or file_metadata.db_object().size), file_metadata.uuid())}
    return client.post(endpoint, data=part_content, content_type='multipart/form-data')
