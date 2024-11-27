import hashlib
import math
import os
from typing import List, Tuple, Type, Union
import uuid

from flask import jsonify
from flask import request
from flask import Response
from flask import send_file
from flask import url_for
from flask_apispec import doc
from flask_apispec import marshal_with
from flask_apispec import MethodResource
from flask_apispec import use_kwargs
import jwt
from marshmallow import fields
from marshmallow import validate

from nexusml.api.resources.base import DuplicateResourceError
from nexusml.api.resources.base import InvalidDataError
from nexusml.api.resources.base import Resource
from nexusml.api.resources.base import ResourceNotFoundError
from nexusml.api.resources.base import UnprocessableRequestError
from nexusml.api.resources.files import OrgFile
from nexusml.api.resources.files import TaskFile
from nexusml.api.resources.organizations import Organization
from nexusml.api.resources.tasks import Task
from nexusml.api.schemas.files import FilePartUploadRequest
from nexusml.api.schemas.files import FilePartUploadResponse
from nexusml.api.schemas.files import FileUploadCompletionRequest
from nexusml.api.schemas.files import OrganizationFileRequest
from nexusml.api.schemas.files import OrganizationFileResponse
from nexusml.api.schemas.files import OrganizationFilesPage
from nexusml.api.schemas.files import TaskFileRequest
from nexusml.api.schemas.files import TaskFileResponse
from nexusml.api.schemas.files import TaskFilesPage
from nexusml.api.utils import API_DOMAIN
from nexusml.api.utils import config
from nexusml.api.utils import generate_tmp_token
from nexusml.api.utils import get_file_storage_backend
from nexusml.api.utils import get_local_file_storage_config
from nexusml.api.utils import save_thumbnail_to_local_file_store
from nexusml.api.views.base import create_view
from nexusml.api.views.core import agent_from_token
from nexusml.api.views.core import capture_request_errors
from nexusml.api.views.core import capture_schema_errors
from nexusml.api.views.core import error_response
from nexusml.api.views.core import get_page_resources
from nexusml.api.views.core import limiter
from nexusml.api.views.core import load_url_resources
from nexusml.api.views.core import process_delete_request
from nexusml.api.views.core import process_get_request
from nexusml.api.views.core import process_post_or_put_request
from nexusml.api.views.core import rate_limits
from nexusml.api.views.core import validate_payload_size
from nexusml.api.views.utils import build_datetime_filter
from nexusml.api.views.utils import build_field_filter
from nexusml.api.views.utils import paging_url_params
from nexusml.api.views.utils import RANGE_MAX
from nexusml.api.views.utils import RANGE_MIN
from nexusml.constants import API_NAME
from nexusml.constants import API_VERSION
from nexusml.constants import HTTP_BAD_REQUEST_STATUS_CODE
from nexusml.constants import MIN_FILE_PART_SIZE
from nexusml.constants import SWAGGER_TAG_FILES
from nexusml.database.core import delete_from_db
from nexusml.database.core import save_to_db
from nexusml.database.files import OrgFileDB
from nexusml.database.files import OrgUpload
from nexusml.database.files import TaskFileDB
from nexusml.database.files import TaskUpload
from nexusml.database.organizations import Agent
from nexusml.database.organizations import ClientDB
from nexusml.database.organizations import UserDB
from nexusml.enums import FileStorageBackend
from nexusml.enums import FileType
from nexusml.enums import OrgFileUse
from nexusml.enums import ResourceAction
from nexusml.enums import TaskFileUse
from nexusml.utils import get_s3_config
from nexusml.utils import s3_client

_OrgOrTaskFile = Union[OrgFile, TaskFile]

################
# Define views #
################

_OrgFileView = create_view(resource_types=[Organization, OrgFile])
_TaskFileView = create_view(resource_types=[Task, TaskFile])

_file_url_params = {'thumbnail': fields.Boolean(description='Get download URL for the thumbnail (only for images)')}
_files_url_params = {
    **paging_url_params(collection_name='files'), 'order_by':
        fields.String(description='Parameter to order by. Default: "created_at"'),
    'order':
        fields.String(description='"asc" (ascending) or "desc" (descending). Default: "desc"'),
    'created_at':
        fields.String(description='Files created at the given datetime'),
    'created_at[min]':
        fields.String(description='Files created after the given datetime (inclusive)'),
    'created_at[max]':
        fields.String(description='Files created before the given datetime (inclusive)'),
    **_file_url_params
}

_org_uses = ' | '.join([f'"{x.name.lower()}"' for x in OrgFileUse])
_org_types = ' | '.join([f'"{x.name.lower()}"' for x in FileType])
_org_files_url_params = {
    **_files_url_params, 'use_for': fields.String(description=f'Files used for: {_org_uses}'),
    'type': fields.String(description=f'Files of type: {_org_types}')
}

_task_uses = ' | '.join([f'"{x.name.lower()}"' for x in TaskFileUse])
_task_types = ' | '.join([f'"{x.name.lower()}"' for x in FileType])
_task_files_url_params = {
    **_files_url_params, 'use_for': fields.String(description=f'Files used for: {_task_uses}'),
    'type': fields.String(description=f'Files of type: {_task_types}')
}

_files_url_params_desc = (
    'To represent AND and OR operators within the value of a query parameter, use "," for AND and "|" for OR. '
    'For example: `type=document|image` (OR)\n\nFor datetimes, use ISO 8601 format'
    '(YYYY-MM-DDTHH:MM:SS), e.g.: `created_at=2021-04-28T16:24:03`')

####################
# Helper functions #
####################


def _get_files(parent: Union[Organization, Task], url_params: dict) -> Response:
    """
    Retrieves files based on the provided parent organization or task and filters.

    The function applies various filters, including datetime filters and type filters,
    to query files associated with the specified parent. It supports both organization
    and task contexts, returning the appropriate file results paginated and ordered
    according to the provided URL parameters.

    Steps:
    1. Identify if the parent is an organization or task and set corresponding models.
    2. Apply filters such as 'created_at', 'use_for', and 'type' from the URL parameters.
    3. Query the database for files associated with the parent.
    4. Order the results based on the 'order_by' and 'order' parameters.
    5. Paginate the results and return them as a JSON response.

    Args:
        parent (Union[Organization, Task]): The parent resource (organization or task) for which to retrieve files.
        url_params (dict): Dictionary of URL parameters for filtering, ordering, and pagination.

    Returns:
        Response: The response containing the paginated list of files.

    Raises:
        UnprocessableRequestError: If the query or order parameters are malformed.
    """
    if isinstance(parent, Organization):
        files_db_model = OrgFileDB
        parent_pk_col = 'organization_id'
        resource_type = OrgFile
    else:
        files_db_model = TaskFileDB
        parent_pk_col = 'task_id'
        resource_type = TaskFile

    # Apply filters
    filters = []

    for param in ['created_at', 'created_at' + RANGE_MIN, 'created_at' + RANGE_MAX]:
        if param not in url_params:
            continue

        try:
            dt_filter = build_datetime_filter(db_model=files_db_model,
                                              datetime_field=param,
                                              datetime_value=url_params[param])
            filters.append(dt_filter)
        except ValueError as e:
            return error_response(code=HTTP_BAD_REQUEST_STATUS_CODE,
                                  message=f'Invalid query. Malformed value for `{param}`: "{str(e)}"')

    if 'use_for' in url_params:
        use_filter = build_field_filter(db_model=files_db_model, field='use_for', value=url_params['use_for'])
        filters.append(use_filter)

    if 'type' in url_params:
        type_filter = build_field_filter(db_model=files_db_model, field='type_', value=url_params['type'])
        filters.append(type_filter)

    db_query = (files_db_model.query().filter(
        getattr(files_db_model, parent_pk_col) == getattr(parent.db_object(), parent_pk_col)))

    if filters:
        db_query = db_query.filter(*filters)

    # Get file ordering
    order_by_field = url_params.get('order_by', 'created_at').lower()
    if order_by_field not in ['created_at', 'use_for', 'type']:
        return error_response(code=HTTP_BAD_REQUEST_STATUS_CODE,
                              message=f'Invalid ordering criterion "{order_by_field}"')
    order_by_col = getattr(files_db_model, order_by_field)

    order = url_params.get('order', 'desc').lower()
    if order not in ['asc', 'desc']:
        return error_response(code=HTTP_BAD_REQUEST_STATUS_CODE, message='Invalid ordering')
    if order == 'desc':
        order_by_col = order_by_col.desc()

    # Return specified page
    # TODO: optimize like `views.examples.Examples.get()`
    res_json: dict = get_page_resources(query=db_query,
                                        page_number=url_params['page'],
                                        per_page=url_params['per_page'],
                                        order_by=order_by_col,
                                        total_count=url_params['total_count'],
                                        resource_type=resource_type,
                                        parents=[parent],
                                        dump_args=({
                                            'thumbnail': True
                                        } if url_params.get('thumbnail') else None))
    return jsonify(res_json)


def _get_file_parts_in_local_store(file: _OrgOrTaskFile) -> List[str]:
    """
    Retrieves file parts from the local storage directory.

    This function checks if the specified file has parts stored in the local directory. If the directory exists,
    it lists all files in the directory and filters out those that match the expected part filenames.

    Args:
        file (_OrgOrTaskFile): The file resource to check for stored parts.

    Returns:
        List[str]: A list of file part paths in the local storage.
    """
    uploads_dir = os.path.dirname(file.path())
    if not os.path.isdir(uploads_dir):
        return []
    all_uploaded_files = [os.path.join(uploads_dir, f) for f in os.listdir(uploads_dir)]
    return [f for f in all_uploaded_files if f.startswith(file.path() + '.part')]


def _post_file_part(file: _OrgOrTaskFile, part_number: int):
    """
    Generates a pre-signed URL for uploading a part of a file in a multipart upload.

    This function checks if the file already exists, verifies that the file is eligible for multipart upload,
    retrieves or creates an upload entry in the database, and generates a pre-signed URL (either for S3 or local
    storage) to upload the specified file part.

    Steps:
    1. Retrieve the storage backend configuration (S3 or local).
    2. Check if the file has already been uploaded, raising an error if it has.
    3. Validate that multipart upload is supported for files larger than the max upload size.
    4. Get or create the upload database entry.
    5. Verify the total size of parts uploaded so far.
    6. Generate and return a pre-signed URL for uploading the file part.

    Args:
        file (_OrgOrTaskFile): The file resource being uploaded.
        part_number (int): The part number of the file to be uploaded.

    Returns:
        Response: The response containing the upload URL.

    Raises:
        DuplicateResourceError: If the file or part has already been uploaded.
        UnprocessableRequestError: If the file is too small for multipart upload or exceeds the allowed size.
    """
    # Get file store config
    file_storage_backend = get_file_storage_backend()
    file_storage_config = get_s3_config(
    ) if file_storage_backend == FileStorageBackend.S3 else get_local_file_storage_config()

    # Get the maximum upload size
    max_upload_size = file_storage_config['max_upload_size']
    max_upload_size_mb = round(max_upload_size / (1024**2))

    # Verify the file was not uploaded already
    _ALREADY_UPLOADED_ERR_MSG = 'File already uploaded'
    if file_storage_backend == FileStorageBackend.S3:
        if 'Contents' in s3_client().list_objects_v2(Bucket=file_storage_config['bucket'], Prefix=file.path()):
            raise DuplicateResourceError(_ALREADY_UPLOADED_ERR_MSG)
    else:
        if os.path.exists(file.path()):
            raise DuplicateResourceError(_ALREADY_UPLOADED_ERR_MSG)

    # Verify multipart upload support for current file
    if file.db_object().size <= max_upload_size:
        raise UnprocessableRequestError(f'Multipart uploads are not supported for files smaller than'
                                        f'{max_upload_size_mb} MB')

    # Get or create the upload database entry
    upload_model = OrgUpload if isinstance(file, OrgFile) else TaskUpload
    file_upload = upload_model.query().filter_by(file_id=file.db_object().file_id).first()
    if not file_upload:
        # Get upload ID
        if file_storage_backend == FileStorageBackend.S3:
            s3_response = s3_client().create_multipart_upload(Bucket=file_storage_config['bucket'], Key=file.path())
            upload_id = s3_response['UploadId']
        else:
            upload_id = str(uuid.uuid4())
        # Save upload entry to database
        file_upload = upload_model(file_id=file.db_object().file_id, upload_id=upload_id)
        save_to_db(file_upload)

    # Verify that the total size of the parts uploaded so far doesn't exceed declared file size.
    _verify_parts_size(file=file, upload=file_upload, is_complete=False)

    # Generate the upload URL
    if file_storage_backend == FileStorageBackend.S3:
        # TODO: Check why presigned URLs don't allow limiting the maximum upload size (unlike presigned POST).
        upload_url = s3_client().generate_presigned_url(ClientMethod='upload_part',
                                                        Params={
                                                            'Bucket': file_storage_config['bucket'],
                                                            'Key': file.path(),
                                                            'UploadId': file_upload.upload_id,
                                                            'PartNumber': part_number
                                                        },
                                                        ExpiresIn=file_storage_config['url_expiration'])
    else:
        parent = file.parents()[0]
        # Generate a temporary token
        token_claims = {
            'file_id': file.uuid(),
            'agent_id': str(file.agent().uuid),
            'agent_type': 'user' if isinstance(file.agent(), UserDB) else 'client'
        }
        expires_in = get_local_file_storage_config()['url_expiration']
        token = generate_tmp_token(agent_uuid=str(file.agent().uuid),
                                   expires_in=expires_in,
                                   custom_claims=token_claims,
                                   custom_claims_key='request')
        # Generate the upload URL
        if isinstance(file, OrgFile):
            upload_endpoint = url_for('files.OrgLocalStoreMultipartUploadView'.lower(),
                                      organization_id=parent.uuid(),
                                      upload_id=file_upload.upload_id,
                                      part_number=part_number,
                                      token=token)
        else:
            upload_endpoint = url_for('files.TaskLocalStoreMultipartUploadView'.lower(),
                                      task_id=parent.uuid(),
                                      upload_id=file_upload.upload_id,
                                      part_number=part_number,
                                      token=token)
        upload_url = API_DOMAIN + upload_endpoint

    # Return the upload URL
    response = jsonify({'upload_url': upload_url})
    response.location = upload_url
    return response


def _complete_multipart_upload(file: _OrgOrTaskFile, uploaded_parts: dict):
    """
    Completes a multipart file upload by verifying the uploaded parts and merging them into the final file.

    This function first retrieves the file store configuration and database entry for the ongoing upload.
    It then verifies that the uploaded parts match the declared file size and completes the upload either
    by merging parts in S3 or the local store, depending on the storage backend in use.

    Steps:
    1. Retrieve the file store configuration and the upload entry from the database.
    2. Verify that the size of the uploaded parts matches the declared file size.
    3. Complete the multipart upload (either in S3 or local store).
    4. Delete the upload entry from the database.
    5. Build and return a response indicating successful completion.

    Args:
        file (_OrgOrTaskFile): The file being uploaded.
        uploaded_parts (dict): A dictionary of uploaded parts with their respective ETags and part numbers.

    Returns:
        Response: A 204 No Content response indicating the successful completion of the upload.

    Raises:
        UnprocessableRequestError: If no upload is found, or if there are inconsistencies with the parts size.
        InvalidDataError: If S3 upload completion returns invalid data.
    """
    # Get file store config
    file_storage_backend = get_file_storage_backend()
    file_storage_config = get_s3_config(
    ) if file_storage_backend == FileStorageBackend.S3 else get_local_file_storage_config()

    # Get the upload database entry
    upload_model = OrgUpload if isinstance(file, OrgFile) else TaskUpload
    file_upload = upload_model.query().filter_by(file_id=file.db_object().file_id).first()
    if file_upload is None:
        raise UnprocessableRequestError('No upload to complete')

    # Verify that parts size matches declared file size
    _verify_parts_size(file=file, upload=file_upload, is_complete=True)

    # Complete multipart upload
    if file_storage_backend == FileStorageBackend.S3:
        # Complete the upload in S3
        uploaded_parts = [{'ETag': x['etag'], 'PartNumber': x['part_number']} for x in uploaded_parts]
        s3_response = s3_client().complete_multipart_upload(Bucket=file_storage_config['bucket'],
                                                            Key=file.path(),
                                                            MultipartUpload={'Parts': uploaded_parts},
                                                            UploadId=file_upload.upload_id)
        if s3_response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise InvalidDataError(f'Invalid ETags: "{s3_response.text}"')
    else:
        # Complete the upload in the local store
        _complete_multipart_upload_in_local_store(file=file, uploaded_parts=uploaded_parts)

    # Delete upload entry from database
    delete_from_db(file_upload)

    # Build response
    response = Response()
    response.status_code = 204
    response.location = file.url()
    return response


def _complete_multipart_upload_in_local_store(file: _OrgOrTaskFile, uploaded_parts: dict):
    """
    Completes the multipart upload by merging all uploaded parts in the local store.

    This function checks if all file parts are uploaded and ensures they are consecutive.
    It then merges the parts into the final file and deletes the part files from the local storage.

    Steps:
    1. Retrieve all uploaded file parts from the local storage.
    2. Verify that all parts are uploaded and in consecutive order.
    3. Merge all parts into the final file.
    4. Delete the individual part files.

    Args:
        file (_OrgOrTaskFile): The file being uploaded.
        uploaded_parts (dict): A dictionary of uploaded parts with their respective part numbers.

    Raises:
        UnprocessableRequestError: If any parts are missing or out of order.
    """
    # Get the uploaded file parts
    file_parts = _get_file_parts_in_local_store(file)

    # Verify that all parts are uploaded
    uploaded_part_numbers = [x['part_number'] for x in uploaded_parts]
    if uploaded_part_numbers != list(range(1, len(file_parts) + 1)):
        raise UnprocessableRequestError('Some of the parts are missing')

    # Verify that the uploaded file parts are consecutive
    expected_parts = [f'{file.path()}.part{idx + 1}' for idx in range(len(file_parts))]
    if file_parts != expected_parts:
        raise UnprocessableRequestError('File parts are not consecutive')

    # Merge parts into the final file
    with open(file.path(), 'wb') as final_file:
        for part_number in uploaded_part_numbers:
            part_path = f'{file.path()}.part{part_number}'
            with open(part_path, 'rb') as file_part:
                final_file.write(file_part.read())
            os.remove(part_path)


def _verify_parts_size(file: _OrgOrTaskFile, upload: Union[OrgUpload, TaskUpload], is_complete: bool):
    """
    Verifies that the parts size is consistent with the declared file size at the specified stage
    (complete or in-progress). In case of inconsistency, the upload is aborted and an error is raised.

    Steps:
    1. Retrieve the storage configuration and declared file size.
    2. Calculate the total size of the uploaded parts.
    3. If the parts size exceeds the declared file size, abort the upload.
    4. If the upload is complete, verify that the parts size matches the declared file size.
    5. For ongoing uploads, ensure the parts size is smaller than the declared file size.

    Args:
        file (_OrgOrTaskFile): The file being uploaded.
        upload (Union[OrgUpload, TaskUpload]): The upload database entry.
        is_complete (bool): Flag indicating whether the upload is complete.

    Raises:
        UnprocessableRequestError: If the parts size exceeds the declared file size or if parts are missing.
    """
    _EXCEEDED_ERR_MSG = 'Parts size exceeds declared file size. Upload aborted'

    # Get file store config
    file_storage_backend = get_file_storage_backend()
    file_storage_config = get_s3_config(
    ) if file_storage_backend == FileStorageBackend.S3 else get_local_file_storage_config()

    # Get declared file size
    file_size = file.db_object().size

    # Get parts size
    if file_storage_backend == FileStorageBackend.S3:
        s3_response = s3_client().list_parts(Bucket=file_storage_config['bucket'],
                                             Key=file.path(),
                                             UploadId=upload.upload_id,
                                             MaxParts=math.ceil(file_size / MIN_FILE_PART_SIZE))
        parts_size = sum(x['Size'] for x in s3_response.get('Parts', []))
    else:
        parts_size = sum(os.path.getsize(p) for p in _get_file_parts_in_local_store(file))

    # If parts size is larger than file size, abort the upload and exit (no matter if it's complete or not)
    if parts_size > file_size:
        _abort_multipart_upload(file=file, upload=upload)
        raise UnprocessableRequestError(_EXCEEDED_ERR_MSG)

    # In remaining cases, complete and in-progress uploads are verified in different ways
    if is_complete:
        # If the upload is complete, verify that the parts size matches the declared file size
        if parts_size < file_size:
            # If the parts size is smaller than the declared file size, raise an error but don't abort the upload
            raise UnprocessableRequestError('File not uploaded completely. Some parts are missing')
    else:
        # If the upload is in progress, verify that the parts size is smaller the declared file size
        if parts_size == file_size:
            _abort_multipart_upload(file=file, upload=upload)
            raise UnprocessableRequestError(_EXCEEDED_ERR_MSG)


def _abort_multipart_upload(file: _OrgOrTaskFile, upload: Union[OrgUpload, TaskUpload]):
    """
    Aborts an ongoing multipart upload by removing any uploaded parts and the database entry.

    This function handles aborting a multipart upload in both S3 and local storage backends.
    It deletes all uploaded parts and removes the upload record from the database to prevent
    any further continuation of the upload process.

    Steps:
    1. Retrieve the file store configuration.
    2. Abort the upload in S3 or delete uploaded file parts in the local store.
    3. Remove the corresponding upload entry from the database.

    Args:
        file (_OrgOrTaskFile): The file being uploaded.
        upload (Union[OrgUpload, TaskUpload]): The upload database entry.

    Raises:
        RuntimeError: If the S3 abort operation fails.
    """
    # Get file store config
    file_storage_backend = get_file_storage_backend()
    file_storage_config = get_s3_config(
    ) if file_storage_backend == FileStorageBackend.S3 else get_local_file_storage_config()

    # Delete file parts in the local store
    if file_storage_backend == FileStorageBackend.S3:
        # Abort the upload in S3
        s3_response = s3_client().abort_multipart_upload(Bucket=file_storage_config['bucket'],
                                                         Key=file.path(),
                                                         UploadId=upload.upload_id)
        if s3_response['ResponseMetadata']['HTTPStatusCode'] != 204:
            raise RuntimeError('Failed to abort the upload in S3')
    else:
        # Delete files from the local store
        for part_path in _get_file_parts_in_local_store(file):
            os.remove(part_path)

    # Delete upload entry from database
    delete_from_db(upload)


######################
# Organization files #
######################


class OrgFilesView(_OrgFileView):
    """
    View for handling requests related to organization files.

    This class provides methods to retrieve organization files or upload metadata for a new file
    and return a pre-signed URL for uploading the actual file content. It uses the organization
    as the parent resource, applying permissions and filters as required.
    """

    @doc(tags=[SWAGGER_TAG_FILES], description=_files_url_params_desc)
    @use_kwargs(_org_files_url_params, location='query')
    @marshal_with(OrganizationFilesPage)
    def get(self, organization_id: str, resources: List[Resource], **kwargs):
        """
        Retrieves files associated with the specified organization based on the provided filters.

        This method uses the organization as the parent context and checks the agent's permissions
        to ensure that the agent has access to the files. It then applies filters and returns a
        paginated list of files.

        Args:
            organization_id (str): The ID of the organization.
            resources (List[Resource]): List of loaded resource objects, where the last is the organization.
            **kwargs: URL query parameters for filtering, ordering, and pagination.

        Returns:
            dict: A paginated list of organization files.
        """
        # Get parent
        organization = resources[-1]
        assert isinstance(organization, Organization)
        # Check permissions
        agent = agent_from_token()
        if isinstance(agent, UserDB):
            OrgFile.check_permissions(organization=organization.db_object(), action=ResourceAction.READ, user=agent)
        # Return files
        return _get_files(parent=organization, url_params=kwargs)

    @doc(tags=[SWAGGER_TAG_FILES], description="Upload file's metadata and get the S3 presigned POST URL")
    @use_kwargs(OrganizationFileRequest, location='json')
    @marshal_with(OrganizationFileResponse)
    def post(self, organization_id: str, resources: List[Resource], **kwargs):
        """
        Uploads metadata for a new file and returns the pre-signed URL to upload the file.

        This method processes the metadata for a new organization file, ensuring that the
        agent has permission to create the file. It returns the pre-signed URL for uploading
        the file content to S3 or local storage.

        Args:
            organization_id (str): The ID of the organization.
            resources (List[Resource]): List of loaded resource objects, where the last is the organization.
            **kwargs: JSON request body containing file metadata.

        Returns:
            dict: The file metadata and pre-signed URL for uploading the file content.
        """
        return process_post_or_put_request(agent=resources[-1].agent(),
                                           resource_or_model=OrgFile,
                                           parents=resources,
                                           json=kwargs)


class OrgFileView(_OrgFileView):
    """
    View for handling requests related to individual organization files.

    This class provides methods to retrieve or delete a specific file associated with an organization.
    """

    @doc(tags=[SWAGGER_TAG_FILES])
    def delete(self, organization_id: str, file_id: str, resources: List[Resource]):
        """
        Deletes the specified file from the organization.

        This method deletes the file resource from the organization after checking the necessary
        permissions.

        Args:
            organization_id (str): The ID of the organization.
            file_id (str): The ID of the file to delete.
            resources (List[Resource]): List of loaded resource objects, where the last is the file.

        Returns:
            Response: A response indicating the result of the deletion.
        """
        return process_delete_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_FILES])
    @use_kwargs(_file_url_params, location='query')
    @marshal_with(OrganizationFileResponse)
    def get(self, organization_id: str, file_id: str, resources: List[Resource], **kwargs):
        """
        Retrieves the specified file associated with the organization.

        This method returns the file's metadata and optionally its thumbnail (if requested) based on
        the provided query parameters.

        Args:
            organization_id (str): The ID of the organization.
            file_id (str): The ID of the file to retrieve.
            resources (List[Resource]): List of loaded resource objects, where the last is the file.
            **kwargs: URL query parameters, including an optional 'thumbnail' parameter.

        Returns:
            dict: The file's metadata and URL.
        """
        dump_args = {'thumbnail': True} if kwargs.get('thumbnail') else None
        return process_get_request(resource=resources[-1], dump_args=dump_args)


class OrgFilePartsView(_OrgFileView):
    """
    View for handling multipart file upload parts for an organization file.

    This class provides a method to retrieve a pre-signed URL for uploading a specific file part
    for a multipart upload.
    """

    @doc(tags=[SWAGGER_TAG_FILES], description='Get the URL to upload a file part (only for multipart uploads)')
    @use_kwargs(FilePartUploadRequest, location='json')
    @marshal_with(FilePartUploadResponse)
    def post(self, organization_id: str, file_id: str, resources: List[Resource], **kwargs):
        """
        Retrieves a pre-signed URL for uploading a part of the organization file.

        This method generates a pre-signed URL to upload a specific part of the file in a multipart
        upload process.

        Args:
            organization_id (str): The ID of the organization.
            file_id (str): The ID of the file.
            resources (List[Resource]): List of loaded resource objects, where the last is the file.
            **kwargs: JSON request body containing the part number.

        Returns:
            dict: A pre-signed URL for uploading the specified file part.
        """
        return _post_file_part(file=resources[-1], part_number=kwargs['part_number'])


class OrgFilePartsCompletionView(_OrgFileView):
    """
    View for completing a multipart upload for an organization file.

    This class provides a method to complete the multipart upload process by verifying
    the uploaded parts and merging them into the final file.
    """

    @doc(tags=[SWAGGER_TAG_FILES], description='Completes a multipart upload')
    @use_kwargs(FileUploadCompletionRequest, location='json')
    def post(self, organization_id: str, file_id: str, resources: List[Resource], **kwargs):
        """
        Completes the multipart upload for the specified organization file.

        This method verifies the uploaded parts and merges them into the final file, completing
        the multipart upload process.

        Args:
            organization_id (str): The ID of the organization.
            file_id (str): The ID of the file.
            resources (List[Resource]): List of loaded resource objects, where the last is the file.
            **kwargs: JSON request body containing the list of uploaded parts.

        Returns:
            Response: A response indicating the successful completion of the upload.
        """
        return _complete_multipart_upload(file=resources[-1], uploaded_parts=kwargs['uploaded_parts'])


##############
# Task files #
##############


class TaskFilesView(_TaskFileView):
    """
    View for handling requests related to task files.

    This class provides methods to retrieve task files or upload metadata for a new file
    and return a pre-signed URL for uploading the actual file content. It uses the task
    as the parent resource, applying permissions and filters as required.
    """

    @doc(tags=[SWAGGER_TAG_FILES], description=_files_url_params_desc)
    @use_kwargs(_task_files_url_params, location='query')
    @marshal_with(TaskFilesPage)
    def get(self, task_id: str, resources: List[Resource], **kwargs):
        """
        Retrieves files associated with the specified task based on the provided filters.

        This method uses the task as the parent context and checks the agent's permissions
        to ensure that the agent has access to the files. It then applies filters and returns
        a paginated list of files.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): List of loaded resource objects, where the last is the task.
            **kwargs: URL query parameters for filtering, ordering, and pagination.

        Returns:
            dict: A paginated list of task files.
        """
        # Get parent
        task = resources[-1]
        assert isinstance(task, Task)
        # Check permissions
        agent = agent_from_token()
        if isinstance(agent, UserDB):
            TaskFile.check_permissions(organization=task.db_object().organization,
                                       action=ResourceAction.READ,
                                       user=agent)
        # Return files
        return _get_files(parent=task, url_params=kwargs)

    @doc(tags=[SWAGGER_TAG_FILES], description="Upload file's metadata and get the S3 presigned POST URL")
    @use_kwargs(TaskFileRequest, location='json')
    @marshal_with(TaskFileResponse)
    def post(self, task_id: str, resources: List[Resource], **kwargs):
        """
        Uploads metadata for a new file and returns the pre-signed URL to upload the file.

        This method processes the metadata for a new task file, ensuring that the
        agent has permission to create the file. It returns the pre-signed URL for uploading
        the file content to S3 or local storage.

        Args:
            task_id (str): The ID of the task.
            resources (List[Resource]): List of loaded resource objects, where the last is the task.
            **kwargs: JSON request body containing file metadata.

        Returns:
            dict: The file metadata and pre-signed URL for uploading the file content.
        """
        return process_post_or_put_request(agent=resources[-1].agent(),
                                           resource_or_model=TaskFile,
                                           parents=resources,
                                           json=kwargs)


class TaskFileView(_TaskFileView):
    """
    View for handling requests related to individual task files.

    This class provides methods to retrieve or delete a specific file associated with a task.
    """

    @doc(tags=[SWAGGER_TAG_FILES])
    def delete(self, task_id: str, file_id: str, resources: List[Resource]):
        """
        Deletes the specified file from the task.

        This method deletes the file resource from the task after checking the necessary
        permissions.

        Args:
            task_id (str): The ID of the task.
            file_id (str): The ID of the file to delete.
            resources (List[Resource]): List of loaded resource objects, where the last is the file.

        Returns:
            Response: A response indicating the result of the deletion.
        """
        return process_delete_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_FILES])
    @use_kwargs(_file_url_params, location='query')
    @marshal_with(TaskFileResponse)
    def get(self, task_id: str, file_id: str, resources: List[Resource], **kwargs):
        """
        Retrieves the specified file associated with the task.

        This method returns the file's metadata and optionally its thumbnail (if requested) based on
        the provided query parameters.

        Args:
            task_id (str): The ID of the task.
            file_id (str): The ID of the file to retrieve.
            resources (List[Resource]): List of loaded resource objects, where the last is the file.
            **kwargs: URL query parameters, including an optional 'thumbnail' parameter.

        Returns:
            dict: The file's metadata and URL.
        """
        dump_args = {'thumbnail': True} if kwargs.get('thumbnail') else None
        return process_get_request(resource=resources[-1], dump_args=dump_args)


class TaskFilePartsView(_TaskFileView):
    """
    View for handling multipart file upload parts for a task file.

    This class provides a method to retrieve a pre-signed URL for uploading a specific file part
    for a multipart upload.
    """

    @doc(tags=[SWAGGER_TAG_FILES], description='Get the URL to upload a file part (only for multipart uploads)')
    @use_kwargs(FilePartUploadRequest, location='json')
    @marshal_with(FilePartUploadResponse)
    def post(self, task_id: str, file_id: str, resources: List[Resource], **kwargs):
        """
        Retrieves a pre-signed URL for uploading a part of the task file.

        This method generates a pre-signed URL to upload a specific part of the file in a multipart
        upload process.

        Args:
            task_id (str): The ID of the task.
            file_id (str): The ID of the file.
            resources (List[Resource]): List of loaded resource objects, where the last is the file.
            **kwargs: JSON request body containing the part number.

        Returns:
            dict: A pre-signed URL for uploading the specified file part.
        """
        return _post_file_part(file=resources[-1], part_number=kwargs['part_number'])


class TaskFilePartsCompletionView(_TaskFileView):
    """
    View for completing a multipart upload for a task file.

    This class provides a method to complete the multipart upload process by verifying
    the uploaded parts and merging them into the final file.
    """

    @doc(tags=[SWAGGER_TAG_FILES], description='Completes a multipart upload')
    @use_kwargs(FileUploadCompletionRequest, location='json')
    def post(self, task_id: str, file_id: str, resources: List[Resource], **kwargs):
        """
        Completes the multipart upload for the specified task file.

        This method verifies the uploaded parts and merges them into the final file, completing
        the multipart upload process.

        Args:
            task_id (str): The ID of the task.
            file_id (str): The ID of the file.
            resources (List[Resource]): List of loaded resource objects, where the last is the file.
            **kwargs: JSON request body containing the list of uploaded parts.

        Returns:
            Response: A response indicating the successful completion of the upload.
        """
        return _complete_multipart_upload(file=resources[-1], uploaded_parts=kwargs['uploaded_parts'])


#####################
# Local store views #
#####################


def _decode_tmp_token(enc_token: str) -> dict:
    """
    Decodes a temporary token used for local file store authentication.

    This function decodes the temporary token using the RSA public key, verifies the token's
    issuer, and checks that the API version matches the current API version. The token must
    be issued by the current API.

    Args:
        enc_token (str): The encoded temporary token.

    Returns:
        dict: The decoded token.

    Raises:
        UnprocessableRequestError: If the token is issued by a different API version.
    """
    dec_token = jwt.decode(jwt=enc_token,
                           key=config.rsa_public_key(),
                           algorithms=['RS256'],
                           issuer=API_NAME,
                           options={'verify_aud': False})
    if dec_token['api_version'] != API_VERSION:
        raise UnprocessableRequestError('Token not issued by current API version')
    return dec_token


def _verify_tmp_token(dec_token: dict, file: _OrgOrTaskFile):
    """
    Verifies the temporary token by checking that it corresponds to the correct file.

    This function compares the file ID in the decoded token with the UUID of the provided file
    to ensure the token is valid for the requested file.

    Args:
        dec_token (dict): The decoded token.
        file (_OrgOrTaskFile): The file being accessed.

    Raises:
        ResourceNotFoundError: If the file ID in the token does not match the file's UUID.
    """
    if dec_token['request']['file_id'] != file.uuid():
        raise ResourceNotFoundError()


def _get_agent_from_tmp_token(dec_token: dict) -> Agent:
    """
    Retrieves the agent (user or client) based on the temporary token's claims.

    This function determines whether the agent is a user or client based on the agent type
    in the decoded token and retrieves the corresponding agent from the database.

    Args:
        dec_token (dict): The decoded token containing the agent type and ID.

    Returns:
        Agent: The agent (user or client) associated with the token.
    """
    agent_db_model = UserDB if dec_token['request']['agent_type'] == 'user' else ClientDB
    return agent_db_model.get_from_id(id_value=dec_token['request']['agent_id'])


def _check_local_store_root_path(func):
    """
    Decorator to check if the local store's root path exists and create it if it doesn't.

    This function ensures that the root directory for the local file store exists before
    performing any file operations. If the directory does not exist, it creates it.

    Args:
        func: The function to wrap with the check for the root path.

    Returns:
        The wrapped function.
    """

    def wrapper(*args, **kwargs):
        # Check if the root path exists and create it if it doesn't
        root_path = get_local_file_storage_config()['root_path']
        if not os.path.isdir(root_path):
            os.makedirs(root_path)
        # Call the function
        return func(*args, **kwargs)

    return wrapper


def _verify_request_data_length(request_data_length: int):
    """
    Verifies that the length of the file upload data does not exceed the maximum allowed size.

    This function compares the file content length to the maximum upload size allowed by the
    local storage configuration. If the size is exceeded, it raises an error.

    Args:
        request_data_length (int): The length of the request data in bytes.

    Raises:
        UnprocessableRequestError: If the request data exceeds the maximum upload size.
    """
    max_upload_size = get_local_file_storage_config()['max_upload_size']
    if request_data_length > max_upload_size:
        max_upload_size_mb = round(max_upload_size / (1024**2))
        raise UnprocessableRequestError(f'Maximum upload size exceeded: {max_upload_size_mb} MB')


def _load_file_and_auth_info(parent_id: str, file_id: str, file_type: Type[_OrgOrTaskFile],
                             token: str) -> Tuple[_OrgOrTaskFile, Agent, dict]:
    """
    Loads file metadata and agent information based on the provided token and file type.

    This function decodes the token, retrieves the associated agent, and loads the file
    metadata based on the parent resource (organization or task). It also verifies the
    token for correctness.

    Args:
        parent_id (str): The ID of the parent organization or task.
        file_id (str): The ID of the file.
        file_type (Type[_OrgOrTaskFile]): The type of the file (OrgFile or TaskFile).
        token (str): The temporary token for authentication.

    Returns:
        Tuple[_OrgOrTaskFile, Agent, dict]: The file, the agent, and the decoded token.
    """
    # Get the agent from the provided token
    dec_token = _decode_tmp_token(enc_token=token)
    agent = _get_agent_from_tmp_token(dec_token=dec_token)

    # Load resources
    parent_type = Organization if file_type == OrgFile else Task
    resources = load_url_resources(agent=agent,
                                   resource_ids=[parent_id, file_id],
                                   resource_types=[parent_type, file_type])

    # Return file metadata, agent, and decoded token
    return resources[-1], agent, dec_token


@_check_local_store_root_path
def _download_file_from_local_store(parent_id: str, file_id: str, file_type: Type[_OrgOrTaskFile], url_params: dict):
    """
    Downloads file content from the local store.

    Note: The maximum file size limited by `app.config['MAX_CONTENT_LENGTH']` does not affect this function.
          `MAX_CONTENT_LENGTH` only affects incoming request bodies and does not limit the size of files being
          sent to clients using `send_file`.

    Args:
        parent_id (str): The ID of the parent organization or task.
        file_id (str): The file ID.
        file_type (Type[_OrgOrTaskFile]): The file type (OrgFile or TaskFile).
        url_params (dict): The URL parameters.

    Returns:
        Response: The response object containing the file to be downloaded.
    """
    # Get file metadata, request agent, and decoded token
    # TODO: Replace `agent` with `_` to remove the pylint disable warning.
    # pylint: disable-next=unused-variable
    file, agent, dec_token = _load_file_and_auth_info(parent_id=parent_id,
                                                      file_id=file_id,
                                                      file_type=file_type,
                                                      token=url_params['token'])

    # Verify the provided token
    _verify_tmp_token(dec_token=dec_token, file=file)

    # Get file path
    file_path = file.path(thumbnail=url_params.get('thumbnail', False))

    # Serve file content
    return send_file(path_or_file=file_path, as_attachment=True, etag=True)


@_check_local_store_root_path
def _upload_file_to_local_store(parent_id: str, file_id: str, file_type: Type[_OrgOrTaskFile], token: str):
    """
    Uploads file content to the local store.

    Note: The maximum file size is also limited by `app.config['MAX_CONTENT_LENGTH']`.

    Args:
        parent_id (str): The ID of the parent organization or task.
        file_id (str): The file ID.
        file_type (Type[_OrgOrTaskFile]): The file type (OrgFile or TaskFile).
        token (str): The temporary token for authentication.

    Returns:
        Response: A 200 OK response indicating successful file upload.

    Raises:
        InvalidDataError: If no file is provided.
        UnprocessableRequestError: If the maximum upload size is exceeded or if the file size is incorrect.
    """
    # Get file metadata, request agent, and decoded token
    file, agent, dec_token = _load_file_and_auth_info(parent_id=parent_id,
                                                      file_id=file_id,
                                                      file_type=file_type,
                                                      token=token)

    # Verify the provided token
    _verify_tmp_token(dec_token=dec_token, file=file)

    # Get the provided file
    if 'file' not in request.files:
        raise InvalidDataError('No file provided')
    request_file = request.files['file']

    # Get the file content length.
    # 1) Move cursor to the end of the file
    request_file.stream.seek(0, 2)
    # 2) Get the current cursor position (size in bytes)
    file_content_length = request_file.stream.tell()
    # 3) Reset the stream position back to the start for actual upload processing
    request_file.stream.seek(0)

    # If the content-length header is available, verify that it matches the actual file size
    if request_file.content_length and request_file.content_length != file_content_length:
        raise UnprocessableRequestError('Content-Length header does not match actual file size')

    # Verify that the maximum upload size is not exceeded
    _verify_request_data_length(request_data_length=file_content_length)

    # Check permissions to upload files
    if isinstance(file, OrgFile):
        organization = file.db_object().organization
    else:
        organization = file.db_object().task.organization

    type(file).check_permissions(organization=organization,
                                 action=ResourceAction.CREATE,
                                 user=agent if isinstance(agent, UserDB) else None)

    # Verify that the file size does not exceed the declared size
    if file_content_length > file.db_object().size:
        raise UnprocessableRequestError('File size exceeds declared size')

    # Create parent directories
    os.makedirs(os.path.dirname(file.path()), exist_ok=True)

    # Save the file content
    request_file.save(file.path())

    # Save the file thumbnail (for the moment, only for images)
    is_image = file.db_object().type_ == FileType.IMAGE
    has_image_extension = file.db_object().filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))
    if is_image or has_image_extension:
        request_file.stream.seek(0)  # Reset stream before saving the thumbnail
        thumbnail_path = file.path(thumbnail=True)
        os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)
        save_thumbnail_to_local_file_store(thumbnail_path=thumbnail_path, file_content=request_file.read())

    # Return response
    return Response(status=200)


@_check_local_store_root_path
def _upload_file_part_to_local_store(parent_id: str, parent_type: Type[Union[Organization, Task]], upload_id: str,
                                     part_number: int, token: str):
    """
    Uploads a file part to the local store during a multipart upload.

    This function handles the uploading of individual file parts for a multipart upload in the local store.
    It verifies the part number, token, and ensures that the total size of the uploaded parts does not exceed
    the declared file size. The part is then saved, and the corresponding ETag is returned in the response.

    Note: The multipart upload view has been designed to be similar to S3's.

    Args:
        parent_id (str): The ID of the parent organization or task.
        parent_type (Type[Union[Organization, Task]]): The parent type (Organization or Task).
        upload_id (str): The upload ID associated with the multipart upload.
        part_number (int): The part number being uploaded.
        token (str): The temporary token for authentication.

    Returns:
        Response: A response with the ETag for the uploaded part.

    Raises:
        InvalidDataError: If the part number is invalid.
        ResourceNotFoundError: If the upload or file metadata cannot be found.
        UnprocessableRequestError: If the maximum upload size is exceeded or
        if there are inconsistencies with part sizes.
    """
    if part_number < 1:
        raise InvalidDataError('Part number must be greater than 0')

    _UPLOAD_NOT_FOUND_ERR_MSG = 'Upload not found'

    # Get the agent from the provided token
    dec_token = _decode_tmp_token(enc_token=token)
    agent = _get_agent_from_tmp_token(dec_token=dec_token)

    # Get the parent organization or task
    parent = parent_type.get(agent=agent, db_object_or_id=parent_id)

    # Load upload database entry
    upload_model = OrgUpload if parent_type == Organization else TaskUpload
    file_upload = upload_model.get(upload_id=upload_id)
    if file_upload is None:
        raise ResourceNotFoundError(_UPLOAD_NOT_FOUND_ERR_MSG)

    # Load file metadata
    file_type = OrgFile if parent_type == Organization else TaskFile
    file_metadata_db_obj = file_type.db_model().get(file_id=file_upload.file_id)
    file_metadata = file_type.get(agent=agent, db_object_or_id=file_metadata_db_obj, parents=[parent])

    # Verify the provided token
    _verify_tmp_token(dec_token=dec_token, file=file_metadata)

    # Verify that the specified upload belongs to the parent
    parent_col_id = 'organization_id' if parent_type == Organization else 'task_id'
    if getattr(file_metadata_db_obj, parent_col_id) != getattr(parent.db_object(), parent_col_id):
        raise ResourceNotFoundError(_UPLOAD_NOT_FOUND_ERR_MSG)

    # Verify that the maximum upload size is not exceeded
    _verify_request_data_length(request_data_length=len(request.data))

    # Verify that the total size of the parts uploaded so far doesn't exceed declared file size
    _verify_parts_size(file=file_metadata, upload=file_upload, is_complete=False)

    # Create parent directories
    os.makedirs(os.path.dirname(file_metadata.path()), exist_ok=True)

    # Save the file part
    part_path = f'{file_metadata.path()}.part{part_number}'
    with open(part_path, 'wb') as part_file:
        part_file.write(request.data)

    # Generate the ETag
    etag = hashlib.md5(request.data).hexdigest()

    # Return response
    response = Response(status=200)
    response.headers['ETag'] = etag
    return response


_local_store_auth_fields = {'token': fields.String(required=True, description='Temporary token')}

_local_store_download_url_params = {**_local_store_auth_fields, **_file_url_params}

_local_store_upload_form_fields = dict(_local_store_auth_fields)

_local_store_multipart_upload_url_params = {
    'part_number': fields.Integer(required=True, validate=validate.Range(min=1), description='Part number'),
    **_local_store_auth_fields
}


class OrgLocalStoreDownloadView(MethodResource):
    """
    View for downloading a file from the local store for an organization.

    This class provides a method to download a file (or thumbnail) from the local store.
    """

    @doc(tags=[SWAGGER_TAG_FILES], description='Download a file from the local store')
    @use_kwargs(_local_store_download_url_params, location='query')
    @validate_payload_size
    @limiter.limit(rate_limits)
    @capture_request_errors
    @capture_schema_errors
    def get(self, organization_id: str, file_id: str, **kwargs):
        """
        Downloads the specified file or thumbnail from the local store for the organization.

        This method retrieves the file content and sends it as a response to the client.

        Args:
            organization_id (str): The ID of the organization.
            file_id (str): The ID of the file to download.
            **kwargs: URL query parameters, including the 'token' for authentication and optional 'thumbnail' flag.

        Returns:
            Response: A response containing the file content.
        """
        return _download_file_from_local_store(parent_id=organization_id,
                                               file_id=file_id,
                                               file_type=OrgFile,
                                               url_params=kwargs)


class OrgLocalStoreUploadView(MethodResource):
    """
    View for uploading a file to the local store for an organization.

    This class provides a method to upload a file to the local store.
    """

    @doc(tags=[SWAGGER_TAG_FILES], description='Upload a file to the local store')
    @use_kwargs(_local_store_upload_form_fields, location='form')
    @validate_payload_size
    @limiter.limit(rate_limits)
    @capture_request_errors
    @capture_schema_errors
    def post(self, organization_id: str, file_id: str, **kwargs):
        """
        Uploads a file to the local store for the specified organization.

        This method handles the file upload process, including verifying permissions and saving the file.

        Args:
            organization_id (str): The ID of the organization.
            file_id (str): The ID of the file to upload.
            **kwargs: Form data, including the 'token' for authentication.

        Returns:
            Response: A 200 OK response indicating successful file upload.
        """
        return _upload_file_to_local_store(parent_id=organization_id,
                                           file_id=file_id,
                                           file_type=OrgFile,
                                           token=kwargs['token'])


class OrgLocalStoreMultipartUploadView(MethodResource):
    """
    View for uploading a file part during a multipart upload to the local store for an organization.

    This class provides a method to upload individual file parts for a multipart upload.
    """

    @doc(tags=[SWAGGER_TAG_FILES], description='Upload a file part to the local store (only for multipart uploads)')
    @use_kwargs(_local_store_multipart_upload_url_params, location='query')
    @validate_payload_size
    @limiter.limit(rate_limits)
    @capture_request_errors
    @capture_schema_errors
    def put(self, organization_id: str, upload_id: str, **kwargs):
        """
        Uploads a part of the file to the local store during a multipart upload for the specified organization.

        This method handles the upload of a file part for a multipart upload.

        Args:
            organization_id (str): The ID of the organization.
            upload_id (str): The upload ID associated with the multipart upload.
            **kwargs: Query parameters, including the part number and 'token' for authentication.

        Returns:
            Response: A response with the ETag of the uploaded part.
        """
        return _upload_file_part_to_local_store(parent_id=organization_id,
                                                parent_type=Organization,
                                                upload_id=upload_id,
                                                part_number=kwargs['part_number'],
                                                token=kwargs['token'])


class TaskLocalStoreDownloadView(MethodResource):
    """
    View for downloading a file from the local store for a task.

    This class provides a method to download a file (or thumbnail) from the local store.
    """

    @doc(tags=[SWAGGER_TAG_FILES], description='Download a file from the local store')
    @use_kwargs(_local_store_download_url_params, location='query')
    @validate_payload_size
    @limiter.limit(rate_limits)
    @capture_request_errors
    @capture_schema_errors
    def get(self, task_id: str, file_id: str, **kwargs):
        """
        Downloads the specified file or thumbnail from the local store for the task.

        This method retrieves the file content and sends it as a response to the client.

        Args:
            task_id (str): The ID of the task.
            file_id (str): The ID of the file to download.
            **kwargs: URL query parameters, including the 'token' for authentication and optional 'thumbnail' flag.

        Returns:
            Response: A response containing the file content.
        """
        return _download_file_from_local_store(parent_id=task_id,
                                               file_id=file_id,
                                               file_type=TaskFile,
                                               url_params=kwargs)


class TaskLocalStoreUploadView(MethodResource):
    """
    View for uploading a file to the local store for a task.

    This class provides a method to upload a file to the local store.
    """

    @doc(tags=[SWAGGER_TAG_FILES], description='Upload a file to the local store')
    @use_kwargs(_local_store_upload_form_fields, location='form')
    @validate_payload_size
    @limiter.limit(rate_limits)
    @capture_request_errors
    @capture_schema_errors
    def post(self, task_id: str, file_id: str, **kwargs):
        """
        Uploads a file to the local store for the specified task.

        This method handles the file upload process, including verifying permissions and saving the file.

        Args:
            task_id (str): The ID of the task.
            file_id (str): The ID of the file to upload.
            **kwargs: Form data, including the 'token' for authentication.

        Returns:
            Response: A 200 OK response indicating successful file upload.
        """
        return _upload_file_to_local_store(parent_id=task_id,
                                           file_id=file_id,
                                           file_type=TaskFile,
                                           token=kwargs['token'])


class TaskLocalStoreMultipartUploadView(MethodResource):
    """
    View for uploading a file part during a multipart upload to the local store for a task.

    This class provides a method to upload individual file parts for a multipart upload.
    """

    @doc(tags=[SWAGGER_TAG_FILES], description='Upload a file part to the local store (only for multipart uploads)')
    @use_kwargs(_local_store_multipart_upload_url_params, location='query')
    @validate_payload_size
    @limiter.limit(rate_limits)
    @capture_request_errors
    @capture_schema_errors
    def put(self, task_id: str, upload_id: str, **kwargs):
        """
        Uploads a part of the file to the local store during a multipart upload for the specified task.

        This method handles the upload of a file part for a multipart upload.

        Args:
            task_id (str): The ID of the task.
            upload_id (str): The upload ID associated with the multipart upload.
            **kwargs: Query parameters, including the part number and 'token' for authentication.

        Returns:
            Response: A response with the ETag of the uploaded part.
        """
        return _upload_file_part_to_local_store(parent_id=task_id,
                                                parent_type=Task,
                                                upload_id=upload_id,
                                                part_number=kwargs['part_number'],
                                                token=kwargs['token'])
