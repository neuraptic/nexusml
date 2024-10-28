from abc import abstractmethod
import functools
import os
from typing import Iterable, Type, Union

from flask import url_for

from nexusml.api.endpoints import ENDPOINT_ORG_FILE
from nexusml.api.endpoints import ENDPOINT_TASK_FILE
from nexusml.api.ext import cache
from nexusml.api.resources.base import PermissionDeniedError
from nexusml.api.resources.base import QuotaError
from nexusml.api.resources.base import Resource
from nexusml.api.resources.base import ResourceNotFoundError
from nexusml.api.schemas.base import ResourceResponseSchema
from nexusml.api.schemas.files import FileRequest
from nexusml.api.schemas.files import FileResponse
from nexusml.api.schemas.files import OrganizationFileRequest
from nexusml.api.schemas.files import OrganizationFileResponse
from nexusml.api.schemas.files import TaskFileRequest
from nexusml.api.schemas.files import TaskFileResponse
from nexusml.api.utils import API_DOMAIN
from nexusml.api.utils import generate_tmp_token
from nexusml.api.utils import get_file_storage_backend
from nexusml.api.utils import get_local_file_storage_config
from nexusml.api.utils import save_thumbnail_to_local_file_store
from nexusml.api.utils import save_thumbnail_to_s3
from nexusml.constants import PREFIX_ORG_PICTURES
from nexusml.constants import PREFIX_ORGANIZATIONS
from nexusml.constants import PREFIX_TASK_INPUTS
from nexusml.constants import PREFIX_TASK_METADATA
from nexusml.constants import PREFIX_TASK_MODELS
from nexusml.constants import PREFIX_TASK_OUTPUTS
from nexusml.constants import PREFIX_TASK_PICTURES
from nexusml.constants import PREFIX_TASKS
from nexusml.constants import PREFIX_THUMBNAILS
from nexusml.database.files import FileDB
from nexusml.database.files import OrgFileDB
from nexusml.database.files import TaskFileDB
from nexusml.database.organizations import Agent
from nexusml.database.organizations import OrganizationDB
from nexusml.database.organizations import UserDB
from nexusml.enums import FileStorageBackend
from nexusml.enums import FileType
from nexusml.enums import NotificationSource
from nexusml.enums import OrgFileUse
from nexusml.enums import ResourceAction
from nexusml.enums import ResourceType
from nexusml.enums import TaskFileUse
from nexusml.utils import get_s3_config
from nexusml.utils import s3_client


def _require_file_storage_backend(file_storage_backend: FileStorageBackend):
    """
    Decorator to require a specific file storage backend.

    This ensures that the wrapped function only runs if the file storage backend in use matches the required backend.

    Args:
        file_storage_backend (FileStorageBackend): The required file storage backend.

    Raises:
        RuntimeError: If the storage backend is not the required one.
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Verify the storage backend
            if get_file_storage_backend() != file_storage_backend:
                raise RuntimeError(
                    f'This method is only available when using the "{file_storage_backend}" storage backend')
            # Run decorated function
            return func(*args, **kwargs)

        return wrapper

    return decorator


class File(Resource):
    """
    Base class for managing files in the system.

    This class defines common behavior for file management, including uploading, downloading, deleting, and
    generating presigned URLs for file operations. It supports both local and S3-based file storage backends.
    """

    @classmethod
    @abstractmethod
    def db_model(cls) -> Type[FileDB]:
        """ Returns the database model associated with the file. """
        pass

    @classmethod
    @abstractmethod
    def load_schema(cls) -> Type[FileRequest]:
        """ Returns the schema used for loading file data from requests. """
        pass

    @classmethod
    @abstractmethod
    def dump_schema(cls) -> Type[FileResponse]:
        """ Returns the schema used for dumping file data to responses. """
        pass

    @classmethod
    def notification_source_type(cls) -> NotificationSource:
        """ Returns the notification source type for files. """
        return NotificationSource.FILE

    @classmethod
    def post(cls,
             agent: Agent,
             data: dict,
             parents: list = None,
             check_permissions: bool = True,
             check_parents: bool = True,
             notify_to: Iterable[UserDB] = None) -> Resource:
        """
        Creates and uploads a file to the system, while checking for space quotas and replacing backslashes with slashes
        in filenames.

        The function first checks if the parent resource has enough space quota available. It then processes the file
        upload, updates the space usage, and ensures that the file fits within the remaining quota. If the quota is
        exceeded, the file is removed, and an exception is raised.

        Args:
            agent (Agent): The agent performing the action.
            data (dict): Data about the file, including metadata such as size and filename.
            parents (list, optional): The parent resources associated with this file.
            check_permissions (bool, optional): If True, permissions will be checked before the operation.
            check_parents (bool, optional): If True, the parent resources will be checked for existence and validity.
            notify_to (Iterable[UserDB], optional): A list of users to notify about this action.

        Returns:
            Resource: The created resource after successfully posting the file.

        Raises:
            QuotaError: If the space usage exceeds the quota.
        """
        # Check limits and quotas in parent resource
        parent = parents[0]
        assert hasattr(parent, 'check_quota_usage')

        parent.check_quota_usage(name='space', cache=True, delta=data['size'])

        # Replace backslashes with slashes in the filename
        data['filename'] = data['filename'].replace('\\', '/')

        # Create and persist the resource in the database
        resource = super().post(agent=agent,
                                data=data,
                                parents=parents,
                                check_permissions=check_permissions,
                                check_parents=check_parents,
                                notify_to=notify_to)

        # Update space usage in parent resource
        parent.update_quota_usage(name='space', delta=data['size'])

        # Check limits and quotas again, just in case more files have been posted during this request.
        # If any limit or quota was exceeded, remove the file from the database
        try:
            parent.check_quota_usage(name='space', cache=False)
        except QuotaError as qe:
            resource.delete()
            raise qe

        return resource

    def delete(self, notify_to: Iterable[UserDB] = None):
        """
        Deletes a file and updates the space quota usage.

        This function removes the file from the storage backend (S3 or local) and updates the parent's quota usage
        by reducing the size of the file. If the file has an associated thumbnail, that is deleted as well.

        Args:
            notify_to (Iterable[UserDB], optional): A list of users to notify about this action.

        Returns:
            None
        """
        parent = self.parents()[0]
        assert hasattr(parent, 'update_quota_usage')

        # Delete file (and its thumbnail if it exists)
        if get_file_storage_backend() == FileStorageBackend.S3:
            self._delete_file_from_s3()
        else:
            self._delete_file_from_local_store()

        # Remove resource from the database
        super().delete(notify_to=notify_to)

        # Update space usage in parent resource
        parent.update_quota_usage(name='space', delta=-self.db_object().size)

    def dump(self,
             serialize=True,
             expand_associations=False,
             reference_parents=False,
             update_sync_state: bool = True,
             thumbnail: bool = False) -> Union[ResourceResponseSchema, dict]:
        """
        Dumps the file metadata and generates download or upload URLs based on the storage backend.

        This method provides details about the file, including download URLs, and handles the generation of presigned
        URLs for both downloading and uploading files. If the file does not exist, it handles the creation of thumbnails
        or returns an error.

        Args:
            serialize (bool, optional): Whether to serialize the dumped data.
            expand_associations (bool, optional): Whether to expand associated resources in the response.
            reference_parents (bool, optional): Whether to include references to parent resources.
            update_sync_state (bool, optional): Whether to update the sync state.
            thumbnail (bool, optional): Whether to return the thumbnail URL.

        Returns:
            Union[ResourceResponseSchema, dict]: The file metadata and associated URLs.
        """
        # Dump resource data
        dumped_data = super().dump(serialize=False,
                                   expand_associations=expand_associations,
                                   reference_parents=reference_parents,
                                   update_sync_state=update_sync_state)

        # Get download/upload URL
        file_storage_backend = get_file_storage_backend()
        try:
            # Try to get the download URL
            if file_storage_backend == FileStorageBackend.S3:
                dumped_data['download_url'] = self._generate_s3_download_url(file_uuid=self.uuid(), thumbnail=thumbnail)
            else:
                dumped_data['download_url'] = self._generate_local_store_download_url(thumbnail=thumbnail)
        except ResourceNotFoundError:
            # If the file was not found:
            #     - If the thumbnail was requested, create it, upload it, and get its download URL.
            #     - Otherwise, generate a URL to upload the file content
            #       (only if the file size doesn't exceed the maximum size for a single upload)
            if thumbnail:
                # Create and upload thumbnail and get its download URL
                if file_storage_backend == FileStorageBackend.S3:
                    save_thumbnail_to_s3(s3_client=s3_client(),
                                         bucket=get_s3_config()['bucket'],
                                         object_key=self.path())
                    dumped_data['download_url'] = self._generate_s3_download_url(file_uuid=self.uuid(), thumbnail=True)
                else:
                    save_thumbnail_to_local_file_store(thumbnail_path=self.path(thumbnail=True),
                                                       file_content=self.read_from_local_store())
                    dumped_data['download_url'] = self._generate_local_store_download_url(thumbnail=True)
            else:
                # Check permissions to upload files in the current organization
                try:
                    parent = self.parents()[0]
                    organization = OrganizationDB.get(organization_id=parent.db_object().organization_id)
                    self.check_permissions(organization=organization, action=ResourceAction.CREATE, resource=self)
                except PermissionDeniedError:
                    raise ResourceNotFoundError(f'Content of file "{self.uuid()}" was not uploaded')
                # Get maximum upload size
                if file_storage_backend == FileStorageBackend.S3:
                    max_upload_size = get_s3_config()['max_upload_size']
                else:
                    max_upload_size = get_local_file_storage_config()['max_upload_size']
                # Generate upload URL only if the file size doesn't exceed the maximum size for a single upload
                if self.db_object().size <= max_upload_size:
                    if file_storage_backend == FileStorageBackend.S3:
                        dumped_data['upload_url'] = self._generate_s3_upload_url(file_uuid=self.uuid())
                    else:
                        dumped_data['upload_url'] = self._generate_local_store_upload_url()

        # Return dumped data
        return self.dump_data(dumped_data, serialize=serialize)

    @classmethod
    @abstractmethod
    def prefix(cls) -> str:
        """ Parent prefix (e.g. 'organizations', 'tasks', etc.). """
        pass

    @classmethod
    @abstractmethod
    def use_prefixes(cls) -> dict:
        """ Prefix for each file use. """
        pass

    @classmethod
    @abstractmethod
    def local_store_view_parent_id_arg(cls) -> str:
        """ Name of the URL argument that identifies the parent resource in the local store views. """
        pass

    @classmethod
    @abstractmethod
    def local_store_download_view(cls) -> str:
        """ Name of the local file store download view. """
        pass

    @classmethod
    @abstractmethod
    def local_store_upload_view(cls) -> str:
        """ Name of the local file store upload view. """
        pass

    def path(self, thumbnail: bool = False) -> str:
        """
        Returns the path to the file in the current storage backend.

        This method constructs the full file path based on the storage backend in use (local or S3) and
        includes logic for generating the path to the file's thumbnail if required.

        Args:
            thumbnail (bool): If True, the path to the thumbnail is returned.

        Returns:
            str: The path to the file in the current storage backend.
        """
        # Get the parent resource
        assert len(self.parents()) == 1
        parent = self.parents()[0]

        # Determine the file path
        use_prefix = self.use_prefixes()[self.db_object().use_for]
        self_prefix = use_prefix if not thumbnail else PREFIX_THUMBNAILS
        file_path = self.prefix() + parent.uuid() + '/' + self_prefix + self.uuid()

        # Return the path in the current storage backend
        if get_file_storage_backend() == FileStorageBackend.S3:
            return file_path
        else:
            return os.path.join(get_local_file_storage_config()['root_path'], file_path.replace('/', os.sep))

    @_require_file_storage_backend(file_storage_backend=FileStorageBackend.LOCAL)
    def read_from_local_store(self, thumbnail: bool = False) -> bytes:
        """
        Reads the file content from the local store.

        This method opens the file from the local file system and returns its content as bytes.

        Args:
            thumbnail (bool): If True, the thumbnail content is read.

        Returns:
            bytes: The file content.

        Raises:
            FileNotFoundError: If the file is not found in the local store.
        """
        with open(self.path(thumbnail=thumbnail), 'rb') as f:
            return f.read()

    @_require_file_storage_backend(file_storage_backend=FileStorageBackend.LOCAL)
    def _get_local_store_view_args(self, thumbnail: bool):
        """
        Helper method to generate URL arguments for accessing the local store view.

        Args:
            thumbnail (bool): If True, generates arguments for accessing the thumbnail.

        Returns:
            dict: The URL arguments for accessing the local store.
        """
        # Get the parent resource
        parent = self.parents()[0]

        # Set URL args
        url_args = {self.local_store_view_parent_id_arg(): parent.uuid(), 'file_id': self.uuid()}
        if thumbnail:
            url_args['thumbnail'] = str(thumbnail).lower()

        return url_args

    @_require_file_storage_backend(file_storage_backend=FileStorageBackend.LOCAL)
    def _generate_local_store_token(self) -> str:
        """
        Generates a token for accessing the local file store.

        This token allows for temporary authenticated access to files in the local store.

        Returns:
            str: The generated token.
        """
        agent_uuid = str(self.agent().uuid)
        claims = {
            'file_id': self.uuid(),
            'agent_id': agent_uuid,
            'agent_type': 'user' if isinstance(self.agent(), UserDB) else 'client'
        }
        expires_in = get_local_file_storage_config()['url_expiration']
        return generate_tmp_token(agent_uuid=agent_uuid,
                                  expires_in=expires_in,
                                  custom_claims=claims,
                                  custom_claims_key='request')

    @_require_file_storage_backend(file_storage_backend=FileStorageBackend.LOCAL)
    def _generate_local_store_download_url(self, thumbnail: bool = False) -> str:
        """
        Returns the URL for downloading a file from the local file store.

        This method generates a URL that allows for authenticated downloading of files from the local file store,
        including logic for downloading thumbnails if requested.

        Args:
            thumbnail (bool): Whether to return the thumbnail URL.

        Returns:
            str: The URL for downloading a file from the local file store.

        Raises:
            ResourceNotFoundError: If the file is not found in the local store.
        """
        if not os.path.isfile(self.path(thumbnail=thumbnail)):
            raise ResourceNotFoundError()
        url_args = self._get_local_store_view_args(thumbnail=thumbnail)
        download_url = API_DOMAIN + url_for(self.local_store_download_view().lower(), **url_args)
        token = self._generate_local_store_token()
        return download_url + ('&' if '?' in download_url else '?') + 'token=' + token

    @_require_file_storage_backend(file_storage_backend=FileStorageBackend.LOCAL)
    def _generate_local_store_upload_url(self) -> dict:
        """
        Returns the URL for uploading a file to the local file store.

        This method generates a URL and necessary fields for authenticated uploading of files to the local store.

        Returns:
            dict: The URL and the fields for uploading a file to the local file store.
        """
        url_args = self._get_local_store_view_args(thumbnail=False)
        return {
            'url': API_DOMAIN + url_for(self.local_store_upload_view().lower(), **url_args),
            'fields': {
                'token': self._generate_local_store_token()
            }
        }

    @_require_file_storage_backend(file_storage_backend=FileStorageBackend.S3)
    @cache.memoize()
    def _generate_s3_download_url(self, file_uuid: str, thumbnail: bool) -> str:
        """
        Generates a presigned S3 URL for downloading a file.

        This method generates a temporary, authenticated URL that allows downloading of the file from S3, using
        a presigned URL.

        Args:
            file_uuid (str): The UUID of the file.
            thumbnail (bool): Whether to generate a URL for the thumbnail.

        Returns:
            str: The presigned S3 download URL.

        Raises:
            ResourceNotFoundError: If the file is not found in the S3 bucket.
        """
        assert file_uuid == self.uuid()  # NOTE: don't remove `file_uuid` as it is used as the cache key
        s3_config = get_s3_config()
        object_key = self.path(thumbnail=thumbnail)
        # Check whether the file exists in S3
        if 'Contents' not in s3_client().list_objects_v2(Bucket=s3_config['bucket'], Prefix=object_key):
            raise ResourceNotFoundError(f'"{object_key}" not found in S3')
        # Generate presigned URL
        return s3_client().generate_presigned_url('get_object',
                                                  Params={
                                                      'Bucket': s3_config['bucket'],
                                                      'Key': object_key
                                                  },
                                                  ExpiresIn=s3_config['url_expiration'])

    @_require_file_storage_backend(file_storage_backend=FileStorageBackend.S3)
    @cache.memoize()
    def _generate_s3_upload_url(
        self,
        file_uuid: str,
    ) -> dict:
        """
        Generates a presigned S3 URL for uploading a file.

        This method generates a temporary, authenticated URL that allows uploading of the file to S3, using
        a presigned URL.

        Args:
            file_uuid (str): The UUID of the file.

        Returns:
            dict: The presigned S3 upload URL and associated fields for uploading the file.
        """
        assert file_uuid == self.uuid()  # NOTE: don't remove `file_uuid` as it is used as the cache key
        s3_config = get_s3_config()
        return s3_client().generate_presigned_post(s3_config['bucket'],
                                                   self.path(),
                                                   Conditions=[['content-length-range', 1,
                                                                self.db_object().size]],
                                                   ExpiresIn=s3_config['url_expiration'])

    @_require_file_storage_backend(file_storage_backend=FileStorageBackend.LOCAL)
    def _delete_file_from_local_store(self):
        """
        Deletes the file from the local store.

        This method removes both the file and its thumbnail from the local storage, if they exist.

        Returns:
            None
        """
        # Delete file
        os.remove(self.path())
        # Delete thumbnail
        try:
            os.remove(self.path(thumbnail=True))
        except FileNotFoundError:
            pass

    @_require_file_storage_backend(file_storage_backend=FileStorageBackend.S3)
    def _delete_file_from_s3(self):
        """
        Deletes the file from S3 storage.

        This method removes both the file and its thumbnail from the S3 bucket.

        Returns:
            None
        """
        s3_config = get_s3_config()
        # Delete file
        s3_response = s3_client().delete_object(Bucket=s3_config['bucket'], Key=self.path())
        if s3_response['ResponseMetadata']['HTTPStatusCode'] != 204:
            raise Exception('File deletion failed')
        # Delete thumbnail
        s3_client().delete_object(Bucket=s3_config['bucket'], Key=self.path(thumbnail=True))


class OrgFile(File):
    """
    Represents a file associated with an organization.

    This class extends the `File` class to provide additional functionality specific to files belonging to
    organizations, such as pictures and other organization-related resources.
    """

    @classmethod
    def db_model(cls):
        return OrgFileDB

    @classmethod
    def load_schema(cls):
        return OrganizationFileRequest

    @classmethod
    def dump_schema(cls):
        return OrganizationFileResponse

    @classmethod
    def location(cls) -> str:
        return ENDPOINT_ORG_FILE

    @classmethod
    def prefix(cls) -> str:
        return PREFIX_ORGANIZATIONS

    @classmethod
    def use_prefixes(cls) -> dict:
        return {OrgFileUse.PICTURE: PREFIX_ORG_PICTURES}

    @classmethod
    def local_store_view_parent_id_arg(cls) -> str:
        return 'organization_id'

    @classmethod
    def local_store_download_view(cls) -> str:
        return 'files.OrgLocalStoreDownloadView'

    @classmethod
    def local_store_upload_view(cls) -> str:
        return 'files.OrgLocalStoreUploadView'

    def _set_data(self, data: dict, notify_to: Iterable[UserDB] = None):
        """
        Sets file data for an organization file, assuming it is always a picture.

        Args:
            data (dict): The file data to set.
            notify_to (Iterable[UserDB], optional): A list of users to notify about this action.

        Returns:
            None
        """
        # Always assume the file is a picture
        assert len(OrgFileUse) == 1
        assert 'use_for' not in data
        assert 'type_' not in data
        data['use_for'] = OrgFileUse.PICTURE
        data['type_'] = FileType.IMAGE
        # Set resource data
        super()._set_data(data=data, notify_to=notify_to)


class TaskFile(File):
    """
    Represents a file associated with a task.

    This class extends the `File` class to provide additional functionality specific to files belonging to tasks,
    such as input files, output files, models, metadata, and task-related pictures.
    """

    @classmethod
    def db_model(cls):
        return TaskFileDB

    @classmethod
    def load_schema(cls):
        return TaskFileRequest

    @classmethod
    def dump_schema(cls):
        return TaskFileResponse

    @classmethod
    def location(cls) -> str:
        return ENDPOINT_TASK_FILE

    @classmethod
    def permission_resource_type(cls) -> ResourceType:
        # For the moment, only task files have permissions
        return ResourceType.FILE

    @classmethod
    def prefix(cls) -> str:
        return PREFIX_TASKS

    @classmethod
    def use_prefixes(cls) -> dict:
        return {
            TaskFileUse.AI_MODEL: PREFIX_TASK_MODELS,
            TaskFileUse.INPUT: PREFIX_TASK_INPUTS,
            TaskFileUse.OUTPUT: PREFIX_TASK_OUTPUTS,
            TaskFileUse.METADATA: PREFIX_TASK_METADATA,
            TaskFileUse.PICTURE: PREFIX_TASK_PICTURES
        }

    @classmethod
    def local_store_view_parent_id_arg(cls) -> str:
        return 'task_id'

    @classmethod
    def local_store_download_view(cls) -> str:
        return 'files.TaskLocalStoreDownloadView'

    @classmethod
    def local_store_upload_view(cls) -> str:
        return 'files.TaskLocalStoreUploadView'
