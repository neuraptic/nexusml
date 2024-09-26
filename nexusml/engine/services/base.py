# TODO: Try to make this module independent from `nexusml.api`

from abc import ABC
from abc import abstractmethod
from datetime import datetime
from typing import Union

from nexusml.api.resources import InvalidDataError
from nexusml.api.resources import UnprocessableRequestError
from nexusml.api.resources.organizations import get_active_subscription
from nexusml.constants import FREE_PLAN_ID
from nexusml.database.services import Service as ServiceDB
from nexusml.database.tasks import TaskDB
from nexusml.engine.buffers import Buffer
from nexusml.enums import ServiceType
from nexusml.statuses import Status


def update_service_status(task_db_obj: TaskDB, service_type: ServiceType, status: dict) -> Union[dict, None]:
    """
    Update the status of a service.

    Args:
        task_db_obj (TaskDB): Database object of the task to which the service belongs
        service_type (ServiceType): Type of service
        status (dict): Status object

    Raises:
        InvalidDataError: If the status code is invalid
        UnprocessableRequestError: If the organization is in the Free Plan
    """
    if get_active_subscription(organization_id=task_db_obj.organization_id).plan_id == FREE_PLAN_ID:
        raise UnprocessableRequestError('Services are disabled in the Free Plan')
    try:
        new_status = Status.from_dict(status)
        service: ServiceDB = ServiceDB.filter_by_task_and_type(task_id=task_db_obj.task_id, type_=service_type)
        service.set_status(status=new_status)
        return new_status.to_dict(include_state=True, expand_status=True)
    except AttributeError:
        raise InvalidDataError(f'Status not found: "{status["code"]}"')
    except ValueError as e:
        err_msg = str(e)
        if err_msg == 'Invalid status code':
            raise InvalidDataError(err_msg)
        else:
            raise e


class Service(ABC):
    """ Abstract base class for services. """

    def __init__(self, buffer: Buffer, service_type: ServiceType):
        """
        Initializes the Service with a buffer and service type. The service operates within the context
        of a task, which is inferred from the buffer.

        Args:
            buffer (Buffer): The buffer from which data will be read.
            service_type (ServiceType): The type of service (e.g., continual_learning, monitoring, etc.).
        """
        self._buffer = buffer
        self._service_type: ServiceType = service_type

    def buffer(self):
        """
        Returns the buffer from which the service reads data.

        Returns:
            Buffer: The buffer object associated with the service.
        """
        return self._buffer

    def task(self) -> TaskDB:
        """
        Returns the task associated with the service. The task is inferred from the buffer.

        Returns:
            TaskDB: The task object tied to the buffer and service.
        """
        return self.buffer().task()

    @abstractmethod
    def service_name(self) -> str:
        """
        Abstract method that must be implemented by subclasses to return a URL-safe name for the service.

        Returns:
            str: A URL-safe name for the service, using "_" instead of "-" to separate words.
        """
        raise NotImplementedError()

    def update_service_status(self,
                              code: str,
                              started_at: datetime = None,
                              updated_at: datetime = None,
                              ended_at: datetime = None,
                              details: dict = None,
                              ignore_errors: bool = False):
        """
        Updates the status of the service by calling the `update_service_status` function with the current task,
        service type, and the new status information. It handles errors gracefully if `ignore_errors` is set
        to True.

        Steps:
        1. Create a status dictionary using the provided status details (e.g., code, started_at).
        2. Call the `update_service_status` function to update the status in the database.
        3. If an error occurs during the update and `ignore_errors` is False, re-raise the error.

        Args:
            code (str): Status code to update the service with.
            started_at (datetime, optional): Start time of the status update.
            updated_at (datetime, optional): Time when the status was updated.
            ended_at (datetime, optional): End time of the status, if applicable.
            details (dict, optional): Additional details to attach to the status.
            ignore_errors (bool, optional): Whether to suppress exceptions during the status update.

        Returns:
            None

        Raises:
            Exception: Re-raises any exceptions if `ignore_errors` is set to False.
        """
        status_items_map: dict = {
            'code': code,
            'started_at': started_at,
            'updated_at': updated_at,
            'ended_at': ended_at,
            'details': details
        }
        status = {k: v for k, v in status_items_map.items() if v is not None}

        try:
            update_service_status(task_db_obj=self.task(), service_type=self._service_type, status=status)
        except Exception as e:
            if not ignore_errors:
                raise e
