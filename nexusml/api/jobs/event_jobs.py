""" Event driven jobs. """
from typing import Optional

from celery import shared_task

from nexusml.api.utils import get_engine_type
from nexusml.database.services import Service
from nexusml.database.tasks import TaskDB
from nexusml.engine.buffers import MonBuffer
from nexusml.engine.buffers import MonBufferIO
from nexusml.engine.services.monitoring import MonitoringService
from nexusml.engine.workers import get_engine
from nexusml.enums import ServiceType
from nexusml.statuses import MONITORING_ANALYZING_STATUS_CODE
from nexusml.statuses import MONITORING_UNKNOWN_ERROR_STATUS_CODE
from nexusml.statuses import MONITORING_WAITING_STATUS_CODE

######################
# MONITORING SERVICE #
######################


@shared_task
def run_mon_service(task_id: int) -> None:
    """
    Runs the Monitoring Service.

    Steps:
    1. Retrieve the monitoring service settings from the database.
    2. Initialize the service with settings such as refresh interval, OOD prediction sensitivity, and smoothing.
    3. Check if the buffer has enough samples to proceed.
    4. Ensure the service is in the waiting status to avoid overlapping execution.
    5. Update the status to analyzing and attempt to detect OOD predictions.
    6. Handle errors by updating the service status to unknown error.
    7. Reset the service status to waiting after completion.

    Args:
        task_id (int): The task ID associated with the monitoring service.

    Returns:
        None

    Raises:
        Any exception raised during the detection of OOD predictions will be caught and handled internally,
        updating the service status to `MONITORING_UNKNOWN_ERROR_STATUS_CODE`.
    """
    default_refresh_interval: int = 100  # in number of predictions

    default_ood_min_sample: int = 100  # in number of predictions / verify max buffer size
    default_ood_sensitivity: float = 0.5
    default_ood_smoothing: float = 0.5

    task = TaskDB.get(task_id=task_id)
    mon_buffer: MonBuffer = MonBuffer(buffer_io=MonBufferIO(task=task))

    # Get service settings
    mon_service_db: Service = Service.filter_by_task_and_type(task_id=mon_buffer.task().task_id,
                                                              type_=ServiceType.MONITORING)

    mon_settings: dict = mon_service_db.settings
    refresh_interval: int = mon_settings.get('refresh_interval', default_refresh_interval)

    ood_predictions: dict = mon_settings.get('ood_predictions', dict())
    ood_min_sample: int = ood_predictions.get('min_sample', default_ood_min_sample)
    ood_sensitivity: float = ood_predictions.get('sensitivity', default_ood_sensitivity)
    ood_smoothing: float = ood_predictions.get('smoothing', default_ood_smoothing)

    # Initialize service
    mon_service: MonitoringService = MonitoringService(buffer=mon_buffer,
                                                       refresh_interval=refresh_interval,
                                                       ood_min_sample=ood_min_sample,
                                                       ood_sensitivity=ood_sensitivity,
                                                       ood_smoothing=ood_smoothing)
    # Check service has enough samples
    samples: list = mon_buffer.buffer_io().read_items()
    if not mon_service.check_if_enough_samples(samples):
        return

    # Check service is not stopped or already running
    # Note: Place this check at the end to minimize concurrency issues by keeping it closer to the status update.
    mon_status_code: str = mon_service_db.status['code']
    if mon_status_code != MONITORING_WAITING_STATUS_CODE:
        return

    mon_service.update_service_status(code=MONITORING_ANALYZING_STATUS_CODE)
    try:
        # Detect out-of-distribution (OOD) predictions
        mon_service.detect_ood_predictions()
    except Exception:
        # NOTE: be more specific about the error and log the error
        mon_service.update_service_status(code=MONITORING_UNKNOWN_ERROR_STATUS_CODE)
        return

    mon_service.update_service_status(code=MONITORING_WAITING_STATUS_CODE)


@shared_task
def train(task_uuid: str, model_uuid: Optional[str] = None, **kwargs):
    engine_type = get_engine_type()
    engine = get_engine(engine_type=engine_type, task_uuid=task_uuid)
    engine.train(model_uuid=model_uuid, **kwargs)
