# TODO: Try to make this module independent from `nexusml.api`

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional
from typing import Type as Type_
from typing import Union

from sqlalchemy import Column
from sqlalchemy import Enum
from sqlalchemy import ForeignKey
from sqlalchemy import JSON
from sqlalchemy import UniqueConstraint
from sqlalchemy.dialects.mysql import MEDIUMINT
from sqlalchemy.orm import relationship

from nexusml.api.utils import config
from nexusml.constants import AL_SERVICE_NAME
from nexusml.constants import CL_SERVICE_NAME
from nexusml.constants import INFERENCE_SERVICE_NAME
from nexusml.constants import MONITORING_SERVICE_NAME
from nexusml.constants import TESTING_SERVICE_NAME
from nexusml.database.base import Entity
from nexusml.database.organizations import client_scopes
from nexusml.database.organizations import ClientDB
from nexusml.database.utils import set_status
from nexusml.enums import ServiceType
from nexusml.statuses import Status


class ServiceSettings(ABC):
    """Abstract base class for service settings.

    This class defines the interface for service settings, providing methods
    for converting settings to and from dictionary format.
    """

    @classmethod
    @abstractmethod
    def from_dict(cls, settings: dict):
        """Create an instance of the settings class from a dictionary.

        Args:
            settings (dict): Dictionary containing the settings.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def to_dict(self) -> dict:
        """Convert the settings instance to a dictionary.

        Returns:
            dict: Dictionary representation of the settings.
        """
        raise NotImplementedError()


class Service(Entity):
    """
    Represents a service running in a task.

    Attributes:
        service_id (PK): Surrogate key of the service.
        client_id (FK, unique): Surrogate key of the client running the service.
        task_id (FK): Task for which the service runs.
        type_: Unique for each `task_id`. Values: "inference", "continual_learning", "active_learning", "monitoring".
        status: JSON containing information about the status.
        settings: JSON containing service-specific settings.
        data: JSON containing service-specific data.
    """

    __tablename__ = 'services'
    __table_args__ = (UniqueConstraint('client_id'), UniqueConstraint('task_id', 'type_'))

    service_id = Column(MEDIUMINT(unsigned=True), primary_key=True, autoincrement=True)
    client_id = Column(MEDIUMINT(unsigned=True), ForeignKey(ClientDB.client_id, ondelete='CASCADE'), nullable=False)
    task_id = Column(MEDIUMINT(unsigned=True), ForeignKey('tasks.task_id', ondelete='CASCADE'), nullable=False)
    # TODO: Consider using `Column('type', Enum(ServiceType))` to keep "type" at SQL level
    type_ = Column(Enum(ServiceType), nullable=False)
    status = Column(JSON(none_as_null=True), nullable=False)
    settings = Column(JSON(none_as_null=True), nullable=False)
    data = Column(JSON(none_as_null=True), nullable=False, default=dict())

    # Parents (Many-to-One relationships)
    client = relationship('ClientDB')
    task = relationship('TaskDB')

    @classmethod
    def filter_by_task(cls, task_id) -> list:
        """Filter services by task ID.

        Args:
            task_id: ID of the task to filter services by.

        Returns:
            list: List of services associated with the given task ID.
        """
        return cls.query().filter_by(task_id=task_id).all()

    @classmethod
    def filter_by_task_and_type(cls, task_id, type_):
        """Filter services by task ID and type.

        Args:
            task_id: ID of the task to filter services by.
            type_: Type of the service.

        Returns:
            Service: The filtered service if found, otherwise None.

        Raises:
            AssertionError: If more than one service is found.
        """
        filtered_services = cls.query().filter_by(task_id=task_id, type_=type_).all()
        if not filtered_services:
            return None
        assert len(filtered_services) == 1
        return filtered_services[0]

    def set_status(self, status: Status, commit: bool = True):
        """Set the status of the service.

        Args:
            status (Status): The new status to set.
            commit (bool, optional): Whether to commit the change. Defaults to True.
        """
        groups = {
            ServiceType.INFERENCE: INFERENCE_SERVICE_NAME,
            ServiceType.CONTINUAL_LEARNING: CL_SERVICE_NAME,
            ServiceType.ACTIVE_LEARNING: AL_SERVICE_NAME,
            ServiceType.MONITORING: MONITORING_SERVICE_NAME,
            ServiceType.TESTING: TESTING_SERVICE_NAME
        }
        set_status(db_object=self, status=status, group=groups[self.type_], commit=commit)

    def set_settings(self, settings: Union[ServiceSettings, dict]):
        """Set the settings for the service.

        Args:
            settings (Union[ServiceSettings, dict]): The new settings to set.
        """
        if isinstance(settings, dict):
            settings = self._settings_class().from_dict(settings)
        else:
            assert isinstance(settings, self._settings_class())
        self.settings = settings.to_dict()

    def to_dict(self) -> dict:
        """Convert the service instance to a dictionary.

        Returns:
            dict: Dictionary representation of the service.
        """
        _dict = super().to_dict()
        _dict['status'] = Status.from_dict(self.status).to_dict(include_state=True, expand_status=True)
        _dict['settings'] = self._settings_class().from_dict(self.settings).to_dict()
        return _dict

    def _settings_class(self) -> Type_[ServiceSettings]:
        """Get the class for the service settings.

        Returns:
            Type_[ServiceSettings]: The class for the service settings.
        """
        settings_classes = {
            ServiceType.INFERENCE: InferenceServiceSettings,
            ServiceType.CONTINUAL_LEARNING: CLServiceSettings,
            ServiceType.ACTIVE_LEARNING: ALServiceSettings,
            ServiceType.MONITORING: MonitoringServiceSettings,
            ServiceType.TESTING: TestingServiceSettings
        }
        return settings_classes[self.type_]


#########################
# Service client scopes #
#########################

inference_client_scopes = [
    'tasks.read',  # Tasks
    'files.read',
    'files.create',  # Files (predictions involving files require files to be uploaded first)
    'models.read',  # AI models
    'predictions.create'  # Predictions
]

cl_client_scopes = [
    'tasks.read',  # Tasks
    'files.read',
    'files.create',  # Files (AI model creation involves file creation)
    'models.read',
    'models.create',  # AI models
    'examples.read'  # Examples
]

al_client_scopes = [
    'tasks.read',  # Tasks
    'files.read',
    'files.create',  # Files (examples involving files require files to be uploaded first)
    'models.read',  # AI models
    'examples.read',
    'examples.create',  # Examples
    'predictions.read'  # Predictions
]

monitoring_client_scopes = [
    'tasks.read',  # Tasks
    'models.read',  # AI models
    'predictions.read'  # Predictions
]

testing_client_scopes = list(inference_client_scopes)

assert (set(inference_client_scopes + cl_client_scopes + al_client_scopes + monitoring_client_scopes +
            testing_client_scopes).issubset(set(client_scopes)))


@dataclass
class InferenceServiceSettings(ServiceSettings):
    """
    Service settings for Inference Service.

    Attributes:
        enabled (bool): Flag for enabling Inference Service.
    """

    enabled: bool

    def __init__(self, enabled: Optional[bool] = None):
        self.enabled = enabled if enabled is not None else self._default_settings()['enabled']

    @staticmethod
    def _default_settings() -> dict:
        """Get the default settings for Inference Service.

        Returns:
            dict: Default settings for Inference Service.
        """
        return config.get('engine')['services']['inference']

    @classmethod
    def from_dict(cls, settings: dict):
        """Create an instance of InferenceServiceSettings from a dictionary.

        Args:
            settings (dict): Dictionary containing the settings.

        Returns:
            InferenceServiceSettings: An instance of InferenceServiceSettings.
        """
        return cls(enabled=settings.get('enabled'))

    def to_dict(self) -> dict:
        """Convert the InferenceServiceSettings instance to a dictionary.

        Returns:
            dict: Dictionary representation of the InferenceServiceSettings.
        """
        return {'enabled': self.enabled}


@dataclass
class CLServiceSettings(ServiceSettings):
    """
    Service settings for Continual Learning Service.

    Attributes:
        enabled (bool): Flag for enabling Continual Learning Service.
        min_days (float): Maximum frequency at which the AI can be retrained.
        max_days (float): Minimum frequency at which the AI should be retrained.
        min_sample (float): Minimum sample size to trigger retraining, relative to current number of examples.
                            Value between 0 and 1, representing the percentage of current number of examples.
        min_cpu_quota (float): Minimum CPU quota to be used (in hours).
        max_cpu_quota (float): Maximum CPU quota to be used (in hours).
        cpu_hard_limit (float): CPU hard limit for guaranteeing quality of service (in hours).
        min_gpu_quota (float): Minimum GPU quota to be used (in hours).
        max_gpu_quota (float): Maximum GPU quota to be used (in hours).
        gpu_hard_limit (float): GPU hard limit for guaranteeing quality of service (in hours).
    """

    enabled: bool
    min_days: float
    max_days: float
    min_sample: float
    min_cpu_quota: float
    max_cpu_quota: float
    cpu_hard_limit: float
    min_gpu_quota: float
    max_gpu_quota: float
    gpu_hard_limit: float

    def __init__(self,
                 enabled: Optional[bool] = None,
                 min_days: Optional[float] = None,
                 max_days: Optional[float] = None,
                 min_sample: Optional[float] = None,
                 min_cpu_quota: Optional[float] = None,
                 max_cpu_quota: Optional[float] = None,
                 cpu_hard_limit: Optional[float] = None,
                 min_gpu_quota: Optional[float] = None,
                 max_gpu_quota: Optional[float] = None,
                 gpu_hard_limit: Optional[float] = None):

        _default_settings = self._default_settings()

        self.enabled = enabled if enabled is not None else _default_settings['enabled']
        self.min_days = min_days if min_days is not None else _default_settings['min_days']
        self.max_days = max_days if max_days is not None else _default_settings['max_days']
        self.min_sample = min_sample if min_sample is not None else _default_settings['min_sample']
        self.min_cpu_quota = min_cpu_quota if min_cpu_quota is not None else _default_settings['min_cpu_quota']
        self.max_cpu_quota = max_cpu_quota if max_cpu_quota is not None else _default_settings['max_cpu_quota']
        self.cpu_hard_limit = cpu_hard_limit if cpu_hard_limit is not None else _default_settings['cpu_hard_limit']
        self.min_gpu_quota = min_gpu_quota if min_gpu_quota is not None else _default_settings['min_gpu_quota']
        self.max_gpu_quota = max_gpu_quota if max_gpu_quota is not None else _default_settings['max_gpu_quota']
        self.gpu_hard_limit = gpu_hard_limit if gpu_hard_limit is not None else _default_settings['gpu_hard_limit']

    @staticmethod
    def _default_settings() -> dict:
        """Get the default settings for Continual Learning Service.

        Returns:
            dict: Default settings for Continual Learning Service.
        """
        return config.get('engine')['services']['continual_learning']

    @classmethod
    def from_dict(cls, settings: dict):
        """Create an instance of CLServiceSettings from a dictionary.

        Args:
            settings (dict): Dictionary containing the settings.

        Returns:
            CLServiceSettings: An instance of CLServiceSettings.
        """
        return cls(
            enabled=settings.get('enabled',
                                 cls._default_settings()['enabled']),
            min_days=settings.get('min_days',
                                  cls._default_settings()['min_days']),
            max_days=settings.get('max_days',
                                  cls._default_settings()['max_days']),
            min_sample=settings.get('min_sample',
                                    cls._default_settings()['min_sample']),
            min_cpu_quota=settings.get('min_cpu_quota',
                                       cls._default_settings()['min_cpu_quota']),
            max_cpu_quota=settings.get('max_cpu_quota',
                                       cls._default_settings()['max_cpu_quota']),
            cpu_hard_limit=settings.get('cpu_hard_limit',
                                        cls._default_settings()['cpu_hard_limit']),
            min_gpu_quota=settings.get('min_gpu_quota',
                                       cls._default_settings()['min_gpu_quota']),
            max_gpu_quota=settings.get('max_gpu_quota',
                                       cls._default_settings()['max_gpu_quota']),
            gpu_hard_limit=settings.get('gpu_hard_limit',
                                        cls._default_settings()['gpu_hard_limit']),
        )

    def to_dict(self) -> dict:
        """Convert the CLServiceSettings instance to a dictionary.

        Returns:
            dict: Dictionary representation of the CLServiceSettings.
        """
        return {
            'enabled': self.enabled,
            'min_days': self.min_days,
            'max_days': self.max_days,
            'min_sample': self.min_sample,
            'min_cpu_quota': self.min_cpu_quota,
            'max_cpu_quota': self.max_cpu_quota,
            'cpu_hard_limit': self.cpu_hard_limit,
            'min_gpu_quota': self.min_gpu_quota,
            'max_gpu_quota': self.max_gpu_quota,
            'gpu_hard_limit': self.gpu_hard_limit,
        }


@dataclass
class ALServiceSettings(ServiceSettings):
    """
    Service settings for Active Learning Service.

    Attributes:
        enabled (bool): Flag for enabling Active Learning Service.
        query_interval (int): Interval in days between active learning queries.
        max_examples_per_query (int): Maximum number of examples to be queried per query.
    """

    enabled: bool
    query_interval: int
    max_examples_per_query: int

    def __init__(self,
                 enabled: Optional[bool] = None,
                 query_interval: Optional[int] = None,
                 max_examples_per_query: Optional[int] = None):

        _def_enabled = self._default_settings()['enabled']
        _def_query_interval = self._default_settings()['query_interval']
        _def_max_ex = self._default_settings()['max_examples_per_query']

        self.enabled = enabled if enabled is not None else _def_enabled
        self.query_interval = query_interval if query_interval is not None else _def_query_interval
        self.max_examples_per_query = max_examples_per_query if max_examples_per_query is not None else _def_max_ex

    @staticmethod
    def _default_settings() -> dict:
        """Get the default settings for Active Learning Service.

        Returns:
            dict: Default settings for Active Learning Service.
        """
        return config.get('engine')['services']['active_learning']

    @classmethod
    def from_dict(cls, settings: dict):
        """Create an instance of ALServiceSettings from a dictionary.

        Args:
            settings (dict): Dictionary containing the settings.

        Returns:
            ALServiceSettings: An instance of ALServiceSettings.
        """
        return cls(
            enabled=settings.get('enabled',
                                 cls._default_settings()['enabled']),
            query_interval=settings.get('query_interval',
                                        cls._default_settings()['query_interval']),
            max_examples_per_query=settings.get('max_examples_per_query',
                                                cls._default_settings()['max_examples_per_query']),
        )

    def to_dict(self) -> dict:
        """Convert the ALServiceSettings instance to a dictionary.

        Returns:
            dict: Dictionary representation of the ALServiceSettings.
        """
        return {
            'enabled': self.enabled,
            'query_interval': self.query_interval,
            'max_examples_per_query': self.max_examples_per_query,
        }


@dataclass
class MonitoringServiceSettings(ServiceSettings):
    """
    Service settings for Monitoring Service.

    Attributes:
        enabled (bool): Flag for enabling Monitoring Service.
        refresh_interval (int): Interval (in number of predictions) in which metrics are refreshed.
        ood_predictions: Settings for detection of out-of-distribution (OOD) predictions.
    """

    @dataclass
    class _OutOfDistributionSettings:
        """
        Attributes:
            min_sample (int): Minimum number of predictions required for running detection.
            sensitivity (float): Sensitivity to anomalies.
            smoothing (float): Smoothing factor.
        """
        min_sample: int
        sensitivity: float
        smoothing: float

        def __init__(self,
                     min_sample: Optional[int] = None,
                     sensitivity: Optional[float] = None,
                     smoothing: Optional[float] = None):

            if sensitivity < 0 or sensitivity > 1:
                raise ValueError('Sensitivity must be greater than or equal to 0 and less than or equal to 1')
            if smoothing < 0 or smoothing > 1:
                raise ValueError('Smoothing must be greater than or equal to 0 and less than or equal to 1')

            self.min_sample = min_sample if min_sample is not None else self._default_settings()['min_sample']
            self.sensitivity = sensitivity if sensitivity is not None else self._default_settings()['sensitivity']
            self.smoothing = smoothing if smoothing is not None else self._default_settings()['smoothing']

        @staticmethod
        def _default_settings() -> dict:
            """Get the default settings for OOD predictions.

            Returns:
                dict: Default settings for OOD predictions.
            """
            return config.get('engine')['services']['monitoring']['ood_predictions']

        @classmethod
        def from_dict(cls, settings: dict):
            """Create an instance of _OutOfDistributionSettings from a dictionary.

            Args:
                settings (dict): Dictionary containing the settings.

            Returns:
                _OutOfDistributionSettings: An instance of _OutOfDistributionSettings.

            Raises:
                ValueError: If sensitivity or smoothing are out of bounds.
            """
            sensitivity = settings.get('sensitivity', cls._default_settings()['sensitivity'])
            smoothing = settings.get('smoothing', cls._default_settings()['smoothing'])
            if sensitivity < 0 or sensitivity > 1:
                raise ValueError('Sensitivity must be greater than or equal to 0 and less than or equal to 1')
            if smoothing < 0 or smoothing > 1:
                raise ValueError('Smoothing must be greater than or equal to 0 and less than or equal to 1')
            return cls(
                min_sample=settings.get('min_sample',
                                        cls._default_settings()['min_sample']),
                sensitivity=sensitivity,
                smoothing=smoothing,
            )

        def to_dict(self) -> dict:
            """Convert the _OutOfDistributionSettings instance to a dictionary.

            Returns:
                dict: Dictionary representation of the _OutOfDistributionSettings.
            """
            return {
                'min_sample': self.min_sample,
                'sensitivity': self.sensitivity,
                'smoothing': self.smoothing,
            }

    enabled: bool
    refresh_interval: int
    ood_predictions: _OutOfDistributionSettings

    def __init__(self,
                 enabled: Optional[bool] = None,
                 refresh_interval: Optional[int] = None,
                 ood_predictions: Optional[_OutOfDistributionSettings] = None):

        _default_enabled = self._default_settings()['enabled']
        _default_refresh_interval = self._default_settings()['refresh_interval']
        _default_ood_predictions = self._default_ood_settings()

        self.enabled = enabled if enabled is not None else _default_enabled
        self.refresh_interval = refresh_interval if refresh_interval is not None else _default_refresh_interval
        self.ood_predictions = ood_predictions if ood_predictions is not None else _default_ood_predictions

    @staticmethod
    def _default_settings() -> dict:
        """Get the default settings for Monitoring Service.

        Returns:
            dict: Default settings for Monitoring Service.
        """
        return config.get('engine')['services']['monitoring']

    @classmethod
    def _default_ood_settings(cls) -> _OutOfDistributionSettings:
        """Get the default OOD settings for Monitoring Service.

        Returns:
            _OutOfDistributionSettings: Default OOD settings.
        """
        return cls._OutOfDistributionSettings.from_dict(cls._default_settings()['ood_predictions'])

    @classmethod
    def from_dict(cls, settings: dict):
        """Create an instance of MonitoringServiceSettings from a dictionary.

        Args:
            settings (dict): Dictionary containing the settings.

        Returns:
            MonitoringServiceSettings: An instance of MonitoringServiceSettings.
        """
        if 'ood_predictions' in settings:
            ood_predictions = cls._OutOfDistributionSettings.from_dict(settings['ood_predictions'])
        else:
            ood_predictions = cls._default_ood_settings()
        return cls(
            enabled=settings.get('enabled',
                                 cls._default_settings()['enabled']),
            refresh_interval=settings.get('refresh_interval',
                                          cls._default_settings()['refresh_interval']),
            ood_predictions=ood_predictions,
        )

    def to_dict(self) -> dict:
        """Convert the MonitoringServiceSettings instance to a dictionary.

        Returns:
            dict: Dictionary representation of the MonitoringServiceSettings.
        """
        return {
            'enabled': self.enabled,
            'refresh_interval': self.refresh_interval,
            'ood_predictions': self.ood_predictions.to_dict(),
        }


@dataclass
class TestingServiceSettings(ServiceSettings):
    """
    Service settings for Testing Service.

    Attributes:
        enabled (bool): Flag for enabling Testing Service.
    """

    enabled: bool

    def __init__(self, enabled: Optional[bool] = None):
        self.enabled = enabled if enabled is not None else self._default_settings()['enabled']

    @staticmethod
    def _default_settings() -> dict:
        """Get the default settings for Testing Service.

        Returns:
            dict: Default settings for Testing Service.
        """
        return config.get('engine')['services']['testing']

    @classmethod
    def from_dict(cls, settings: dict):
        """Create an instance of TestingServiceSettings from a dictionary.

        Args:
            settings (dict): Dictionary containing the settings.

        Returns:
            TestingServiceSettings: An instance of TestingServiceSettings.
        """
        return cls(enabled=settings.get('enabled'))

    def to_dict(self) -> dict:
        """Convert the TestingServiceSettings instance to a dictionary.

        Returns:
            dict: Dictionary representation of the TestingServiceSettings.
        """
        return {'enabled': self.enabled}
