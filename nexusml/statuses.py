"""
This module defines NexusML states/statuses (see `README.md`)
"""

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import ClassVar, Dict

from nexusml.constants import AL_SERVICE_NAME
from nexusml.constants import CL_SERVICE_NAME
from nexusml.constants import DATETIME_FORMAT
from nexusml.constants import INFERENCE_SERVICE_NAME
from nexusml.constants import MONITORING_SERVICE_NAME
from nexusml.constants import TESTING_SERVICE_NAME

###############
# State codes #
###############

STATE_CODE_LENGTH = 4  # up to 100 groups/prefixes and 100 states per group/prefix

# Task (00xx)
TASK_SETUP_STATE_CODE = '0000'
TASK_ACTIVE_STATE_CODE = '0001'
TASK_INACTIVE_STATE_CODE = '0002'

# Inference Service (01xx)
INFERENCE_STOPPED_STATE_CODE = '0100'
INFERENCE_RUNNING_STATE_CODE = '0101'

# CL Service (02xx)
CL_STOPPED_STATE_CODE = '0200'
CL_RUNNING_STATE_CODE = '0201'

# AL Service (03xx)
AL_STOPPED_STATE_CODE = '0300'
AL_RUNNING_STATE_CODE = '0301'

# Monitoring Service (04xx)
MONITORING_STOPPED_STATE_CODE = '0400'
MONITORING_RUNNING_STATE_CODE = '0401'

# Testing Service (05xx)
TESTING_STOPPED_STATE_CODE = '0500'
TESTING_RUNNING_STATE_CODE = '0501'

# Unknown (8xxx)
TASK_UNKNOWN_STATE_CODE = '8000'
INFERENCE_UNKNOWN_STATE_CODE = '8010'
CL_UNKNOWN_STATE_CODE = '8020'
AL_UNKNOWN_STATE_CODE = '8030'
MONITORING_UNKNOWN_STATE_CODE = '8040'
TESTING_UNKNOWN_STATE_CODE = '8050'

# Errors (9xxx)
TASK_ERROR_STATE_CODE = '9000'
INFERENCE_ERROR_STATE_CODE = '9010'
CL_ERROR_STATE_CODE = '9020'
AL_ERROR_STATE_CODE = '9030'
MONITORING_ERROR_STATE_CODE = '9040'
TESTING_ERROR_STATE_CODE = '9050'
"""
Status codes
"""

STATUS_CODE_LENGTH = 5  # up to 100 groups/prefixes and 1000 statuses per group/prefix

# Task (00xxy)
TASK_CREATED_STATUS_CODE = '00000'
TASK_COPYING_STATUS_CODE = '00001'
TASK_ACTIVE_STATUS_CODE = '00010'
TASK_PAUSED_STATUS_CODE = '00020'
TASK_RESUMING_STATUS_CODE = '00021'
TASK_SUSPENDED_STATUS_CODE = '00022'
TASK_CANCELED_STATUS_CODE = '00023'

# Inference Service (01xxy)
INFERENCE_STOPPED_STATUS_CODE = '01000'
INFERENCE_WAITING_STATUS_CODE = '01010'
INFERENCE_PROCESSING_STATUS_CODE = '01011'

# CL Service (02xxy)
CL_STOPPED_STATUS_CODE = '02000'
CL_WAITING_STATUS_CODE = '02010'
CL_INITIALIZING_TRAINING_STATUS_CODE = '02011'  # starting EC2 instances, etc.
CL_TRAINING_STATUS_CODE = '02012'  # retraining AI model
CL_DEPLOYING_STATUS_CODE = '02013'  # deploying AI model in production

# AL Service (03xxy)
AL_STOPPED_STATUS_CODE = '03000'
AL_WAITING_STATUS_CODE = '03010'
AL_ANALYZING_STATUS_CODE = '03011'

# Monitoring Service (04xxy)
MONITORING_STOPPED_STATUS_CODE = '04000'
MONITORING_WAITING_STATUS_CODE = '04010'
MONITORING_ANALYZING_STATUS_CODE = '04011'

# Testing Service (05xxy)
TESTING_STOPPED_STATUS_CODE = '05000'
TESTING_WAITING_STATUS_CODE = '05010'
TESTING_SETUP_STATUS_CODE = '05011'
TESTING_PROCESSING_STATUS_CODE = '05012'

# Unknown (8xxxy)
TASK_UNKNOWN_STATUS_CODE = '80000'
INFERENCE_UNKNOWN_STATUS_CODE = '80100'
CL_UNKNOWN_STATUS_CODE = '80200'
AL_UNKNOWN_STATUS_CODE = '80300'
MONITORING_UNKNOWN_STATUS_CODE = '80400'
TESTING_UNKNOWN_STATUS_CODE = '80500'

# Task errors (900xy)
TASK_UNKNOWN_ERROR_STATUS_CODE = '90000'

# Inference errors (901xx)
INFERENCE_UNKNOWN_ERROR_STATUS_CODE = '90100'
INFERENCE_ENVIRONMENT_ERROR_STATUS_CODE = '90101'
INFERENCE_CONNECTION_ERROR_STATUS_CODE = '90102'
INFERENCE_DATA_ERROR_STATUS_CODE = '90103'
INFERENCE_SCHEMA_ERROR_STATUS_CODE = '90104'
INFERENCE_AI_MODEL_ERROR_STATUS_CODE = '90105'

# CL errors (902xy)
CL_UNKNOWN_ERROR_STATUS_CODE = '90200'
CL_ENVIRONMENT_ERROR_STATUS_CODE = '90201'
CL_CONNECTION_ERROR_STATUS_CODE = '90202'
CL_DATA_ERROR_STATUS_CODE = '90203'
CL_SCHEMA_ERROR_STATUS_CODE = '90204'
CL_AI_MODEL_ERROR_STATUS_CODE = '90205'

# AL errors (903xy)
AL_UNKNOWN_ERROR_STATUS_CODE = '90300'
AL_CONNECTION_ERROR_STATUS_CODE = '90301'
AL_DATA_ERROR_STATUS_CODE = '90302'

# Monitoring errors (904xy)
MONITORING_UNKNOWN_ERROR_STATUS_CODE = '90400'
MONITORING_CONNECTION_ERROR_STATUS_CODE = '90401'
MONITORING_DATA_ERROR_STATUS_CODE = '90402'

# Testing errors (905xy)
TESTING_UNKNOWN_ERROR_STATUS_CODE = '90500'
TESTING_ENVIRONMENT_ERROR_STATUS_CODE = '90501'
TESTING_CONNECTION_ERROR_STATUS_CODE = '90502'
TESTING_DATA_ERROR_STATUS_CODE = '90503'
TESTING_SCHEMA_ERROR_STATUS_CODE = '90504'
TESTING_AI_MODEL_ERROR_STATUS_CODE = '90505'
"""
State/Status data classes
"""

status_group_prefixes = {
    'task': ('00', '800', '900'),
    INFERENCE_SERVICE_NAME: ('01', '801', '901'),
    CL_SERVICE_NAME: ('02', '802', '902'),
    AL_SERVICE_NAME: ('03', '803', '903'),
    MONITORING_SERVICE_NAME: ('04', '804', '904'),
    TESTING_SERVICE_NAME: ('05', '805', '905')
}


@dataclass(frozen=True)
class State:
    """
    Class representing a state.

    Attributes:
        - code (str): state code
        - name (str): name
        - display_name (str): name to show
        - description (str): text describing the state
    """
    _states: ClassVar[Dict] = dict()  # dictionary of (key = state code, value = state)

    code: str
    name: str
    display_name: str
    description: str

    def __post_init__(self):
        assert len(self.code) == STATE_CODE_LENGTH
        assert self.code not in self._states
        self._states[self.code] = self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def get_state(cls, state_code: str):
        return cls._states.get(state_code)


@dataclass(frozen=True)
class StatusTemplate:
    """
    Class representing a status within a state, such as the transition into the state
    or the outcome of an action at a particular point in time.

    Note: unlike states, statuses might be continuously updated during their lifecycle.
          That's why we create hardcoded status templates.

    Attributes:
        - code (str): status code
        - name (str): name
        - display_name (str): name to show
        - description (str): text describing the status
        - state (State): state to which the status is linked
    """

    _statuses: ClassVar[Dict] = dict()  # dictionary of (key = status code, value = status)

    code: str
    name: str
    display_name: str
    description: str
    state: State

    def __post_init__(self):
        assert len(self.code) == STATUS_CODE_LENGTH
        assert self.code not in self._statuses
        self._statuses[self.code] = self

    def to_dict(self) -> dict:
        _dict = asdict(self)
        _dict['state'] = self.state.to_dict()
        return _dict

    @classmethod
    def get_status(cls, status_code: str):
        return cls._statuses.get(status_code)


@dataclass
class Status:
    """
    Represents an instance of a status template.

    Attributes:
        - code (str): status code
        - name (str): name
        - display_name (str): name to show
        - description (str): text describing the status
        - state (State): state to which the status is linked
        - started_at (datetime): start UTC datetime
        - updated_at (datetime): last update UTC datetime
        - ended_at (datetime): end UTC datetime
        - details (dict): any status-specific details (e.g. progress percentage)
        - prev_status (str): code of the previous status
    """

    code: str
    name: str
    display_name: str
    description: str
    state: State
    started_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = None
    ended_at: datetime = None
    details: dict = None
    prev_status: str = None

    def __init__(self, template: StatusTemplate):
        self.code = template.code
        self.name = template.name
        self.display_name = template.display_name
        self.description = template.description
        self.state = template.state
        # TODO: any way to avoid duplicating initialization code below?
        self.started_at = datetime.utcnow()
        self.updated_at = None
        self.ended_at = None
        self.details = None
        self.prev_status = None

    @classmethod
    def from_dict(cls, status: dict):
        _status = Status(StatusTemplate.get_status(status.get('code') or status.get('status_code')))
        _status.started_at = status.get('started_at', datetime.utcnow())
        _status.updated_at = status.get('updated_at')
        _status.ended_at = status.get('ended_at')
        _status.details = status.get('details')
        _status.prev_status = status.get('prev_status')
        return _status

    def to_dict(self, include_state=False, expand_status=False, expand_state=False) -> dict:
        _dict = asdict(self)
        # Status
        if include_state and not expand_state:
            _dict['status_code'] = _dict.pop('code')
        if not expand_status:
            _dict.pop('name')
            _dict.pop('display_name')
            _dict.pop('description')
        # State
        if include_state:
            if expand_state:
                _dict['state'] = self.state.to_dict()
            else:
                _dict['state_code'] = _dict.pop('state')['code']
        else:
            _dict.pop('state')
        # Convert datetimes to strings
        if isinstance(self.started_at, datetime):
            _dict['started_at'] = self.started_at.strftime(DATETIME_FORMAT)
        if self.updated_at is not None and isinstance(self.updated_at, datetime):
            _dict['updated_at'] = self.updated_at.strftime(DATETIME_FORMAT)
        if self.ended_at is not None and isinstance(self.ended_at, datetime):
            _dict['ended_at'] = self.ended_at.strftime(DATETIME_FORMAT)
        return _dict


#####################
# Predefined states #
#####################

# Task
task_setup_state = State(code=TASK_SETUP_STATE_CODE,
                         name='TaskSetup',
                         display_name='Setting Up',
                         description='Task is being set up')
task_active_state = State(code=TASK_ACTIVE_STATE_CODE,
                          name='TaskActive',
                          display_name='Task active',
                          description='Task is active')
task_inactive_state = State(code=TASK_INACTIVE_STATE_CODE,
                            name='TaskInactive',
                            display_name='Task inactive',
                            description='Task is inactive')

# Inference Service
inference_stopped_state = State(code=INFERENCE_STOPPED_STATE_CODE,
                                name='InferenceStopped',
                                display_name='Stopped',
                                description='Inference Service is stopped')
inference_running_state = State(code=INFERENCE_RUNNING_STATE_CODE,
                                name='InferenceRunning',
                                display_name='Running',
                                description='Inference Service is running')

# CL Service
cl_stopped_state = State(code=CL_STOPPED_STATE_CODE,
                         name='CLStopped',
                         display_name='Stopped',
                         description='Continual Learning Service is stopped')
cl_running_state = State(code=CL_RUNNING_STATE_CODE,
                         name='CLRunning',
                         display_name='Running',
                         description='Continual Learning Service is running')

# AL Service
al_stopped_state = State(code=AL_STOPPED_STATE_CODE,
                         name='ALStopped',
                         display_name='Stopped',
                         description='Active Learning Service is stopped')
al_running_state = State(code=AL_RUNNING_STATE_CODE,
                         name='ALRunning',
                         display_name='Running',
                         description='Active Learning Service is running')

# Monitoring Service
monitoring_stopped_state = State(code=MONITORING_STOPPED_STATE_CODE,
                                 name='MonitoringStopped',
                                 display_name='Stopped',
                                 description='Monitoring Service is stopped')
monitoring_running_state = State(code=MONITORING_RUNNING_STATE_CODE,
                                 name='MonitoringRunning',
                                 display_name='Running',
                                 description='Monitoring Service is running')

# Testing Service
testing_stopped_state = State(code=TESTING_STOPPED_STATE_CODE,
                              name='TestingStopped',
                              display_name='Stopped',
                              description='Testing Service is stopped')
testing_running_state = State(code=TESTING_RUNNING_STATE_CODE,
                              name='TestingRunning',
                              display_name='Running',
                              description='Testing Service is running')

# Unknown states
task_unknown_state = State(code=TASK_UNKNOWN_STATE_CODE,
                           name='TaskUnknown',
                           display_name='Unknown',
                           description='Cannot determine task state')
inference_unknown_state = State(code=INFERENCE_UNKNOWN_STATE_CODE,
                                name='InferenceUnknown',
                                display_name='Unknown',
                                description='Cannot determine Inference Service state')
cl_unknown_state = State(code=CL_UNKNOWN_STATE_CODE,
                         name='CLUnknown',
                         display_name='Unknown',
                         description='Cannot determine Continual Learning Service state')
al_unknown_state = State(code=AL_UNKNOWN_STATE_CODE,
                         name='ALUnknown',
                         display_name='Unknown',
                         description='Cannot determine Active Learning Service state')
monitoring_unknown_state = State(code=MONITORING_UNKNOWN_STATE_CODE,
                                 name='MonitoringUnknown',
                                 display_name='Unknown',
                                 description='Cannot determine Monitoring Service state')
testing_unknown_state = State(code=TESTING_UNKNOWN_STATE_CODE,
                              name='TestingUnknown',
                              display_name='Unknown',
                              description='Cannot determine Testing Service state')

# Errors
task_error_state = State(code=TASK_ERROR_STATE_CODE,
                         name='TaskError',
                         display_name='Error',
                         description='Something is wrong with the task')
inference_error_state = State(code=INFERENCE_ERROR_STATE_CODE,
                              name='InferenceError',
                              display_name='Error',
                              description='An error occurred in Inference Service')
cl_error_state = State(code=CL_ERROR_STATE_CODE,
                       name='CLError',
                       display_name='Error',
                       description='An error occurred in Continual Learning Service')
al_error_state = State(code=AL_ERROR_STATE_CODE,
                       name='ALError',
                       display_name='Error',
                       description='An error occurred in Active Learning Service')
monitoring_error_state = State(code=MONITORING_ERROR_STATE_CODE,
                               name='MonitoringError',
                               display_name='Error',
                               description='An error occurred in Monitoring Service')
testing_error_state = State(code=TESTING_ERROR_STATE_CODE,
                            name='TestingError',
                            display_name='Error',
                            description='An error occurred in Testing Service')
"""
Status templates
"""

# Task
task_created_status = StatusTemplate(code=TASK_CREATED_STATUS_CODE,
                                     name='TaskCreated',
                                     display_name='Created',
                                     description='Task has been created. Waiting for enough examples to train the AI',
                                     state=task_setup_state)
task_copying_status = StatusTemplate(code=TASK_COPYING_STATUS_CODE,
                                     name='TaskCopying',
                                     display_name='Copying',
                                     description='Task is being copied',
                                     state=task_setup_state)
task_active_status = StatusTemplate(code=TASK_ACTIVE_STATUS_CODE,
                                    name='TaskActive',
                                    display_name='Active',
                                    description='Task is active',
                                    state=task_active_state)
task_paused_status = StatusTemplate(code=TASK_PAUSED_STATUS_CODE,
                                    name='TaskPaused',
                                    display_name='Paused',
                                    description='Task is paused',
                                    state=task_inactive_state)
task_resuming_status = StatusTemplate(code=TASK_RESUMING_STATUS_CODE,
                                      name='TaskResuming',
                                      display_name='Resuming',
                                      description='Task is being resumed',
                                      state=task_inactive_state)
task_suspended_status = StatusTemplate(code=TASK_SUSPENDED_STATUS_CODE,
                                       name='TaskSuspended',
                                       display_name='Suspended',
                                       description='Task is suspended',
                                       state=task_inactive_state)
task_canceled_status = StatusTemplate(code=TASK_CANCELED_STATUS_CODE,
                                      name='TaskCanceled',
                                      display_name='Canceled',
                                      description='Task is canceled',
                                      state=task_inactive_state)

# Inference Service
inference_stopped_status = StatusTemplate(code=INFERENCE_STOPPED_STATUS_CODE,
                                          name='InferenceStopped',
                                          display_name='Stopped',
                                          description='Inference Service is not running and will not make any '
                                          'prediction',
                                          state=inference_stopped_state)
inference_waiting_status = StatusTemplate(code=INFERENCE_WAITING_STATUS_CODE,
                                          name='InferenceWaiting',
                                          display_name='Waiting',
                                          description='Inference Service is ready to make predictions',
                                          state=inference_running_state)
inference_processing_status = StatusTemplate(code=INFERENCE_PROCESSING_STATUS_CODE,
                                             name='InferenceProcessing',
                                             display_name='Processing',
                                             description='Inference Service is making predictions',
                                             state=inference_running_state)

# CL Service
cl_stopped_status = StatusTemplate(code=CL_STOPPED_STATUS_CODE,
                                   name='CLStopped',
                                   display_name='Stopped',
                                   description='Continual Learning Service is not running and will not train the AI',
                                   state=cl_stopped_state)
cl_waiting_status = StatusTemplate(code=CL_WAITING_STATUS_CODE,
                                   name='CLWaiting',
                                   display_name='Waiting',
                                   description='Continual Learning Service is waiting for enough examples to train '
                                   'the AI',
                                   state=cl_running_state)
cl_initializing_training_status = StatusTemplate(code=CL_INITIALIZING_TRAINING_STATUS_CODE,
                                                 name='CLInitializingTraining',
                                                 display_name='Initializing Training',
                                                 description='Continual Learning Service is initializing the AI '
                                                 'training',
                                                 state=cl_running_state)
cl_training_status = StatusTemplate(code=CL_TRAINING_STATUS_CODE,
                                    name='CLTraining',
                                    display_name='Training',
                                    description='Continual Learning Service is training the AI',
                                    state=cl_running_state)
cl_deploying_status = StatusTemplate(code=CL_DEPLOYING_STATUS_CODE,
                                     name='CLDeploying',
                                     display_name='Deploying AI',
                                     description='Continual Learning Service is deploying the AI',
                                     state=cl_running_state)

# AL Service
al_stopped_status = StatusTemplate(code=AL_STOPPED_STATUS_CODE,
                                   name='ALStopped',
                                   display_name='Stopped',
                                   description='Active Learning Service is not running and will not ask experts for '
                                   'data labels',
                                   state=al_stopped_state)
al_waiting_status = StatusTemplate(code=AL_WAITING_STATUS_CODE,
                                   name='ALWaiting',
                                   display_name='Waiting',
                                   description='Active Learning Service is waiting for enough AI predictions',
                                   state=al_running_state)
al_analyzing_status = StatusTemplate(code=AL_ANALYZING_STATUS_CODE,
                                     name='ALAnalyzing',
                                     display_name='Analyzing',
                                     description='Active Learning Service is analyzing AI predictions',
                                     state=al_running_state)

# Monitoring Service
monitoring_stopped_status = StatusTemplate(code=MONITORING_STOPPED_STATUS_CODE,
                                           name='MonitoringStopped',
                                           display_name='Stopped',
                                           description='Monitoring Service is not running and will not monitor AI '
                                           'activity',
                                           state=monitoring_stopped_state)
monitoring_waiting_status = StatusTemplate(code=MONITORING_WAITING_STATUS_CODE,
                                           name='MonitoringWaiting',
                                           display_name='Waiting',
                                           description='Monitoring Service is waiting for enough AI activity',
                                           state=monitoring_running_state)
monitoring_analyzing_status = StatusTemplate(code=MONITORING_ANALYZING_STATUS_CODE,
                                             name='MonitoringAnalyzing',
                                             display_name='Analyzing',
                                             description='Monitoring Service is analyzing AI activity',
                                             state=monitoring_running_state)
# Testing Service
testing_stopped_status = StatusTemplate(code=TESTING_STOPPED_STATUS_CODE,
                                        name='TestingStopped',
                                        display_name='Stopped',
                                        description='Testing Service is not running',
                                        state=testing_stopped_state)
testing_waiting_status = StatusTemplate(code=TESTING_WAITING_STATUS_CODE,
                                        name='TestingWaiting',
                                        display_name='Waiting',
                                        description='Testing Service is waiting for input data',
                                        state=testing_running_state)
testing_setup_status = StatusTemplate(code=TESTING_SETUP_STATUS_CODE,
                                      name='TestingSetup',
                                      display_name='Setting Up',
                                      description='Testing Service is setting up tests',
                                      state=testing_running_state)
testing_processing_status = StatusTemplate(code=TESTING_PROCESSING_STATUS_CODE,
                                           name='TestingProcessing',
                                           display_name='Processing',
                                           description='Testing Service is making predictions',
                                           state=testing_running_state)

# Unknown statuses
task_unknown_status = StatusTemplate(code=TASK_UNKNOWN_STATUS_CODE,
                                     name='TaskUnknown',
                                     display_name='Unknown',
                                     description='Cannot determine task status',
                                     state=task_unknown_state)
inference_unknown_status = StatusTemplate(code=INFERENCE_UNKNOWN_STATUS_CODE,
                                          name='InferenceUnknown',
                                          display_name='Unknown',
                                          description='Cannot determine Inference Service status',
                                          state=inference_unknown_state)
cl_unknown_status = StatusTemplate(code=CL_UNKNOWN_STATUS_CODE,
                                   name='CLUnknown',
                                   display_name='Unknown',
                                   description='Cannot determine Continual Learning Service status',
                                   state=cl_unknown_state)
al_unknown_status = StatusTemplate(code=AL_UNKNOWN_STATUS_CODE,
                                   name='ALUnknown',
                                   display_name='Unknown',
                                   description='Cannot determine Active Learning Service status',
                                   state=al_unknown_state)
monitoring_unknown_status = StatusTemplate(code=MONITORING_UNKNOWN_STATUS_CODE,
                                           name='MonitoringUnknown',
                                           display_name='Unknown',
                                           description='Cannot determine Monitoring Service status',
                                           state=monitoring_unknown_state)
testing_unknown_status = StatusTemplate(code=TESTING_UNKNOWN_STATUS_CODE,
                                        name='TestingUnknown',
                                        display_name='Unknown',
                                        description='Cannot determine Testing Service status',
                                        state=testing_unknown_state)

# Task errors
task_unknown_error_status = StatusTemplate(code=TASK_UNKNOWN_ERROR_STATUS_CODE,
                                           name='TaskUnknownError',
                                           display_name='Unknown Error',
                                           description='Unknown error in task',
                                           state=task_error_state)

# Inference errors
inference_unknown_error_status = StatusTemplate(code=INFERENCE_UNKNOWN_ERROR_STATUS_CODE,
                                                name='InferenceUnknownError',
                                                display_name='Unknown Error',
                                                description='Unknown error in Inference Service',
                                                state=inference_error_state)
inference_environment_error_status = StatusTemplate(code=INFERENCE_ENVIRONMENT_ERROR_STATUS_CODE,
                                                    name='InferenceEnvironmentError',
                                                    display_name='Environment Error',
                                                    description='Error with the environment in Inference Service',
                                                    state=inference_error_state)
inference_connection_error_status = StatusTemplate(code=INFERENCE_CONNECTION_ERROR_STATUS_CODE,
                                                   name='InferenceConnectionError',
                                                   display_name='Connection Error',
                                                   description='Failed connection in Inference Service',
                                                   state=inference_error_state)
inference_data_error_status = StatusTemplate(code=INFERENCE_DATA_ERROR_STATUS_CODE,
                                             name='InferenceDataError',
                                             display_name='Data Error',
                                             description='Invalid data in Inference Service',
                                             state=inference_error_state)
inference_schema_error_status = StatusTemplate(code=INFERENCE_SCHEMA_ERROR_STATUS_CODE,
                                               name='InferenceSchemaError',
                                               display_name='Schema Error',
                                               description='Invalid schema in Inference Service',
                                               state=inference_error_state)
inference_ai_model_error_status = StatusTemplate(code=INFERENCE_AI_MODEL_ERROR_STATUS_CODE,
                                                 name='InferenceAIModelError',
                                                 display_name='AI Model Error',
                                                 description='Something is wrong with the AI model used by Inference '
                                                 'Service',
                                                 state=inference_error_state)

# CL errors
cl_unknown_error_status = StatusTemplate(code=CL_UNKNOWN_ERROR_STATUS_CODE,
                                         name='CLUnknownError',
                                         display_name='Unknown Error',
                                         description='Unknown error in Continual Learning Service',
                                         state=cl_error_state)
cl_environment_error_status = StatusTemplate(code=CL_ENVIRONMENT_ERROR_STATUS_CODE,
                                             name='CLEnvironmentError',
                                             display_name='Environment Error',
                                             description='Error with the environment in Continual Learning Service',
                                             state=cl_error_state)
cl_connection_error_status = StatusTemplate(code=CL_CONNECTION_ERROR_STATUS_CODE,
                                            name='CLConnectionError',
                                            display_name='Connection Error',
                                            description='Failed connection in Continual Learning Service',
                                            state=cl_error_state)
cl_data_error_status = StatusTemplate(code=CL_DATA_ERROR_STATUS_CODE,
                                      name='CLDataError',
                                      display_name='Data Error',
                                      description='Invalid data in Continual Learning Service',
                                      state=cl_error_state)
cl_schema_error_status = StatusTemplate(code=CL_SCHEMA_ERROR_STATUS_CODE,
                                        name='CLSchemaError',
                                        display_name='Schema Error',
                                        description='Invalid schema in Continual Learning Service',
                                        state=cl_error_state)
cl_ai_model_error_status = StatusTemplate(code=CL_AI_MODEL_ERROR_STATUS_CODE,
                                          name='CLAIModelError',
                                          display_name='AI Model Error',
                                          description='Something is wrong with the AI model used by Continual '
                                          'Learning Service',
                                          state=cl_error_state)

# AL errors
al_unknown_error_status = StatusTemplate(code=AL_UNKNOWN_ERROR_STATUS_CODE,
                                         name='ALUnknownError',
                                         display_name='Unknown Error',
                                         description='Unknown error in Active Learning Service',
                                         state=al_error_state)
al_connection_error_status = StatusTemplate(code=AL_CONNECTION_ERROR_STATUS_CODE,
                                            name='ALConnectionError',
                                            display_name='Connection Error',
                                            description='Failed connection in Active Learning Service',
                                            state=al_error_state)
al_data_error_status = StatusTemplate(code=AL_DATA_ERROR_STATUS_CODE,
                                      name='ALDataError',
                                      display_name='Data Error',
                                      description='Invalid data in Active Learning Service',
                                      state=al_error_state)

# Monitoring errors
monitoring_unknown_error_status = StatusTemplate(code=MONITORING_UNKNOWN_ERROR_STATUS_CODE,
                                                 name='MonitoringUnknownError',
                                                 display_name='Unknown Error',
                                                 description='Unknown error in Monitoring Service',
                                                 state=monitoring_error_state)
monitoring_connection_error_status = StatusTemplate(code=MONITORING_CONNECTION_ERROR_STATUS_CODE,
                                                    name='MonitoringConnectionError',
                                                    display_name='Connection Error',
                                                    description='Failed connection in Monitoring Service',
                                                    state=monitoring_error_state)
monitoring_data_error_status = StatusTemplate(code=MONITORING_DATA_ERROR_STATUS_CODE,
                                              name='MonitoringDataError',
                                              display_name='Data Error',
                                              description='Invalid data in Monitoring Service',
                                              state=monitoring_error_state)

# Testing errors
testing_unknown_error_status = StatusTemplate(code=TESTING_UNKNOWN_ERROR_STATUS_CODE,
                                              name='TestingUnknownError',
                                              display_name='Unknown Error',
                                              description='Unknown error in Testing Service',
                                              state=testing_error_state)
testing_environment_error_status = StatusTemplate(code=TESTING_ENVIRONMENT_ERROR_STATUS_CODE,
                                                  name='TestingEnvironmentError',
                                                  display_name='Environment Error',
                                                  description='Error with the environment in Testing Service',
                                                  state=testing_error_state)
testing_connection_error_status = StatusTemplate(code=TESTING_CONNECTION_ERROR_STATUS_CODE,
                                                 name='TestingConnectionError',
                                                 display_name='Connection Error',
                                                 description='Failed connection in Testing Service',
                                                 state=testing_error_state)
testing_data_error_status = StatusTemplate(code=TESTING_DATA_ERROR_STATUS_CODE,
                                           name='TestingDataError',
                                           display_name='Data Error',
                                           description='Invalid data in Testing Service	',
                                           state=testing_error_state)
testing_schema_error_status = StatusTemplate(code=TESTING_SCHEMA_ERROR_STATUS_CODE,
                                             name='TestingSchemaError',
                                             display_name='	Schema Error',
                                             description='Invalid schema in Testing Service',
                                             state=testing_error_state)
testing_ai_model_error_status = StatusTemplate(code=TESTING_AI_MODEL_ERROR_STATUS_CODE,
                                               name='TestingAIModelError',
                                               display_name='AI Model Error',
                                               description='Something is wrong with the AI model used by Testing '
                                               'Service',
                                               state=testing_error_state)
