# States and Statuses

In general, a system is in a certain *state* at a point in time when the system meets a predefined set of criteria.
A *status* describes the transition into a state or the outcome of an action at a particular point in time.

In this context, NexusML is a system composed of multiple modules (subsystems). In each task, a module is run by a 
service that may go through different states/statuses:

- *Inference Service*. AI making predictions in production.
- *Continual Learning (CL) Service*. Performs periodic retraining of the AI.
- *Active Learning (AL) Service*. Asks experts for data labels.
- *Monitoring Service*. Monitors the Inference Service.
- *Testing Service*. Provides on-demand testing for both deployed and non-deployed AI models.

In addition to these services, the task itself has its own state/status (e.g. "created", "active", "canceled", etc.).

There are some important properties and requirements to keep in mind:

- A state must have at least one status. This requirement is also present in some platforms like 
[Microsoft Dynamics 365](https://docs.microsoft.com/en-us/dynamics365/customerengagement/on-premises/developer/define-custom-state-model-transitions?view=op-9-1#what-is-the-state-model).
- A status is always linked to a specific state.
- States are predefined and immutable, while the status of an entity may be updated during the whole lifecycle 
(e.g. progress percentage).

### State codes

#### Task (00xx)

| Code | Name         | Display name | Description          |
|------|--------------|--------------|----------------------|
| 0000 | TaskSetup    | Setting Up   | Task is being set up |
| 0001 | TaskActive   | Active       | Task is active       |
| 0002 | TaskInactive | Inactive     | Task is inactive     |

#### Inference Service (01xx)

| Code | Name             | Display name | Description                  |
|------|------------------|--------------|------------------------------|
| 0100 | InferenceStopped | Stopped      | Inference Service is stopped |
| 0101 | InferenceRunning | Running      | Inference Service is running |

#### CL Service (02xx)

| Code | Name      | Display name | Description                           |
|------|-----------|--------------|---------------------------------------|
| 0200 | CLStopped | Stopped      | Continual Learning Service is stopped |
| 0201 | CLRunning | Running      | Continual Learning Service is running |

#### AL Service (03xx)

| Code | Name      | Display name | Description                        |
|------|-----------|--------------|------------------------------------|
| 0300 | ALStopped | Stopped      | Active Learning Service is stopped |
| 0301 | ALRunning | Running      | Active Learning Service is running |

#### Monitoring Service (04xx)

| Code | Name              | Display name | Description                   |
|------|-------------------|--------------|-------------------------------|
| 0400 | MonitoringStopped | Stopped      | Monitoring Service is stopped |
| 0401 | MonitoringRunning | Running      | Monitoring Service is running |

#### Testing Service (05xx)

| Code | Name           | Display name | Description                |
|------|----------------|--------------|----------------------------|
| 0500 | TestingStopped | Stopped      | Testing Service is stopped |
| 0501 | TestingRunning | Running      | Testing Service is running |

#### Unknown (8xxx)

*Note: Unknown states, in turn, use prefixes for tasks (800), Inference Service (801), CL Service (802), 
AL Service (803), Monitoring Service (804), and Testing Service (805)*.

| Code | Name              | Display name | Description                                       |
|------|-------------------|--------------|---------------------------------------------------|
| 8000 | TaskUnknown       | Unknown      | Cannot determine task state                       |
| 8010 | InferenceUnknown  | Unknown      | Cannot determine Inference Service state          |
| 8020 | CLUnknown         | Unknown      | Cannot determine Continual Learning Service state |
| 8030 | ALUnknown         | Unknown      | Cannot determine Active Learning Service state    |
| 8040 | MonitoringUnknown | Unknown      | Cannot determine Monitoring Service state         |
| 8050 | TestingUnknown    | Unknown      | Cannot determine Testing Service state            |

#### Errors (9xxx)

*Note: Error states, in turn, use prefixes for tasks (900), Inference Service (901), CL Service (902), 
AL Service (903), Monitoring Service (904), and Testing Service (905)*.

| Code | Name            | Display name | Description                                     |
|------|-----------------|--------------|-------------------------------------------------|
| 9000 | TaskError       | Error        | Something is wrong with the task                |
| 9010 | InferenceError  | Error        | An error occurred in Inference Service          |
| 9020 | CLError         | Error        | An error occurred in Continual Learning Service |
| 9030 | ALError         | Error        | An error occurred in Active Learning Service    |
| 9040 | MonitoringError | Error        | An error occurred in Monitoring Service         |
| 9050 | TestingError    | Error        | An error occurred in Testing Service            |

### Status codes

*Note: Statuses are derived by combining a base state code with a unique identifier. For example, 
the `0001` state can produce status codes like `00010`, `00011`, `00012`, and so on.*


#### Task (00xxy)

| State code | Status code | Name          | Display name | Description                                                        | Details |
|------------|-------------|---------------|--------------|--------------------------------------------------------------------|---------|
| 0000       | 00000       | TaskCreated   | Created      | Task has been created. Waiting for enough examples to train the AI | `null`  |
| 0000       | 00001       | TaskCopying   | Copying      | Task is being copied                                               | `null`  |
| 0001       | 00010       | TaskActive    | Active       | Task is active                                                     | `null`  |
| 0002       | 00020       | TaskPaused    | Paused       | Task is paused                                                     | `null`  |
| 0002       | 00021       | TaskResuming  | Resuming     | Task is being resumed                                              | `null`  |
| 0002       | 00022       | TaskSuspended | Suspended    | Task is suspended                                                  | `null`  |
| 0002       | 00023       | TaskCanceled  | Canceled     | Task is canceled                                                   | `null`  |


#### Inference Service (01xxy)

| State code | Status code | Name                | Display name | Description                                                       | Details |
|------------|-------------|---------------------|--------------|-------------------------------------------------------------------|---------|
| 0100       | 01000       | InferenceStopped    | Stopped      | Inference Service is not running and will not make any prediction | `null`  |
| 0101       | 01010       | InferenceWaiting    | Waiting      | Inference Service is ready to make predictions                    | `null`  |
| 0101       | 01011       | InferenceProcessing | Processing   | Inference Service is making predictions                           | `null`  |

#### CL Service (02xxy)

| State code | Status code | Name                   | Display name          | Description                                                               | Details                    |
|------------|-------------|------------------------|-----------------------|---------------------------------------------------------------------------|----------------------------|
| 0200       | 02000       | CLStopped              | Stopped               | Continual Learning Service is not running and will not train the AI       | `null`                     |
| 0201       | 02010       | CLWaiting              | Waiting               | Continual Learning Service is waiting for enough examples to train the AI | `null`                     |
| 0201       | 02011       | CLInitializingTraining | Initializing Training | Continual Learning Service is initializing the AI training                | `null`                     |
| 0201       | 02012       | CLTraining             | Training              | Continual Learning Service is training the AI                             | `int: progress percentage` |
| 0201       | 02013       | CLDeploying            | Deploying AI          | Continual Learning Service is deploying the AI                            | `null`                     |

#### AL Service (03xxy)

| State code | Status code | Name        | Display name | Description                                                                     | Details |
|------------|-------------|-------------|--------------|---------------------------------------------------------------------------------|---------|
| 0300       | 03000       | ALStopped   | Stopped      | Active Learning Service is not running and will not ask experts for data labels | `null`  |
| 0301       | 03010       | ALWaiting   | Waiting      | Active Learning Service is waiting for enough AI predictions                    | `null`  |
| 0301       | 03011       | ALAnalyzing | Analyzing    | Active Learning Service is analyzing AI predictions                             | `null`  |

#### Monitoring Service (04xxy)

| State code | Status code | Name                | Display name | Description                                                        | Details |
|------------|-------------|---------------------|--------------|--------------------------------------------------------------------|---------|
| 0400       | 04000       | MonitoringStopped   | Stopped      | Monitoring Service is not running and will not monitor AI activity | `null`  |
| 0401       | 04010       | MonitoringWaiting   | Waiting      | Monitoring Service is waiting for enough AI activity               | `null`  |
| 0401       | 04011       | MonitoringAnalyzing | Analyzing    | Monitoring Service is analyzing AI activity                        | `null`  |

#### Testing Service (05xxy)

| State code | Status code | Name              | Display name | Description                               | Details                    |
|------------|-------------|-------------------|--------------|-------------------------------------------|----------------------------|
| 0500       | 05000       | TestingStopped    | Stopped      | Testing Service is not running            | `null`                     |
| 0501       | 05010       | TestingWaiting    | Waiting      | Testing Service is waiting for input data | `null`                     |
| 0501       | 05011       | TestingSetup      | Setting Up   | Testing Service is setting up tests       | `null`                     |
| 0501       | 05012       | TestingProcessing | Processing   | Testing Service is making predictions     | `int: progress percentage` |

#### Unknown (8xxxy)

*Note: Unknown statuses, in turn, use prefixes for tasks (800), Inference Service (801), CL Service (802), 
AL Service (803), Monitoring Service (804), and Testing Service (805)*.

| State code | Status code | Name              | Display name | Description                                        | Details |
|------------|-------------|-------------------|--------------|----------------------------------------------------|---------|
| 8000       | 80000       | TaskUnknown       | Unknown      | Cannot determine task status                       | `null`  |
| 8010       | 80100       | InferenceUnknown  | Unknown      | Cannot determine Inference Service status          | `null`  |
| 8020       | 80200       | CLUnknown         | Unknown      | Cannot determine Continual Learning Service status | `null`  |
| 8030       | 80300       | ALUnknown         | Unknown      | Cannot determine Active Learning Service status    | `null`  |
| 8040       | 80400       | MonitoringUnknown | Unknown      | Cannot determine Monitoring Service status         | `null`  |
| 8050       | 80500       | TestingUnknown    | Unknown      | Cannot determine Testing Service status            | `null`  |

#### Errors (9xxxy)

*Note: Error statuses, in turn, use prefixes for tasks (900), Inference Service (901), CL Service (902), 
AL Service (903), Monitoring Service (904), and Testing Service (905)*.

| State code | Status code | Name                      | Display name      | Description                                                             | Details                  |
|------------|-------------|---------------------------|-------------------|-------------------------------------------------------------------------|--------------------------|
| 9000       | 90000       | TaskUnknownError          | Unknown Error     | Unknown error in task                                                   | `str: error description` |
| 9010       | 90100       | InferenceUnknownError     | Unknown Error     | Unknown error in Inference Service                                      | `str: error description` |
| 9010       | 90101       | InferenceEnvironmentError | Environment Error | Error with the environment in Inference Service                         | `str: error description` |
| 9010       | 90102       | InferenceConnectionError  | Connection Error  | Failed connection in Inference Service                                  | `str: error description` |
| 9010       | 90103       | InferenceDataError        | Data Error        | Invalid data in Inference Service                                       | `str: error description` |
| 9010       | 90104       | InferenceSchemaError      | Schema Error      | Invalid schema in Inference Service                                     | `str: error description` |
| 9010       | 90105       | InferenceAIModelError     | AI Model Error    | Something is wrong with the AI model used by Inference Service          | `str: error description` |
| 9020       | 90200       | CLUnknownError            | Unknown Error     | Unknown error in Continual Learning Service                             | `str: error description` |
| 9020       | 90201       | CLEnvironmentError        | Environment Error | Error with the environment in Continual Learning Service                | `str: error description` |
| 9020       | 90202       | CLConnectionError         | Connection Error  | Failed connection in Continual Learning Service                         | `str: error description` |
| 9020       | 90203       | CLDataError               | Data Error        | Invalid data in Continual Learning Service                              | `str: error description` |
| 9020       | 90204       | CLSchemaError             | Schema Error      | Invalid schema in Continual Learning Service                            | `str: error description` |
| 9020       | 90205       | CLAIModelError            | AI Model Error    | Something is wrong with the AI model used by Continual Learning Service | `str: error description` |
| 9030       | 90300       | ALUnknownError            | Unknown Error     | Unknown error in Active Learning Service                                | `str: error description` |
| 9030       | 90301       | ALConnectionError         | Connection Error  | Failed connection in Active Learning Service                            | `str: error description` |
| 9030       | 90302       | ALDataError               | Data Error        | Invalid data in Active Learning Service                                 | `str: error description` |
| 9040       | 90400       | MonitoringUnknownError    | Unknown Error     | Unknown error in Monitoring Service                                     | `str: error description` |
| 9040       | 90401       | MonitoringConnectionError | Connection Error  | Failed connection in Monitoring Service                                 | `str: error description` |
| 9040       | 90402       | MonitoringDataError       | Data Error        | Invalid data in Monitoring Service                                      | `str: error description` |
| 9050       | 90500       | TestingUnknownError       | Unknown Error     | Unknown error in Testing Service                                        | `str: error description` |
| 9050       | 90501       | TestingEnvironmentError   | Environment Error | Error with the environment in Testing Service                           | `str: error description` |
| 9050       | 90502       | TestingConnectionError    | Connection Error  | Failed connection in Testing Service                                    | `str: error description` |
| 9050       | 90503       | TestingDataError          | Data Error        | Invalid data in Testing Service                                         | `str: error description` |
| 9050       | 90504       | TestingSchemaError        | Schema Error      | Invalid schema in Testing Service                                       | `str: error description` |
| 9050       | 90505       | TestingAIModelError       | AI Model Error    | Something is wrong with the AI model used by Testing Service            | `str: error description` |
