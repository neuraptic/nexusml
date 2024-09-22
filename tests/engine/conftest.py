from datetime import datetime
from datetime import timedelta
import os
import shutil

import pytest

from nexusml.constants import AL_SERVICE_NAME
from nexusml.constants import INFERENCE_SERVICE_NAME
from nexusml.constants import MONITORING_SERVICE_NAME
from nexusml.database.ai import AIModelDB
from nexusml.database.ai import PredictionDB
from nexusml.database.core import db
from nexusml.database.core import db_commit
from nexusml.database.core import empty_table
from nexusml.database.core import save_to_db
from nexusml.database.files import TaskFileDB
from nexusml.database.organizations import ClientDB
from nexusml.database.services import al_client_scopes
from nexusml.database.services import ALServiceSettings
from nexusml.database.services import inference_client_scopes
from nexusml.database.services import InferenceServiceSettings
from nexusml.database.services import monitoring_client_scopes
from nexusml.database.services import MonitoringServiceSettings
from nexusml.database.services import Service
from nexusml.database.subscriptions import Plan
from nexusml.database.subscriptions import SubscriptionDB
from nexusml.database.tasks import CategoryDB
from nexusml.database.tasks import ElementDB
from nexusml.database.tasks import TaskDB
from nexusml.enums import AIEnvironment
from nexusml.enums import BillingCycle
from nexusml.enums import Currency
from nexusml.enums import ElementType
from nexusml.enums import ElementValueType
from nexusml.enums import PredictionState
from nexusml.enums import ServiceType
from nexusml.enums import TaskFileUse
from nexusml.enums import TrainingDevice
from nexusml.statuses import al_waiting_status
from nexusml.statuses import inference_waiting_status
from nexusml.statuses import monitoring_waiting_status
from nexusml.statuses import Status

###############
# Basic stuff #
###############


@pytest.fixture(scope='session', autouse=True)
def engine_artifacts_dir(artifacts_dir) -> str:
    # Setup: Create or empty directory
    engine_artifacts_path = os.path.join(artifacts_dir, 'engine')
    os.mkdir(engine_artifacts_path)

    # Run tests
    yield engine_artifacts_path

    # Teardown: Delete directory
    shutil.rmtree(engine_artifacts_path)


############
# Database #
############


@pytest.fixture(scope='function', autouse=True)
def _restore_db(app_context, app_data_dir):
    # Setup: Create or empty database
    try:
        empty_table(TaskDB)
    except Exception:
        db.create_all()

    # Run test function
    yield

    # Teardown: Close database connection
    db.engine.dispose()


@pytest.fixture
def task_and_prediction() -> tuple[TaskDB, PredictionDB]:
    task_schema = {
        'inputs': [{
            'type': 'text'
        }],
        'outputs': [{
            'element': 'be93cbf8-6638-41d6-9808-fad09e89471e',
            'type': 'category'
        }, {
            'element': '659617b0-40d4-4a4b-9500-e6c3dbef78b8',
            'type': 'float'
        }, {
            'element': 'be93cbf8-6638-41d6-9808-fad09e89471e',
            'type': 'category'
        }]
    }

    task_uuids = ['8700d81d-be65-4b5b-8284-807d812f31cc']
    output_uuids = [
        'be93cbf8-6638-41d6-9808-fad09e89471e', '659617b0-40d4-4a4b-9500-e6c3dbef78b8',
        '05c5c405-8a95-46b5-9f50-2f554ae3f20a'
    ]
    input_uuids = [
        'be93cbf8-1111-41d6-9808-fad09e89471e', '659617b0-2222-4a4b-9500-e6c3dbef78b8',
        '05c5c405-3333-46b5-9f50-2f554ae3f20a'
    ]
    cat_uuids_1: list = [
        '73799fb9-cd5f-4f0c-bbd5-817340acfa72', '3151109a-502d-4e53-926e-d0584e324a4b',
        'ed4549d1-6fa7-4f8d-9636-4f44cc033f54', 'e18fee08-c6aa-4a1e-b89c-e322f4927946'
    ]
    cat_uuids_2: list = [
        '9220bb77-9088-42c6-9b2b-9aaf496d6cfa', 'f638b6c5-e2c7-43ce-9888-fd8ebc9ef2b4',
        'fb6af3d7-acf6-4c6f-83f1-62c7a218be2b', 'f95392ba-b5fc-4229-bc8a-03b8f6c1efd0'
    ]

    plan_db_obj: Plan = Plan(
        name='test_plan',
        organization_id=1,
        price=1.1,
        currency=Currency.EURO,
        billing_cycle=BillingCycle.ANNUAL,
        max_tasks=10,
        max_deployments=10,
        max_predictions=10,
        max_gpu_hours=1.2,
        max_cpu_hours=1.1,
        max_examples=100,
        space_limit=30000,
        max_users=10,
        max_roles=10,
        max_collaborators=10,
        max_clients=10,
    )
    save_to_db(plan_db_obj)
    SubscriptionDB.query().delete()
    db_commit()
    subscription_db_obj: SubscriptionDB = SubscriptionDB(organization_id=1,
                                                         start_at=datetime.utcnow() - timedelta(days=1),
                                                         end_at=datetime.utcnow() + timedelta(days=1),
                                                         cancel_at=datetime.utcnow() + timedelta(days=1),
                                                         plan_id=plan_db_obj.plan_id)
    save_to_db(subscription_db_obj)
    if not task_uuids:
        raise Exception('No task uuid found in tasks test creation')

    tasks: list = list()
    element_list: list = list()
    for task_idx, task_uuid in enumerate(task_uuids):
        task = TaskDB(name=f'task_id_{task_idx}', uuid=task_uuid, organization_id=1)

        save_to_db(task)
        tasks.append(task)

        al_client = ClientDB(organization_id=1, name=AL_SERVICE_NAME)
        save_to_db(al_client)
        al_client.update_api_key(scopes=al_client_scopes, never_expire=True)

        al_service: Service = Service(client_id=al_client.client_id,
                                      task_id=task.task_id,
                                      type_=ServiceType.ACTIVE_LEARNING,
                                      status=Status(template=al_waiting_status).to_dict(),
                                      settings=ALServiceSettings().to_dict())

        mon_client = ClientDB(organization_id=1, name=MONITORING_SERVICE_NAME)
        save_to_db(mon_client)
        mon_client.update_api_key(scopes=monitoring_client_scopes, never_expire=True)

        mon_service: Service = Service(client_id=mon_client.client_id,
                                       task_id=task.task_id,
                                       type_=ServiceType.MONITORING,
                                       status=Status(template=monitoring_waiting_status).to_dict(),
                                       settings=MonitoringServiceSettings().to_dict())

        inf_client = ClientDB(organization_id=1, name=INFERENCE_SERVICE_NAME)
        save_to_db(inf_client)
        inf_client.update_api_key(scopes=inference_client_scopes, never_expire=True)

        inf_service: Service = Service(client_id=inf_client.client_id,
                                       task_id=task.task_id,
                                       type_=ServiceType.INFERENCE,
                                       status=Status(template=inference_waiting_status).to_dict(),
                                       settings=InferenceServiceSettings().to_dict())

        save_to_db(al_service)
        save_to_db(mon_service)
        save_to_db(inf_service)

    ElementDB.query().delete()
    db_commit()

    for elem_idx, output_uuid in enumerate(output_uuids):
        categories: list = list()
        value_type = ElementValueType.CATEGORY
        if elem_idx == 0:
            categories = cat_uuids_1
        elif elem_idx == 2:
            categories = cat_uuids_2

        if not categories:
            value_type = ElementValueType.FLOAT
        element_db_obj: ElementDB = ElementDB(task_id=tasks[0].task_id,
                                              uuid=output_uuid,
                                              name=f'output_name_{elem_idx}',
                                              element_type=ElementType.OUTPUT,
                                              value_type=value_type)

        save_to_db(element_db_obj)
        for pos, cat_uuid in enumerate(categories):
            category: CategoryDB = CategoryDB(element_id=element_db_obj.element_id,
                                              name=f'name_{elem_idx}_{pos}',
                                              uuid=cat_uuid)
            save_to_db(category)

        element_list.append(element_db_obj)

    for elem_idx, input_uuid in enumerate(input_uuids):
        element_db_obj: ElementDB = ElementDB(task_id=tasks[0].task_id,
                                              uuid=input_uuid,
                                              name=f'input_name_{elem_idx+1}',
                                              element_type=ElementType.INPUT,
                                              value_type=ElementValueType.TEXT)
        save_to_db(element_db_obj)

    file_db_obj: TaskFileDB = TaskFileDB(task_id=tasks[0].task_id,
                                         filename='file_test_name',
                                         size=1,
                                         use_for=TaskFileUse.OUTPUT)
    save_to_db(objects=file_db_obj)
    AIModelDB.query().delete()
    db_commit()
    ai_model: AIModelDB = AIModelDB(task_id=tasks[0].task_id,
                                    version='version',
                                    file_id=file_db_obj.file_id,
                                    task_schema=task_schema,
                                    training_time=1.0,
                                    training_device=TrainingDevice.CPU,
                                    uuid='52a4f64d-37b5-49f9-85e6-a793cd8271e5')
    save_to_db(objects=ai_model)
    # Set model on production for tasks
    for task in tasks:
        task.prod_model_id = ai_model.model_id
    save_to_db(tasks)
    prediction: PredictionDB = PredictionDB(model_id=ai_model.model_id,
                                            task_id=tasks[0].task_id,
                                            environment=AIEnvironment.PRODUCTION,
                                            state=PredictionState.COMPLETE,
                                            size=613)
    save_to_db(objects=prediction)
    return task, prediction
