# TODO: Try to make this module independent from `nexusml.api`

from datetime import datetime
from datetime import timedelta
import os
import shutil
from typing import List, Type

import boto3
from flask import Flask
from moto import mock_s3 as moto_mock_s3
import pytest

from nexusml.api import create_app
from nexusml.api.buffers import ALBuffer
from nexusml.api.buffers import ALBufferIO
from nexusml.api.buffers import MonBuffer
from nexusml.api.buffers import MonBufferIO
from nexusml.api.resources.ai import PredictionLog
from nexusml.api.resources.tasks import Task
from nexusml.api.utils import config
from nexusml.database.ai import PredictionDB
from nexusml.database.buffers import ALBufferItemDB
from nexusml.database.buffers import BufferItemDB
from nexusml.database.buffers import MonBufferItemDB
from nexusml.database.core import save_to_db
from nexusml.database.organizations import ClientDB
from nexusml.database.services import Service
from nexusml.database.tasks import ElementDB
from nexusml.database.tasks import TaskDB
from nexusml.enums import AIEnvironment
from nexusml.enums import ElementType
from nexusml.enums import PredictionState
from tests.api.conftest import set_app_config

BUFFER_ITEMS_PER_TASK = 10
MOCK_FILE_SIZE = 350
"""
App
"""


@pytest.fixture(scope='session', autouse=True)
def app(app_data_dir) -> Flask:
    try:
        app_ = create_app()
    except RuntimeError:
        # TODO: Resolve the `RuntimeError: NexusML config already initialized` issue.
        #       This error emerges when an app has already been created by integration tests,
        #       due to the API's reliance on a centralized config object.
        #       A temporary workaround involves overriding that object.
        #       Decentralizing config would be a more permanent solution.
        config._app = None
        app_ = create_app()

    set_app_config(app=app_)

    return app_


@pytest.fixture(scope='function', autouse=True)
def app_context(app):
    with app.app_context():
        yield


@pytest.fixture(scope='session', autouse=True)
def app_data_dir(engine_artifacts_dir) -> str:
    # TODO: What is this for?
    # Setup: Create or empty directory
    app_data_path = os.path.join(engine_artifacts_dir, 'edge')
    os.mkdir(app_data_path)

    # Run tests
    yield app_data_path

    # Teardown: Delete directory
    shutil.rmtree(app_data_path)


@pytest.fixture(scope='session')
def tasks_dir(app_data_dir) -> str:
    return os.path.join(app_data_dir, 'tasks')


###########
# Mock S3 #
###########


def mock_s3(func):
    """ Decorator for mocking S3 server. """

    def wrapper(*args, **kwargs):
        with moto_mock_s3():
            conn = boto3.resource('s3')
            conn.create_bucket(Bucket='test_bucket')
            return func(*args, **kwargs)

    return wrapper


########################
# Services and Buffers #
########################


@pytest.fixture
def al_local_buffers(task_and_prediction) -> list[ALBuffer]:
    return [ALBuffer(buffer_io=ALBufferIO(task=task_and_prediction[0]))]


@pytest.fixture
def mon_local_buffers(task_and_prediction) -> list[MonBuffer]:
    return [MonBuffer(buffer_io=MonBufferIO(task=task_and_prediction[0]))]


def _buffer_items(task_db_obj: TaskDB, prediction: PredictionDB, item_type: Type[BufferItemDB]) -> list[BufferItemDB]:
    input_elements: list[ElementDB] = ElementDB.query().filter_by(task_id=task_db_obj.task_id,
                                                                  element_type=ElementType.INPUT).all()
    data = [
        # 1
        {
            'inputs': [{
                'element': x.name,
                'value': f'value_{task_db_obj.task_id}_{idx + 1}'
            } for idx, x in enumerate(input_elements)],
            'outputs': [{
                'element': 'be93cbf8-6638-41d6-9808-fad09e89471e',
                'value': {
                    'category': '3151109a-502d-4e53-926e-d0584e324a4b',
                    'scores': {
                        '73799fb9-cd5f-4f0c-bbd5-817340acfa72': 2.4658748928110463,
                        '3151109a-502d-4e53-926e-d0584e324a4b': 5.86673195311106,
                        'ed4549d1-6fa7-4f8d-9636-4f44cc033f54': 0.11161469330433826,
                        'e18fee08-c6aa-4a1e-b89c-e322f4927946': 3.695551011127666,
                    }
                }
            }, {
                'element': '659617b0-40d4-4a4b-9500-e6c3dbef78b8',
                'value': 32.93063275833592
            }, {
                'element': '05c5c405-8a95-46b5-9f50-2f554ae3f20a',
                'value': {
                    'category': 'f95392ba-b5fc-4229-bc8a-03b8f6c1efd0',
                    'scores': {
                        '9220bb77-9088-42c6-9b2b-9aaf496d6cfa': 7.404816774277608,
                        'f638b6c5-e2c7-43ce-9888-fd8ebc9ef2b4': 3.7394433390205197,
                        'fb6af3d7-acf6-4c6f-83f1-62c7a218be2b': 5.955647968846527,
                        'f95392ba-b5fc-4229-bc8a-03b8f6c1efd0': 9.782921384831504
                    }
                }
            }],
            'relevance': 1.2058271327301169,
            'state': PredictionState.COMPLETE
        },
        # 2
        {
            'inputs': [{
                'element': x.name,
                'value': f'value_{task_db_obj.task_id}_{idx + 1}'
            } for idx, x in enumerate(input_elements)],
            'outputs': [{
                'element': 'be93cbf8-6638-41d6-9808-fad09e89471e',
                'value': {
                    'category': '73799fb9-cd5f-4f0c-bbd5-817340acfa72',
                    'scores': {
                        '73799fb9-cd5f-4f0c-bbd5-817340acfa72': 8.715222095892393,
                        '3151109a-502d-4e53-926e-d0584e324a4b': 5.730860817267085,
                        'ed4549d1-6fa7-4f8d-9636-4f44cc033f54': 1.9243928298074187,
                        'e18fee08-c6aa-4a1e-b89c-e322f4927946': 1.4038939468522138
                    }
                }
            }, {
                'element': '659617b0-40d4-4a4b-9500-e6c3dbef78b8',
                'value': 44.28476659152101
            }, {
                'element': '05c5c405-8a95-46b5-9f50-2f554ae3f20a',
                'value': {
                    'category': 'f638b6c5-e2c7-43ce-9888-fd8ebc9ef2b4',
                    'scores': {
                        '9220bb77-9088-42c6-9b2b-9aaf496d6cfa': 7.449631089354747,
                        'f638b6c5-e2c7-43ce-9888-fd8ebc9ef2b4': 8.591272151212497,
                        'fb6af3d7-acf6-4c6f-83f1-62c7a218be2b': 3.5932454358316077,
                        'f95392ba-b5fc-4229-bc8a-03b8f6c1efd0': 4.179711982441031
                    }
                }
            }],
            'relevance': 1.238853062217331,
            'state': PredictionState.COMPLETE
        },
        # 3
        {
            'inputs': [{
                'element': x.name,
                'value': f'value_{task_db_obj.task_id}_{idx + 1}'
            } for idx, x in enumerate(input_elements)],
            'outputs': [{
                'element': 'be93cbf8-6638-41d6-9808-fad09e89471e',
                'value': {
                    'category': '73799fb9-cd5f-4f0c-bbd5-817340acfa72',
                    'scores': {
                        '73799fb9-cd5f-4f0c-bbd5-817340acfa72': 8.568894883186042,
                        '3151109a-502d-4e53-926e-d0584e324a4b': 4.254953236584313,
                        'ed4549d1-6fa7-4f8d-9636-4f44cc033f54': 2.2005423704287375,
                        'e18fee08-c6aa-4a1e-b89c-e322f4927946': 4.420816116062728
                    }
                }
            }, {
                'element': '659617b0-40d4-4a4b-9500-e6c3dbef78b8',
                'value': 13.801683433027822
            }, {
                'element': '05c5c405-8a95-46b5-9f50-2f554ae3f20a',
                'value': {
                    'category': 'fb6af3d7-acf6-4c6f-83f1-62c7a218be2b',
                    'scores': {
                        '9220bb77-9088-42c6-9b2b-9aaf496d6cfa': 4.672241775982648,
                        'f638b6c5-e2c7-43ce-9888-fd8ebc9ef2b4': 0.7351751057266376,
                        'fb6af3d7-acf6-4c6f-83f1-62c7a218be2b': 8.91219619055992,
                        'f95392ba-b5fc-4229-bc8a-03b8f6c1efd0': 6.610586048851999
                    }
                }
            }],
            'relevance': 1.228438100478867,
            'state': PredictionState.COMPLETE
        },
        # 4
        {
            'inputs': [{
                'element': x.name,
                'value': f'value_{task_db_obj.task_id}_{idx + 1}'
            } for idx, x in enumerate(input_elements)],
            'outputs': [{
                'element': 'be93cbf8-6638-41d6-9808-fad09e89471e',
                'value': {
                    'category': '73799fb9-cd5f-4f0c-bbd5-817340acfa72',
                    'scores': {
                        '73799fb9-cd5f-4f0c-bbd5-817340acfa72': 9.804218970500463,
                        '3151109a-502d-4e53-926e-d0584e324a4b': 5.163988539807452,
                        'ed4549d1-6fa7-4f8d-9636-4f44cc033f54': 4.875280767715207,
                        'e18fee08-c6aa-4a1e-b89c-e322f4927946': 2.610566312796477
                    }
                }
            }, {
                'element': '659617b0-40d4-4a4b-9500-e6c3dbef78b8',
                'value': 22.093303571884686
            }, {
                'element': '05c5c405-8a95-46b5-9f50-2f554ae3f20a',
                'value': {
                    'category': 'f638b6c5-e2c7-43ce-9888-fd8ebc9ef2b4',
                    'scores': {
                        '9220bb77-9088-42c6-9b2b-9aaf496d6cfa': 7.1001984713638775,
                        'f638b6c5-e2c7-43ce-9888-fd8ebc9ef2b4': 7.650785315964628,
                        'fb6af3d7-acf6-4c6f-83f1-62c7a218be2b': 1.5744025302886944,
                        'f95392ba-b5fc-4229-bc8a-03b8f6c1efd0': 6.556799761547164
                    }
                }
            }],
            'relevance': 1.2766745334761382,
            'state': PredictionState.COMPLETE
        },
        # 5
        {
            'inputs': [{
                'element': x.name,
                'value': f'value_{task_db_obj.task_id}_{idx + 1}'
            } for idx, x in enumerate(input_elements)],
            'outputs': [{
                'element': 'be93cbf8-6638-41d6-9808-fad09e89471e',
                'value': {
                    'category': 'ed4549d1-6fa7-4f8d-9636-4f44cc033f54',
                    'scores': {
                        '73799fb9-cd5f-4f0c-bbd5-817340acfa72': 4.680712035225271,
                        '3151109a-502d-4e53-926e-d0584e324a4b': 0.649219792303457,
                        'ed4549d1-6fa7-4f8d-9636-4f44cc033f54': 5.015917871449242,
                        'e18fee08-c6aa-4a1e-b89c-e322f4927946': 1.1864383417041113
                    }
                }
            }, {
                'element': '659617b0-40d4-4a4b-9500-e6c3dbef78b8',
                'value': 43.93264056965831
            }, {
                'element': '05c5c405-8a95-46b5-9f50-2f554ae3f20a',
                'value': {
                    'category': 'f95392ba-b5fc-4229-bc8a-03b8f6c1efd0',
                    'scores': {
                        '9220bb77-9088-42c6-9b2b-9aaf496d6cfa': 0.9174453284817485,
                        'f638b6c5-e2c7-43ce-9888-fd8ebc9ef2b4': 3.3802124773628774,
                        'fb6af3d7-acf6-4c6f-83f1-62c7a218be2b': 0.6123348933118888,
                        'f95392ba-b5fc-4229-bc8a-03b8f6c1efd0': 7.274589316060615
                    }
                }
            }],
            'relevance': 1.0663574348904161,
            'state': PredictionState.COMPLETE
        },
        # 6
        {
            'inputs': [{
                'element': x.name,
                'value': f'value_{task_db_obj.task_id}_{idx + 1}'
            } for idx, x in enumerate(input_elements)],
            'outputs': [{
                'element': 'be93cbf8-6638-41d6-9808-fad09e89471e',
                'value': {
                    'category': 'e18fee08-c6aa-4a1e-b89c-e322f4927946',
                    'scores': {
                        '73799fb9-cd5f-4f0c-bbd5-817340acfa72': 3.2640251510178264,
                        '3151109a-502d-4e53-926e-d0584e324a4b': 6.708503013902952,
                        'ed4549d1-6fa7-4f8d-9636-4f44cc033f54': 4.903981593165146,
                        'e18fee08-c6aa-4a1e-b89c-e322f4927946': 9.536412839583013
                    }
                }
            }, {
                'element': '659617b0-40d4-4a4b-9500-e6c3dbef78b8',
                'value': 7.018648423763585
            }, {
                'element': '05c5c405-8a95-46b5-9f50-2f554ae3f20a',
                'value': {
                    'category': 'f95392ba-b5fc-4229-bc8a-03b8f6c1efd0',
                    'scores': {
                        '9220bb77-9088-42c6-9b2b-9aaf496d6cfa': 7.151021242225093,
                        'f638b6c5-e2c7-43ce-9888-fd8ebc9ef2b4': 5.018488087915367,
                        'fb6af3d7-acf6-4c6f-83f1-62c7a218be2b': 8.1209362099993,
                        'f95392ba-b5fc-4229-bc8a-03b8f6c1efd0': 8.297953947476
                    }
                }
            }],
            'relevance': 1.3410998337948232,
            'state': PredictionState.COMPLETE
        },
        # 7
        {
            'inputs': [{
                'element': x.name,
                'value': f'value_{task_db_obj.task_id}_{idx + 1}'
            } for idx, x in enumerate(input_elements)],
            'outputs': [{
                'element': 'be93cbf8-6638-41d6-9808-fad09e89471e',
                'value': {
                    'category': 'e18fee08-c6aa-4a1e-b89c-e322f4927946',
                    'scores': {
                        '73799fb9-cd5f-4f0c-bbd5-817340acfa72': 7.5423335439412975,
                        '3151109a-502d-4e53-926e-d0584e324a4b': 6.1965979815007435,
                        'ed4549d1-6fa7-4f8d-9636-4f44cc033f54': 2.3092245328128627,
                        'e18fee08-c6aa-4a1e-b89c-e322f4927946': 9.076170204787385
                    }
                }
            }, {
                'element': '659617b0-40d4-4a4b-9500-e6c3dbef78b8',
                'value': 0.27015797593606394
            }, {
                'element': '05c5c405-8a95-46b5-9f50-2f554ae3f20a',
                'value': {
                    'category': 'fb6af3d7-acf6-4c6f-83f1-62c7a218be2b',
                    'scores': {
                        '9220bb77-9088-42c6-9b2b-9aaf496d6cfa': 6.4649351953822665,
                        'f638b6c5-e2c7-43ce-9888-fd8ebc9ef2b4': 5.51516255955795,
                        'fb6af3d7-acf6-4c6f-83f1-62c7a218be2b': 9.667408611500168,
                        'f95392ba-b5fc-4229-bc8a-03b8f6c1efd0': 7.5475662384129585
                    }
                }
            }],
            'relevance': 1.32902474902979,
            'state': PredictionState.COMPLETE
        },
        # 8
        {
            'inputs': [{
                'element': x.name,
                'value': f'value_{task_db_obj.task_id}_{idx + 1}'
            } for idx, x in enumerate(input_elements)],
            'outputs': [{
                'element': 'be93cbf8-6638-41d6-9808-fad09e89471e',
                'value': {
                    'category': 'ed4549d1-6fa7-4f8d-9636-4f44cc033f54',
                    'scores': {
                        '73799fb9-cd5f-4f0c-bbd5-817340acfa72': 3.5756566234935194,
                        '3151109a-502d-4e53-926e-d0584e324a4b': 6.534097081070122,
                        'ed4549d1-6fa7-4f8d-9636-4f44cc033f54': 7.278212938259689,
                        'e18fee08-c6aa-4a1e-b89c-e322f4927946': 6.063746502406877
                    }
                }
            }, {
                'element': '659617b0-40d4-4a4b-9500-e6c3dbef78b8',
                'value': 28.44276760151732
            }, {
                'element': '05c5c405-8a95-46b5-9f50-2f554ae3f20a',
                'value': {
                    'category': 'fb6af3d7-acf6-4c6f-83f1-62c7a218be2b',
                    'scores': {
                        '9220bb77-9088-42c6-9b2b-9aaf496d6cfa': 3.6438236919094935,
                        'f638b6c5-e2c7-43ce-9888-fd8ebc9ef2b4': 6.830019946762672,
                        'fb6af3d7-acf6-4c6f-83f1-62c7a218be2b': 9.34228521596468,
                        'f95392ba-b5fc-4229-bc8a-03b8f6c1efd0': 2.406989467884687
                    }
                }
            }],
            'relevance': 1.3098934899270094,
            'state': PredictionState.COMPLETE
        },
        # 9
        {
            'inputs': [{
                'element': x.name,
                'value': f'value_{task_db_obj.task_id}_{idx + 1}'
            } for idx, x in enumerate(input_elements)],
            'outputs': [{
                'element': 'be93cbf8-6638-41d6-9808-fad09e89471e',
                'value': {
                    'category': '73799fb9-cd5f-4f0c-bbd5-817340acfa72',
                    'scores': {
                        '73799fb9-cd5f-4f0c-bbd5-817340acfa72': 7.396193801250888,
                        '3151109a-502d-4e53-926e-d0584e324a4b': 5.231275501201499,
                        'ed4549d1-6fa7-4f8d-9636-4f44cc033f54': 2.814914229045656,
                        'e18fee08-c6aa-4a1e-b89c-e322f4927946': 6.375315958271768
                    }
                }
            }, {
                'element': '659617b0-40d4-4a4b-9500-e6c3dbef78b8',
                'value': 36.35511852228802
            }, {
                'element': '05c5c405-8a95-46b5-9f50-2f554ae3f20a',
                'value': {
                    'category': '9220bb77-9088-42c6-9b2b-9aaf496d6cfa',
                    'scores': {
                        '9220bb77-9088-42c6-9b2b-9aaf496d6cfa': 6.134722758615937,
                        'f638b6c5-e2c7-43ce-9888-fd8ebc9ef2b4': 2.6655752082612825,
                        'fb6af3d7-acf6-4c6f-83f1-62c7a218be2b': 2.495180768997013,
                        'f95392ba-b5fc-4229-bc8a-03b8f6c1efd0': 0.31972257184463504
                    }
                }
            }],
            'relevance': 1.2185222218112743,
            'state': PredictionState.COMPLETE
        },
        # 10
        {
            'inputs': [{
                'element': x.name,
                'value': f'value_{task_db_obj.task_id}_{1}'
            } for idx, x in enumerate(input_elements)],
            'outputs': [{
                'element': 'be93cbf8-6638-41d6-9808-fad09e89471e',
                'value': {
                    'category': 'ed4549d1-6fa7-4f8d-9636-4f44cc033f54',
                    'scores': {
                        '73799fb9-cd5f-4f0c-bbd5-817340acfa72': 0.7410973356609696,
                        '3151109a-502d-4e53-926e-d0584e324a4b': 6.894738329407564,
                        'ed4549d1-6fa7-4f8d-9636-4f44cc033f54': 7.0433655918229565,
                        'e18fee08-c6aa-4a1e-b89c-e322f4927946': 2.3194877279857073
                    }
                }
            }, {
                'element': '659617b0-40d4-4a4b-9500-e6c3dbef78b8',
                'value': 16.429801285438828
            }, {
                'element': '05c5c405-8a95-46b5-9f50-2f554ae3f20a',
                'value': {
                    'category': 'f95392ba-b5fc-4229-bc8a-03b8f6c1efd0',
                    'scores': {
                        '9220bb77-9088-42c6-9b2b-9aaf496d6cfa': 0.000673232327635942,
                        'f638b6c5-e2c7-43ce-9888-fd8ebc9ef2b4': 2.4793335045551004,
                        'fb6af3d7-acf6-4c6f-83f1-62c7a218be2b': 1.767820697371113,
                        'f95392ba-b5fc-4229-bc8a-03b8f6c1efd0': 9.066756188520891
                    }
                }
            }],
            'relevance': 0.9913351886542403,
            'state': PredictionState.COMPLETE
        },
    ]
    service_db: Service = Service.filter_by_task(task_id=task_db_obj.task_id)[0]
    client_db_agent: ClientDB = ClientDB.get(client_id=service_db.client_id)
    task: Task = Task.get(agent=client_db_agent, db_object_or_id=task_db_obj, check_permissions=False)
    updated_predictions: list[PredictionLog] = PredictionLog.post_batch(data=data,
                                                                        task=task,
                                                                        environment=AIEnvironment.PRODUCTION)

    buffer_items = []
    now = datetime.utcnow()
    pred_idx = 0
    for prediction_log, prediction_ in zip(updated_predictions, data):
        assert len(data) == BUFFER_ITEMS_PER_TASK
        pred_timestamp = now + timedelta(seconds=pred_idx)
        pred_relevance = prediction_['relevance']
        pred_size = prediction_log.size()

        buffer_item = item_type(task_id=task_db_obj.task_id,
                                prediction_id=prediction_log.db_object().prediction_id,
                                timestamp=pred_timestamp,
                                size=pred_size)
        if item_type == ALBufferItemDB:
            buffer_item.relevance = pred_relevance
            task_db_obj.al_buffer_items += 1
            task_db_obj.al_buffer_bytes += pred_size
        else:
            assert item_type == MonBufferItemDB
            buffer_item.relevance = pred_timestamp.timestamp()
            task_db_obj.mon_buffer_items += 1
            task_db_obj.mon_buffer_bytes += pred_size
        buffer_items.append(buffer_item)
        pred_idx += 1

    save_to_db(objects=buffer_items)
    save_to_db(objects=task_db_obj)
    return buffer_items


@pytest.fixture
def al_buffer_items(task_and_prediction) -> List[ALBufferItemDB]:
    return _buffer_items(task_db_obj=task_and_prediction[0],
                         prediction=task_and_prediction[1],
                         item_type=ALBufferItemDB)


@pytest.fixture
def mon_buffer_items(task_and_prediction) -> List[MonBufferItemDB]:
    return _buffer_items(task_db_obj=task_and_prediction[0],
                         prediction=task_and_prediction[1],
                         item_type=MonBufferItemDB)
