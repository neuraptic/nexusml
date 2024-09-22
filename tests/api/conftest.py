import copy
from datetime import datetime
import os
import random
import shutil
import sys
from typing import Iterable, List, Optional, Union
import uuid

from flask import Flask
import pytest
from sqlalchemy import bindparam
from sqlalchemy import text as sql_text
import yaml

from nexusml.api.resources.ai import PredictionLog
from nexusml.api.resources.examples import Example
from nexusml.api.resources.tasks import OutputElement
from nexusml.api.resources.tasks import Task
from nexusml.api.utils import config
from nexusml.constants import ADMIN_ROLE
from nexusml.constants import CONFIG_FILE
from nexusml.constants import MAINTAINER_ROLE
from nexusml.constants import NUM_RESERVED_CLIENTS
from nexusml.database.ai import AIModelDB
from nexusml.database.ai import PredictionDB
from nexusml.database.ai import PredScores
from nexusml.database.core import db
from nexusml.database.core import db_commit
from nexusml.database.core import db_execute
from nexusml.database.core import db_rollback
from nexusml.database.core import empty_table
from nexusml.database.core import save_to_db
from nexusml.database.examples import CommentDB
from nexusml.database.examples import ex_tags
from nexusml.database.examples import ExampleDB
from nexusml.database.examples import ExBoolean
from nexusml.database.examples import ExCategory
from nexusml.database.examples import ExFile
from nexusml.database.examples import ExFloat
from nexusml.database.examples import ExInteger
from nexusml.database.examples import ExText
from nexusml.database.examples import ShapeCategory
from nexusml.database.examples import ShapeDB
from nexusml.database.examples import ShapeFloat
from nexusml.database.examples import SliceCategory
from nexusml.database.examples import SliceDB
from nexusml.database.examples import SliceFloat
from nexusml.database.files import OrgFileDB
from nexusml.database.files import TaskFileDB
from nexusml.database.myaccount import AccountClientSettings
from nexusml.database.myaccount import AccountSettings
from nexusml.database.notifications import AggregatedNotificationDB
from nexusml.database.notifications import NotificationDB
from nexusml.database.organizations import ClientDB
from nexusml.database.organizations import CollaboratorDB
from nexusml.database.organizations import create_known_clients_and_reserved_clients
from nexusml.database.organizations import create_main_admin_and_maintainer
from nexusml.database.organizations import create_main_organization
from nexusml.database.organizations import InvitationDB
from nexusml.database.organizations import OrganizationDB
from nexusml.database.organizations import RoleDB
from nexusml.database.organizations import user_roles
from nexusml.database.organizations import UserDB
from nexusml.database.organizations import WaitList
from nexusml.database.permissions import RolePermission
from nexusml.database.permissions import UserPermission
from nexusml.database.subscriptions import create_default_plans
from nexusml.database.subscriptions import Plan
from nexusml.database.subscriptions import PlanExtra
from nexusml.database.subscriptions import SubscriptionDB
from nexusml.database.subscriptions import SubscriptionDiscount
from nexusml.database.subscriptions import SubscriptionExtra
from nexusml.database.tags import TagDB
from nexusml.database.tasks import CategoryDB
from nexusml.database.tasks import demo_tasks
from nexusml.database.tasks import ElementDB
from nexusml.database.tasks import TaskDB
from nexusml.enums import AIEnvironment
from nexusml.enums import BillingCycle
from nexusml.enums import Currency
from nexusml.enums import ElementMultiValue
from nexusml.enums import ElementType
from nexusml.enums import ElementValueType
from nexusml.enums import FileType
from nexusml.enums import LabelingStatus
from nexusml.enums import OrgFileUse
from nexusml.enums import PredictionState
from nexusml.enums import TaskFileUse
from nexusml.enums import TrainingDevice
from nexusml.env import AWS_S3_BUCKET
from nexusml.env import ENV_API_DOMAIN
from nexusml.env import ENV_DB_NAME
from nexusml.env import ENV_DB_PASSWORD
from nexusml.env import ENV_DB_USER
from nexusml.utils import FILE_TYPES
from tests.api.constants import TEST_CONFIG
from tests.api.env import ENV_MOCK_CLIENT_ID
from tests.api.env import ENV_SESSION_USER_AUTH0_ID
from tests.api.env import ENV_SESSION_USER_UUID

###############
# Basic stuff #
###############


@pytest.fixture(scope='session', autouse=True)
def api_config(tests_dir) -> dict:
    # TODO: Save config to an `api` subdir.
    #       Issue: `nexusml.api._set_app_config()` loads config from working directory (`tests_dir`).
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(TEST_CONFIG, f)
    return TEST_CONFIG


@pytest.fixture(scope='function', autouse=True)
def api_domain(monkeypatch) -> str:
    _api_domain = 'http://' + os.environ[ENV_API_DOMAIN]

    monkeypatch.setenv(os.environ[ENV_API_DOMAIN], _api_domain)

    monkeypatch.setattr('nexusml.api.resources.base.API_DOMAIN', _api_domain)
    monkeypatch.setattr('nexusml.api.resources.files.API_DOMAIN', _api_domain)
    monkeypatch.setattr('nexusml.api.views.files.API_DOMAIN', _api_domain)
    monkeypatch.setattr('tests.api.integration.conftest.API_DOMAIN', _api_domain)
    monkeypatch.setattr('tests.api.integration.test_api_keys.API_DOMAIN', _api_domain)
    monkeypatch.setattr('tests.api.integration.test_myaccount.API_DOMAIN', _api_domain)
    monkeypatch.setattr('tests.api.integration.utils.API_DOMAIN', _api_domain)

    return _api_domain


@pytest.fixture(scope='session', autouse=True)
def api_artifacts_dir(artifacts_dir) -> str:
    # Setup: Create or empty directory
    api_artifacts_path = os.path.join(artifacts_dir, 'api')
    os.mkdir(api_artifacts_path)

    # Run tests
    yield api_artifacts_path

    # Teardown: Delete directory
    shutil.rmtree(api_artifacts_path)


#######
# App #
#######


def set_app_config(app: Flask):

    def _set_db_uri(template_db_uri: str) -> str:
        _db_connection = {
            '<database>': ENV_DB_NAME,
            '<user>': ENV_DB_USER,
            '<password>': ENV_DB_PASSWORD,
        }

        for param, value in _db_connection.items():
            if value not in os.environ:
                print(f'FATAL: environment variable "{value}" has not been set. Exiting')
                sys.exit(1)
            else:
                template_db_uri = template_db_uri.replace(param, os.environ[value])

        return template_db_uri

    test_config = copy.deepcopy(TEST_CONFIG)
    test_config['storage']['files']['s3']['bucket'] = os.environ[AWS_S3_BUCKET]
    config.set(test_config)

    app.config['SQLALCHEMY_DATABASE_URI'] = _set_db_uri(template_db_uri=test_config['storage']['database']['uri'])
    app.config['MAX_CONTENT_LENGTH'] = test_config['limits']['requests']['max_payload']


############
# Database #
############


def populate_db():
    # TODO: Do we really need to populate so many tables?
    # TODO: Decompose this function into multiple functions. It's too long.

    _default_creator = {'created_by_user': 1, 'modified_by_user': 1, 'synced_by_users': [1]}

    def _create_elements(element_type: ElementType) -> List[ElementDB]:
        are_inputs = element_type == ElementType.INPUT

        multi_value_types = {
            ElementValueType.INTEGER: ElementMultiValue.ORDERED,
            ElementValueType.FLOAT: ElementMultiValue.TIME_SERIES,
            ElementValueType.CATEGORY: ElementMultiValue.UNORDERED,
        }

        elements = []

        idx = 1
        for value_type in ElementValueType:
            if element_type == ElementType.OUTPUT and value_type not in OutputElement.ALLOWED_VALUE_TYPES:
                continue

            required = not (value_type in FILE_TYPES or value_type in [ElementValueType.SHAPE, ElementValueType.SLICE])

            multi_value = (are_inputs or value_type == ElementValueType.CATEGORY)

            element = ElementDB(task_id=1,
                                element_id=(idx + (element_type.value * len(ElementValueType))),
                                name=f'{element_type.name.lower()}_{idx}',
                                display_name=f'{element_type.name.lower()} {idx}',
                                element_type=element_type,
                                value_type=value_type,
                                required=required,
                                nullable=(not required),
                                multi_value=(multi_value_types.get(value_type) if multi_value else None),
                                **_default_creator)

            elements.append(element)

            idx += 1

        return elements

    def _get_image_element(elements: Iterable[ElementDB]) -> ElementDB:
        return [x for x in elements if x.value_type == ElementValueType.IMAGE_FILE][0]

    def _get_sequence_element(elements: Iterable[ElementDB]) -> ElementDB:
        return [
            x for x in elements if (x.value_type in [ElementValueType.INTEGER, ElementValueType.FLOAT] and
                                    x.multi_value in [ElementMultiValue.ORDERED, ElementMultiValue.TIME_SERIES])
        ][0]

    def _set_element_values(db_object: Union[ExampleDB, PredictionDB],
                            elements: Iterable[ElementDB],
                            categories: List[CategoryDB] = None) -> list:

        db_model = type(db_object)
        id_col = 'example_id' if db_model == ExampleDB else 'prediction_id'
        db_object_id = getattr(db_object, id_col)

        elem_values = {
            ElementType.INPUT: {
                ElementValueType.BOOLEAN: (lambda x: x % 2 == 0),
                ElementValueType.INTEGER: (lambda x: 153 * x),
                ElementValueType.FLOAT: (lambda x: 7.45 * x),
                ElementValueType.TEXT: (lambda x: f'test_{db_model}_input_value_{x}'),
                ElementValueType.DATETIME:
                    (lambda x: datetime(year=2021 - x, month=x % 12 + 1, day=x % 30 + 1, hour=x % 24)),
                ElementValueType.CATEGORY: (lambda x: max(1, x % len(categories)) if categories else None),
                ElementValueType.DOCUMENT_FILE: (lambda x: 2 if x % 2 == 0 else 4),
                ElementValueType.IMAGE_FILE: (lambda x: 3 if x % 2 == 0 else 3),
                ElementValueType.VIDEO_FILE: (lambda x: 4 if x % 2 == 0 else 2),
                ElementValueType.AUDIO_FILE: (lambda x: 5 if x % 2 == 0 else 1),
            },
            ElementType.OUTPUT: {
                ElementValueType.INTEGER: (lambda x: 63 * x),
                ElementValueType.FLOAT: (lambda x: 14.952 * x),
                ElementValueType.CATEGORY: (lambda x: max(1, x % len(categories)) if categories else None)
            },
            ElementType.METADATA: {
                ElementValueType.BOOLEAN: (lambda x: x % 2 == 0),
                ElementValueType.INTEGER: (lambda x: 4 * x),
                ElementValueType.FLOAT: (lambda x: 33.47 * x),
                ElementValueType.TEXT: (lambda x: f'test_{db_model}_metadata_value_{x}'),
                ElementValueType.DATETIME:
                    (lambda x: datetime(year=2014 - x, month=x % 12 + 1, day=x % 30 + 1, hour=x % 24)),
                ElementValueType.CATEGORY: (lambda x: max(1, x % len(categories)) if categories else None)
            }
        }

        if db_model == PredictionDB:

            def _mock_category_json(*args, **kwargs) -> Optional[dict]:
                if not categories:
                    return None
                # Get random scores
                random_scores = [random.uniform(0, 1) for _ in range(len(categories))]
                scores_sum = sum(random_scores)
                normalized_scores = [x / scores_sum for x in random_scores]
                max_score = max(normalized_scores)
                # Assign scores to categories
                cat_scores = dict()
                winner_category_name = None
                for category, score in zip(categories, normalized_scores):
                    cat_scores[category.name] = score
                    if score == max_score:
                        winner_category_name = category.name
                return {
                    'category': winner_category_name,
                    'scores': cat_scores,
                }

            def _mock_shape_json(x: int) -> dict:
                task = TaskDB.get(task_id=db_object.task_id)
                input_element = _get_image_element(elements=task.input_elements())
                return {
                    'element': input_element.name,
                    'polygon': [{
                        'x': 136 + x,
                        'y': 14 + x
                    }, {
                        'x': 201 + x,
                        'y': 78 + x
                    }, {
                        'x': 166 + x,
                        'y': 49 + x
                    }]
                }

            def _mock_slice_json(x: int) -> dict:
                task = TaskDB.get(task_id=db_object.task_id)
                input_element = _get_sequence_element(elements=task.input_elements())
                return {'element': input_element.name, 'start_index': 2, 'end_index': 4}

            elem_values[ElementType.OUTPUT][ElementValueType.CATEGORY] = _mock_category_json
            elem_values[ElementType.OUTPUT][ElementValueType.SHAPE] = _mock_shape_json
            elem_values[ElementType.OUTPUT][ElementValueType.SLICE] = _mock_slice_json

        element_values = []
        for element in elements:
            elem_type = element.element_type
            value_type = element.value_type

            if value_type not in elem_values[elem_type]:
                continue

            if (db_model == PredictionDB and elem_type == ElementType.OUTPUT and
                    value_type == ElementValueType.CATEGORY):
                value_db_model = PredScores
            else:
                value_db_model = db_model.value_type_models()[value_type]

            for index in range(1, (6 if element.multi_value is not None else 2)):
                elem_value = value_db_model(element_id=element.element_id,
                                            value=elem_values[elem_type][value_type](db_object_id),
                                            index=index,
                                            **{id_col: db_object_id})
                element_values.append(elem_value)

        return element_values

    def _set_shape_or_slice_output_values(db_object: Union[ShapeDB, SliceDB],
                                          outputs: Iterable[ElementDB],
                                          categories: Iterable[CategoryDB] = None) -> list:
        output_values = {
            ElementValueType.FLOAT: (lambda x: 14.952 * x),
            ElementValueType.CATEGORY: (lambda x: max(1, x % len(categories)) if categories else None)
        }

        output_db_models = {
            ElementValueType.FLOAT: ShapeFloat if isinstance(db_object, ShapeDB) else SliceFloat,
            ElementValueType.CATEGORY: ShapeCategory if isinstance(db_object, ShapeDB) else SliceCategory
        }

        pk_col = 'shape_id' if isinstance(db_object, ShapeDB) else 'slice_id'
        pk_val = getattr(db_object, pk_col)

        elem_values_db_objs = []
        for output in outputs:
            value_type = output.value_type
            if value_type not in output_values:
                continue
            output_db_model = output_db_models[value_type]
            elem_value = output_db_model(**{pk_col: pk_val},
                                         element_id=output.element_id,
                                         value=output_values[value_type](pk_val))
            elem_values_db_objs.append(elem_value)

        return elem_values_db_objs

    db_commit()

    ###################################################################
    # Organizations                                                   #

    # Note: `organization_id=1` is reserved for the main organization #
    ###################################################################
    empty_table(OrganizationDB)
    create_main_organization()
    organization_2 = OrganizationDB(organization_id=2,
                                    trn='org_2_trn',
                                    name='Organization 2',
                                    domain='org2.com',
                                    address='Organization Address 2')
    organization_3 = OrganizationDB(organization_id=3,
                                    trn='org_3_trn',
                                    name='Organization 3',
                                    domain='org3.com',
                                    address='Organization Address 3')
    save_to_db([organization_2, organization_3])

    #################
    # Subscriptions #
    #################
    empty_table(Plan)
    create_default_plans()

    plan = Plan(
        plan_id=2,  # 1 is reserved for the Free Plan
        name='Test Plan Name',
        description='Test Plan Description',
        price=12000,
        currency=Currency.DOLLAR,
        billing_cycle=BillingCycle.ANNUAL,
        max_tasks=20,
        max_deployments=500,
        max_users=100,
        max_roles=10,
        max_collaborators=50,
        max_clients=3)
    save_to_db(plan)

    plan_extra = PlanExtra(extra_id=1,
                           plan_id=2,
                           price=2200,
                           extra_tasks=3,
                           extra_users=5,
                           extra_roles=3,
                           extra_collaborators=8,
                           extra_clients=2)
    save_to_db(plan_extra)

    subscription = SubscriptionDB(subscription_id=1, organization_id=2, plan_id=2)
    save_to_db(subscription)

    discount = SubscriptionDiscount(discount_id=1, subscription_id=1, percentage=20)
    save_to_db(discount)

    extra = SubscriptionExtra(sub_extra_id=1, subscription_id=1, extra_id=1)
    save_to_db(extra)

    ###############
    # Invitations #
    ###############

    # Empty tables
    empty_table(InvitationDB)

    # Create invitation
    invitation = InvitationDB(organization_id=2, email='test@org2.com')
    save_to_db(invitation)

    #########
    # Users #
    #########

    # Empty tables
    empty_table(UserDB)

    # Create users
    user_uuids = [os.environ[ENV_SESSION_USER_UUID]] + [str(uuid.uuid4()) for _ in range(7)]
    user_auth0_ids = ([os.environ[ENV_SESSION_USER_AUTH0_ID]] +
                      [f'auth0|{111111111111111111111111 + (111111111111111111111111 * i)}' for i in range(1, 8)])
    users = [
        UserDB(user_id=(idx + 1),
               uuid=user_uuid,
               organization_id=(2 if idx < 4 else 3),
               auth0_id=str(user_auth0_ids[idx])) for idx, user_uuid in enumerate(user_uuids)
    ]
    save_to_db(users)

    # Set default user
    user = UserDB.get(user_id=1)
    assert user.uuid == os.environ[ENV_SESSION_USER_UUID]

    #########
    # Roles #
    #########
    # Empty tables
    empty_table(RoleDB)
    create_main_admin_and_maintainer()

    # Create roles
    admin_role = RoleDB(organization_id=2, name=ADMIN_ROLE, description='Test Admin Role')
    maintainer_role = RoleDB(organization_id=2, name=MAINTAINER_ROLE, description='Test Maintainer Role')
    dasci_role = RoleDB(organization_id=2, name='Dasci', description='Test Dasci Role')
    devs_role = RoleDB(organization_id=2, name='Devs', description='Test Devs Role')

    roles = [admin_role, maintainer_role, dasci_role, devs_role]
    save_to_db(roles)

    # Assign roles to users
    users_roles = {1: [admin_role.role_id], 2: [dasci_role.role_id, devs_role.role_id], 3: [dasci_role.role_id]}
    user_role_insert = user_roles.insert().values(user_id=bindparam('user_id'), role_id=bindparam('role_id'))
    user_roles_rows = [{'user_id': u, 'role_id': role_id} for u, roles in users_roles.items() for role_id in roles]

    db_execute(user_role_insert, user_roles_rows)
    db_commit()

    #################
    # Collaborators #
    #################
    empty_table(CollaboratorDB)
    collaborators = [CollaboratorDB(organization_id=2, user_id=x.user_id) for x in users if x.organization_id != 2]
    save_to_db(collaborators)

    ##################
    # Clients (apps) #
    ##################
    empty_table(ClientDB)
    create_known_clients_and_reserved_clients()

    # Create mock client
    mock_client = ClientDB(client_id=(NUM_RESERVED_CLIENTS + 1),
                           organization_id=2,
                           uuid=os.environ[ENV_MOCK_CLIENT_ID],
                           name='Mock Client (Organization 2)')

    clients = [
        ClientDB(client_id=idx,
                 organization_id=2 if idx % 2 == 1 else 3,
                 name=f'Client {idx}',
                 description=f'Description {idx}')
        for idx in range((NUM_RESERVED_CLIENTS + 2), (NUM_RESERVED_CLIENTS + 11))
    ]

    clients.insert(0, mock_client)

    save_to_db(clients)
    for client in clients:
        client.update_api_key()
    save_to_db(clients)

    ###############
    # Permissions #
    ###############
    empty_table(UserPermission)
    empty_table(RolePermission)

    #################
    # User Settings #
    #################
    user_settings = AccountSettings(user_id=1)
    user_client_settings = AccountClientSettings(user_id=1,
                                                 client_id=mock_client.client_id,
                                                 client_version='0.1.0',
                                                 settings={
                                                     'Parameter 1': 'Value 1',
                                                     'Parameter N': 'Value N'
                                                 })
    save_to_db(user_settings)
    save_to_db(user_client_settings)

    ######################
    # Organization Files #
    ######################

    NUM_ORG_FILES = 6

    # Empty tables
    empty_table(OrgFileDB)

    # Prepare rows
    org_files = []
    files_space_usage = 0
    for idx in range(1, NUM_ORG_FILES + 1):
        file = OrgFileDB(organization_id=2,
                         file_id=idx,
                         filename=(f'test_org_file_{idx}'),
                         size=20 + 10 * idx,
                         use_for=OrgFileUse.PICTURE,
                         type_=FileType.IMAGE,
                         created_by_user=1)
        org_files.append(file)
        files_space_usage += file.size

    # Save rows
    save_to_db(org_files)

    #########
    # Tasks #
    #########
    # Empty tables
    empty_table(TaskDB)

    # Prepare rows
    tasks = [
        TaskDB(task_id=idx,
               organization_id=(2 if idx % 2 != 0 else 3),
               name=f'Mock task {idx}',
               description=f'Mock description {idx}',
               **_default_creator) for idx in range(1, 4)
    ]
    task = tasks[0]  # default task

    input_elements = _create_elements(ElementType.INPUT)
    output_elements = _create_elements(ElementType.OUTPUT)
    metadata_elements = _create_elements(ElementType.METADATA)

    input_cat_element = [x for x in input_elements if x.value_type == ElementValueType.CATEGORY][0]
    output_cat_element = [x for x in output_elements if x.value_type == ElementValueType.CATEGORY][0]
    metadata_cat_element = [x for x in metadata_elements if x.value_type == ElementValueType.CATEGORY][0]

    cat_colors = ['B52B2B', '2BB535', '27468F']

    input_categories = [
        CategoryDB(element_id=input_cat_element.element_id,
                   category_id=idx,
                   name=f'Input Category {(idx - 1) % len(cat_colors) + 1}',
                   color=cat_colors[(idx - 1) % len(cat_colors)],
                   **_default_creator) for idx in range(1,
                                                        len(cat_colors) + 1)
    ]

    output_categories = [
        CategoryDB(element_id=output_cat_element.element_id,
                   category_id=idx,
                   name=f'Output Category {(idx - 1) % len(cat_colors) + 1}',
                   color=cat_colors[(idx - 1) % len(cat_colors)],
                   **_default_creator) for idx in range(len(cat_colors) + 1, 2 * len(cat_colors) + 1)
    ]

    metadata_categories = [
        CategoryDB(element_id=metadata_cat_element.element_id,
                   category_id=idx,
                   name=f'Metadata Category {(idx - 1) % len(cat_colors) + 1}',
                   color=cat_colors[(idx - 1) % len(cat_colors)],
                   **_default_creator) for idx in range(2 * len(cat_colors) + 1, 3 * len(cat_colors) + 1)
    ]

    tags = [
        TagDB(task_id=1, tag_id=idx, name=f'Tag {idx}', color=cat_colors[idx % len(cat_colors)], **_default_creator)
        for idx in range(1, 7)
    ]

    # Save rows
    save_to_db(tasks)
    save_to_db(input_elements)
    save_to_db(output_elements)
    save_to_db(metadata_elements)
    save_to_db(input_categories)
    save_to_db(output_categories)
    save_to_db(metadata_categories)
    save_to_db(tags)

    # Create default task's services
    task.init_task()

    ##############
    # Task Files #
    ##############
    NUM_AI_FILES = 3
    NUM_EX_INPUT_FILES = 3
    NUM_PRED_INPUT_FILES = 3

    num_files = NUM_AI_FILES + NUM_EX_INPUT_FILES + NUM_PRED_INPUT_FILES

    # Empty tables
    empty_table(TaskFileDB)

    # Prepare rows
    task_files = []
    files_space_usage = 0
    first_ai_file_idx = NUM_EX_INPUT_FILES + NUM_PRED_INPUT_FILES + 1

    for idx in range(1, num_files + 1):
        file_use = TaskFileUse.INPUT if idx < first_ai_file_idx else TaskFileUse.AI_MODEL
        filename = f'test_input_file_{idx}' if idx < first_ai_file_idx else f'ai_model_file_{idx}.tar.gz'
        file = TaskFileDB(task_id=1,
                          file_id=idx,
                          filename=filename,
                          size=60 + 10 * idx,
                          use_for=file_use,
                          type_=(FileType.IMAGE if file_use == TaskFileUse.INPUT else None),
                          created_by_user=1)
        task_files.append(file)
        files_space_usage += file.size

    # Save rows
    save_to_db(task_files)

    #############
    # AI models #
    #############
    # Empty table
    empty_table(AIModelDB)

    # Prepare rows
    ai_models = []
    for idx in range(1, NUM_AI_FILES + 1):
        file_id = idx + first_ai_file_idx - 1
        ai_model = AIModelDB(task_id=1,
                             model_id=idx,
                             file_id=file_id,
                             version=f'0.{file_id}.0',
                             task_schema={
                                 'inputs': [],
                                 'outputs': [],
                                 'metadata': [],
                             },
                             training_time=3.0,
                             training_device=TrainingDevice.GPU,
                             extra_metadata={
                                 'regularization': (idx % 2 == 0),
                                 'architecture': 'resnet-34'
                             },
                             created_by_user=1)
        ai_models.append(ai_model)

    # Save rows
    save_to_db(ai_models)

    # Set production and testing models
    task.prod_model_id = ai_models[0].model_id

    task.test_model_id = ai_models[1].model_id

    save_to_db(task)

    ############
    # Examples #
    ############
    NUM_EXAMPLES = 6

    image_element_id = _get_image_element(input_elements).element_id
    seq_element_id = _get_sequence_element(input_elements).element_id

    # Empty table
    empty_table(ExampleDB)

    # Set examples
    examples = [
        ExampleDB(task_id=1,
                  example_id=(idx + 1),
                  labeling_status=LabelingStatus(idx % len(LabelingStatus)),
                  trained=(idx % 3 == 0),
                  **_default_creator) for idx in range(NUM_EXAMPLES)
    ]

    save_to_db(examples)

    # Set element values
    example_values = []

    for example in examples:
        example_values += _set_element_values(db_object=example, elements=input_elements, categories=input_categories)

        example_values += _set_element_values(db_object=example, elements=output_elements, categories=output_categories)

        example_values += _set_element_values(db_object=example,
                                              elements=metadata_elements,
                                              categories=metadata_categories)

    save_to_db(example_values)

    # Add comments
    comments = [
        CommentDB(example_id=1, comment_id=1, message='test message 1.1', created_by_user=1),
        CommentDB(example_id=1, comment_id=2, message='test message 1.2', created_by_user=1),
        CommentDB(example_id=1, comment_id=3, message='test message 1.3', created_by_user=1),
        CommentDB(example_id=2, comment_id=4, message='test message 2', created_by_user=1),
        CommentDB(example_id=4, comment_id=5, message='test message 4', created_by_user=1),
        CommentDB(example_id=5, comment_id=6, message='test message 5', created_by_user=1)
    ]

    save_to_db(comments)

    # Add tags
    examples_tags = {1: [1, 3, 5, 6], 2: [3, 4], 3: [2, 5], 5: [3, 6], 6: [4, 6]}

    example_tag_insert = ex_tags.insert().values(example_id=bindparam('example_id'), tag_id=bindparam('tag_id'))
    example_tags_rows = [{
        'example_id': example_id,
        'tag_id': tag_id
    } for example_id, tag_ids in examples_tags.items() for tag_id in tag_ids]
    db_execute(example_tag_insert, example_tags_rows)
    db_commit()

    # Add shapes
    shapes = [
        ShapeDB(example_id=1,
                element_id=image_element_id,
                shape_id=1,
                polygon=[{
                    'x': 120,
                    'y': 52
                }, {
                    'x': 120,
                    'y': 531
                }, {
                    'x': 469,
                    'y': 52
                }, {
                    'x': 469,
                    'y': 531
                }],
                **_default_creator),
        ShapeDB(example_id=1,
                element_id=image_element_id,
                shape_id=2,
                pixels=[{
                    'x': 501,
                    'y': 4
                }, {
                    'x': 2,
                    'y': 68
                }, {
                    'x': 93,
                    'y': 107
                }],
                **_default_creator)
    ]

    save_to_db(shapes)

    shape_outputs = _set_shape_or_slice_output_values(db_object=shapes[0],
                                                      outputs=output_elements,
                                                      categories=output_categories)

    save_to_db(shape_outputs)

    # Add slices
    slices = [
        SliceDB(example_id=1, element_id=seq_element_id, slice_id=1, start_index=1, end_index=3, **_default_creator),
        SliceDB(example_id=1, element_id=seq_element_id, slice_id=2, start_index=2, end_index=4, **_default_creator)
    ]

    save_to_db(slices)

    slice_outputs = _set_shape_or_slice_output_values(db_object=slices[0],
                                                      outputs=output_elements,
                                                      categories=output_categories)

    save_to_db(slice_outputs)

    ###############
    # Predictions #
    ###############
    NUM_PREDICTIONS = 6

    # Empty table
    empty_table(PredictionDB)

    # Prepare rows
    predictions = [
        PredictionDB(task_id=1,
                     prediction_id=(idx + 1),
                     model_id=task.prod_model_id,
                     environment=(AIEnvironment.PRODUCTION if idx % 2 == 0 else AIEnvironment.TESTING),
                     state=PredictionState.COMPLETE,
                     created_by_user=1) for idx in range(NUM_PREDICTIONS)
    ]
    # TODO: set shapes and slices

    prediction_values = []
    for prediction in predictions:
        # Set element values
        input_values = _set_element_values(db_object=prediction, elements=input_elements, categories=input_categories)

        output_values = _set_element_values(db_object=prediction,
                                            elements=output_elements,
                                            categories=output_categories)

        metadata_values = _set_element_values(db_object=prediction,
                                              elements=metadata_elements,
                                              categories=metadata_categories)
        # Set target values
        target_values = copy.deepcopy(output_values)
        for element_value in target_values:
            element_value.is_target = True
            if isinstance(element_value.value, dict) and set(element_value.value.keys()) == {'category', 'scores'}:
                element_value.value.pop('scores')
        # Add element values
        prediction_values += input_values
        prediction_values += output_values
        prediction_values += metadata_values
        prediction_values += target_values

    # Save rows
    save_to_db(predictions)
    save_to_db(prediction_values)

    ###################################
    # Update subscription quota usage #
    ###################################
    # Compute examples and predictions' space usage
    parent_task = Task.get(agent=user, db_object_or_id=task, check_permissions=False)

    example_resources = [
        Example.get(agent=user, db_object_or_id=x, parents=[parent_task], check_permissions=False) for x in examples
    ]

    examples_space_usage = sum(example.size(refresh=True) for example in example_resources)

    prediction_resources = [
        PredictionLog.get(agent=user, db_object_or_id=x, parents=[parent_task], check_permissions=False)
        for x in predictions
    ]

    predictions_space_usage = sum(prediction.size(refresh=True) for prediction in prediction_resources)

    # Users, Roles, Collaborators, and Clients (apps)
    subscription.num_users = len(users)
    subscription.num_roles = len([x for x in roles if x.name not in [ADMIN_ROLE, MAINTAINER_ROLE]])
    subscription.num_collaborators = len(collaborators)
    subscription.num_clients = len(clients) / 2

    # Number of Tasks
    subscription.num_tasks = len(tasks)

    # CPU/GPU hours
    subscription.num_cpu_hours = 0
    subscription.num_gpu_hours = 75
    task.num_cpu_hours = 0
    task.num_gpu_hours = 75

    # Number of examples
    subscription.num_examples = len(examples)
    subscription.num_predictions = len(predictions)
    task.num_examples = len(examples)
    task.num_predictions = len(predictions)

    # Space usage
    space_usage = files_space_usage + examples_space_usage + predictions_space_usage
    subscription.space_usage = space_usage
    task.space_usage = space_usage

    # Save database entries
    save_to_db([subscription, task])

    #################
    # Notifications #
    #################
    empty_table(NotificationDB)
    empty_table(AggregatedNotificationDB)

    ##############
    # Demo Tasks #
    ##############
    _demo_tasks = demo_tasks()

    # Define task schemas
    for (idx, task) in enumerate(_demo_tasks):
        name = f'demo_{idx + 1}'
        # Elements (inputs, outputs, metadata)
        inputs = [
            ElementDB(task_id=task.task_id,
                      name=('input_1_' + name),
                      display_name=('Input 1 ' + task.name),
                      description='No description',
                      element_type=ElementType.INPUT,
                      value_type=ElementValueType.TEXT),
            ElementDB(task_id=task.task_id,
                      name=('input_2_' + name),
                      display_name=('Input 2 ' + task.name),
                      description='No description',
                      element_type=ElementType.INPUT,
                      value_type=ElementValueType.CATEGORY),
            ElementDB(task_id=task.task_id,
                      name=('input_3_' + name),
                      display_name=('Input 3 ' + task.name),
                      description='No description',
                      element_type=ElementType.INPUT,
                      value_type=ElementValueType.IMAGE_FILE)
        ]
        outputs = [
            ElementDB(task_id=task.task_id,
                      name=('output_1_' + name),
                      display_name=('Output 1 ' + task.name),
                      description='No description',
                      element_type=ElementType.OUTPUT,
                      value_type=ElementValueType.CATEGORY),
            ElementDB(task_id=task.task_id,
                      name=('output_2_' + name),
                      display_name=('Output 2 ' + task.name),
                      description='No description',
                      element_type=ElementType.OUTPUT,
                      value_type=ElementValueType.FLOAT),
            ElementDB(task_id=task.task_id,
                      name=('output_3_' + name),
                      display_name=('Output 3 ' + task.name),
                      description='No description',
                      element_type=ElementType.OUTPUT,
                      value_type=ElementValueType.INTEGER)
        ]
        metadata = [
            ElementDB(task_id=task.task_id,
                      name=('metadata_1_' + name),
                      display_name=('Metadata 1 ' + task.name),
                      description='No description',
                      element_type=ElementType.METADATA,
                      value_type=ElementValueType.CATEGORY),
            ElementDB(task_id=task.task_id,
                      name=('metadata_2_' + name),
                      display_name=('Metadata 2 ' + task.name),
                      description='No description',
                      element_type=ElementType.METADATA,
                      value_type=ElementValueType.BOOLEAN),
            ElementDB(task_id=task.task_id,
                      name=('metadata_3_' + name),
                      display_name=('Metadata 3 ' + task.name),
                      description='No description',
                      element_type=ElementType.METADATA,
                      value_type=ElementValueType.FLOAT)
        ]
        save_to_db(inputs + outputs + metadata)
        # Categories
        input_cats = [
            CategoryDB(element_id=inputs[1].element_id,
                       name=('input_category_1_' + name),
                       display_name=('Input Category 1 ' + task.name),
                       description='No description',
                       color='010000'),
            CategoryDB(element_id=inputs[1].element_id,
                       name=('input_category_2_' + name),
                       display_name=('Input Category 2 ' + task.name),
                       description='No description',
                       color='020000'),
            CategoryDB(element_id=inputs[1].element_id,
                       name=('input_category_3_' + name),
                       display_name=('Input Category 3 ' + task.name),
                       description='No description',
                       color='030000')
        ]
        output_cats = [
            CategoryDB(element_id=outputs[0].element_id,
                       name=('output_category_1_' + name),
                       display_name=('Output Category 1 ' + task.name),
                       description='No description',
                       color='011000'),
            CategoryDB(element_id=outputs[0].element_id,
                       name=('output_category_2_' + name),
                       display_name=('Output Category 2 ' + task.name),
                       description='No description',
                       color='022000'),
            CategoryDB(element_id=outputs[0].element_id,
                       name=('output_category_3_' + name),
                       display_name=('Output Category 3 ' + task.name),
                       description='No description',
                       color='033000')
        ]
        metadata_cats = [
            CategoryDB(element_id=metadata[0].element_id,
                       name=('metadata_category_1_' + name),
                       display_name=('Metadata Category 1 ' + task.name),
                       description='No description',
                       color='010100'),
            CategoryDB(element_id=metadata[0].element_id,
                       name=('metadata_category_2_' + name),
                       display_name=('Metadata Category 2 ' + task.name),
                       description='No description',
                       color='020200'),
            CategoryDB(element_id=metadata[0].element_id,
                       name=('metadata_category_3_' + name),
                       display_name=('Metadata Category 3 ' + task.name),
                       description='No description',
                       color='030300')
        ]
        save_to_db(input_cats + output_cats + metadata_cats)

    # Create files' database rows
    demo_files = []
    for (t_idx, task) in enumerate(_demo_tasks):
        for f_idx in range(1, 6):
            file = TaskFileDB(task_id=task.task_id,
                              filename=(f'input_{t_idx}_{f_idx}'),
                              use_for=TaskFileUse.INPUT,
                              size=1024 * f_idx)
            demo_files.append(file)
        for f_idx in range(1, 4):
            file = TaskFileDB(task_id=task.task_id,
                              filename=(f'ai_model_{t_idx}_{f_idx}'),
                              use_for=TaskFileUse.AI_MODEL,
                              size=2048 * f_idx)
            demo_files.append(file)
    save_to_db(demo_files)

    # Create examples
    for (t_idx, task) in enumerate(_demo_tasks):
        db.session.expire(task)
        examples = [
            ExampleDB(task_id=task.task_id,
                      labeling_status=LabelingStatus(idx % len(LabelingStatus)),
                      trained=(idx % 3 == 0),
                      **_default_creator) for _ in range(1, 11)
        ]
        save_to_db(examples)
        inputs = task.input_elements()
        outputs = task.output_elements()
        metadata = task.metadata_elements()
        input_files = TaskFileDB.query().filter_by(task_id=task.task_id, use_for=TaskFileUse.INPUT).all()
        for (ex_idx, example) in enumerate(examples):
            input_cat = CategoryDB.query().filter_by(element_id=inputs[1].element_id).all()[1]
            output_cat = CategoryDB.query().filter_by(element_id=outputs[0].element_id).all()[1]
            metadata_cat = CategoryDB.query().filter_by(element_id=metadata[0].element_id).all()[1]
            elem_values = [
                ExText(example_id=example.example_id,
                       element_id=inputs[0].element_id,
                       value=f'Value for (task {t_idx}, example {ex_idx})'),
                ExCategory(example_id=example.example_id, element_id=inputs[1].element_id, value=input_cat.category_id),
                ExFile(example_id=example.example_id,
                       element_id=inputs[2].element_id,
                       value=input_files[(ex_idx % 5)].file_id),
                ExCategory(example_id=example.example_id,
                           element_id=outputs[0].element_id,
                           value=output_cat.category_id),
                ExFloat(example_id=example.example_id, element_id=outputs[1].element_id, value=0.369 * ex_idx + t_idx),
                ExInteger(example_id=example.example_id, element_id=outputs[2].element_id, value=1 * ex_idx + t_idx),
                ExCategory(example_id=example.example_id,
                           element_id=metadata[0].element_id,
                           value=metadata_cat.category_id),
                ExBoolean(example_id=example.example_id, element_id=metadata[1].element_id, value=ex_idx % 2 == 0),
                ExFloat(example_id=example.example_id, element_id=metadata[2].element_id, value=0.411 * ex_idx + t_idx),
            ]
            save_to_db(elem_values)

    # Create shapes
    shapes = [
        ShapeDB(example_id=examples[-1].example_id,
                element_id=inputs[-1].element_id,
                polygon=[{
                    'x': 121,
                    'y': 53
                }, {
                    'x': 121,
                    'y': 532
                }, {
                    'x': 470,
                    'y': 53
                }, {
                    'x': 470,
                    'y': 532
                }],
                **_default_creator),
        ShapeDB(example_id=examples[-1].example_id,
                element_id=inputs[-1].element_id,
                pixels=[{
                    'x': 502,
                    'y': 5
                }, {
                    'x': 3,
                    'y': 69
                }, {
                    'x': 94,
                    'y': 108
                }],
                **_default_creator),
        ShapeDB(example_id=examples[-2].example_id,
                element_id=inputs[-1].element_id,
                pixels=[{
                    'x': 269,
                    'y': 40
                }, {
                    'x': 88,
                    'y': 6
                }, {
                    'x': 130,
                    'y': 99
                }],
                **_default_creator)
    ]
    save_to_db(shapes)

    shape_outputs = []
    for idx, shape in enumerate(shapes):
        if idx % 2 == 1:
            continue
        shape_output = ShapeCategory(shape_id=shape.shape_id,
                                     element_id=outputs[0].element_id,
                                     value=output_cat.category_id)
        shape_outputs.append(shape_output)
    save_to_db(shape_outputs)

    # TODO: create slices

    # Create AI models
    for (t_idx, task) in enumerate(_demo_tasks):
        ai_model_files = TaskFileDB.query().filter_by(task_id=task.task_id, use_for=TaskFileUse.AI_MODEL).all()
        ai_models = [
            AIModelDB(task_id=task.task_id,
                      file_id=ai_model_files[(m_idx % 3)].file_id,
                      version=f'0.{file_id}.{m_idx}',
                      task_schema={
                          'inputs': [],
                          'outputs': [],
                          'metadata': [],
                      },
                      training_time=1.5,
                      training_device=TrainingDevice.GPU,
                      extra_metadata={
                          'regularization': (idx % 2 == 0),
                          'architecture': 'resnet-34'
                      },
                      created_by_user=1) for m_idx in range(1, 4)
        ]
        save_to_db(ai_models)

    #############
    # WAIT LIST #
    #############
    empty_table(WaitList)

    entries = []
    for i in range(1, 11):
        entry = WaitList(uuid=uuid.uuid4(),
                         email=f'test_email_{i}@testorg.com',
                         first_name=f'First Name {i}',
                         last_name=f'Last Name {i}',
                         company=f'Company {i}')
        entries.append(entry)

    save_to_db(entries)


def restore_db():
    db_rollback()
    db.session.expire_all()
    populate_db()


def empty_db(app: Flask):
    """
    Deletes all database tables.
    """
    with app.app_context():
        # Disable foreign key constraint
        db.engine.execute(sql_text('SET FOREIGN_KEY_CHECKS=0'))
        # Drop all tables
        # TODO: Getting the following error when calling `db.drop_all()`:
        #       "sqlalchemy.exc.CircularDependencyError: Can't sort tables for DROP; an unresolvable foreign key
        #       dependency exists between tables: clients, org_files, organizations, task_files, tasks, users.  Please
        #       ensure that the ForeignKey and ForeignKeyConstraint objects involved in the cycle have names so that
        #       they can be dropped using DROP CONSTRAINT."
        # db.drop_all()
        # Re-enable foreign key constraint
        db.engine.execute(sql_text('SET FOREIGN_KEY_CHECKS=1'))
