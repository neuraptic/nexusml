from datetime import datetime
from enum import Enum
from typing import Iterable, Set, Type, Union
import uuid

import boto3

from nexusml.api.ext import cache
from nexusml.api.resources.base import Resource
from nexusml.api.resources.organizations import Organization
from nexusml.api.resources.organizations import Role
from nexusml.api.resources.organizations import User
from nexusml.api.resources.tasks import Task
from nexusml.api.utils import get_s3_config
from nexusml.constants import DATETIME_FORMAT
from nexusml.constants import PREFIX_ORG_PICTURES
from nexusml.constants import PREFIX_ORGANIZATIONS
from nexusml.constants import PREFIX_TASK_INPUTS
from nexusml.constants import PREFIX_TASK_METADATA
from nexusml.constants import PREFIX_TASK_MODELS
from nexusml.constants import PREFIX_TASK_OUTPUTS
from nexusml.constants import PREFIX_TASK_PICTURES
from nexusml.constants import PREFIX_TASKS
from nexusml.database.ai import PredictionDB
from nexusml.database.base import Entity
from nexusml.database.core import db
from nexusml.database.core import db_commit
from nexusml.database.core import db_rollback
from nexusml.database.core import save_to_db
from nexusml.database.examples import ExampleDB
from nexusml.database.examples import ShapeDB
from nexusml.database.examples import SliceDB
from nexusml.database.organizations import ClientDB
from nexusml.database.organizations import OrganizationDB
from nexusml.database.organizations import UserDB
from nexusml.database.subscriptions import get_active_subscription
from nexusml.database.subscriptions import quotas
from nexusml.database.tasks import TaskDB


def db_commit_and_expire():
    db_commit()
    db.session.expire_all()


def db_rollback_and_expire():
    db_rollback()
    db.session.expire_all()


def load_default_resource(resource_type: Type[Resource],
                          parents_types: Iterable[Type[Resource]] = None,
                          check_permissions: bool = True) -> Resource:
    """ Loads the first database object (pk_col=1) of the provided resource model and its parents' models. """

    def _load_resource(resource_type: Type[Resource], parents: Iterable[Resource]) -> Resource:
        if resource_type == Organization:
            db_object = Organization.db_model().get(organization_id=2)  # 1 is reserved for the main organization
        else:
            pk = {col: 1 for col in resource_type.db_model().primary_key_columns()}
            db_object = resource_type.db_model().get(**pk)
        return resource_type.get(agent=user,
                                 db_object_or_id=db_object,
                                 parents=list(parents),
                                 cache=False,
                                 check_permissions=check_permissions,
                                 check_parents=check_permissions)

    user = UserDB.get(user_id=1)

    parents = []
    for parent_type in parents_types or []:
        parents.append(_load_resource(resource_type=parent_type, parents=parents))

    return _load_resource(resource_type=resource_type, parents=parents)


def get_json_from_db_object(db_object: Entity,
                            reference_related_entities=False,
                            ignore_columns: Iterable = None) -> dict:
    ignore_columns = ['synced_by_users', 'synced_by_clients'] + list(ignore_columns or [])
    json_ = dict()
    for column in db_object.columns() - db_object.primary_key_columns() - db_object.foreign_keys_columns():
        if column in ignore_columns:
            continue
        json_[column] = getattr(db_object, column)
        # Convert internal user/client ID to UUID + Public ID
        if column.startswith(('created_by', 'modified_by')):
            agent_id = json_.pop(column)
            if agent_id is None:
                continue
            agent = UserDB.get(user_id=agent_id) if 'user' in column else ClientDB.get(client_id=agent_id)
            column = 'created_by' if column.startswith('created_by') else 'modified_by'
            json_[column] = {
                'id': agent.public_id,
                'uuid': agent.uuid,
                'type': 'user' if isinstance(agent, UserDB) else 'client'
            }
        # Value transformations
        if isinstance(json_[column], uuid.UUID):
            json_[column] = str(json_[column])
        elif isinstance(json_[column], datetime):
            json_[column] = json_[column].strftime(DATETIME_FORMAT) + 'Z'
        elif isinstance(json_[column], Enum):
            json_[column] = json_[column].name.lower()
        # Fields renaming
        if column == 'public_id':
            json_['id'] = json_.pop(column)
        if column.endswith('_'):
            json_[column[:-1]] = json_.pop(column)
    if reference_related_entities:
        for relationship in db_object.relationships():
            rel = getattr(db_object, relationship)
            if isinstance(rel, Entity) and rel is not None:
                json_[relationship] = getattr(rel, rel.id_column() or 'public_id')
            elif isinstance(rel, Iterable) and rel and all(isinstance(x, Entity) for x in rel):
                json_[relationship] = [getattr(x, x.id_column() or 'public_id') for x in rel]
    return json_


def get_shape_or_slice_json_from_db_object(db_object: Union[ShapeDB, SliceDB]) -> dict:
    json_ = get_json_from_db_object(db_object=db_object, reference_related_entities=True)
    json_.pop('example')
    return json_


def get_user(user_id: int) -> User:
    session_user = UserDB.get(user_id=1)
    user = UserDB.get(user_id=user_id)
    org = Organization.get(agent=session_user, db_object_or_id=OrganizationDB.get(organization_id=user.organization_id))
    return User.get(agent=session_user, db_object_or_id=user, parents=[org])


def get_role(organization_id: int, name: str) -> Role:
    session_user = UserDB.get(user_id=1)
    organization_db_object = OrganizationDB.get(organization_id=organization_id)
    organization = Organization.get(agent=session_user, db_object_or_id=organization_db_object)
    return Role.get(agent=session_user, db_object_or_id=name, parents=[organization])


def set_quota_usage(db_object: Union[OrganizationDB, TaskDB], quota: str, usage: int):
    db_commit_and_expire()
    subscription = get_active_subscription(organization_id=db_object.organization_id)
    setattr(subscription, quotas[quota]['usage'], usage)
    save_to_db(subscription)
    if isinstance(db_object, TaskDB):
        setattr(db_object, quotas[quota]['usage'], usage)
        save_to_db(db_object)


def set_quota_limit(db_object: Union[OrganizationDB, TaskDB], quota: str, limit: int):
    subscription = get_active_subscription(organization_id=db_object.organization_id)
    setattr(subscription.plan, quotas[quota]['limit'], limit)
    save_to_db(subscription.plan)
    if isinstance(db_object, TaskDB):
        setattr(db_object, quotas[quota]['limit'], limit)
        save_to_db(db_object)


def verify_quota_usage(db_object: Union[OrganizationDB, TaskDB], quota: str, expected_usage: int):
    cache.clear()
    db_commit_and_expire()
    subscription = get_active_subscription(organization_id=db_object.organization_id)
    assert getattr(subscription, quotas[quota]['usage']) == expected_usage
    if isinstance(db_object, TaskDB):
        assert getattr(db_object, quotas[quota]['usage']) == expected_usage


def get_files_in_s3(parent_resource: Union[Organization, Task]) -> Set[str]:
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(get_s3_config()['bucket'])
    if isinstance(parent_resource, Task):
        root_prefix = PREFIX_TASKS
        collection_prefixes = [
            PREFIX_TASK_MODELS, PREFIX_TASK_INPUTS, PREFIX_TASK_OUTPUTS, PREFIX_TASK_METADATA, PREFIX_TASK_PICTURES
        ]
    else:
        root_prefix = PREFIX_ORGANIZATIONS
        collection_prefixes = [PREFIX_ORG_PICTURES]
    files = []
    for collection_prefix in collection_prefixes:
        prefix = root_prefix + parent_resource.uuid() + '/' + collection_prefix
        files += [o.key.replace(prefix, '') for o in bucket.objects.filter(Prefix=prefix, Delimiter='/')]
    return set(files)


def assert_same_element_values(db_object_1: Union[ExampleDB, PredictionDB], db_object_2: Union[ExampleDB,
                                                                                               PredictionDB]):

    assert isinstance(db_object_1, type(db_object_2))

    db_model = type(db_object_1)

    db_commit_and_expire()

    for column in db_model.columns():
        assert getattr(db_object_1, column) == getattr(db_object_2, column)

    for r in db_model.relationships():
        r_object_1 = getattr(db_object_1, r)
        r_object_2 = getattr(db_object_2, r)
        if isinstance(r_object_1, list):
            assert len(r_object_1) == len(r_object_2)
            for r_object_1_item, r_object_2_item in zip(r_object_1, r_object_2):
                assert all(
                    getattr(r_object_1_item, c) == getattr(r_object_2_item, c) for c in r_object_1_item.columns())
        else:
            assert all(getattr(r_object_1, c) == getattr(r_object_2, c) for c in r_object_1.columns())
