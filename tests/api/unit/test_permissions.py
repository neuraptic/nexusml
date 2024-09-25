"""
TODO: Mock database connection. Otherwise, these tests cannot be considered as unit tests.
"""

from typing import List, Type, Union
import uuid

import pytest
from sqlalchemy import bindparam

from nexusml.api.ext import cache
from nexusml.api.resources.ai import AIModel
from nexusml.api.resources.base import Permission
from nexusml.api.resources.base import Resource
from nexusml.api.resources.base import users_permissions
from nexusml.api.resources.files import TaskFile
from nexusml.api.resources.tasks import Task
from nexusml.constants import NULL_UUID
from nexusml.database.ai import AIModelDB
from nexusml.database.core import db_commit
from nexusml.database.core import db_execute
from nexusml.database.core import empty_table
from nexusml.database.core import save_to_db
from nexusml.database.files import TaskFileDB
from nexusml.database.organizations import OrganizationDB
from nexusml.database.organizations import RoleDB
from nexusml.database.organizations import user_roles
from nexusml.database.organizations import UserDB
from nexusml.database.permissions import RolePermission
from nexusml.database.permissions import UserPermission
from nexusml.database.tasks import TaskDB
from nexusml.enums import ResourceAction
from nexusml.enums import ResourceType
from tests.api.conftest import restore_db
from tests.api.utils import load_default_resource

pytestmark = [pytest.mark.unit, pytest.mark.slow]  # TODO: Mark as "fast" when the database connection is mocked

_ORG_ID = 2
_NUM_USERS = 5
_NUM_ROLES = 3


def _setup_db(mock_client_id: str, session_user_id: str, session_user_auth0_id: str):
    # Restore database
    restore_db(mock_client_id=mock_client_id,
               session_user_id=session_user_id,
               session_user_auth0_id=session_user_auth0_id)
    # Set users
    empty_table(UserDB)
    users = [
        UserDB(user_id=idx,
               uuid=str(uuid.uuid4()),
               auth0_id=f'auth0|{111111111111111111111111 + (111111111111111111111111 * idx)}',
               organization_id=_ORG_ID) for idx in range(1, _NUM_USERS + 1)
    ]
    save_to_db(users)

    # Set roles
    empty_table(RoleDB)
    roles = [RoleDB(role_id=idx, name=(f'Role {idx}'), organization_id=_ORG_ID) for idx in range(1, _NUM_ROLES + 1)]
    save_to_db(roles)

    # Set default task's organization
    for task in TaskDB.query().all():
        if task.task_id < 3:
            task.organization_id = _ORG_ID
            break
        save_to_db(task)


def _setup_user_perms() -> List[UserPermission]:
    """
    Prepare database entries for testing:

    User permissions.
    """
    cache.clear()

    empty_table(UserPermission)
    empty_table(RolePermission)

    user_perms = [
        # User 1
        UserPermission(user_id=1,
                       organization_id=_ORG_ID,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.READ,
                       allow=True),
        UserPermission(user_id=1,
                       organization_id=_ORG_ID,
                       resource_type=ResourceType.FILE,
                       action=ResourceAction.CREATE,
                       allow=True),
        UserPermission(user_id=1,
                       organization_id=_ORG_ID,
                       resource_type=ResourceType.AI_MODEL,
                       action=ResourceAction.DELETE,
                       allow=False),
        # User 3
        UserPermission(user_id=3,
                       organization_id=_ORG_ID,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.UPDATE,
                       allow=True),
        UserPermission(user_id=3,
                       organization_id=_ORG_ID,
                       resource_type=ResourceType.FILE,
                       action=ResourceAction.READ,
                       allow=False),
        UserPermission(user_id=3,
                       organization_id=_ORG_ID,
                       resource_type=ResourceType.AI_MODEL,
                       action=ResourceAction.READ,
                       allow=False),
        # User 4
        UserPermission(user_id=4,
                       organization_id=_ORG_ID,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.DELETE,
                       allow=False),
        UserPermission(user_id=4,
                       organization_id=_ORG_ID,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.UPDATE,
                       allow=True),
    ]

    save_to_db(user_perms)

    return user_perms


def _setup_role_perms() -> List[RolePermission]:
    """
    Prepare database entries for testing:

    Role permissions.
    """
    cache.clear()

    empty_table(UserPermission)
    empty_table(RolePermission)
    empty_table(user_roles)

    # Add permissions to roles
    role_perms = [
        # Role 1
        RolePermission(role_id=1, resource_type=ResourceType.TASK, action=ResourceAction.DELETE, allow=True),
        RolePermission(role_id=1, resource_type=ResourceType.AI_MODEL, action=ResourceAction.READ, allow=True),
        RolePermission(role_id=1, resource_type=ResourceType.FILE, action=ResourceAction.CREATE, allow=False),
        # Role 3
        RolePermission(role_id=3, resource_type=ResourceType.TASK, action=ResourceAction.READ, allow=False),
    ]
    save_to_db(role_perms)

    # Assign roles to users
    users_roles = {1: [3], 2: [2], 3: [1, 3], 4: [1]}
    user_role_insert = user_roles.insert().values(user_id=bindparam('user_id'), role_id=bindparam('role_id'))
    user_roles_rows = [{'user_id': u, 'role_id': role_id} for u, roles in users_roles.items() for role_id in roles]

    db_execute(user_role_insert, user_roles_rows)
    db_commit()

    return role_perms


def _setup_user_vs_role() -> List[Union[UserPermission, RolePermission]]:
    """
    Prepare database entries for testing:

    User vs. Role permissions.
    """
    cache.clear()

    empty_table(UserPermission)
    empty_table(RolePermission)
    empty_table(user_roles)

    user_perms = [
        # User 1
        UserPermission(user_id=1,
                       organization_id=_ORG_ID,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.READ,
                       allow=True),
        UserPermission(user_id=1,
                       organization_id=_ORG_ID,
                       resource_type=ResourceType.FILE,
                       action=ResourceAction.CREATE,
                       allow=True),
        UserPermission(user_id=1,
                       organization_id=_ORG_ID,
                       resource_type=ResourceType.AI_MODEL,
                       action=ResourceAction.DELETE,
                       allow=False),
        # User 4
        UserPermission(user_id=4,
                       organization_id=_ORG_ID,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.DELETE,
                       allow=False),
        UserPermission(user_id=4,
                       organization_id=_ORG_ID,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.UPDATE,
                       allow=True),
    ]

    save_to_db(user_perms)

    # Add permissions to roles
    role_perms = [
        # Role 1
        RolePermission(role_id=1, resource_type=ResourceType.TASK, action=ResourceAction.READ, allow=False),
        RolePermission(role_id=1, resource_type=ResourceType.FILE, action=ResourceAction.CREATE, allow=False),
        RolePermission(role_id=1, resource_type=ResourceType.AI_MODEL, action=ResourceAction.DELETE, allow=True),
        # Role 3
        RolePermission(role_id=3, resource_type=ResourceType.FILE, action=ResourceAction.CREATE, allow=True),
    ]
    save_to_db(role_perms)

    # Assign roles to users
    users_roles = {1: [1], 2: [2], 3: [1, 3]}
    user_role_insert = user_roles.insert().values(user_id=bindparam('user_id'), role_id=bindparam('role_id'))
    user_roles_rows = [{'user_id': u, 'role_id': role_id} for u, roles in users_roles.items() for role_id in roles]

    db_execute(user_role_insert, user_roles_rows)
    db_commit()

    return role_perms


def _setup_generic_vs_rl() -> List[Union[UserPermission, RolePermission]]:
    """
    Prepare database entries for testing:

    Generic vs. Resource-level permissions.
    """
    task = TaskDB.get(task_id=1)

    cache.clear()

    empty_table(UserPermission)
    empty_table(RolePermission)

    user_perms = [
        # User 1
        UserPermission(user_id=1,
                       organization_id=_ORG_ID,
                       resource_uuid=task.uuid,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.READ,
                       allow=False),
        UserPermission(user_id=1,
                       organization_id=_ORG_ID,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.READ,
                       allow=True),
        # User 4
        UserPermission(user_id=4,
                       organization_id=_ORG_ID,
                       resource_uuid=task.uuid,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.READ,
                       allow=True),
        UserPermission(user_id=4,
                       organization_id=_ORG_ID,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.READ,
                       allow=False),
    ]

    save_to_db(user_perms)

    return user_perms


def _setup_user_vs_role_plus_generic_vs_rl() -> List[Union[UserPermission, RolePermission]]:
    """
    Prepare database entries for testing:

    User vs. Role permissions + Generic vs. Resource-level permissions..
    """
    task = TaskDB.get(task_id=1)
    file = TaskFileDB.query().filter_by(task_id=1).first()
    ai_model = AIModelDB.query().filter_by(task_id=1).first()

    cache.clear()

    empty_table(UserPermission)
    empty_table(RolePermission)
    empty_table(user_roles)

    user_perms = [
        # User 1
        UserPermission(user_id=1,
                       organization_id=_ORG_ID,
                       resource_uuid=file.uuid,
                       resource_type=ResourceType.FILE,
                       action=ResourceAction.READ,
                       allow=True),
        UserPermission(user_id=1,
                       organization_id=_ORG_ID,
                       resource_type=ResourceType.AI_MODEL,
                       action=ResourceAction.READ,
                       allow=False),
        # User 4
        UserPermission(user_id=4,
                       organization_id=_ORG_ID,
                       resource_uuid=task.uuid,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.READ,
                       allow=False),
    ]

    save_to_db(user_perms)

    # Add permissions to roles
    role_perms = [
        # Role 1
        RolePermission(role_id=1, resource_type=ResourceType.FILE, action=ResourceAction.READ, allow=False),
        RolePermission(role_id=1,
                       resource_uuid=ai_model.uuid,
                       resource_type=ResourceType.AI_MODEL,
                       action=ResourceAction.READ,
                       allow=True),
        # Role 3
        RolePermission(role_id=3, resource_type=ResourceType.TASK, action=ResourceAction.READ, allow=True),
    ]
    save_to_db(role_perms)

    # Assign roles to users
    users_roles = {1: [1], 2: [2], 4: [1, 3]}
    user_role_insert = user_roles.insert().values(user_id=bindparam('user_id'), role_id=bindparam('role_id'))
    user_roles_rows = [{'user_id': u, 'role_id': role_id} for u, roles in users_roles.items() for role_id in roles]

    db_execute(user_role_insert, user_roles_rows)
    db_commit()

    return user_perms + role_perms


def test_filter_effective_permissions(mock_client_id: str, session_user_id: str, session_user_auth0_id: str):
    """ Tests for `resources.base.filter_effective_permissions()`. """
    _setup_db(mock_client_id=mock_client_id,
              session_user_id=session_user_id,
              session_user_auth0_id=session_user_auth0_id)
    pass  # TODO


def test_check_permissions(mock_client_id: str, session_user_id: str, session_user_auth0_id: str):
    """ Tests for `resources.base.Resource.check_permissions()`. """

    def _check_permissions(resource_or_type: Union[Resource, Type[Resource]], action: ResourceAction, user: UserDB,
                           should_allow: bool):
        try:
            resource = resource_or_type if isinstance(resource_or_type, Resource) else None
            resource_or_type.check_permissions(organization=org,
                                               action=action,
                                               resource=resource,
                                               user=user,
                                               check_parents=False)
            has_permissions = True
        except Exception:
            has_permissions = False
        assert has_permissions == should_allow

    _setup_db(mock_client_id=mock_client_id,
              session_user_id=session_user_id,
              session_user_auth0_id=session_user_auth0_id)

    # Load resources
    org = OrganizationDB.get(organization_id=_ORG_ID)

    user = UserDB.get(user_id=1)
    other_user = UserDB.get(user_id=4)

    file_db_obj = TaskFileDB.query().filter_by(task_id=1).first()
    ai_model_db_obj = AIModelDB.query().filter_by(task_id=1).first()

    task = load_default_resource(resource_type=Task, check_permissions=False)
    other_task = Task.get(agent=user, db_object_or_id=TaskDB.get(task_id=2), check_permissions=False)

    other_user_task = Task.get(agent=other_user, db_object_or_id=TaskDB.get(task_id=1), check_permissions=False)
    other_user_other_task = Task.get(agent=other_user, db_object_or_id=TaskDB.get(task_id=2), check_permissions=False)

    file = TaskFile.get(agent=user,
                        db_object_or_id=file_db_obj,
                        parents=[task],
                        check_parents=False,
                        check_permissions=False)

    ai_model = AIModel.get(agent=user,
                           db_object_or_id=ai_model_db_obj,
                           parents=[task],
                           check_parents=False,
                           check_permissions=False)
    ###################################
    # User permissions (no conflicts) #
    ###################################
    _setup_user_perms()

    _check_permissions(resource_or_type=Task, action=ResourceAction.READ, user=user, should_allow=True)
    _check_permissions(resource_or_type=TaskFile, action=ResourceAction.CREATE, user=user, should_allow=True)
    _check_permissions(resource_or_type=AIModel, action=ResourceAction.DELETE, user=user, should_allow=False)
    ##########################################################
    # Role permissions (inherited permissions, no conflicts) #
    ##########################################################
    _setup_role_perms()
    _check_permissions(resource_or_type=Task, action=ResourceAction.READ, user=user, should_allow=False)
    #############################
    # User vs. Role permissions #
    #############################
    _setup_user_vs_role()
    _check_permissions(resource_or_type=Task, action=ResourceAction.READ, user=user, should_allow=True)
    _check_permissions(resource_or_type=TaskFile, action=ResourceAction.CREATE, user=user, should_allow=True)
    _check_permissions(resource_or_type=AIModel, action=ResourceAction.DELETE, user=user, should_allow=False)
    ##########################################
    # Generic vs. Resource-level permissions #
    ##########################################
    _setup_generic_vs_rl()

    _check_permissions(resource_or_type=task, action=ResourceAction.READ, user=user, should_allow=False)
    _check_permissions(resource_or_type=other_task, action=ResourceAction.READ, user=user, should_allow=True)

    _check_permissions(resource_or_type=other_user_task, action=ResourceAction.READ, user=other_user, should_allow=True)
    _check_permissions(resource_or_type=other_user_other_task,
                       action=ResourceAction.READ,
                       user=other_user,
                       should_allow=False)
    ######################################################################
    # User vs. Role permissions + Generic vs. Resource-level permissions #
    ######################################################################
    _setup_user_vs_role_plus_generic_vs_rl()

    _check_permissions(resource_or_type=other_user_task,
                       action=ResourceAction.READ,
                       user=other_user,
                       should_allow=False)
    _check_permissions(resource_or_type=file, action=ResourceAction.READ, user=user, should_allow=True)
    _check_permissions(resource_or_type=ai_model, action=ResourceAction.READ, user=user, should_allow=False)


def test_users_permissions(mock_client_id: str, session_user_id: str, session_user_auth0_id: str):
    """ Tests for `resources.base.users_permissions()`. """

    _setup_db(mock_client_id=mock_client_id,
              session_user_id=session_user_id,
              session_user_auth0_id=session_user_auth0_id)

    org = OrganizationDB.get(organization_id=_ORG_ID)
    task = TaskDB.get(task_id=1)
    file = TaskFileDB.query().filter_by(task_id=1).first()
    ai_model = AIModelDB.query().filter_by(task_id=1).first()

    user_1 = UserDB.get(user_id=1)
    user_3 = UserDB.get(user_id=3)
    user_4 = UserDB.get(user_id=4)
    ###################################
    # User permissions (no conflicts) #
    ###################################
    _setup_user_perms()

    perms = {k: set(v) for k, v in users_permissions(organization=org).items()}

    assert perms == {
        user_1: {
            Permission(resource_uuid=NULL_UUID, resource_type=ResourceType.TASK, action=ResourceAction.READ,
                       allow=True),
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.FILE,
                       action=ResourceAction.CREATE,
                       allow=True),
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.AI_MODEL,
                       action=ResourceAction.DELETE,
                       allow=False),
        },
        user_3: {
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.UPDATE,
                       allow=True),
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.FILE,
                       action=ResourceAction.READ,
                       allow=False),
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.AI_MODEL,
                       action=ResourceAction.READ,
                       allow=False),
        },
        user_4: {
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.DELETE,
                       allow=False),
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.UPDATE,
                       allow=True),
        }
    }
    ##########################################################
    # Role permissions (inherited permissions, no conflicts) #
    ##########################################################
    _setup_role_perms()

    perms = {k: set(v) for k, v in users_permissions(organization=org).items()}

    assert perms == {
        user_1: {
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.READ,
                       allow=False)
        },
        user_3: {
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.DELETE,
                       allow=True),
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.AI_MODEL,
                       action=ResourceAction.READ,
                       allow=True),
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.FILE,
                       action=ResourceAction.CREATE,
                       allow=False),
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.READ,
                       allow=False)
        },
        user_4: {
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.DELETE,
                       allow=True),
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.AI_MODEL,
                       action=ResourceAction.READ,
                       allow=True),
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.FILE,
                       action=ResourceAction.CREATE,
                       allow=False)
        }
    }
    #############################
    # User vs. Role permissions #
    #############################
    _setup_user_vs_role()

    perms = {k: set(v) for k, v in users_permissions(organization=org).items()}

    assert perms == {
        user_1: {
            Permission(resource_uuid=NULL_UUID, resource_type=ResourceType.TASK, action=ResourceAction.READ,
                       allow=True),
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.FILE,
                       action=ResourceAction.CREATE,
                       allow=True),
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.AI_MODEL,
                       action=ResourceAction.DELETE,
                       allow=False)
        },
        user_3: {
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.READ,
                       allow=False),
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.FILE,
                       action=ResourceAction.CREATE,
                       allow=False),
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.AI_MODEL,
                       action=ResourceAction.DELETE,
                       allow=True),
        },
        user_4: {
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.DELETE,
                       allow=False),
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.UPDATE,
                       allow=True),
        }
    }
    ##########################################
    # Generic vs. Resource-level permissions #
    ##########################################
    _setup_generic_vs_rl()

    perms = {k: set(v) for k, v in users_permissions(organization=org).items()}

    assert perms == {
        user_1: {
            Permission(resource_uuid=task.uuid,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.READ,
                       allow=False),
            Permission(resource_uuid=NULL_UUID, resource_type=ResourceType.TASK, action=ResourceAction.READ,
                       allow=True),
        },
        user_4: {
            Permission(resource_uuid=task.uuid, resource_type=ResourceType.TASK, action=ResourceAction.READ,
                       allow=True),
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.READ,
                       allow=False),
        },
    }
    ######################################################################
    # User vs. Role permissions + Generic vs. Resource-level permissions #
    ######################################################################
    _setup_user_vs_role_plus_generic_vs_rl()

    perms = {k: set(v) for k, v in users_permissions(organization=org).items()}

    assert perms == {
        user_1: {
            Permission(resource_uuid=file.uuid, resource_type=ResourceType.FILE, action=ResourceAction.READ,
                       allow=True),
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.AI_MODEL,
                       action=ResourceAction.READ,
                       allow=False),
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.FILE,
                       action=ResourceAction.READ,
                       allow=False),
        },
        user_4: {
            Permission(resource_uuid=task.uuid,
                       resource_type=ResourceType.TASK,
                       action=ResourceAction.READ,
                       allow=False),
            Permission(resource_uuid=NULL_UUID,
                       resource_type=ResourceType.FILE,
                       action=ResourceAction.READ,
                       allow=False),
            Permission(resource_uuid=ai_model.uuid,
                       resource_type=ResourceType.AI_MODEL,
                       action=ResourceAction.READ,
                       allow=True),
            Permission(resource_uuid=NULL_UUID, resource_type=ResourceType.TASK, action=ResourceAction.READ,
                       allow=True),
        },
    }


def test_collaborators_permissions(mock_client_id: str, session_user_id: str, session_user_auth0_id: str):
    """ Tests for `resources.base.collaborators_permissions()`. """
    _setup_db(mock_client_id=mock_client_id,
              session_user_id=session_user_id,
              session_user_auth0_id=session_user_auth0_id)
    pass  # TODO
