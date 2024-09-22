from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import Enum
from sqlalchemy import ForeignKey
from sqlalchemy import UniqueConstraint
from sqlalchemy.dialects.mysql import INTEGER
from sqlalchemy.dialects.mysql import MEDIUMINT

from nexusml.constants import NULL_UUID
from nexusml.database.base import BinaryUUID
from nexusml.database.base import DBModel
from nexusml.database.core import db
from nexusml.database.organizations import OrganizationER
from nexusml.database.organizations import RoleDB
from nexusml.database.organizations import UserDB
from nexusml.enums import ResourceAction
from nexusml.enums import ResourceType


class _Permission(DBModel):
    """
    Permissions assigned to a user or a role.

    Attributes:
        resource_uuid (BinaryUUID): UUID of the resource (only for resource-level permissions).
                                    WARNING: use `NULL_UUID` instead of `NULL`, since `NULL` is not a value in MySQL
                                    and cannot be compared. As a result, any uniqueness constraint referencing a
                                    nullable column won't work.
        resource_type (Enum): Type of resource.
        action (Enum): Action type ("create", "read", "update", "delete").
        allow (Boolean): Boolean value indicating whether the action is allowed.
    """
    __abstract__ = True

    resource_uuid = Column(BinaryUUID, index=True, nullable=False, default=NULL_UUID)
    resource_type = Column(Enum(ResourceType), nullable=False)
    action = Column(Enum(ResourceAction), nullable=False)
    allow = Column(Boolean, nullable=False)


class UserPermission(_Permission, OrganizationER):
    """
    User permissions.

    Attributes:
        id_ (INTEGER): Primary key, surrogate key.
        user_id (INTEGER): Foreign key to user's surrogate key.
        organization_id (INTEGER): Foreign key to organization's surrogate key.
    """
    __tablename__ = 'user_permissions'
    __table_args__ = (UniqueConstraint('user_id', 'organization_id', 'resource_uuid', 'resource_type', 'action'),)

    id_ = Column(INTEGER(unsigned=True), primary_key=True, autoincrement=True)
    user_id = Column(INTEGER(unsigned=True), ForeignKey(UserDB.user_id, ondelete='CASCADE'), nullable=False)


class RolePermission(_Permission):
    """
    Role permissions.

    Attributes:
        id_ (MEDIUMINT): Primary key, surrogate key.
        role_id (MEDIUMINT): Foreign key to role's surrogate key.
    """
    __tablename__ = 'role_permissions'
    __table_args__ = (UniqueConstraint('role_id', 'resource_uuid', 'resource_type', 'action'),)

    id_ = Column(MEDIUMINT(unsigned=True), primary_key=True, autoincrement=True)
    role_id = Column(MEDIUMINT(unsigned=True), ForeignKey(RoleDB.role_id, ondelete='CASCADE'), nullable=False)


################################################################################
# Set permission relationships in users and roles                              #
# (due to circular dependencies, this cannot be done inside class definitions) #
################################################################################
UserDB.permissions = db.relationship('UserPermission', backref='user', cascade='all, delete-orphan', lazy='dynamic')
RoleDB.permissions = db.relationship('RolePermission', backref='role', cascade='all, delete-orphan', lazy='dynamic')
