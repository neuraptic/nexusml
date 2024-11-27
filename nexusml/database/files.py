from datetime import datetime

from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Enum
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.dialects.mysql import BIGINT
from sqlalchemy.dialects.mysql import INTEGER
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship

from nexusml.database.base import DBModel
from nexusml.database.organizations import ClientDB
from nexusml.database.organizations import ImmutableEntity
from nexusml.database.organizations import OrganizationDB
from nexusml.database.organizations import OrganizationER
from nexusml.database.tasks import TaskDB
from nexusml.database.tasks import TaskER
from nexusml.enums import FileFormat
from nexusml.enums import FileType
from nexusml.enums import OrgFileUse
from nexusml.enums import TaskFileUse


class FileDB(ImmutableEntity):
    """
    Base class for files, representing various attributes and types of files.

    Attributes:
        file_id (PK): surrogate key
        filename: name of the file. Prefixes can be used to represent directories
        size: file size in bytes
        type_: file type
        format_: file format (usually corresponds to filename extension)
    """
    __abstract__ = True

    file_id = Column(INTEGER(unsigned=True), primary_key=True, autoincrement=True)
    filename = Column(String(128), nullable=False)
    size = Column(BIGINT(unsigned=True), nullable=False)
    type_ = Column(Enum(FileType))  # TODO: Consider using `Column('type', Enum(Type))` to keep "type" at SQL level
    format_ = Column(Enum(FileFormat))  # "format" is a Python reserved word


class OrgFileDB(OrganizationER, FileDB):
    """
    Class representing organization-specific files, inheriting from FileDB.

    Attributes:
        organization_id (FK): parent organization's surrogate key
        use_for: for the moment, "picture" is the only use
    """
    __tablename__ = 'org_files'

    use_for = Column(Enum(OrgFileUse), nullable=False)

    @declared_attr
    def organization(cls):
        """
        Establishes a relationship to the OrganizationDB. This method is overridden to avoid ambiguity errors
        when determining join conditions between parent and child tables.
        If we don't override this attribute, we get:
            "sqlalchemy.exc.AmbiguousForeignKeysError: Could not determine join condition between parent/child tables
            on relationship OrgFileDB.organization - there are multiple foreign key paths linking the tables.
            Specify the 'foreign_keys' argument, providing a list of those columns which should be counted as
            containing a foreign key reference to the parent table."
        """
        return relationship('OrganizationDB', foreign_keys='OrgFileDB.organization_id')


# We set the organization logo here due to the circular dependency
OrganizationDB.logo = Column(INTEGER(unsigned=True), ForeignKey(OrgFileDB.file_id, ondelete='SET NULL'))
# We set the client (app) icon here due to the circular dependency
ClientDB.icon = Column(INTEGER(unsigned=True), ForeignKey(OrgFileDB.file_id, ondelete='SET NULL'))


class TaskFileDB(TaskER, FileDB):
    """
    Class representing task-specific files, inheriting from FileDB.

    Attributes:
        task_id (FK): parent task's surrogate key
        use_for: "ai_model", "input", "output", "metadata", "picture"
    """
    __tablename__ = 'task_files'

    use_for = Column(Enum(TaskFileUse), nullable=False)

    @declared_attr
    def task(cls):
        """
        Establishes a relationship to the TaskDB. This method is overridden to avoid ambiguity errors
        when determining join conditions between parent and child tables.
        If we don't override this attribute, we get:
            "sqlalchemy.exc.AmbiguousForeignKeysError: Could not determine join condition between parent/child tables
            on relationship TaskFileDB.task - there are multiple foreign key paths linking the tables. Specify the
            'foreign_keys' argument, providing a list of those columns which should be counted as containing a foreign
            key reference to the parent table."
        """
        return relationship('TaskDB', foreign_keys='TaskFileDB.task_id')


# We set the task icon here due to the circular dependency
TaskDB.icon = Column(INTEGER(unsigned=True), ForeignKey(TaskFileDB.file_id, ondelete='SET NULL'))


class _Upload(DBModel):
    """
    Abstract base class for uploads, capturing common attributes.

    Attributes:
        upload_id: upload identifier
        init_at: initialization datetime
    """
    __abstract__ = True

    upload_id = Column(String(256), primary_key=True, nullable=False)
    init_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class OrgUpload(_Upload):
    """
    Class representing organization-specific uploads, inheriting from _Upload.

    Attributes:
        file_id (PK, FK): file's surrogate key
    """
    __tablename__ = 'org_uploads'

    file_id = Column(INTEGER(unsigned=True),
                     ForeignKey(OrgFileDB.file_id, ondelete='CASCADE'),
                     unique=True,
                     nullable=False)


class TaskUpload(_Upload):
    """
    Class representing task-specific uploads, inheriting from _Upload.

    Attributes:
        file_id (PK, FK): file's surrogate key
    """
    __tablename__ = 'task_uploads'

    file_id = Column(INTEGER(unsigned=True),
                     ForeignKey(TaskFileDB.file_id, ondelete='CASCADE'),
                     unique=True,
                     nullable=False)
