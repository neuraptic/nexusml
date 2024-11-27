from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import UniqueConstraint
from sqlalchemy.dialects.mysql import MEDIUMINT

from nexusml.database.organizations import MutableEntity
from nexusml.database.tasks import TaskER


class TagDB(MutableEntity, TaskER):
    """ Tags defined in each task.

    Attributes:
        - tag_id (PK): surrogate key
        - task_id (FK): parent task's surrogate key
        - name: tag name. Unique in each task
        - description: tag description
        - color: hexadecimal color assigned to the category
    """
    __tablename__ = 'tags'
    __table_args__ = (UniqueConstraint('name', 'task_id'),)

    tag_id = Column(MEDIUMINT(unsigned=True), primary_key=True, autoincrement=True)
    name = Column(String(32), nullable=False)
    description = Column(String(256))
    color = Column(String(6))
