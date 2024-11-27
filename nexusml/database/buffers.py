from sqlalchemy import case
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import DECIMAL
from sqlalchemy import ForeignKey
from sqlalchemy import func
from sqlalchemy import Integer
from sqlalchemy.dialects.mysql import INTEGER
from sqlalchemy.dialects.mysql import MEDIUMINT
from sqlalchemy.orm import declared_attr

from nexusml.database.ai import PredictionDB
from nexusml.database.base import DBModel
from nexusml.database.tasks import TaskDB


def _default_buffer_item_relevance(context):
    return context.get_current_parameters()['timestamp'].timestamp()


class BufferItemDB(DBModel):
    """
    Abstract base class representing buffer items stored in the database.

    Buffer items typically represent data points associated with a prediction, and they are
    characterized by properties like size, relevance, and timestamp. This class is designed to be extended
    by specific buffer types (e.g., for active learning or monitoring). It provides a general structure for
    handling items related to predictions, along with a method to retrieve and sort these items by relevance.

    Attributes:
        id_ (int): The primary key identifying the buffer item.
        size (int): The size of the buffer item.
        relevance (float): The relevance of the item, indexed for performance and sorted in descending order.
        timestamp (DateTime): The timestamp when the item was created, indexed for fast lookups.
        task_id (int): Foreign key to the task this buffer item is associated with.
        prediction_id (int): Foreign key to the prediction this buffer item is associated with.
    """

    __abstract__ = True

    id_ = Column(Integer, primary_key=True)
    size = Column(Integer, nullable=False)
    relevance = Column(DECIMAL(precision=14, scale=2, asdecimal=False),
                       index=True,
                       nullable=False,
                       default=_default_buffer_item_relevance)
    timestamp = Column(DateTime, index=True, nullable=False, server_default=func.now())

    @declared_attr
    def task_id(self):
        """
        Returns a column for storing the task ID associated with the buffer item.

        This is a foreign key referencing the `TaskDB` and is marked for deletion on cascade. The foreign key can be
        removed to improve insertion speed if needed.

        Returns:
            Column: The task ID column.
        """
        # consider removing FK to speed up insertion
        return Column(MEDIUMINT(unsigned=True), ForeignKey(TaskDB.task_id, ondelete='CASCADE'), nullable=False)

    @declared_attr
    def prediction_id(self):
        """
        Returns a column for storing the prediction ID associated with the buffer item.

        This is a foreign key referencing the `PredictionDB` and is marked for deletion on cascade.

        Returns:
            Column: The prediction ID column.
        """
        return Column(INTEGER(unsigned=True),
                      ForeignKey(PredictionDB.prediction_id, ondelete='CASCADE'),
                      nullable=False)

    @classmethod
    def get_items(cls, task_id: int) -> list:
        """
        Retrieves buffer items associated with a specific task, sorted by relevance in descending order.

        Items are first sorted based on whether the relevance value is `None` (with `None` values pushed last)
        and then in descending order of relevance.

        Args:
            task_id (int): The ID of the task for which buffer items are to be retrieved.

        Returns:
            list: A list of buffer items sorted by relevance.
        """
        items = cls.query().filter_by(task_id=task_id).order_by(case((cls.relevance.is_(None), 1), else_=0),
                                                                cls.relevance.desc()).all()
        return items


class ALBufferItemDB(BufferItemDB):
    """
    Represents buffer items for Active Learning (AL) Service.

    This class extends `BufferItemDB` and defines the table for storing buffer items specific to
    the Active Learning Service. It inherits all the columns and functionality from the parent class.
    """
    __tablename__ = 'al_buffer'


class MonBufferItemDB(BufferItemDB):
    """
    Represents buffer items for Monitoring Service.

    This class extends `BufferItemDB` and defines the table for storing buffer items specific to
    the Monitoring Service. It inherits all the columns and functionality from the parent class.
    """
    __tablename__ = 'mon_buffer'
