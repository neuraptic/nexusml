from typing import Dict

from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import DECIMAL
from sqlalchemy import Enum
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import UniqueConstraint
from sqlalchemy.dialects.mysql import INTEGER
from sqlalchemy.dialects.mysql import MEDIUMINT
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import declared_attr
from sqlalchemy.orm import relationship
from sqlalchemy.orm import RelationshipProperty
from sqlalchemy.sql.type_api import TypeEngine

from nexusml.database.base import Association
from nexusml.database.base import DBModel
from nexusml.database.files import TaskFileDB as FileDB
from nexusml.database.organizations import ImmutableEntity
from nexusml.database.tasks import CategoryDB
from nexusml.database.tasks import ElementDB
from nexusml.database.tasks import TaskER
from nexusml.enums import AIEnvironment
from nexusml.enums import ElementValueType
from nexusml.enums import PredictionState
from nexusml.enums import TrainingDevice


class AIModelDB(ImmutableEntity, TaskER):
    """
    Attributes:
        - model_id (PK): surrogate key
        - task_id (FK): parent task's surrogate key
        - file_id (FK): surrogate key of the file containing the model.
        - version: version of the AI model in X.Y.Z format
        - task_schema: task schema for which the model is built
        - training_time: time needed to train the model (in hours)
        - training_device: device used for training the model
        - extra_metadata: model-specific extra metadata
    """
    __tablename__ = 'ai_models'

    __table_args__ = (
        UniqueConstraint('file_id', 'task_id'),
        UniqueConstraint('version', 'task_id')  # Ensure version is unique for the same task
    )

    model_id = Column(INTEGER(unsigned=True), primary_key=True, autoincrement=True)
    file_id = Column(INTEGER(unsigned=True), ForeignKey(FileDB.file_id, ondelete='CASCADE'), nullable=False)
    version = Column(String(16), nullable=False)
    task_schema = Column(JSON(none_as_null=True), nullable=False)
    training_time = Column(DECIMAL(precision=7, scale=2, asdecimal=False), nullable=False)
    training_device = Column(Enum(TrainingDevice), nullable=False)
    extra_metadata = Column(JSON(none_as_null=True))

    # Parents (Many-to-One relationships)
    file = relationship('TaskFileDB')


def _pred_value_relationship(db_model_name: str) -> RelationshipProperty:
    # - Info about which relationship loading strategy to use:
    #   https://docs.sqlalchemy.org/en/13/orm/loading_relationships.html#what-kind-of-loading-to-use
    # - Info about "Select IN eager loading":
    #   https://docs.sqlalchemy.org/en/13/orm/loading_relationships.html#select-in-loading
    # Note: don't pass `backref="prediction"` because `PredictionER` defines a parent (many-to-one) relationship
    return relationship(db_model_name, cascade='all, delete-orphan', lazy='selectin')


class PredictionDB(ImmutableEntity, TaskER):
    """ Represents a prediction made by an AI model.

    Attributes:
        - prediction_id (PK): surrogate key
        - model_id (FK): AI model which made the prediction
        - task_id (FK): parent task's surrogate key
        - environment: "production" if the prediction was made in production
                       or "testing" if the prediction was made in testing environment.
                       Note: we don't index this column because of its low selectivity (only 2 possible values).
        - state: "pending", "in_progress", or "complete"
        - size: total size of the prediction in bytes
        - invalid_data: if the prediction couldn't be saved due to integrity errors, this field will save all the data
        - removed_elements: JSON containing values predicted for elements removed from task schema
    """
    __tablename__ = 'predictions'

    prediction_id = Column(INTEGER(unsigned=True), primary_key=True, autoincrement=True)
    model_id = Column(INTEGER(unsigned=True), ForeignKey(AIModelDB.model_id, ondelete='SET NULL'))
    environment = Column(Enum(AIEnvironment), nullable=False)
    state = Column(Enum(PredictionState), nullable=False)
    size = Column(INTEGER(unsigned=True))
    invalid_data = Column(JSON(none_as_null=True))
    removed_elements = Column(MutableList.as_mutable(JSON), nullable=False, default=list)

    # Parents (Many-to-One relationships)
    ai_model = relationship('AIModelDB')

    # Association Objects
    pred_booleans = _pred_value_relationship('PredBoolean')
    pred_integers = _pred_value_relationship('PredInteger')
    pred_floats = _pred_value_relationship('PredFloat')
    pred_texts = _pred_value_relationship('PredText')
    pred_datetimes = _pred_value_relationship('PredDatetime')
    pred_categories = _pred_value_relationship('PredCategory')
    pred_scores = _pred_value_relationship('PredScores')
    pred_files = _pred_value_relationship('PredFile')
    pred_shapes = _pred_value_relationship('PredShape')
    pred_slices = _pred_value_relationship('PredSlice')

    def values(self) -> list:
        values_ = []
        for collection in self.value_collections():
            values_ += getattr(self, collection)
        return values_

    @staticmethod
    def value_collections() -> list:
        return [
            'pred_booleans', 'pred_integers', 'pred_floats', 'pred_texts', 'pred_datetimes', 'pred_categories',
            'pred_scores', 'pred_files', 'pred_shapes', 'pred_slices'
        ]

    @staticmethod
    def value_type_models() -> Dict[ElementValueType, Association]:
        return {
            ElementValueType.BOOLEAN: PredBoolean,
            ElementValueType.INTEGER: PredInteger,
            ElementValueType.FLOAT: PredFloat,
            ElementValueType.TEXT: PredText,
            ElementValueType.DATETIME: PredDatetime,
            ElementValueType.CATEGORY: PredCategory,  # Watch out: Keep in mind `PredScores`
            ElementValueType.DOCUMENT_FILE: PredFile,
            ElementValueType.IMAGE_FILE: PredFile,
            ElementValueType.VIDEO_FILE: PredFile,
            ElementValueType.AUDIO_FILE: PredFile,
            ElementValueType.SHAPE: PredShape,
            ElementValueType.SLICE: PredSlice
        }

    @staticmethod
    def value_type_collections() -> Dict[ElementValueType, str]:
        return {
            ElementValueType.BOOLEAN: 'pred_booleans',
            ElementValueType.INTEGER: 'pred_integers',
            ElementValueType.FLOAT: 'pred_floats',
            ElementValueType.TEXT: 'pred_texts',
            ElementValueType.DATETIME: 'pred_datetimes',
            ElementValueType.CATEGORY: 'pred_categories',  # Watch out: Keep in mind 'pred_scores'
            ElementValueType.DOCUMENT_FILE: 'pred_files',
            ElementValueType.IMAGE_FILE: 'pred_files',
            ElementValueType.VIDEO_FILE: 'pred_files',
            ElementValueType.AUDIO_FILE: 'pred_files',
            ElementValueType.SHAPE: 'pred_shapes',
            ElementValueType.SLICE: 'pred_slices'
        }


class PredictionER(DBModel):
    """ Represents an entity or an association of a prediction.

    Attributes:
        - prediction_id (PK, FK): prediction's surrogate key
    """
    __abstract__ = True

    @declared_attr
    def prediction_id(cls):
        return Column(INTEGER(unsigned=True),
                      ForeignKey(PredictionDB.prediction_id, ondelete='CASCADE'),
                      primary_key=True,
                      nullable=False)

    @declared_attr
    def prediction(cls):
        return relationship('PredictionDB')


class PredValue(Association, PredictionER):
    """
    Value for a given element.

    Attributes:
        - prediction_id (PK, FK): surrogate key of the prediction to which the value refers
        - element_id (PK, FK): surrogate key of the element to which the value refers
        - index (PK): 1-based index of the value (only used for multi-value assignments)
        - is_target (PK): boolean flag indicating whether the value is a target value
                          used as a reference for stats and metrics computed in tests.
        - value: assigned value
    """
    __abstract__ = True

    @declared_attr
    def element_id(cls):
        return Column(INTEGER(unsigned=True),
                      ForeignKey(ElementDB.element_id, ondelete='CASCADE'),
                      primary_key=True,
                      nullable=False)

    @declared_attr
    def index(cls):
        return Column(MEDIUMINT(unsigned=True), primary_key=True, nullable=False, default=1)

    @declared_attr
    def is_target(cls):
        # TODO: consider replacing this column with a "target" column of the same type of "value" column.
        #       In this alternative approach, the predicted value and the target value will be given in the same row.
        #       Pro of this approach: the primary key index will be smaller (faster writes).
        #       Con of this approach: fixed-length columns use the same space for NULL values (takes more space).
        return Column(Boolean, primary_key=True, nullable=False, default=False)

    @declared_attr
    def value(cls):
        return Column(TypeEngine)

    @declared_attr
    def element(cls):
        return relationship('ElementDB')


class PredBoolean(PredValue):
    __tablename__ = 'pred_booleans'

    value = Column(Boolean)


class PredInteger(PredValue):
    __tablename__ = 'pred_integers'

    value = Column(Integer)


class PredFloat(PredValue):
    __tablename__ = 'pred_floats'

    # MySQL docs recommend using the DECIMAL type in place of FLOAT or DOUBLE:
    # https://dev.mysql.com/doc/refman/8.0/en/problems-with-float.html
    # https://dev.mysql.com/doc/refman/8.0/en/precision-math-decimal-characteristics.html
    value = Column(DECIMAL(precision=13, scale=4, asdecimal=False))


class PredText(PredValue):
    __tablename__ = 'pred_texts'

    value = Column(Text)


class PredDatetime(PredValue):
    __tablename__ = 'pred_datetimes'

    value = Column(DateTime)


class PredCategory(PredValue):
    __tablename__ = 'pred_categories'

    value = Column(MEDIUMINT(unsigned=True), ForeignKey(CategoryDB.category_id, ondelete='CASCADE'))

    # Parents (Many-to-One relationships)
    category = relationship('CategoryDB')


class PredScores(PredValue):
    """
    Category scores.

    Format:

        ```
        {
            "category": "<predicted_category>",
            "scores": {
                "<class_1>": <score_1>,
                "<class_2>": <score_2>,
                ...
                "<class_N>": <score_N>,
            }
        }
        ```
    """
    __tablename__ = 'pred_scores'

    value = Column(JSON)


class PredFile(PredValue):
    __tablename__ = 'pred_files'

    value = Column(INTEGER(unsigned=True), ForeignKey(FileDB.file_id, ondelete='CASCADE'))

    # Parents (Many-to-One relationships)
    file = relationship('TaskFileDB')


class PredShape(PredValue):
    __tablename__ = 'pred_shapes'

    value = Column(JSON)


class PredSlice(PredValue):
    __tablename__ = 'pred_slices'

    value = Column(JSON)
