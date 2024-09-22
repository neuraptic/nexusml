from datetime import datetime
from typing import Dict

from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import DECIMAL
from sqlalchemy import Enum
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import Table
from sqlalchemy import Text
from sqlalchemy.dialects.mysql import INTEGER
from sqlalchemy.dialects.mysql import MEDIUMINT
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship
from sqlalchemy.orm import RelationshipProperty
from sqlalchemy.sql.type_api import TypeEngine

from nexusml.database.base import Association
from nexusml.database.base import DBModel
from nexusml.database.core import db
from nexusml.database.core import db_commit
from nexusml.database.files import TaskFileDB as FileDB
from nexusml.database.organizations import ImmutableEntity
from nexusml.database.organizations import MutableEntity
from nexusml.database.tags import TagDB
from nexusml.database.tasks import CategoryDB
from nexusml.database.tasks import ElementDB
from nexusml.database.tasks import TaskDB
from nexusml.database.tasks import TaskER
from nexusml.enums import ElementValueType
from nexusml.enums import LabelingStatus


def _ex_value_relationship(db_model_name: str) -> RelationshipProperty:
    # - Info about which relationship loading strategy to use:
    #   https://docs.sqlalchemy.org/en/13/orm/loading_relationships.html#what-kind-of-loading-to-use
    # - Info about "Select IN eager loading":
    #   https://docs.sqlalchemy.org/en/13/orm/loading_relationships.html#select-in-loading
    # Note: don't pass `backref="example"` because `ExampleER` defines a parent (many-to-one) relationship
    return relationship(db_model_name, cascade='all, delete-orphan', lazy='selectin')


class ExampleDB(MutableEntity, TaskER):
    """ Represents an example used for AI training.

    Attributes:
        - example_id (PK): surrogate key
        - task_id (FK): parent task's surrogate key
        - size: total size of the example in bytes, considering element values, tags, comments, and shapes
        - labeling_status: labeling status ("unlabeled", "pending_review", "labeled", "rejected")
        - trained: indicates whether the AI used the example's current version for training
        - activity_at: last activity datetime
    """
    __tablename__ = 'examples'

    example_id = Column(INTEGER(unsigned=True), primary_key=True, autoincrement=True)
    size = Column(INTEGER(unsigned=True))
    labeling_status = Column(Enum(LabelingStatus), nullable=False, default=LabelingStatus.UNLABELED)
    trained = Column(Boolean, nullable=False, default=False)
    activity_at = Column(DateTime, nullable=False, default=datetime.utcnow)  # TODO: add index for faster queries?

    # Children (One-to-Many relationships)
    # Note: don't pass `backref="example"` because `ExampleER` defines a parent (many-to-one) relationship
    comments = relationship('CommentDB', cascade='all, delete-orphan')
    shapes = relationship('ShapeDB', cascade='all, delete-orphan', lazy='selectin')  # all shapes in the example
    slices = relationship('SliceDB', cascade='all, delete-orphan', lazy='selectin')  # all slices in the example

    # Many-to-Many relationships
    tags = relationship('TagDB', secondary='ex_tags')

    # Association Objects
    ex_booleans = _ex_value_relationship('ExBoolean')
    ex_integers = _ex_value_relationship('ExInteger')
    ex_floats = _ex_value_relationship('ExFloat')
    ex_texts = _ex_value_relationship('ExText')
    ex_datetimes = _ex_value_relationship('ExDatetime')
    ex_categories = _ex_value_relationship('ExCategory')
    ex_files = _ex_value_relationship('ExFile')
    ex_shapes = _ex_value_relationship('ExShape')  # 'element-shape' assignments
    ex_slices = _ex_value_relationship('ExSlice')  # 'element-slice' assignments

    def values(self) -> list:
        values_ = []
        for collection in self.value_collections():
            values_ += getattr(self, collection)
        return values_

    @staticmethod
    def value_collections() -> list:
        return [
            'ex_booleans', 'ex_integers', 'ex_floats', 'ex_texts', 'ex_datetimes', 'ex_categories', 'ex_files',
            'ex_shapes', 'ex_slices'
        ]

    @staticmethod
    def value_type_models() -> Dict[ElementValueType, Association]:
        return {
            ElementValueType.BOOLEAN: ExBoolean,
            ElementValueType.INTEGER: ExInteger,
            ElementValueType.FLOAT: ExFloat,
            ElementValueType.TEXT: ExText,
            ElementValueType.DATETIME: ExDatetime,
            ElementValueType.CATEGORY: ExCategory,
            ElementValueType.DOCUMENT_FILE: ExFile,
            ElementValueType.IMAGE_FILE: ExFile,
            ElementValueType.VIDEO_FILE: ExFile,
            ElementValueType.AUDIO_FILE: ExFile,
            ElementValueType.SHAPE: ExShape,
            ElementValueType.SLICE: ExSlice
        }

    @staticmethod
    def value_type_collections() -> Dict[ElementValueType, str]:
        return {
            ElementValueType.BOOLEAN: 'ex_booleans',
            ElementValueType.INTEGER: 'ex_integers',
            ElementValueType.FLOAT: 'ex_floats',
            ElementValueType.TEXT: 'ex_texts',
            ElementValueType.DATETIME: 'ex_datetimes',
            ElementValueType.CATEGORY: 'ex_categories',
            ElementValueType.DOCUMENT_FILE: 'ex_files',
            ElementValueType.IMAGE_FILE: 'ex_files',
            ElementValueType.VIDEO_FILE: 'ex_files',
            ElementValueType.AUDIO_FILE: 'ex_files',
            ElementValueType.SHAPE: 'ex_shapes',
            ElementValueType.SLICE: 'ex_slices'
        }

    @classmethod
    def set_examples_training_flag(cls, task: TaskDB, trained: bool):
        cls.query().filter_by(task_id=task.task_id).update({ExampleDB.trained: trained})
        db_commit()


class ExampleER(DBModel):
    """ Represents an entity or an association of an example.

    Attributes:
        - example_id (PK, FK): example's surrogate key
    """
    __abstract__ = True

    @declared_attr
    def example_id(cls):
        return Column(INTEGER(unsigned=True),
                      ForeignKey(ExampleDB.example_id, ondelete='CASCADE'),
                      primary_key=True,
                      nullable=False)

    @declared_attr
    def example(cls):
        return relationship('ExampleDB')


class ExValue(Association, ExampleER):
    """ Value assigned to example.
    Attributes:
        - example_id (PK, FK): surrogate key of the example to which the value refers
        - element_id (PK, FK): surrogate key of the element to which the value refers
        - index (PK): 1-based index of the value (only used for multi-value assignments)
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
        return Column(MEDIUMINT(unsigned=True), primary_key=True, default=1)

    @declared_attr
    def value(cls):
        return Column(TypeEngine)

    @declared_attr
    def element(cls):
        return relationship('ElementDB')


class ExBoolean(ExValue):
    __tablename__ = 'ex_booleans'

    value = Column(Boolean)


class ExInteger(ExValue):
    __tablename__ = 'ex_integers'

    value = Column(Integer)


class ExFloat(ExValue):
    __tablename__ = 'ex_floats'

    # MySQL docs recommend using the DECIMAL type in place of FLOAT or DOUBLE:
    # https://dev.mysql.com/doc/refman/8.0/en/problems-with-float.html
    # https://dev.mysql.com/doc/refman/8.0/en/precision-math-decimal-characteristics.html
    value = Column(DECIMAL(precision=13, scale=4, asdecimal=False))


class ExText(ExValue):
    __tablename__ = 'ex_texts'

    value = Column(Text)


class ExDatetime(ExValue):
    __tablename__ = 'ex_datetimes'

    value = Column(DateTime)


class ExCategory(ExValue):
    __tablename__ = 'ex_categories'

    value = Column(MEDIUMINT(unsigned=True), ForeignKey(CategoryDB.category_id, ondelete='CASCADE'))

    # Parents (Many-to-One relationships)
    category = relationship('CategoryDB')


class ExFile(ExValue):
    __tablename__ = 'ex_files'

    value = Column(INTEGER(unsigned=True), ForeignKey(FileDB.file_id, ondelete='CASCADE'))

    # Parents (Many-to-One relationships)
    file = relationship('TaskFileDB')


class ExShape(ExValue):
    __tablename__ = 'ex_shapes'

    value = Column(INTEGER(unsigned=True), ForeignKey('shapes.shape_id', ondelete='CASCADE'))

    # Parents (Many-to-One relationships)
    shape = relationship('ShapeDB')


class ExSlice(ExValue):
    __tablename__ = 'ex_slices'

    value = Column(INTEGER(unsigned=True), ForeignKey('slices.slice_id', ondelete='CASCADE'))

    # Parents (Many-to-One relationships)
    slice = relationship('SliceDB')


ex_tags = Table('ex_tags',
                db.Model.metadata,
                Column('example_id',
                       INTEGER(unsigned=True),
                       ForeignKey(ExampleDB.example_id, ondelete='CASCADE'),
                       primary_key=True,
                       nullable=False),
                Column('tag_id',
                       MEDIUMINT(unsigned=True),
                       ForeignKey(TagDB.tag_id, ondelete='CASCADE'),
                       primary_key=True,
                       nullable=False),
                mysql_engine='InnoDB')


class CommentDB(ImmutableEntity, ExampleER):
    """
    Attributes:
        - comment_id (PK): surrogate key
        - example_id (FK): surrogate key of the example on which the comment was made
        - message: message
    """
    __tablename__ = 'ex_comments'

    comment_id = Column(INTEGER(unsigned=True), primary_key=True, autoincrement=True)
    message = Column(Text, nullable=False)

    @declared_attr
    def example_id(cls):
        # Override method to NOT include `example_id` in the primary key
        return Column(INTEGER(unsigned=True), ForeignKey(ExampleDB.example_id, ondelete='CASCADE'), nullable=False)


class ShapeDB(MutableEntity, ExampleER):
    """
    Attributes:
        - shape_id (PK): surrogate key
        - element_id (FK): surrogate key of the element (file) in which the shape is drawn
        - example_id (FK): example's surrogate key
        - polygon: JSON containing a list of vertices defined by `x` and `y` fields (coordinates).
                   Used for geometric shapes described by a finite number of connected straight line segments
        - path: string defining an SVG <path> element.
                Used for free shapes
        - pixels: JSON containing a list of pixels defined by `x` and `y` fields (coordinates).
                  Used for segmentation
    """
    __tablename__ = 'shapes'

    shape_id = Column(INTEGER(unsigned=True), primary_key=True, autoincrement=True)
    element_id = Column(INTEGER(unsigned=True), ForeignKey(ElementDB.element_id, ondelete='CASCADE'), nullable=False)

    polygon = Column(JSON(none_as_null=True))
    path = Column(Text)
    pixels = Column(JSON(none_as_null=True))

    # Parents (Many-to-One relationships)
    element = relationship('ElementDB')

    # Children (One-to-Many relationships)
    # Note: don't pass `backref="shape"` because `ShapeValue` defines a parent (many-to-one) relationship
    shape_floats = relationship('ShapeFloat', cascade='all, delete-orphan')
    shape_categories = relationship('ShapeCategory', cascade='all, delete-orphan')

    @declared_attr
    def example_id(cls):
        # Override method to NOT include `example_id` in the primary key
        return Column(INTEGER(unsigned=True), ForeignKey(ExampleDB.example_id, ondelete='CASCADE'), nullable=False)

    def values(self) -> list:
        values_ = []
        for collection in self.value_collections():
            values_ += getattr(self, collection)
        return values_

    @staticmethod
    def value_collections() -> list:
        return ['shape_floats', 'shape_categories']

    @staticmethod
    def value_type_models() -> Dict[ElementValueType, Association]:
        return {ElementValueType.FLOAT: ShapeFloat, ElementValueType.CATEGORY: ShapeCategory}

    @staticmethod
    def value_type_collections() -> Dict[ElementValueType, str]:
        return {ElementValueType.FLOAT: 'shape_floats', ElementValueType.CATEGORY: 'shape_categories'}


class ShapeValue(Association):
    """ Value assigned to shape.
    Attributes:
        - shape_id (PK, FK): shape's surrogate key
        - element_id (PK, FK): surrogate key of the element to which the value refers
        - index (PK): 1-based index of the value (only used for multi-value assignments)
        - value: assigned value
    """
    __abstract__ = True

    @declared_attr
    def shape_id(cls):
        return Column(INTEGER(unsigned=True),
                      ForeignKey(ShapeDB.shape_id, ondelete='CASCADE'),
                      primary_key=True,
                      nullable=False)

    @declared_attr
    def element_id(cls):
        return Column(INTEGER(unsigned=True),
                      ForeignKey(ElementDB.element_id, ondelete='CASCADE'),
                      primary_key=True,
                      nullable=False)

    @declared_attr
    def index(cls):
        return Column(MEDIUMINT(unsigned=True), primary_key=True, default=1)

    @declared_attr
    def value(cls):
        return Column(TypeEngine)

    @declared_attr
    def shape(cls):
        return relationship('ShapeDB')

    @declared_attr
    def element(cls):
        return relationship('ElementDB')


class ShapeFloat(ShapeValue):
    __tablename__ = 'shape_floats'

    # MySQL docs recommend using the DECIMAL type in place of FLOAT or DOUBLE:
    # https://dev.mysql.com/doc/refman/8.0/en/problems-with-float.html
    # https://dev.mysql.com/doc/refman/8.0/en/precision-math-decimal-characteristics.html
    value = Column(DECIMAL(precision=13, scale=4, asdecimal=False))


class ShapeCategory(ShapeValue):
    __tablename__ = 'shape_categories'

    value = Column(MEDIUMINT(unsigned=True), ForeignKey(CategoryDB.category_id, ondelete='CASCADE'))

    # Parents (Many-to-One relationships)
    category = relationship('CategoryDB')


class SliceDB(MutableEntity, ExampleER):
    """
    Attributes:
        - slice_id (PK): surrogate key
        - element_id (FK): surrogate key of the element (file) to which the slice refers
        - example_id (FK): example's surrogate key
        - start_index: start index
        - end_index: end index
    """
    __tablename__ = 'slices'

    slice_id = Column(INTEGER(unsigned=True), primary_key=True, autoincrement=True)
    element_id = Column(INTEGER(unsigned=True), ForeignKey(ElementDB.element_id, ondelete='CASCADE'), nullable=False)

    start_index = Column(MEDIUMINT(unsigned=True), nullable=False)
    end_index = Column(MEDIUMINT(unsigned=True), nullable=False)

    # Parents (Many-to-One relationships)
    element = relationship('ElementDB')

    # Children (One-to-Many relationships)
    # Note: don't pass `backref="slice"` because `SliceValue` defines a parent (many-to-one) relationship
    slice_floats = relationship('SliceFloat', cascade='all, delete-orphan')
    slice_categories = relationship('SliceCategory', cascade='all, delete-orphan')

    @declared_attr
    def example_id(cls):
        # Override method to NOT include `example_id` in the primary key
        return Column(INTEGER(unsigned=True), ForeignKey(ExampleDB.example_id, ondelete='CASCADE'), nullable=False)

    def values(self) -> list:
        values_ = []
        for collection in self.value_collections():
            values_ += getattr(self, collection)
        return values_

    @staticmethod
    def value_collections() -> list:
        return ['slice_floats', 'slice_categories']

    @staticmethod
    def value_type_models() -> Dict[ElementValueType, Association]:
        return {ElementValueType.FLOAT: SliceFloat, ElementValueType.CATEGORY: SliceCategory}

    @staticmethod
    def value_type_collections() -> Dict[ElementValueType, str]:
        return {ElementValueType.FLOAT: 'slice_floats', ElementValueType.CATEGORY: 'slice_categories'}


class SliceValue(Association):
    """ Value assigned to slice.
    Attributes:
        - slice_id (PK, FK): slice's surrogate key
        - element_id (PK, FK): surrogate key of the element to which the value refers
        - index (PK): 1-based index of the value (only used for multi-value assignments)
        - value: assigned value
    """
    __abstract__ = True

    @declared_attr
    def slice_id(cls):
        return Column(INTEGER(unsigned=True),
                      ForeignKey(SliceDB.slice_id, ondelete='CASCADE'),
                      primary_key=True,
                      nullable=False)

    @declared_attr
    def element_id(cls):
        return Column(INTEGER(unsigned=True),
                      ForeignKey(ElementDB.element_id, ondelete='CASCADE'),
                      primary_key=True,
                      nullable=False)

    @declared_attr
    def index(cls):
        return Column(MEDIUMINT(unsigned=True), primary_key=True, default=1)

    @declared_attr
    def value(cls):
        return Column(TypeEngine)

    @declared_attr
    def slice(cls):
        return relationship('SliceDB')

    @declared_attr
    def element(cls):
        return relationship('ElementDB')


class SliceFloat(SliceValue):
    __tablename__ = 'slice_floats'

    # MySQL docs recommend using the DECIMAL type in place of FLOAT or DOUBLE:
    # https://dev.mysql.com/doc/refman/8.0/en/problems-with-float.html
    # https://dev.mysql.com/doc/refman/8.0/en/precision-math-decimal-characteristics.html
    value = Column(DECIMAL(precision=13, scale=4, asdecimal=False))


class SliceCategory(SliceValue):
    __tablename__ = 'slice_categories'

    value = Column(MEDIUMINT(unsigned=True), ForeignKey(CategoryDB.category_id, ondelete='CASCADE'))

    # Parents (Many-to-One relationships)
    category = relationship('CategoryDB')
