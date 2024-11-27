# TODO: Try to make this module independent from `nexusml.api`
"""
IMPORTANT NOTES:

- All entities have a UUID for identifying the resource universally and
  a public ID for identifying the resource in the API.
- Defining foreign key constraints in abstract models prevent subclasses from defining composite foreign keys.
- To handle large collections (One-to-Many parent-child relationships):
  - Pass `lazy="dynamic"` to `relationship()` (or `backref()`) to return a `Query` object instead of a list of
    Python objects. This way, collection items can be filtered, paginated, etc. before loading in memory.
  - Pass `query_class=flask_sqlalchemy.BaseQuery` to `relationship()` (or `backref()`) to paginate query results.
    Using the `relationship()` and `backref()` wrapped by Flask-SQLAlchemy (`db.relationship()` and `db.backref()`)
    works as well.
  - See https://docs.sqlalchemy.org/en/14/orm/collections.html#working-with-large-collections
"""
import base64
import binascii
import random
import string
from typing import Dict, Iterable, List, Optional, Set, Tuple, TYPE_CHECKING, Union
import uuid

from sqlalchemy import BINARY
from sqlalchemy import Column
from sqlalchemy import event
from sqlalchemy import String
from sqlalchemy import TypeDecorator
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.exc import StatementError
from sqlalchemy.inspection import inspect
from sqlalchemy.orm.base import MANYTOMANY
from sqlalchemy.orm.base import MANYTOONE
from sqlalchemy.orm.base import ONETOMANY
from sqlalchemy.orm.dynamic import AppenderQuery

from nexusml.api.utils import config
from nexusml.constants import NULL_UUID
from nexusml.constants import UUID_VERSION
from nexusml.database.core import db
from nexusml.database.core import db_commit
from nexusml.database.core import db_execute
from nexusml.database.core import db_query
from nexusml.database.core import retry_on_deadlock
from nexusml.database.core import save_to_db
from nexusml.enums import DBRelationshipType

if TYPE_CHECKING:
    from sqlalchemy.orm import Query

__all__ = [
    'BinaryUUID',
    'DBModel',
    'Association',
    'Entity',
]

# Define digit <=> letter mapping for encoding/decoding entities' public IDs
_pub_id_letters = [x for x in string.ascii_lowercase + string.ascii_uppercase if x.lower() != 'o']
_pub_id_digit_letters = {str(digit): [] for digit in range(10)}
for idx, letter in enumerate(_pub_id_letters):
    _pub_id_digit_letters[str(idx % 10)].append(letter)


# Note: The following pylint warning is disable because:
# Method `process_literal_param` is abstract in class `TypeDecorator` but is not
# overridden in child class `BinaryUUID` (abstract-method)
# Method `python_type` is abstract in class `TypeEngine` but
# is not overridden in child class `BinaryUUID` (abstract-method)
# pylint: disable=abstract-method
class BinaryUUID(TypeDecorator):
    """
    This class handles the conversion of UUIDs to binary format for storage in the database.
    The UUID is stored as a 16-byte sequence and returned as a string.

    Inspired by:
        - https://gist.github.com/craigdmckenna/b52cb7207a413672c5576e554c527111
        - http://mysqlserverteam.com/storing-uuid-values-in-mysql-tables/

    Attributes:
        impl: The underlying SQLAlchemy type to use.
        cache_ok: Whether caching is enabled for this type.
        _null_uuid_bytes: The byte representation of the NULL_UUID.
    """

    impl = BINARY(16)

    cache_ok = True

    _null_uuid_bytes = uuid.UUID(NULL_UUID).bytes

    def process_bind_param(self, value: Union[str, uuid.UUID], dialect) -> Union[bytes, str]:
        """
        Convert the UUID value to its binary representation for storage in the database.

        Steps:
        1. If the value is NULL_UUID, return the byte representation of NULL_UUID.
        2. If the value has a bytes attribute, return the bytes.
        3. If the value can be converted to a UUID, return the bytes.
        4. Otherwise, return the value as-is.

        Args:
            value: The UUID value to be converted.
            dialect: The database dialect being used.

        Returns:
            Union[bytes, str]: The binary representation of the UUID, or the value as-is if conversion fails.
        """
        if value == NULL_UUID:
            return self._null_uuid_bytes
        try:
            return value.bytes
        except AttributeError:
            try:
                return uuid.UUID(value, version=UUID_VERSION).bytes
            except TypeError:
                return value

    def process_result_value(self, value, dialect) -> str:
        """
        Convert the binary representation of the UUID back to a string representation.

        Steps:
        1. If the value is None, return None.
        2. If the value is the byte representation of NULL_UUID, return NULL_UUID.
        3. Otherwise, return the string representation of the UUID.

        Args:
            value: The binary representation of the UUID.
            dialect: The database dialect being used.

        Returns:
            str: The string representation of the UUID.
        """
        if value is None:
            return
        if value == self._null_uuid_bytes:
            return NULL_UUID
        return str(uuid.UUID(bytes=value, version=UUID_VERSION))


# pylint: enable=abstract-method


class DBModel(db.Model):
    """
    This abstract base class represents a database entity/relationship.

    Methods:
        query: Returns a query object for the model.
        get: Retrieves an instance by primary key values.
        columns: Returns a set of all column names.
        primary_key_columns: Returns a set of primary key column names.
        foreign_keys_columns: Returns a set of foreign key column names.
        id_column: Returns the name of the unique identifying column.
        relationships: Returns a set of all relationship names.
        relationship_types: Returns a dictionary of relationship types and their corresponding relationships.
        is_parent: Checks if the current model is a parent of another model.
        force_relationship_loading: Forces the loading of relationships into memory.
        update_numeric_value: Updates a numeric value at the SQL level.
        to_dict: Converts the instance to a dictionary.
        duplicate: Duplicates the instance and its children recursively.
    """

    __abstract__ = True

    ####################################################################################################################
    # TODO: the commented `query()` function below leads to the following error in
    #       `sqlalchemy.orm.path_registry` (line 127) with SQLAlchemy 1.3.24:
    #
    #                   [path[i].key for i in range(1, len(path), 2)] + [None]
    #               AttributeError: 'str' object has no attribute 'key'
    ####################################################################################################################
    # @classmethod
    # def query(cls, load_relationships: Optional[bool] = None):
    #     """
    #     Setting `load_relationships` to `True` or `False` can force relationships (not) to be loaded regardless
    #     of the configured loading strategy. Useful for dynamic or sporadic situations.
    #     """
    #     if load_relationships is None:
    #         return db_query(cls)
    #     else:
    #         if load_relationships:
    #             return db_query(cls).options(Load(cls).selectinload('*'))  # TODO: leads to an error
    #         else:
    #             return db_query(cls).options(Load(cls).lazyload('*'))  # TODO: leads to an error
    ####################################################################################################################

    @classmethod
    def query(cls) -> 'Query':
        """
        Returns a query object for the model.

        Returns:
            Query: A query object for the model.
        """
        return db_query(cls)

    @classmethod
    def get(cls, **primary_key_values):
        """
        Retrieves an instance by primary key values.

        Args:
            primary_key_values: Dictionary of primary key values.

        Returns:
            DBModel: The instance matching the primary key values, or None if not found.
        """
        return cls.query().get(primary_key_values)

    @classmethod
    def columns(cls) -> Set[str]:
        """
        Returns a set of all column names.

        Returns:
            Set[str]: A set of all column names.
        """
        return set(cls.__table__.columns.keys())

    @classmethod
    def primary_key_columns(cls) -> Set[str]:
        """
        Returns a set of primary key column names.

        Returns:
            Set[str]: A set of primary key column names.
        """
        return {k.name for k in cls.__mapper__.primary_key}

    @classmethod
    def foreign_keys_columns(cls) -> Set[str]:
        """
        Returns a set of foreign key column names.

        Returns:
            Set[str]: A set of foreign key column names.
        """
        return {fk.column.name for fk in cls.__table__.foreign_keys}

    @classmethod
    def id_column(cls) -> Optional[str]:
        """
        Returns the column that allows identifying an object with its value
        (and optionally plus the value of the foreign keys). If the column
        is part of the primary key or any of the foreign keys, it is not
        returned.

        For example, an element belonging to a task schema can be identified based on its name
        plus the task ID (foreign key), since element names are unique in each task.

        If there is no such a column or there are multiple columns like this, `None` is returned.

        Steps:
        1. Retrieves unique columns that are not part of the primary key or foreign keys.
        2. Returns the column if there is exactly one unique column, otherwise returns None.

        Returns:
            Optional[str]: The unique identifying column, or None if not found.
        """
        unique_non_pk_cols = []
        for unique_cols in inspect(db.engine).get_unique_constraints(cls.__tablename__):
            unique_non_pk_cols += [
                x for x in unique_cols['column_names']
                if x not in cls.primary_key_columns().union(cls.foreign_keys_columns())
            ]
        return unique_non_pk_cols[0] if len(unique_non_pk_cols) == 1 else None

    @classmethod
    def relationships(cls) -> Set[str]:
        """
        Returns a set of all relationship names.

        Returns:
            Set[str]: A set of all relationship names.
        """
        return set(inspect(cls).relationships.keys())

    @classmethod
    def relationship_types(cls) -> Dict[DBRelationshipType, List[str]]:
        """
        Returns a dictionary of relationship types and their corresponding relationships.

        Steps:
        1. Initializes a dictionary with empty lists for each relationship type.
        2. Iterates over the relationships and classifies them into the appropriate types.

        Returns:
            Dict[DBRelationshipType, List[str]]: A dictionary mapping relationship types to lists of relationship names.
        """
        relationships = {x: [] for x in DBRelationshipType}
        for rel_name, rel_props in inspect(cls).relationships.items():
            rel_db_model = rel_props.mapper.class_
            # Association Object (Many-to-Many relationship with additional fields)
            if issubclass(rel_db_model, Association):
                assert rel_props.secondary is None
                relationships[DBRelationshipType.ASSOCIATION_OBJECT].append(rel_name)
            # Many-to-Many relationship
            elif rel_props.secondary is not None:
                assert rel_props.direction == MANYTOMANY
                relationships[DBRelationshipType.MANY_TO_MANY].append(rel_name)
            # Parent-Child relationship (One-to-Many/Many-to-One)
            else:
                # One-to-Many relationship (child side)
                if cls.is_parent(rel_db_model) and not rel_db_model.is_parent(cls):
                    assert rel_props.direction == ONETOMANY
                    relationships[DBRelationshipType.CHILD].append(rel_name)
                # Many-to-One relationship (parent side)
                else:
                    assert rel_props.direction == MANYTOONE
                    relationships[DBRelationshipType.PARENT].append(rel_name)
        return relationships

    @classmethod
    def is_parent(cls, child_db_model) -> bool:
        """
        Checks if the current model is a parent of another model.

        Steps:
        1. Retrieves the foreign keys of the child model that reference the current model's table.
        2. Checks if the primary keys of the current model match the foreign keys.

        Args:
            child_db_model: The child model to check.

        Returns:
            bool: True if the current model is a parent of the child model, False otherwise.
        """
        child_fks = [fk for fk in child_db_model.__table__.foreign_keys if fk.column.table == cls.__table__]
        return set(cls.__mapper__.primary_key) == set().union(*[fk.column.base_columns for fk in child_fks])

    def force_relationship_loading(self, max_depth: int = None):
        """
        Forces relationships to be loaded in memory. Useful when the instance has already been loaded
        and its relationships are configured to be lazily loaded.

        WARNING: try to avoid this function as much as possible. Instead, ensure the `lazy` argument is correctly
                 passed to `sqlalchemy.orm.relationship` to set the right default loading strategy or
                 use `query().options(Load(cls).selectinload('*'))` or `query().options(Load(cls).lazyload('*'))`
                 to dynamically change the relationship loading strategy before loading the instance in memory.
                 This function should only be used when the instance of this class has already been loaded while
                 its relationships have not.

        Args:
            max_depth: Maximum recursion depth.

        """

        def _force_relationship_loading(db_object: DBModel, depth: int, loaded_objects: list):
            if depth > (max_depth or float('inf')):
                return
            depth += 1
            for relationship in db_object.relationships():
                rel_attr = getattr(db_object, relationship)
                if isinstance(rel_attr, AppenderQuery):
                    continue  # don't force loading of relationships configured with `lazy="dynamic"`
                if isinstance(rel_attr, Iterable):
                    for obj in [x for x in rel_attr if x not in loaded_objects]:
                        loaded_objects.append(obj)
                        _force_relationship_loading(db_object=obj, depth=depth, loaded_objects=loaded_objects)
                elif rel_attr not in loaded_objects:
                    loaded_objects.append(rel_attr)
                    _force_relationship_loading(db_object=rel_attr, depth=depth, loaded_objects=loaded_objects)

        _force_relationship_loading(db_object=self, depth=1, loaded_objects=[])

    @retry_on_deadlock
    def update_numeric_value(self, column: str, delta: Union[int, float]):
        """
        Increases/Decreases a numeric value at SQL-level instead of Python-level to avoid race conditions.

        Steps:
        1. Retrieves the column attribute.
        2. Gets the primary key values of the instance.
        3. Updates the column value by the delta using SQL.
        4. Commits the transaction.
        5. Expires the session to refresh the instance.

        Args:
            column (str): The name of the column to update.
            delta (Union[int, float]): The amount to increase or decrease the column value by.
        """
        col_attr = getattr(type(self), column)
        pk = {pk_col: getattr(self, pk_col) for pk_col in self.primary_key_columns()}
        self.query().filter_by(**pk).update({column: col_attr + delta})
        db_commit()
        try:
            db.session.expire(self)
        except InvalidRequestError:
            pass  # the object may not be persistent within the Session

    def to_dict(self) -> dict:
        """
        Converts the instance to a dictionary.

        Steps:
        1. Retrieves the column names.
        2. Creates a dictionary with column names as keys and column values as values.

        Returns:
            dict: A dictionary representation of the instance.
        """
        return {col: getattr(self, col) for col in self.columns()}

    def duplicate(self, parent_pk: dict = None, fixed_values: dict = None) -> Tuple[object, dict]:
        """
        Duplicate the database object and its children recursively.

        Steps:
        1. Creates a new instance of the object's class.
        2. Copies the object's columns to the new instance.
        3. Sets the parent primary key values.
        4. Saves the new object to the database.
        5. Creates a mapping of new primary keys.
        6. Duplicates the object's children recursively and updates the primary key mapping.

        Args:
            parent_pk (dict): Parent's primary key.
            fixed_values (dict): Columns that will always take a certain value (in children as well).

        Returns:
            object: New copy of this object.
            dict: New primary keys (including children's). For example: {'task_id': {1: 106, 2: 107}} indicates the
                  copy of `task_id = 1` is `task_id = 106`.
        """
        # Create a new instance of the object's class
        new_obj = self.__class__()
        pk_maps = dict()

        # Copy the object's columns
        id_cols = {'uuid', 'public_id'}
        for column in self.columns() - id_cols - self.primary_key_columns() - self.foreign_keys_columns():
            if column in (fixed_values or {}):
                value = fixed_values[column]
            else:
                value = getattr(self, column)
            setattr(new_obj, column, value)
        for parent_col, parent_value in (parent_pk or dict()).items():
            setattr(new_obj, parent_col, parent_value)

        # Save the new object to database
        # TODO: how to avoid committing changes each time a new object is created while getting its primary key?
        save_to_db(new_obj)

        # New primary keys' mapping
        new_obj_pk = dict()
        for pk_col in self.primary_key_columns():
            full_pk_col_name = str(self.__class__.__name__) + '.' + pk_col
            new_obj_pk_col = getattr(new_obj, pk_col)
            new_obj_pk[pk_col] = new_obj_pk_col
            if pk_col not in pk_maps:
                pk_maps[full_pk_col_name] = dict()
            pk_maps[full_pk_col_name].update({getattr(self, pk_col): new_obj_pk_col})

        # Copy the object's children
        child_relationships = self.relationship_types()[DBRelationshipType.CHILD]
        for child_relationship in child_relationships:
            children = getattr(self, child_relationship)
            duplicated_children = [c.duplicate(parent_pk=new_obj_pk, fixed_values=fixed_values) for c in children]
            duplicated_objects = [x[0] for x in duplicated_children]
            setattr(new_obj, child_relationship, duplicated_objects)
            children_pk_maps = [x[1] for x in duplicated_children]
            for child_pk_maps in children_pk_maps:
                for col, child_pk_map in child_pk_maps.items():
                    if col not in pk_maps:
                        pk_maps[col] = dict()
                    pk_maps[col].update(child_pk_map)

        return new_obj, pk_maps


class Association(DBModel):
    """
    Association Object used in Many-to-Many relationships containing additional fields beyond those which are foreign
    keys to the left and right tables.

    More info at https://docs.sqlalchemy.org/en/14/orm/basic_relationships.html#association-object

    WARNING: flask-sqlalchemy docs say:

    "If you want to use Many-to-Many relationships you will need to define a helper table that is used for the
    relationship. For this helper table it is strongly recommended to NOT use a model but an actual table"

    More info at https://flask-sqlalchemy.palletsprojects.com/en/2.x/models/#Many-to-Many-relationships
    """
    __abstract__ = True
    pass  # add any column/function common to all associations


class Entity(DBModel):
    """
    Represents an entity with a UUID and a public ID.

    Attributes:
        uuid: Universal identifier of the resource.
        public_id: Public identifier of the resource in the API.

    Methods:
        get_from_uuid: Retrieves an instance by UUID.
        get_from_public_id: Retrieves an instance by public ID.
        get_from_id: Retrieves an instance by UUID, public ID, or unique column value.
        equivalent_public_ids: Checks if two public IDs are equivalent.
        _get_pk_value_from_public_id: Decodes the primary key value from a public ID.
    """
    __abstract__ = True

    # The maximum unsigned value that can be represented by MySQL's INT type is 4294967295 (10 digits).
    # To represent MySQL's BIGINT with public IDs, use more characters in the `public_id` column.
    # See https://dev.mysql.com/doc/refman/8.0/en/integer-types.html for more info.
    _PUB_ID_MAX_DIGITS = 10

    uuid = Column(BinaryUUID, nullable=False, default=uuid.uuid4)
    # Note: Base64 encoder produces a string of 4n/3 Base64 characters, where `n` is the length of the original string.
    public_id = Column(String(int(4 * _PUB_ID_MAX_DIGITS / 3)))  # padding ("=") is ignored

    @classmethod
    def get_from_uuid(cls, uuid: str) -> Optional[DBModel]:
        """
        Retrieves an instance by UUID.

        Args:
            uuid: The UUID of the instance to retrieve.

        Returns:
            Optional[DBModel]: The instance matching the UUID, or None if not found.
        """
        return cls.query().filter_by(uuid=uuid).first()

    @classmethod
    def get_from_public_id(cls, public_id: str) -> Optional[DBModel]:
        """
        Retrieves an instance by public ID. Public IDs only support single-integer primary keys.

        Steps:
        1. Asserts that the model has a single primary key column.
        2. Decodes the primary key value from the public ID.
        3. Retrieves the instance by the primary key value.

        Args:
            public_id: The public ID of the instance to retrieve.

        Returns:
            Optional[DBModel]: The instance matching the public ID, or None if not found.
        """
        assert len(cls.primary_key_columns()) == 1
        pk_col = list(cls.primary_key_columns())[0]
        pk_value = cls._get_pk_value_from_public_id(public_id=public_id)
        return cls.get(**{pk_col: pk_value})

    @classmethod
    def get_from_id(cls, id_value: str, parent=None) -> Optional[DBModel]:
        """
        Returns the database object identified by the provided UUID, public ID, or unique column value.

        Steps:
        1. Checks if the id_value is a UUID and retrieves the instance.
        2. If not, checks if the id_value is a public ID and retrieves the instance.
        3. If not, checks if the id_value is a unique column value and retrieves the instance.

        Args:
            id_value (str): UUID, public ID, or a unique column value.
            parent (Entity): Instance of this class. Required only for identifying an entity based on a value that is
            unique in the corresponding collection of the parent entity.

        Returns:
            Optional[DBModel]: The instance matching the id_value, or None if not found.
        """

        def _valid_parent(child: Entity) -> bool:
            if parent is None or child is None:
                return True
            return all(getattr(child, col) == getattr(parent, col) for col in parent.primary_key_columns())

        if id_value is None or not id_value.strip():
            return None

        if parent is not None:
            assert parent.is_parent(cls)

        # Case 1: `id_value` is a UUID
        try:
            instance = cls.get_from_uuid(uuid=id_value)
        except (ValueError, StatementError):
            # Case 2: `id_value` is a public ID
            try:
                instance = cls.get_from_public_id(public_id=id_value)
                assert instance.public_id == id_value
            except Exception:
                # Case 3: `id_value` is a unique string
                id_col = cls.id_column()
                if id_col is None:
                    return None
                if len(inspect(db.engine).get_unique_constraints(cls.__tablename__)[0]['column_names']) > 1:
                    # `id_value` itself may not identify the object (it may require also parent's primary key)
                    # Note: whenever `get_unique_constraints()` returns multiple unique constraints,
                    #       `id_column()` returns `None`. That's why we always access the first item.
                    assert parent is not None
                fks = {col: getattr(parent, col) for col in parent.primary_key_columns()} if parent is not None else {}
                instances = cls.query().filter_by(**{id_col: id_value, **fks}).all()
                assert len(instances) <= 1
                instance = instances[0] if instances else None

        return instance if _valid_parent(child=instance) else None

    @classmethod
    def equivalent_public_ids(cls, public_id_1: str, public_id_2: str) -> bool:
        """
        Checks if two public IDs are equivalent.

        Args:
            public_id_1: The first public ID.
            public_id_2: The second public ID.

        Returns:
            bool: True if the primary key values are equivalent, False otherwise.
        """
        return cls._get_pk_value_from_public_id(public_id_1) == cls._get_pk_value_from_public_id(public_id_2)

    @classmethod
    def _get_pk_value_from_public_id(cls, public_id: str) -> int:
        """
        Decodes the primary key value from a public ID.

        Steps:
        1. Decodes the base64-encoded public ID.
        2. Unobfuscates the primary key value.

        Args:
            public_id: The public ID to decode.

        Returns:
            int: The primary key value.
        """
        base64_str = public_id
        dec_id = None
        dec_errors = 0
        while dec_errors < 3:  # encoded string must be a multiple of three
            try:
                base64_bytes = base64_str.encode('ascii')
                dec_id_bytes = base64.b64decode(base64_bytes)
                dec_id = dec_id_bytes.decode('ascii')
                break
            except binascii.Error:
                base64_str += '='  # add padding
                dec_errors += 1
        if dec_id is None:
            raise ValueError('Invalid public ID')

        # Unobfuscate the obfuscated primary key value
        obf_pk = dec_id[:dec_id.index('-')] if '-' in dec_id else dec_id
        pk_digits = []
        for letter in obf_pk:
            for digit, letters in _pub_id_digit_letters.items():
                if letter in letters:
                    pk_digits.append(digit)
                    break

        return int(''.join(pk_digits))


@event.listens_for(db.session, 'pending_to_persistent')
def _generate_public_id(session, instance):
    """
    Generates a base64-encoded string containing an obfuscated version of the primary key value.

    The obfuscation consists in converting digits into letters. If there are fewer digits than a certain number `N`,
    remaining indices will be filled with random letters. To distinguish between meaningful and random letters, the "-"
    character is used as a separator. Meaningful letters are always on the left of "-".

    Since ASCII alphabet has 52 letters considering lower and upper cases, each digit can be uniquely represented by
    5 letters, leaving the "o" and "O" letters apart (i.e., 50 letters for representing 10 digits).

    Example: the ID "nVudkL-sPqlNtXeh" obfuscates the value "359306" (in substring "nVudkL"), considering the
             following digit<=>letters mapping:

             0: ['a', 'k', 'v', 'F', 'Q']
             1: ['b', 'l', 'w', 'G', 'R']
             2: ['c', 'm', 'x', 'H', 'S']
             3: ['d', 'n', 'y', 'I', 'T']
             4: ['e', 'p', 'z', 'J', 'U']
             5: ['f', 'q', 'A', 'K', 'V']
             6: ['g', 'r', 'B', 'L', 'W']
             7: ['h', 's', 'C', 'M', 'X']
             8: ['i', 't', 'D', 'N', 'Y']
             9: ['j', 'u', 'E', 'P', 'Z']

    WARNING: if the primary key doesn't consist of a single integer, the public ID will be `None`.

    NOTE: the pad characters ("=") of the base64-encoded string are omitted.

    Steps:
    1. If the instance is not a subclass of Entity or already has a public ID, return.
    2. If the primary key is not a single integer, return.
    3. Obfuscate the primary key value.
    4. Encode the obfuscated value to Base64 and set the public ID.
    5. Update the database row with the new public ID.

    Args:
        session: The SQLAlchemy session.
        instance: The instance being persisted.
    """

    def _random_letters(n: int) -> str:
        if n < 1:
            return ''
        return ''.join(random.choices(_pub_id_letters, k=n))

    model = type(instance)
    if not issubclass(model, Entity):
        return

    if instance.public_id is not None:
        return

    table = model.__table__

    # If the primary key doesn't consist of a single integer, the public ID is not set
    pk_cols = model.primary_key_columns()
    if len(pk_cols) > 1:
        return
    pk_col = list(pk_cols)[0]
    pk_value = getattr(instance, pk_col)
    if not isinstance(pk_value, int):
        return

    # Obfuscate primary key value
    min_len = config.get('security')['public_id']['min_length']
    obf_pk = ''.join(random.choice(_pub_id_digit_letters[x]) for x in str(pk_value))
    obf_pk_len = len(obf_pk)
    if obf_pk_len < min_len:
        obf_pk += '-' + _random_letters(min_len - obf_pk_len - 1)

    # Encode to Base64 and set public ID
    public_id = base64.b64encode(obf_pk.encode('ascii')).decode('ascii')
    while public_id[-1] == '=':  # remove pad chars
        public_id = public_id[:-1]

    instance.public_id = public_id

    # Update database row
    update_stmt = (table.update().where(getattr(table.c, pk_col) == getattr(instance, pk_col)).values(
        public_id=public_id))
    db_execute(update_stmt)
    db_commit()
