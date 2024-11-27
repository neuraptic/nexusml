# TODO: Try to make this module independent from `nexusml.api`

from typing import Iterable, List, Type, Union

from sqlalchemy import text as sql_text
from sqlalchemy.engine import Row
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql.elements import TextClause

from nexusml.api.utils import camel_to_snake
from nexusml.database.base import DBModel
from nexusml.database.base import Entity
from nexusml.database.core import db_rollback
from nexusml.database.core import save_to_db
from nexusml.statuses import Status
from nexusml.statuses import status_group_prefixes


def save_or_ignore_duplicate(db_object: DBModel) -> bool:
    """
    Saves a database object to the database, ignoring any IntegrityError exceptions
    that may be raised due to duplicate entries.

    Args:
        db_object (DBModel): The database object to save.

    Returns:
        bool: True if the object was saved successfully, False if it was discarded due to an IntegrityError.
    """
    try:
        save_to_db(db_object)
        return True
    except IntegrityError as e:
        db_rollback()
        if e.orig.args[0] != 1062:
            raise e
        return False


def get_children(parent_object: DBModel, child_model: Type[DBModel]) -> List[Type[DBModel]]:
    """
    Returns all the children of the parent database object based on the relationship defined
    by foreign key columns.

    Steps:
    1. Identify relationship keys by intersecting foreign keys of the child model and primary keys of the parent object.
    2. Create a dictionary of foreign values by mapping relationship keys to their corresponding values in the parent
       object.
    3. Query the child model using the foreign values to get all children related to the parent object.

    Args:
        parent_object (DBModel): The parent database object.
        child_model (Type[DBModel]): The type of the child database model.

    Returns:
        List[Type[DBModel]]: A list of children related to the parent object.
    """
    relationship_keys = child_model.foreign_keys_columns().intersection(parent_object.primary_key_columns())
    foreign_values = dict(map(lambda k: (k, getattr(parent_object, k)), relationship_keys))
    return child_model.query().filter_by(**foreign_values).all()


def to_dict(db_objects: Union[DBModel, Row, Iterable[Union[DBModel, Row]]],
            preferential_model: Type[DBModel] = None) -> Union[dict, List[dict]]:
    """
    Converts database objects or query results into dictionaries, handling conflicts by prefixing
    field names with the model name when necessary.

    Steps:
    1. Define a helper function to convert query results to dictionaries.
    2. Handle conflicts by identifying columns with the same name and prefixing them.
    3. Convert each database object or query result to a dictionary, applying the conflict resolution logic.

    Args:
        db_objects (Union[DBModel, Row, Iterable[Union[DBModel, Row]]]): The database objects or query results to
                                                                         convert.
        preferential_model (Type[DBModel], optional): The model to prefer in case of naming conflicts.

    Returns:
        Union[dict, List[dict]]: A single dictionary or a list of dictionaries representing the database objects.
    """

    def _query_to_dict(query_result: Row) -> dict:
        # Find columns with same name
        column_names = []
        conflicts = []
        for result in query_result._asdict().values():
            for column in result.columns():
                if column in column_names:
                    conflicts.append(column)
                else:
                    column_names.append(column)
        # Convert query result to dictionary
        result_dict = {}
        for model, result in query_result._asdict().items():
            object_dict = result.to_dict()
            for field, value in object_dict.items():
                if field not in conflicts or (preferential_model is not None and preferential_model.__name__ == model):
                    result_dict[field] = value
                else:
                    result_dict[camel_to_snake(model) + '_' + field] = value

        return result_dict

    if isinstance(db_objects, list):
        _to_dict = lambda x: x.to_dict() if isinstance(x, DBModel) else _query_to_dict(x)
        return list(map(_to_dict, db_objects))
    elif isinstance(db_objects, DBModel):
        return db_objects.to_dict()
    else:
        return _query_to_dict(db_objects)


def serialize(db_objects: Union[DBModel, Row, Iterable[Union[DBModel, Row]]]) -> Union[dict, List[dict]]:
    """
    Dumps database objects into JSON objects using Marshmallow schemas.

    Steps:
    1. Validate the input, ensuring it is not None or an empty list.
    2. Ensure the Marshmallow schema is set for the object class.
    3. Dump objects using the Marshmallow schema.
    4. If objects are query results, dump each class involved separately.
    5. Return a single dictionary if only one object was provided.

    Args:
        db_objects (Union[DBModel, Row, Iterable[Union[DBModel, Row]]]): The database objects or query results to
                                                                         serialize. If a list is given, all its objects
                                                                         must be of the same type.

    Returns:
        Union[dict, List[dict]]: A single dictionary or a list of dictionaries if `db_objects` is a list.
    """
    if db_objects is None or db_objects == []:
        return {}
    # Ensure the Marshmallow schema has been set for the object class
    list_given = isinstance(db_objects, list)
    db_objects = db_objects if list_given else [db_objects]
    object_prototype = db_objects[0]
    assert isinstance(object_prototype, (DBModel, Row))
    assert not any(map(lambda x: x.__class__ != object_prototype.__class__, db_objects))
    if isinstance(object_prototype, Row):
        for object_subtable in object_prototype:  # loop over all classes involved in the query result
            assert hasattr(object_subtable, '__marshmallow__')
    else:
        assert hasattr(object_prototype, '__marshmallow__')
    # Dump objects
    # If objects are query results (e.g. table joins), separately dump each class involved.
    if isinstance(object_prototype, Row):
        metadata_schemas = dict(map(lambda x: (x.__class__, x.__class__.__marshmallow__()), object_prototype))
        result = []
        # Note: if `__marshmallow__(many=True)` preserves objects order, dumping all objects at once for each class
        # and then zipping results might be more efficient than looping over all objects one by one
        for db_object in db_objects:
            dumped_object = {}
            for object_subtable in db_object:
                dumped_subtable = metadata_schemas[object_subtable.__class__].dump(object_subtable)
                dumped_object = {**dumped_object, **dumped_subtable}
            result.append(dumped_object)
    # Otherwise dump all objects at once
    else:
        metadata_schema = object_prototype.__class__.__marshmallow__(many=True)
        result = metadata_schema.dump(db_objects)
    # If only one object was provided (not a list of them), return a single dictionary
    if not list_given:
        assert len(result) == 1
        result = result[0]

    return result


def set_status(db_object: Entity, status: Status, group: str, commit: bool = True):
    """
    Sets the status of a database object, validating the status code and
    optionally committing the change to the database.

    Steps:
    1. Validate that the database object has a 'status' attribute.
    2. Validate that the status group is correct.
    3. Validate the provided status code.
    4. Save the current status code.
    5. Update the status of the database object.
    6. Optionally commit the changes to the database.

    Args:
        db_object (Entity): The database object whose status is to be set.
        status (Status): The new status to set.
        group (str): The group to which the status belongs.
        commit (bool, optional): Whether to commit the change to the database. Default is True.

    Raises:
        ValueError: If the status code is invalid.
    """
    assert hasattr(db_object, 'status')
    assert group in status_group_prefixes
    # Validate provided status code
    if not status.code.startswith(status_group_prefixes[group]):
        raise ValueError('Invalid status code')
    # Save current status code
    status.prev_status = db_object.status['code']
    # Update status
    db_object.status = status.to_dict()
    if commit:
        save_to_db(db_object)


def mysql_uuid_v4() -> TextClause:
    """
    Generates a MySQL UUID version 4 using a custom SQL query.

    Steps:
    1. Construct the SQL text for generating a UUID version 4.
    2. Return the constructed SQL text as a TextClause.

    Returns:
        TextClause: The SQL clause to generate a UUID version 4.
    """
    text_ = ('UUID_TO_BIN(LOWER(CONCAT('
             'HEX(RANDOM_BYTES(4)),'
             "'-', HEX(RANDOM_BYTES(2)),"
             "'-4', SUBSTR(HEX(RANDOM_BYTES(2)), 2, 3),"
             "'-', CONCAT(HEX(FLOOR(ASCII(RANDOM_BYTES(1)) / 64)+8), SUBSTR(HEX(RANDOM_BYTES(2)), 2, 3)),"
             "'-', HEX(RANDOM_BYTES(6))"
             ')))')
    return sql_text(text_)
