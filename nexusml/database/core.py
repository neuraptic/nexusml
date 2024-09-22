from typing import Iterable, Union

from flask_sqlalchemy import DefaultMeta
from flask_sqlalchemy import SQLAlchemy

from nexusml.sqlalchemy_utils import delete_from_db as _delete_from_db
from nexusml.sqlalchemy_utils import empty_table as _empty_table
from nexusml.sqlalchemy_utils import retry_on_deadlock as _retry_on_deadlock
from nexusml.sqlalchemy_utils import save_to_db as _save_to_db

__all__ = [
    'db',
    'retry_on_deadlock',
    'db_commit',
    'db_rollback',
    'db_execute',
    'db_query',
    'save_to_db',
    'delete_from_db',
    'create_tables',
    'empty_table',
    'model_from_tablename',
]

db = SQLAlchemy(session_options={'expire_on_commit': False})


def retry_on_deadlock(func):
    """
    Decorates a function to automatically retry on database deadlock.

    Wraps the given function with retry logic to handle deadlock errors by
    retrying the execution of the function using the session from the SQLAlchemy `db`.

    Args:
        func: The function to be decorated.

    Returns:
        The decorated function with retry logic.
    """
    return _retry_on_deadlock(session=db.session)(func)


@retry_on_deadlock
def db_commit():
    """
    Commits the current transaction to the database.

    Uses the session from the SQLAlchemy `db` to commit the current transaction,
    ensuring that any changes made are saved to the database and handling any necessary retries on deadlock.

    Returns:
        None
    """
    return db.session.commit()


@retry_on_deadlock
def db_rollback():
    """
    Rolls back the current transaction in the database.

    Uses the session from the SQLAlchemy `db` to rollback the current transaction,
    undoing any changes made during the transaction and handling any necessary retries on deadlock.

    Returns:
        None
    """
    return db.session.rollback()


@retry_on_deadlock
def db_execute(statement, *args, **kwargs) -> SQLAlchemy.Query:
    """
    Executes a database statement.

    Uses the session from the SQLAlchemy `db` to execute the given statement with the provided
    arguments and keyword arguments, handling any necessary retries on deadlock.

    Args:
        statement: The SQL statement to be executed.
        *args: Additional positional arguments for the execute method.
        **kwargs: Additional keyword arguments for the execute method.

    Returns:
        Query: The result of the executed statement.
    """
    return db.session.execute(statement, *args, **kwargs)


@retry_on_deadlock
def db_query(*entities, **kwargs) -> SQLAlchemy.Query:
    """
    Queries the database for the specified entities.

    Uses the session from the SQLAlchemy `db` to perform a query for the provided entities,
    handling any necessary retries on deadlock.

    Args:
        *entities: The entities to query.
        **kwargs: Additional keyword arguments for the query method.

    Returns:
        Query: The result of the query.
    """
    return db.session.query(*entities, **kwargs)


def save_to_db(objects: Union[db.Model, Iterable[db.Model]], max_retries: int = 5):
    """
    Saves objects to the database.

    Uses the session from the SQLAlchemy `db` to save the provided objects to the database,
    with a specified maximum number of retries in case of deadlock.

    Args:
        objects: The object or iterable of objects to be saved.
        max_retries: The maximum number of retry attempts on deadlock (default is 5).

    Returns:
        None
    """
    return _save_to_db(session=db.session, objects=objects, max_retries=max_retries)


def delete_from_db(objects: Union[db.Model, Iterable[db.Model]], max_retries: int = 5):
    """
    Deletes objects from the database.

    Uses the session from the SQLAlchemy `db` to delete the provided objects from the database,
    with a specified maximum number of retries in case of deadlock.

    Args:
        objects: The object or iterable of objects to be deleted.
        max_retries: The maximum number of retry attempts on deadlock (default is 5).

    Returns:
        None
    """
    return _delete_from_db(session=db.session, objects=objects, max_retries=max_retries)


def create_tables():
    """
    Creates all tables in the database with InnoDB as the storage engine.

    Iterates over all mapped models in the SQLAlchemy `db`, updating their `__table_args__`
    to use InnoDB as the storage engine if they do not already specify an engine. Then, it
    creates all tables in the database.

    Returns:
        None
    """
    inno_db_args = {'mysql_engine': 'InnoDB'}

    for mapper in db.Model.registry.mappers:
        model = mapper.class_
        if not isinstance(model, DefaultMeta):
            continue
        if hasattr(model, '__table_args__'):
            if isinstance(model.__table_args__, tuple):
                model.__table_args__ += (inno_db_args,)
            else:
                assert isinstance(model.__table_args__, dict)
                model.__table_args__.update(inno_db_args)
        else:
            setattr(model, '__table_args__', inno_db_args)

    db.create_all()


def empty_table(model: db.Model):
    """
    Empties the contents of a table for a given model.

    Uses the session from the SQLAlchemy `db` to delete all records from the table
    corresponding to the provided model.

    Args:
        model: The SQLAlchemy model whose table is to be emptied.

    Returns:
        None
    """
    return _empty_table(session=db.session, model=model)


def model_from_tablename(tablename: str) -> db.Model:
    """
    Retrieves a model class based on the table name.

    Searches the SQLAlchemy `db` registry for a model whose table name matches the provided
    table name and returns the corresponding model class.

    Args:
        tablename (str): The name of the table for which to retrieve the model class.

    Returns:
        db.Model: The model class corresponding to the given table name, or None if no match is found.
    """
    for c in db.Model._decl_class_registry.values():
        if hasattr(c, '__table__') and c.__table__.name == tablename:
            return c
