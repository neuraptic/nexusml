import functools
import random
import time
from typing import Iterable, Union

from flask_sqlalchemy import Model
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import ResourceClosedError
from sqlalchemy.orm import scoped_session


def retry_on_deadlock(session: scoped_session, max_retries: int = 10):
    """
    Decorator to retry a function in case of a database deadlock.

    Args:
        session (scoped_session): The database session to use for rollback.
        max_retries (int, optional): Maximum number of retries. Defaults to 10.

    Returns:
        function: The wrapped function with retry logic.
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            num_retries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except OperationalError as e:
                    # If the error is not related to a deadlock, raise it
                    if 'deadlock' not in str(e).lower():
                        raise e
                    # Roll back the transaction
                    session.rollback()
                    # Wait for the next retry
                    num_retries += 1
                    if num_retries > max_retries:
                        raise e
                    retry_after = random.randint(1, 9) / 1000 * 2**(num_retries - 1)
                    time.sleep(retry_after)

        return wrapper

    return decorator


def save_to_db(session: scoped_session, objects: Union[Model, Iterable[Model]], max_retries: int = 5):
    """
    Save given objects to the database with retry logic on deadlocks.

    Args:
        session (scoped_session): The database session to use.
        objects (Union[Model, Iterable[Model]]): The object or list of objects to save.
        max_retries (int, optional): Maximum number of retries in case of deadlock. Defaults to 5.
    """

    def _add_object(obj: Model):
        try:
            if obj not in session:
                session.add(obj)
        except InvalidRequestError as e:
            if 'is already present in this session' not in str(e):
                raise e

    for retries in range(max_retries):
        try:
            if isinstance(objects, Iterable):
                for obj in objects:
                    _add_object(obj)
            else:
                _add_object(objects)
            retry_on_deadlock(session=session, max_retries=max_retries)(session.commit())
            break
        except ResourceClosedError as e:
            # This error is raised when `_generate_public_id()` saves the public ID of an entity
            retry_on_deadlock(session=session, max_retries=max_retries)(session.rollback())
            if retries >= max_retries:
                raise e


def delete_from_db(session: scoped_session, objects: Union[Model, Iterable[Model]], max_retries: int = 5):
    """
    Delete given objects from the database with retry logic on deadlocks.

    Args:
        session (scoped_session): The database session to use.
        objects (Union[Model, Iterable[Model]]): The object or list of objects to delete.
        max_retries (int, optional): Maximum number of retries in case of deadlock. Defaults to 5.
    """
    if isinstance(objects, Iterable):
        for obj in objects:
            session.delete(obj)
    else:
        session.delete(objects)
    retry_on_deadlock(session=session, max_retries=max_retries)(session.commit())


def empty_table(session: scoped_session, model: Model):
    """
    Empty the contents of a table.

    Args:
        session (scoped_session): The database session to use.
        model (Model): The model class representing the table to be emptied.
    """
    session.query(model).delete()
    session.commit()
