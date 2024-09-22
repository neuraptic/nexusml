import os
import shutil

import pytest


@pytest.fixture(scope='session', autouse=True)
def tests_dir() -> str:
    tests_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(tests_path)
    return tests_path


@pytest.fixture(scope='session', autouse=True)
def artifacts_dir(tests_dir) -> str:
    # Setup: Create or empty directory
    artifacts_path = os.path.join(tests_dir, 'artifacts')

    if os.path.isdir(artifacts_path):
        shutil.rmtree(artifacts_path)

    os.mkdir(artifacts_path)

    # Run tests
    yield artifacts_path

    # Teardown: Delete directory
    shutil.rmtree(artifacts_path)


class Singleton(type):
    """
    Metaclass used for Singleton
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
