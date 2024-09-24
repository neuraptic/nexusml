# TODO: The current project uses setup.py, but the standard for new projects is pyproject.toml.
#       Consider switching to pyproject.toml in future releases.
#       For more information, refer to:
#       https://packaging.python.org/en/latest/guides/modernize-setup-py-project/#modernize-setup-py-project

import os

from setuptools import setup, find_packages

# Set basic information
_NAME = 'nexusml'
_VERSION = '0.1.0'
_AUTHOR = 'Neuraptic AI'
_AUTHOR_EMAIL = 'support@neuraptic.ai'
_DESCRIPTION = 'A multimodal AutoML platform for classification and regression tasks'
_URL = 'https://github.com/neuraptic/nexusml'
_COPYRIGHT = '2024, Neuraptic AI'

if __name__ == '__main__':
    # Read the long description from README.md
    with open('README.md', 'r') as f:
        long_description = f.read()

    # Read engine requirements
    with open(os.path.join('requirements', 'engine-requirements.txt')) as f:
        engine_requirements = f.read().splitlines()

    # Read API requirements
    with open(os.path.join('requirements', 'api-requirements.txt')) as f:
        api_requirements = f.read().splitlines()

    # Setup
    setup(
        name=_NAME,
        version=_VERSION,
        author=_AUTHOR,
        author_email=_AUTHOR_EMAIL,
        description=_DESCRIPTION,
        long_description=long_description,
        long_description_content_type='text/markdown',
        url=_URL,
        python_requires='>=3.10',
        install_requires=(engine_requirements + api_requirements),
        packages=find_packages(include=['nexusml*']),
        include_package_data=True,
        package_data={'nexusml': ['api/templates/email_notifications.html',
                                  'api/templates/service_notifications.html',
                                  'api/templates/user_invitation.html',
                                  'api/templates/waitlist_notification.html']}
    )
