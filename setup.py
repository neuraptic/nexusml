"""
TODO: The current project uses setup.py, but the standard for new projects is pyproject.toml.
      Consider switching to pyproject.toml in future releases.
      For more information, refer to:
      https://packaging.python.org/en/latest/guides/modernize-setup-py-project/#modernize-setup-py-project

Variations:

1. Base Installation:
   - Command: `python -m pip install nexusml`
   - Description: Installs the base packages required by the engine.
                  Use this option if your primary need is to import and use the engine in your code.

2. API Installation:
   - Command: `python -m pip install nexusml[api]`
   - Description: Adds all dependencies required by the API to the base engine packages.
                  Use this option if deploying the API is part of your requirement.
"""

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

    # Read base (engine) requirements
    with open(os.path.join('requirements', 'engine-requirements.txt')) as f:
        base_requirements = f.read().splitlines()

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
        packages=find_packages(include=['nexusml*']),
        include_package_data=True,
        install_requires=base_requirements,
        extras_require={
            'api': api_requirements
        },
        python_requires='>=3.10',
        package_data={'nexusml': ['api/templates/email_notifications.html',
                               'api/templates/service_notifications.html',
                               'api/templates/user_invitation.html',
                               'api/templates/waitlist_notification.html']}
    )
