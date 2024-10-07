from unittest.mock import MagicMock

import pytest

from nexusml.api.external.auth0 import Auth0Manager
from nexusml.api.resources import ResourceNotFoundError
from nexusml.api.resources.organizations import User
from nexusml.env import ENV_AUTH0_CLIENT_ID
from nexusml.env import ENV_AUTH0_CLIENT_SECRET
from nexusml.env import ENV_AUTH0_DOMAIN


@pytest.fixture(scope='function', autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv(ENV_AUTH0_DOMAIN, 'test_domain')
    monkeypatch.setenv(ENV_AUTH0_CLIENT_ID, 'test_client_id')
    monkeypatch.setenv(ENV_AUTH0_CLIENT_SECRET, 'test_client_secret')


@pytest.fixture(scope='function')
def mock_get_auth0_user_data(mocker):
    return mocker.patch.object(Auth0Manager, 'get_auth0_user_data', retutn_value={})


@pytest.fixture(scope='function')
def mock_get_auth0_management_api_token(mocker):
    return mocker.patch.object(Auth0Manager, '_get_auth0_management_api_token', return_value='auth0_token')


class TestDownloadAuth0UserData:

    def test_download_auth0_user_data(self, mock_get_auth0_user_data, mock_get_auth0_management_api_token):
        """
        This function tests the behavior of the download_auth0_user_data method in the User class, ensuring it
        correctly retrieves and processes account data based on a provided user UUID or email.
        It mocks the dependencies get_auth0_user_data and get_auth0_management_api_token to simulate
        the retrieval of user data and an Auth0 token, respectively.

        Args:
            mock_get_auth0_user_data: Mock for the get_auth0_user_data function.
            mock_get_auth0_management_api_token: Mock for the get_auth0_management_api_token function.

        Returns:
            None
        """

        mock_account_data: dict = {
            'user_id': 'test_id',
            'email': 'example@example.com',
            'given_name': 'John',
            'family_name': 'Doe',
            'email_verified': True
        }
        mock_get_auth0_user_data.return_value = mock_account_data
        mock_get_auth0_management_api_token.return_value = 'mocked_access_token'

        result = User.download_auth0_user_data(auth0_id_or_email='test_auth0_id_or_email')

        assert result == {
            'auth0_id': 'test_id',
            'email': 'example@example.com',
            'first_name': 'John',
            'last_name': 'Doe',
            'email_verified': True
        }

    def test_download_auth0_user_data_reduced(self, mock_get_auth0_user_data, mock_get_auth0_management_api_token):
        """
        This function tests the behavior of the download_auth0_user_data method in the User class when provided with
        minimal account data (user ID and email only). It mocks the dependencies get_auth0_user_data and
        get_auth0_management_api_token to simulate the retrieval of user data and an Auth0 token, respectively.

        Args:
            mock_get_auth0_user_data: Mock for the get_auth0_user_data function.
            mock_get_auth0_management_api_token: Mock for the get_auth0_management_api_token function.

        Returns:
            None
        """

        mock_account_data = {'user_id': 'test_id', 'email': 'example@example.com', 'email_verified': True}
        mock_get_auth0_user_data.return_value = mock_account_data
        mock_get_auth0_management_api_token.return_value = 'mocked_access_token'

        MagicMock(is_admin=False, is_maintainer=False)
        result = User.download_auth0_user_data(auth0_id_or_email='test_auth0_id_or_email')

        assert result == {
            'auth0_id': 'test_id',
            'email': 'example@example.com',
            'first_name': None,
            'last_name': None,
            'email_verified': True
        }

    def test_download_auth0_user_data_no_data(self, mock_get_auth0_user_data, mock_get_auth0_management_api_token):
        """
        Test function for User.download_auth0_user_data with no data.

        This function tests the behavior of the download_auth0_user_data method in the User class when no account
        data is returned. It mocks the dependencies get_auth0_user_data and get_auth0_management_api_token to
        simulate the scenario where no user data is found and an Auth0 token is retrieved.

        Args:
            mock_get_auth0_user_data: Mock for the get_auth0_user_data function.
            mock_get_auth0_management_api_token: Mock for the get_auth0_management_api_token function.

        Returns:
            None
        """

        mock_get_auth0_user_data.return_value = {}
        mock_get_auth0_management_api_token.return_value = 'mocked_access_token'

        with pytest.raises(ResourceNotFoundError):
            User.download_auth0_user_data(auth0_id_or_email='test_auth0_id_or_email')

    def test_download_auth0_user_data_invalid_uuid_or_email(self, mock_get_auth0_user_data,
                                                            mock_get_auth0_management_api_token):
        """
        Test function for User.download_auth0_user_data with invalid UUID or email.

        This function tests the behavior of the download_auth0_user_data method in the User class when an invalid
        UUID or email is provided. It mocks the dependencies get_auth0_user_data and
        get_auth0_management_api_token to simulate the scenario where no user data is found due to an invalid UUID
        or email, and an Auth0 token is retrieved.

        Args:
            mock_get_auth0_user_data: Mock for the get_auth0_user_data function.
            mock_get_auth0_management_api_token: Mock for the get_auth0_management_api_token function.

        Returns:
            None
        """

        mock_get_auth0_user_data.return_value = None
        mock_get_auth0_management_api_token.return_value = 'mocked_access_token'

        with pytest.raises(ResourceNotFoundError):
            User.download_auth0_user_data('test_invalid_uuid_or_email')
