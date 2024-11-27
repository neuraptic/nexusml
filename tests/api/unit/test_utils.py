import os
from unittest.mock import MagicMock

import pytest

from nexusml.api.utils import BadRequest
from nexusml.api.utils import delete_auth0_user
from nexusml.api.utils import get_auth0_management_api_token
from nexusml.api.utils import get_auth0_user_data
from nexusml.env import ENV_AUTH0_DOMAIN


@pytest.fixture(scope='function', autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv(ENV_AUTH0_DOMAIN, 'test_domain')


def test_get_auth0_management_api_token(mocker):
    """
    Test get_auth0_management_api_token function for retrieving Auth0 management API token.

    This test verifies that the get_auth0_management_api_token function correctly sends a POST
    request to retrieve the Auth0 management API token and returns the token value.

    Test Steps:
    1. Patch the 'os.environ' to mock environment variables.
    2. Patch the 'requests.post' method to mock the POST request response.
    3. Mock the response of the POST request with a JSON response containing the access token.
    4. Call get_auth0_management_api_token.

    Assertions:
    - The POST request is sent to the correct endpoint.
    - The function returns the correct Auth0 management API token retrieved from the mocked response.
    """

    auth0_json_response: dict = {'access_token': 'access_token_value'}

    mocker.patch('os.environ')

    mock_post = mocker.patch('requests.post')
    mock_post_response: MagicMock = MagicMock()
    mock_post_response.json.return_value = auth0_json_response
    mock_post.return_value = mock_post_response

    result = get_auth0_management_api_token()

    mock_post.assert_called_once()
    assert result == 'access_token_value'


@pytest.mark.parametrize('auth0_id_or_email, expected_url_extension',
                         [('example@example.com', '?q=email:example%40example.com&search_engine=v3'),
                          ('UUID-UUID-UUID-UUID', '/UUID-UUID-UUID-UUID')])
def test_get_auth0_user_data(mocker, auth0_id_or_email, expected_url_extension):
    """
    Parameterized test for get_auth0_user_data function.

    This test verifies the behavior of get_auth0_user_data function for different
    user identifiers (email and UUID). It checks that the function constructs the correct URL
    based on the provided user identifier and sends the appropriate GET request.

    Test Steps:
    1. Mock the Auth0 access token, expected URL (including extension), and mock account data.
    2. Patch the 'requests.get' method to mock the GET request response.
    3. Mock the response of the GET request with a 200 status code and mock account data.
    4. Call get_auth0_user_data with the access token and user identifier.

    Parameters:
    - user_uuid_or_email: The user identifier (email or UUID) used to construct the URL.
    - expected_url_extension: The expected URL extension based on the user identifier.

    Assertions:
    - The GET request is sent to the correct URL with the appropriate headers.
    - The function returns the correct user account data retrieved from the mocked response.
    """
    access_token = 'dummy_access_token'
    expected_url: str = f'https://{os.environ[ENV_AUTH0_DOMAIN]}/api/v2/users{expected_url_extension}'
    mock_account_data: list = ['account_data']

    # Mocking the response of the GET request
    mock_get = mocker.patch('requests.get')
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_account_data
    mock_get.return_value = mock_response

    result = get_auth0_user_data(access_token, auth0_id_or_email)

    mock_get.assert_called_once_with(expected_url, headers={'Authorization': 'Bearer ' + access_token})
    assert result == mock_account_data[0]


def test_get_auth0_user_data_raises_error(mocker):
    """
    Test get_auth0_user_data raises BadRequest error.

    This test verifies that the function get_auth0_user_data correctly raises
    a BadRequest error when the external API call returns a 400 status code.

    Test Steps:
    1. Create a mock access token and user identifier (UUID or email).
    2. Patch the 'requests.get' method to return a mock response with a 400 status code.
    3. Call get_auth0_user_data with the mock access token and user identifier.
    4. Verify that the function raises a BadRequest error.

    Assertions:
    - The function raises a BadRequest error when the API response status code is 400.
    """

    access_token = 'dummy_access_token'
    auth0_id_or_email = 'example@example.com'

    mock_get = mocker.patch('requests.get')
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_get.return_value = mock_response

    with pytest.raises(BadRequest):
        get_auth0_user_data(access_token, auth0_id_or_email)


def test_delete_auth0_user(mocker):
    """
    Test delete_auth0_user function behavior for successful and failed delete operations.

    This test verifies that the delete_auth0_user function sends the correct DELETE request
    and handles both successful (204 No Content) and failed (400 Bad Request) responses
    appropriately.

    Test Steps:
    1. Create mock Auth0 token and ID, and construct the expected URL.
    2. Patch the 'requests.delete' method to mock the DELETE request response.
    3. Mock the response of the DELETE request with a 204 status code (success).
    4. Call delete_auth0_user with the mock token and ID.
    5. Assert the DELETE request was made with the correct URL and headers.
    6. Change the mock response status code to 400 (bad request).
    7. Verify that delete_auth0_user raises a BadRequest error when the response status code is 400.

    Assertions:
    - The DELETE request is sent to the correct URL with the appropriate headers.
    - The function does not raise an error for a 204 No Content response.
    - The function raises a BadRequest error for a 400 Bad Request response.
    """

    auth0_token = 'dummy_auth0_token'
    auth0_id = 'dummy_auth0_id'
    expected_url = f'https://{os.environ[ENV_AUTH0_DOMAIN]}/api/v2/users/{auth0_id}'

    # Mocking the response of the DELETE request
    mock_delete = mocker.patch('requests.delete')
    mock_response = MagicMock()
    mock_response.status_code = 204
    mock_delete.return_value = mock_response

    mock_auth0_token = mocker.patch('nexusml.api.utils.get_auth0_management_api_token')
    mock_auth0_token.return_value = auth0_token

    delete_auth0_user(auth0_id)

    # Assertions
    mock_delete.assert_called_once_with(expected_url, headers={'Authorization': 'Bearer ' + auth0_token})

    mock_response.status_code = 400
    with pytest.raises(AssertionError):
        delete_auth0_user(auth0_id)
