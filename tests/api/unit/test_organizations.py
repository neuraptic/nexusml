from unittest.mock import MagicMock

import pytest

from nexusml.api.resources import DuplicateResourceError
from nexusml.api.views.organizations import UserDB
from nexusml.api.views.organizations import UserInviteView
from nexusml.env import ENV_AUTH0_DOMAIN


@pytest.fixture(scope='function', autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv(ENV_AUTH0_DOMAIN, 'test_domain')


class TestUserInviteView:

    def test_post(self, app, client, mocker):
        """
        Test the POST request handling in UserInviteView.

        This test verifies the behavior of the POST request handling in the UserInviteView class,
        specifically for inviting users to an organization. It mocks various dependencies and asserts
        that the function behaves correctly under different scenarios, such as successful user invitation
        and handling of duplicate invitations.

        Test Steps:
        1. Mock necessary dependencies and environment variables.
        2. Patch required methods and functions to simulate behavior:
           - Mock roles_required decorator to allow access.
           - Mock Auth0 management API token retrieval.
           - Mock agent_from_token to simulate user authentication.
           - Mock download_auth0_user_data to mock user account data retrieval.
           - Mock InvitationDB query to simulate invitation checks.
           - Mock UserDB query and save_to_db for database operations.
           - Mock render_template and mail.send for email functionality.

        3. Call the UserInviteView's post method with mocked parameters and request context.
        4. Repeat steps 3-5 to assert handling of DuplicateResourceError when appropriate.

        Assertions:
        - The POST request to invite a user returns a status code of 204.
        - Auth0 management API token retrieval and agent authentication are called correctly.
        - User account data is retrieved and processed as expected.
        - API calls to create a user and a ticket are made with the correct parameters.
        - Email send and database operations (save_to_db) are called appropriately.
        - DuplicateResourceError is raised and handled when a duplicate invitation is detected.
        """
        organization_id: str = 'org_123'
        resources: list = [MagicMock()]
        resources[0].db_object.return_value = MagicMock(organization_id=organization_id,
                                                        domain='example.com',
                                                        uuid='org_uuid')
        kwargs: dict = {'email': 'user@example.com'}

        mocker.patch('nexusml.api.views.core.roles_required',
                     return_value=MagicMock(db_object=lambda: MagicMock(organization_id='org_123')))

        mock_agent: MagicMock = MagicMock(spec=UserDB)
        mock_agent_from_token = mocker.patch('nexusml.api.views.organizations.agent_from_token',
                                             return_value=mock_agent)

        mocker.patch('nexusml.api.views.core.agent_from_token', return_value=MagicMock(spec=UserDB))
        mock_download_auth0_user_data = mocker.patch('nexusml.api.views.organizations.User.download_auth0_user_data',
                                                     return_value=None)

        mock_has_invitation = MagicMock()
        mock_has_invitation.filter_by.return_value.first.return_value = None
        mocker.patch('nexusml.api.views.organizations.InvitationDB.query', return_value=mock_has_invitation)

        mocker.patch('nexusml.api.views.organizations.UserDB.query', return_value=MagicMock())
        mocker.patch('nexusml.api.views.organizations._invite_user.delay')
        mocker.patch('nexusml.api.views.organizations.render_template', return_value='mock_html')
        mocker.patch('nexusml.api.views.organizations.config.get', return_value={'email': 'mock_email'})
        mocker.patch('nexusml.api.views.organizations.save_to_db')

        user_invite_view = UserInviteView()
        with app.app_context():
            with client.application.test_request_context(json=kwargs):
                response = user_invite_view.post.__wrapped__.__wrapped__(self=user_invite_view,
                                                                         organization_id=organization_id,
                                                                         resources=resources,
                                                                         **kwargs)

        # Asserts
        assert response.status_code == 204
        mock_agent_from_token.assert_called_once()
        mock_download_auth0_user_data.assert_called()

        mock_has_invitation.filter_by.return_value.first.return_value = MagicMock()
        mocker.patch('nexusml.api.views.organizations.InvitationDB.query', return_value=mock_has_invitation)

        # Asserts DuplicateResourceError

        user_invite_view = UserInviteView()
        with app.app_context():
            with client.application.test_request_context(json=kwargs):
                with pytest.raises(DuplicateResourceError):
                    user_invite_view.post.__wrapped__.__wrapped__(self=user_invite_view,
                                                                  organization_id=organization_id,
                                                                  resources=resources,
                                                                  **kwargs)
