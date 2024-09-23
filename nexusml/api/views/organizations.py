from datetime import datetime
import os
from typing import Dict, List

from celery import shared_task
from flask import current_app
from flask import g
from flask import jsonify
from flask import render_template
from flask import Response
from flask_apispec import doc
from flask_apispec import marshal_with
from flask_apispec import use_kwargs
from flask_mail import Message

from nexusml.api.ext import mail
from nexusml.api.resources.base import dump
from nexusml.api.resources.base import DuplicateResourceError
from nexusml.api.resources.base import InvalidDataError
from nexusml.api.resources.base import PermissionDeniedError
from nexusml.api.resources.base import Resource
from nexusml.api.resources.base import ResourceError
from nexusml.api.resources.base import ResourceNotFoundError
from nexusml.api.resources.base import UnprocessableRequestError
from nexusml.api.resources.files import TaskFile
from nexusml.api.resources.organizations import check_last_admin_deletion
from nexusml.api.resources.organizations import Client
from nexusml.api.resources.organizations import Collaborator
from nexusml.api.resources.organizations import get_user_roles
from nexusml.api.resources.organizations import load_active_subscription
from nexusml.api.resources.organizations import Organization
from nexusml.api.resources.organizations import Role
from nexusml.api.resources.organizations import User
from nexusml.api.schemas.organizations import APIKeyRequestSchema
from nexusml.api.schemas.organizations import APIKeyResponseSchema
from nexusml.api.schemas.organizations import AppRequestSchema
from nexusml.api.schemas.organizations import AppResponseSchema
from nexusml.api.schemas.organizations import CollaboratorRequestSchema
from nexusml.api.schemas.organizations import CollaboratorResponseSchema
from nexusml.api.schemas.organizations import CollaboratorsPage
from nexusml.api.schemas.organizations import OrganizationPOSTRequestSchema
from nexusml.api.schemas.organizations import OrganizationPUTRequestSchema
from nexusml.api.schemas.organizations import OrganizationResponseSchema
from nexusml.api.schemas.organizations import RoleRequestSchema
from nexusml.api.schemas.organizations import RoleResponseSchema
from nexusml.api.schemas.organizations import SubscriptionSchema
from nexusml.api.schemas.organizations import UserRequestSchema
from nexusml.api.schemas.organizations import UserResponseSchema
from nexusml.api.schemas.organizations import UserRolesRequestSchema
from nexusml.api.schemas.organizations import UserRolesResponseSchema
from nexusml.api.schemas.organizations import UsersPage
from nexusml.api.utils import config
from nexusml.api.utils import decode_api_key
from nexusml.api.views.base import create_view
from nexusml.api.views.common import PermissionAssignmentView
from nexusml.api.views.core import agent_from_token
from nexusml.api.views.core import error_response
from nexusml.api.views.core import get_page_resources
from nexusml.api.views.core import process_delete_request
from nexusml.api.views.core import process_get_request
from nexusml.api.views.core import process_post_or_put_request
from nexusml.api.views.core import roles_required
from nexusml.api.views.utils import paging_url_params
from nexusml.constants import ADMIN_ROLE
from nexusml.constants import API_NAME
from nexusml.constants import DATETIME_FORMAT
from nexusml.constants import ENDPOINT_CLIENT_API_KEY
from nexusml.constants import GENERIC_DOMAINS
from nexusml.constants import HTTP_BAD_REQUEST_STATUS_CODE
from nexusml.constants import HTTP_DELETE_STATUS_CODE
from nexusml.constants import HTTP_GET_STATUS_CODE
from nexusml.constants import HTTP_NOT_FOUND_STATUS_CODE
from nexusml.constants import HTTP_POST_STATUS_CODE
from nexusml.constants import HTTP_PUT_STATUS_CODE
from nexusml.constants import HTTP_SERVICE_UNAVAILABLE
from nexusml.constants import MAINTAINER_ROLE
from nexusml.constants import NUM_RESERVED_CLIENTS
from nexusml.constants import SWAGGER_TAG_ORGANIZATIONS
from nexusml.database.ai import AIModelDB
from nexusml.database.core import delete_from_db
from nexusml.database.core import save_to_db
from nexusml.database.examples import ExampleDB
from nexusml.database.examples import ExCategory
from nexusml.database.examples import ExFile
from nexusml.database.examples import ShapeCategory
from nexusml.database.examples import ShapeDB
from nexusml.database.files import TaskFileDB
from nexusml.database.organizations import ClientDB
from nexusml.database.organizations import CollaboratorDB
from nexusml.database.organizations import InvitationDB
from nexusml.database.organizations import OrganizationDB
from nexusml.database.organizations import RoleDB
from nexusml.database.organizations import UserDB
from nexusml.database.organizations import WaitList
from nexusml.database.permissions import RolePermission
from nexusml.database.permissions import UserPermission
from nexusml.database.services import Service
from nexusml.database.subscriptions import SubscriptionDB
from nexusml.database.tasks import CategoryDB
from nexusml.database.tasks import copy_task_to_organization
from nexusml.database.tasks import demo_tasks
from nexusml.database.tasks import ElementDB
from nexusml.database.tasks import TaskDB
from nexusml.enums import DBRelationshipType
from nexusml.enums import InviteStatus
from nexusml.env import ENV_AUTH0_SIGN_UP_REDIRECT_URL
from nexusml.env import ENV_NOTIFICATION_EMAIL
from nexusml.env import ENV_SUPPORT_EMAIL
from nexusml.statuses import Status
from nexusml.statuses import task_active_status
from nexusml.statuses import task_copying_status
from nexusml.utils import get_s3_config
from nexusml.utils import s3_client

# Note: This pylint directive is disabled because we are passing `user_id`, `role_id`, or `collaborator_id`
# to the `agent_id` parameter for greater specificity.
# pylint: disable=arguments-renamed

################
# Define views #
################

_no_api_keys_write = {'reject_api_keys': ['DELETE', 'POST', 'PUT']}  # reject write operations for API keys

_OrganizationsView = create_view(resource_types=[Organization], load_resources=False, reject_api_keys=['POST'])
_OrganizationView = create_view(resource_types=[Organization], reject_api_keys=['DELETE', 'PUT'])
_SubscriptionView = create_view(resource_types=[Organization], reject_api_keys=['DELETE', 'PUT'])
_UserView = create_view(resource_types=[Organization, User], **_no_api_keys_write)
_UserRoleView = create_view(resource_types=[Organization, User], load_resources=False, **_no_api_keys_write)
_UserInviteView = create_view(resource_types=[Organization], reject_api_keys=['POST'])
_RoleView = create_view(resource_types=[Organization, Role], **_no_api_keys_write)
_CollaboratorView = create_view(resource_types=[Organization, Collaborator], **_no_api_keys_write)
_ClientView = create_view(resource_types=[Organization, Client], **_no_api_keys_write)
"""
Organizations
"""


def _update_task_copy_progress(task: TaskDB, progress: int):
    """
    Updates the task copy progress status.

    Args:
        task (TaskDB): The task database object whose status is being updated.
        progress (int): The progress percentage to be updated.
    """
    copy_status = Status(template=task_copying_status)
    copy_status.details = {'progress': progress}
    task.set_status(status=copy_status)


@shared_task
def _populate_demo_tasks(user_id: int, pk_maps: Dict[str, list]):
    """
    Creates files, examples, and AI models in the user organization's demo tasks.

    Notes on progress info update:
        - Files copy complete: 50%
            - Database rows copied: 5%
            - S3 files copied: 50% (updated every 30 files)
        - Examples copy complete: 80%
            - Examples' entries copied: 60%
            - Examples' values copied: 80%
        - Shapes copy complete: 95%
        - AI models copy complete: 100%

    Args:
        user_id (int): Surrogate key of the user who created the organization.
        pk_maps (dict): Dictionary mapping demo task IDs to copied task IDs and their
                        primary key mappings. The format is as follows:
                        {
                            <demo_task_id_1> (str): [
                                <task_copy_id_1> (int),
                                {
                                    <pk_col_1> (str): {
                                        <demo_pk_val_1> (str): <copy_pk_val_1> (int),
                                        ...
                                        <demo_pk_val_N> (str): <copy_pk_val_N> (int)
                                    },
                                    ...
                                    <pk_col_N> (str): {
                                        <demo_pk_val_1> (str): <copy_pk_val_1> (int),
                                        ...
                                        <demo_pk_val_N> (str): <copy_pk_val_N> (int)
                                    }
                                }
                            ],
                            ...
                            <demo_task_id_N> (str): [
                                <task_copy_id_N> (int),
                                {
                                    ...
                                }
                            ]
                        }
    """
    pk_maps = {
        int(k): (v[0], {
            vk: {
                int(vvk): vvv for vvk, vvv in vv.items()
            } for vk, vv in v[1].items()
        }) for k, v in pk_maps.items()
    }

    user = UserDB.get(user_id=user_id)

    s3_config = get_s3_config()

    for demo_task_id, pk_map_ in pk_maps.items():
        demo_task = TaskDB.get(task_id=demo_task_id)
        task_copy = TaskDB.get(task_id=pk_map_[0])

        # Primary key mappings
        pk_map = pk_map_[1]

        elem_full_pk_name = ElementDB.__name__ + '.element_id'
        cat_full_pk_name = CategoryDB.__name__ + '.category_id'

        # Copy files
        # TODO: files should be copied at SQL level to scale to large number of files
        files = TaskFileDB.query().filter_by(task_id=demo_task_id).all()
        files_copy = []
        files_pks = []
        cols_to_copy = TaskFileDB.columns() - {'file_id', 'task_id', 'uuid', 'public_id'}
        for file in files:
            # Copy file's database entry
            file_copy = TaskFileDB(**{col: getattr(file, col) for col in cols_to_copy})
            file_copy.task_id = task_copy.task_id
            file_copy.created_by_user = user.user_id
            file_copy.modified_by_user = user.user_id
            file_copy.synced_by_users = [user.user_id]
            files_copy.append(file_copy)
            files_pks.append(file.file_id)
        save_to_db(files_copy)

        _update_task_copy_progress(task=task_copy, progress=5)

        for idx, (file, file_copy) in enumerate(zip(files, files_copy)):
            # Copy file in S3
            assert file.use_for == file_copy.use_for
            src = (s3_config['bucket'] + '/' + TaskFile.prefix() + str(demo_task.uuid) + '/' +
                   TaskFile.use_prefixes()[file.use_for] + str(file.uuid))
            dst = (TaskFile.prefix() + str(task_copy.uuid) + '/' + TaskFile.use_prefixes()[file_copy.use_for] +
                   str(file_copy.uuid))
            s3_client().copy_object(CopySource=src, Bucket=s3_config['bucket'], Key=dst)
            # Update task status
            if (idx + 1) % 30 == 0:
                _update_task_copy_progress(task=task_copy, progress=(5 + round(45 * (idx + 1) / len(files))))

        # Copy examples
        # TODO: examples should be copied at SQL level to scale to large number of examples
        examples = ExampleDB.query().filter_by(task_id=demo_task.task_id).all()
        examples_copy = []
        cols_to_copy = ExampleDB.columns() - {'task_id', 'example_id', 'uuid', 'public_id'}
        for ex in examples:
            # Copy example's entry
            ex_copy = ExampleDB(**{col: getattr(ex, col) for col in cols_to_copy})
            ex_copy.task_id = task_copy.task_id
            ex_copy.created_by_user = user.user_id
            ex_copy.modified_by_user = user.user_id
            ex_copy.synced_by_users = [user.user_id]
            examples_copy.append(ex_copy)
        save_to_db(examples_copy)

        _update_task_copy_progress(task=task_copy, progress=60)

        examples_pks = dict()
        ex_values_copy = []
        for ex, ex_copy in zip(examples, examples_copy):
            examples_pks[ex.example_id] = ex_copy.example_id
            # Copy example's values
            values = [getattr(ex, r) for r in ExampleDB.relationship_types()[DBRelationshipType.ASSOCIATION_OBJECT]]
            for value_list in values:
                for value in value_list:
                    value_copy = value.__class__(**{col: getattr(value, col) for col in value.columns()})
                    value_copy.example_id = ex_copy.example_id
                    value_copy.element_id = pk_map[elem_full_pk_name][value.element_id]
                    # Update reference to files and categories
                    if value.value is not None:
                        if isinstance(value, ExFile):
                            value_copy.value = files_copy[files_pks.index(value.value)].file_id
                        elif isinstance(value, ExCategory):
                            value_copy.value = pk_map[cat_full_pk_name][value.value]
                    ex_values_copy.append(value_copy)
        save_to_db(ex_values_copy)

        _update_task_copy_progress(task=task_copy, progress=80)

        # Copy shapes
        shapes = ShapeDB.query().filter(ShapeDB.example_id.in_(examples_pks.keys())).all()
        shapes_copy = []
        cols_to_copy = ShapeDB.columns() - {'example_id', 'element_id', 'shape_id', 'uuid', 'public_id'}
        for shape in shapes:
            # Copy shape's entry
            shape_copy = ShapeDB(**{col: getattr(shape, col) for col in cols_to_copy})
            shape_copy.example_id = examples_pks[shape.example_id]
            shape_copy.element_id = pk_map[elem_full_pk_name][shape.element_id]
            shape_copy.created_by_user = user.user_id
            shape_copy.modified_by_user = user.user_id
            shape_copy.synced_by_users = [user.user_id]
            shapes_copy.append(shape_copy)
        save_to_db(shapes_copy)

        shape_outputs_copy = []
        for shape, shape_copy in zip(shapes, shapes_copy):
            # Copy shape's output values
            for output_value in shape.shape_floats + shape.shape_categories:
                output_type = type(output_value)
                output_value_copy = output_type(**{c: getattr(output_value, c) for c in output_type.columns()})
                output_value_copy.shape_id = shape_copy.shape_id
                output_value_copy.element_id = pk_map[elem_full_pk_name][output_value.element_id]
                shape_outputs_copy.append(output_value_copy)
                if output_value.value is not None and isinstance(output_value, ShapeCategory):
                    output_value_copy.value = pk_map[cat_full_pk_name][output_value.value]
        save_to_db(shape_outputs_copy)

        _update_task_copy_progress(task=task_copy, progress=95)

        # Copy AI models
        ai_models = AIModelDB.query().filter_by(task_id=demo_task.task_id).all()
        ai_models_copy = []
        cols_to_copy = AIModelDB.columns() - {'model_id', 'task_id', 'uuid', 'public_id'}
        for ai_model in ai_models:
            ai_model_copy = AIModelDB(**{col: getattr(ai_model, col) for col in cols_to_copy})
            try:
                ai_model_copy.file_id = files_copy[files_pks.index(ai_model.file_id)].file_id
            except Exception:
                pass
            ai_model_copy.task_id = task_copy.task_id
            ai_model_copy.created_by_user = user.user_id
            ai_model_copy.modified_by_user = user.user_id
            ai_model_copy.synced_by_users = [user.user_id]
            ai_models_copy.append(ai_model_copy)
        save_to_db(ai_models_copy)

        # Update task status
        task_copy.set_status(status=Status(template=task_active_status))


class OrganizationsView(_OrganizationsView):

    @staticmethod
    def _set_organization(user_auth0_id: str, org_db_object: OrganizationDB, copy_demo_tasks: bool) -> Organization:
        """
        Sets up an organization's basic configuration. It subscribes the organization to the Free Plan,
        creates predefined roles ("admin" and "maintainer"), assigns the admin role to the user,
        and optionally copies demo tasks.

        Args:
            user_uuid (str): UUID of the user creating the organization.
            org_db_object (OrganizationDB): The organization database object.
            copy_demo_tasks (bool): Whether to copy demo tasks for the organization.

        Returns:
            Organization: The organization object with the basic configuration set.
        """
        ##############################
        # Subscribe to the Free Plan #
        ##############################
        subscription = SubscriptionDB(organization_id=org_db_object.organization_id, plan_id=1)
        #########################################################################################################
        # Create "admin" and "maintainer" roles.                                                                #

        # NOTE: these two roles are not taken into account for the maximum number of roles offered by the Plan. #
        #########################################################################################################
        admin_role = RoleDB(organization_id=org_db_object.organization_id, name=ADMIN_ROLE, description='Administrator')
        maintainer_role = RoleDB(organization_id=org_db_object.organization_id,
                                 name=MAINTAINER_ROLE,
                                 description='Maintainer')
        save_to_db([admin_role, maintainer_role])
        ###################################################################################
        # Add the first user ( session user) to the organization and assign "admin" role #
        ###################################################################################
        user_db_obj: UserDB = UserDB(auth0_id=user_auth0_id, organization_id=org_db_object.organization_id)
        save_to_db(user_db_obj)

        # Assign "admin" role to session user_db_obj
        user_db_obj.roles.append(admin_role)
        save_to_db(admin_role)

        # If user email's domain doesn't match organization's, delete the organization
        # TODO: user email should be checked before creating the organization.
        #       We are doing this way because we need a `session_agent` for creating the organization.
        auth0_user_data: dict = User.download_auth0_user_data(auth0_id_or_email=user_db_obj.auth0_id)
        if auth0_user_data['email'].split('@')[-1] != org_db_object.domain:
            raise UnprocessableRequestError("Domains don't match")

        # Update user count
        subscription.num_users = 1
        save_to_db(subscription)
        ############################################################
        # Copy demo tasks in the background (asynchronous process) #
        ############################################################
        if copy_demo_tasks:
            # Create and init tasks
            pk_maps = dict()
            for demo_task in demo_tasks():
                # Copy task schema
                task_copy, pk_map = copy_task_to_organization(src_task=demo_task, agent=user_db_obj)
                pk_maps[demo_task.task_id] = (task_copy.task_id, pk_map)
                # Initialize task
                task_copy.init_task()
                # Update task status
                _update_task_copy_progress(task=task_copy, progress=0)

            # Populate tasks in the background
            _populate_demo_tasks.delay(user_db_obj.user_id, pk_maps)

        return Organization.get(agent=user_db_obj, db_object_or_id=org_db_object)

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS])
    @use_kwargs(OrganizationPOSTRequestSchema, location='json')
    @marshal_with(OrganizationResponseSchema)
    def post(self, **kwargs):
        """
        Creates a new organization. It checks the system capacity, verifies user and organization details,
        and sets up the organization with a subscription, predefined roles, and optionally demo tasks.

        Args:
            **kwargs: Keyword arguments containing organization details from the request.

        Returns:
            Response: The response with the organization details and appropriate status code.
        """
        # Only users can create organizations
        if g.token_type != 'auth0_token':
            raise PermissionDeniedError('Only users can create organizations')

        # Check the number of organizations created so far. If the limit is reached, add user to the wait list.
        max_num_orgs = config.get('limits')['organizations']['num_organizations']
        if OrganizationDB.query().count() >= max_num_orgs:
            try:
                max_waitlist_len = config.get('limits')['organizations']['waitlist']
                if WaitList.query().count() >= max_waitlist_len:
                    oldest_entry = WaitList.query().order_by(WaitList.id_).first()
                    delete_from_db(oldest_entry)
                db_entry = WaitList(uuid=g.agent_uuid,
                                    email=g.token['email'],
                                    first_name=g.token['given_name'],
                                    last_name=g.token['family_name'],
                                    company=g.token['company'])
                save_to_db(db_entry)
            except Exception:
                pass
            err_msg = ('System capacity exceeded due to unexpectedly high demand. '
                       'We are working hard to scale our system to better serve you. '
                       'In the meantime, you have been added to our wait list and '
                       'will be notified as soon as we can accommodate your request.')
            return error_response(code=HTTP_SERVICE_UNAVAILABLE, message=err_msg)

        # Get user from token and check whether he/she belongs to another organization
        if UserDB.get_from_uuid(g.agent_uuid) is not None:
            raise DuplicateResourceError('You already belong to another organization')

        # Check if the organization already exists
        if OrganizationDB.get_from_id(kwargs['trn']) is not None:
            raise DuplicateResourceError(f'Organization "{kwargs["trn"]}" already exists')

        # Check organization's domain
        if kwargs['domain'][:kwargs['domain'].rindex('.')] in GENERIC_DOMAINS:
            raise UnprocessableRequestError('Generic domains like Gmail, Hotmail, Outlook, etc. are not supported')

        # Reject logo image file
        if 'logo' in kwargs:
            raise UnprocessableRequestError('You must create the organization before uploading its logo')

        # Save organization to database
        org_db_object = OrganizationDB(**kwargs)
        save_to_db(org_db_object)

        # Set organization (subscription, admin user, predefined roles, demo tasks)
        try:
            organization = OrganizationsView._set_organization(
                user_auth0_id=g.user_auth0_id,
                org_db_object=org_db_object,
                copy_demo_tasks=config.get('general')['enable_demo_tasks'])
        except Exception as e:
            delete_from_db(org_db_object)
            raise e

        # Set response
        response = jsonify(organization.dump())
        response.status_code = HTTP_POST_STATUS_CODE
        response.headers['Location'] = organization.url()
        return response


class OrganizationView(_OrganizationView):

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS], description='Note: only available to admins')
    @roles_required(roles=[ADMIN_ROLE])
    def delete(self, organization_id: str, resources: List[Resource]):
        """
        Deletes an organization. Only available to admins.

        Args:
            organization_id (str): The ID of the organization to delete.
            resources (List[Resource]): List of resource objects.

        Returns:
            Response: The response with appropriate status code.
        """
        return process_delete_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS])
    @marshal_with(OrganizationResponseSchema)
    def get(self, organization_id: str, resources: List[Resource]):
        """
        Retrieves organization details.

        Args:
            organization_id (str): The ID of the organization to retrieve.
            resources (List[Resource]): List of resource objects.

        Returns:
            Response: The response with organization details.
        """
        return process_get_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS], description='Note: only available to admins')
    @use_kwargs(OrganizationPUTRequestSchema, location='json')
    @marshal_with(OrganizationResponseSchema)
    @roles_required(roles=[ADMIN_ROLE])
    def put(self, organization_id: str, resources: List[Resource], **kwargs):
        """
        Updates an organization. Only available to admins.

        Args:
            organization_id (str): The ID of the organization to update.
            resources (List[Resource]): List of resource objects.
            **kwargs: Keyword arguments containing organization details from the request.

        Returns:
            Response: The response with updated organization details.
        """
        return process_post_or_put_request(agent=agent_from_token(), resource_or_model=resources[-1], json=kwargs)


class SubscriptionView(_SubscriptionView):

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS])
    @marshal_with(SubscriptionSchema)
    def get(self, organization_id: str, resources: List[Resource]):
        """
        Retrieves subscription details for an organization.

        Args:
            organization_id (str): The ID of the organization.
            resources (List[Resource]): List of resource objects.

        Returns:
            Response: The response with subscription details.
        """
        # Get subscription
        org = resources[-1]
        org_id = org.db_object().organization_id
        try:
            subscription = load_active_subscription(organization_id=org_id)
        except ResourceNotFoundError as e:
            return error_response(code=HTTP_NOT_FOUND_STATUS_CODE, message=str(e))
        except ResourceError as e:
            return error_response(code=HTTP_BAD_REQUEST_STATUS_CODE, message=str(e))

        # Fill response JSON
        plan_json = subscription.plan.to_dict()
        subscription_json = subscription.to_dict()

        usage_json = {
            'num_tasks': subscription_json.pop('num_tasks'),
            'num_deployments': subscription_json.pop('num_deployments'),
            'space_usage': subscription_json.pop('space_usage'),
            'num_users': subscription_json.pop('num_users'),
            'num_roles': subscription_json.pop('num_roles'),
            'num_collaborators': subscription_json.pop('num_collaborators'),
            'num_apps': subscription_json.pop('num_clients')
        }

        res_json = {'plan': plan_json, 'usage': usage_json, **subscription_json}

        now = datetime.utcnow()

        active_discounts = [
            x for x in subscription.discounts
            if (x.end_at is None or x.end_at > now) and (x.cancel_at is None or x.cancel_at > now)
        ]
        if len(active_discounts) > 0:
            res_json['discounts'] = [x.to_dict() for x in active_discounts]

        active_extras = [
            x for x in subscription.extras
            if (x.end_at is None or x.end_at > now) and (x.cancel_at is None or x.cancel_at > now)
        ]
        if len(active_extras) > 0:
            res_json['extras'] = [{**x.extra.to_dict(), **x.to_dict()} for x in active_extras]

        return jsonify(SubscriptionSchema().dump(res_json))


#########
# Users #
#########


class UsersView(_UserView):
    _url_params = paging_url_params(collection_name='users')

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS])
    @use_kwargs(_url_params, location='query')
    @marshal_with(UsersPage)
    def get(self, organization_id: str, resources: List[Resource], **kwargs):
        """
        Retrieves a list of users in an organization.

        Args:
            organization_id (str): The ID of the organization.
            resources (List[Resource]): List of resource objects.
            **kwargs: Keyword arguments for pagination (page number, per page, total count).

        Returns:
            Response: The response with paginated list of users.
        """
        query = UserDB.query().filter_by(organization_id=resources[-1].db_object().organization_id)
        # TODO: the line above will probably raise a `PermissionDeniedError`
        users = get_page_resources(query=query,
                                   page_number=kwargs['page'],
                                   per_page=kwargs['per_page'],
                                   total_count=kwargs['total_count'],
                                   resource_type=User,
                                   parents=resources)
        return jsonify(users)


class UserView(_UserView):

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS], description='Note: only available to admins and maintainers')
    @roles_required(roles=[ADMIN_ROLE, MAINTAINER_ROLE], require_all=False)
    def delete(self, organization_id: str, user_id: str, resources: List[Resource]):
        """
        Deletes a user from an organization. Only available to admins and maintainers.

        Args:
            organization_id (str): The ID of the organization.
            user_id (str): The ID of the user to delete.
            resources (List[Resource]): List of resource objects.

        Returns:
            Response: The response with appropriate status code.
        """
        return process_delete_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS])
    @marshal_with(UserResponseSchema)
    def get(self, organization_id: str, user_id: str, resources: List[Resource]):
        """
        Retrieves user details.

        Args:
            organization_id (str): The ID of the organization.
            user_id (str): The ID of the user to retrieve.
            resources (List[Resource]): List of resource objects.

        Returns:
            Response: The response with user details.
        """
        user: User = resources[-1]
        user_data: dict = user.dump(serialize=False)
        response = jsonify(user_data)
        response.headers['Location'] = user.url()
        return response


class UserRolesView(_UserView):

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS], description='Note: only available to admins and maintainers')
    @roles_required(roles=[ADMIN_ROLE, MAINTAINER_ROLE], require_all=False)
    def delete(self, organization_id: str, user_id: str, resources: List[Resource]):
        """
        Removes all roles from a user. Only available to admins and maintainers.

        Args:
            organization_id (str): The ID of the organization.
            user_id (str): The ID of the user whose roles are to be removed.
            resources (List[Resource]): List of resource objects.

        Returns:
            Response: The response with appropriate status code.
        """
        user = resources[-1]
        session_user = agent_from_token()
        is_session_user = user.db_object().user_id == session_user.user_id

        user_roles = get_user_roles(user=user.db_object())
        session_user_roles = get_user_roles(user=session_user)

        if ADMIN_ROLE in user_roles and not is_session_user:
            raise PermissionDeniedError('Admin role can only be removed by the corresponding user')
        elif MAINTAINER_ROLE in user_roles and not (ADMIN_ROLE in session_user_roles or is_session_user):
            raise PermissionDeniedError('Maintainer role can only be removed by an admin or by the corresponding user')

        user.db_object().roles.clear()
        user.persist()

        return Response(status=HTTP_DELETE_STATUS_CODE)

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS])
    @marshal_with(UserRolesResponseSchema)
    def get(self, organization_id: str, user_id: str, resources: List[Resource]):
        """
        Retrieves roles assigned to a user.

        Args:
            organization_id (str): The ID of the organization.
            user_id (str): The ID of the user whose roles are to be retrieved.
            resources (List[Resource]): List of resource objects.

        Returns:
            Response: The response with the roles assigned to the user.
        """
        organization = resources[0]
        user = resources[1]
        user_roles = [
            Role.get(agent=user.user(), db_object_or_id=x, parents=[organization]) for x in user.db_object().roles
        ]
        return {'user': user.public_id(), 'roles': dump(user_roles)}

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS],
         description='Assigns roles to user. Only available to admins and maintainers')
    @use_kwargs(UserRolesRequestSchema, location='json')
    @marshal_with(UserRolesResponseSchema)
    @roles_required(roles=[ADMIN_ROLE, MAINTAINER_ROLE], require_all=False)
    def post(self, organization_id: str, user_id: str, resources: List[Resource], **kwargs):
        """
        Assigns roles to a user. Only available to admins and maintainers.

        Args:
            organization_id (str): The ID of the organization.
            user_id (str): The ID of the user to assign roles to.
            resources (List[Resource]): List of resource objects.
            **kwargs: Keyword arguments containing roles to assign from the request.

        Returns:
            Response: The response with the roles assigned to the user and appropriate status code.
        """
        organization = resources[0]

        user = resources[1]
        user_roles = get_user_roles(user=user.db_object())

        session_user = agent_from_token()
        session_user_roles = get_user_roles(user=session_user)

        if ADMIN_ROLE in user_roles or MAINTAINER_ROLE in user_roles:
            raise PermissionDeniedError('Admins and maintainers cannot have more roles')

        if user == session_user:
            raise PermissionDeniedError('Roles cannot be auto-assigned')

        new_roles = [
            Role.get(agent=session_user, db_object_or_id=x, parents=[organization]).db_object()
            for x in kwargs.get('roles', [])
        ]

        admin_role = None
        maintainer_role = None

        for new_role in new_roles:

            # Skip already assigned roles
            if new_role in user.db_object().roles:
                continue

            # If maintainer role was assigned, skip remaining roles (except for admin role)
            if maintainer_role is not None and new_role.name != ADMIN_ROLE:
                continue

            # Only admins can assign the admin role
            if new_role.name == ADMIN_ROLE and ADMIN_ROLE not in session_user_roles:
                raise PermissionDeniedError('Admin roles can only be assigned by an admin')

            # Assign role (only if it's not the admin/maintainer role)
            if new_role.name == ADMIN_ROLE:
                admin_role = new_role
                break
            elif new_role.name == MAINTAINER_ROLE:
                maintainer_role = new_role
            else:
                user.db_object().roles.append(new_role)

        # If admin/maintainer role was assigned, clear user permissions and roles
        if admin_role is not None or maintainer_role is not None:

            user.db_object().roles.clear()
            user.db_object().permissions.delete()

            if admin_role is not None:
                user.db_object().roles.append(admin_role)
            else:
                user.db_object().roles.append(maintainer_role)

        # Persist changes
        user.persist()

        user_roles = [
            Role.get(agent=session_user, db_object_or_id=x, parents=[organization]) for x in user.db_object().roles
        ]

        return {'user': user.public_id(), 'roles': dump(user_roles)}, HTTP_POST_STATUS_CODE


class UserRoleView(_UserRoleView):

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS], description='Note: only available to admins and maintainers')
    @roles_required(roles=[ADMIN_ROLE, MAINTAINER_ROLE], require_all=False)
    def delete(self, organization_id: str, user_id: str, role_id: str):
        """
        Deletes a specific role from a user. Only available to admins and maintainers.

        Args:
            organization_id (str): The ID of the organization.
            user_id (str): The ID of the user whose role is to be deleted.
            role_id (str): The ID of the role to delete.

        Returns:
            Response: The response with appropriate status code.
        """
        session_user = agent_from_token()
        session_user_roles = get_user_roles(user=session_user)

        organization = Organization.get(agent=session_user, db_object_or_id=organization_id)
        user = User.get(agent=session_user, db_object_or_id=user_id, parents=[organization])
        role = Role.get(agent=session_user, db_object_or_id=role_id, parents=[organization])

        is_session_user = user.db_object().user_id == session_user.user_id

        if role.db_object().name == ADMIN_ROLE and not is_session_user:
            raise PermissionDeniedError('Admin role can only be removed by the corresponding user')
        elif role.db_object().name == MAINTAINER_ROLE and not (ADMIN_ROLE in session_user_roles or is_session_user):
            raise PermissionDeniedError('Maintainer role can only be removed by an admin or by the corresponding user')

        check_last_admin_deletion(user=user, user_roles=[role.db_object().name])

        if role.db_object() not in user.db_object().roles:
            raise ResourceNotFoundError('Role not assigned to user')

        user.db_object().roles.remove(role.db_object())
        user.persist()

        return Response(status=HTTP_DELETE_STATUS_CODE)


class UserPermissionsView(_UserView, PermissionAssignmentView):
    db_model = UserPermission

    def delete(self, organization_id: str, user_id: str, resources: List[Resource], **kwargs):
        """
        Deletes a user's permissions.

        Args:
            organization_id (str): The ID of the organization.
            user_id (str): The ID of the user whose permissions are to be deleted.
            resources (List[Resource]): List of resource objects.
            **kwargs: Additional keyword arguments.

        Returns:
            Response: The response with appropriate status code.
        """
        return super().delete(organization_id=organization_id, agent_id=user_id, resources=resources, **kwargs)

    def get(self, organization_id: str, user_id: str, resources: List[Resource], **kwargs):
        """
        Retrieves a user's permissions.

        Args:
            organization_id (str): The ID of the organization.
            user_id (str): The ID of the user whose permissions are to be retrieved.
            resources (List[Resource]): List of resource objects.
            **kwargs: Additional keyword arguments.

        Returns:
            Response: The response with user's permissions.
        """
        return super().get(organization_id=organization_id, agent_id=user_id, resources=resources, **kwargs)

    def post(self, organization_id: str, user_id: str, resources: List[Resource], **kwargs):
        """
        Assigns permissions to a user.

        Args:
            organization_id (str): The ID of the organization.
            user_id (str): The ID of the user to assign permissions to.
            resources (List[Resource]): List of resource objects.
            **kwargs: Additional keyword arguments.

        Returns:
            Response: The response with user's assigned permissions.
        """
        return super().post(organization_id=organization_id, agent_id=user_id, resources=resources, **kwargs)


@shared_task
def _invite_user(recipient_email: str, org_name: str, sender_user_auth0_id: str) -> None:
    """
    Sends an email invitation to a user to join an organization.

    This asynchronous function handles sending an email invitation to a user, inviting them to join a specified
    organization. It uses the sender's Auth0 user data to personalize the invitation and render an HTML email template.
    The invitation includes a link for the recipient to sign up.

    Args:
        recipient_email (str): The email address of the recipient being invited.
        org_name (str): The name of the organization the recipient is being invited to join.
        sender_user_auth0_id (str): The Auth0 ID of the user sending the invitation.

    Returns:
        None

    Raises:
        UnprocessableRequestError: If sending the email fails due to an invalid email address or other issues.
    """
    url: str = os.environ[ENV_AUTH0_SIGN_UP_REDIRECT_URL]
    sender_auth0_user_data: dict = User.download_auth0_user_data(auth0_id_or_email=sender_user_auth0_id)
    html = render_template('user_invitation.html',
                           app_name=API_NAME,
                           sender_name=f'{sender_auth0_user_data["first_name"]} {sender_auth0_user_data["last_name"]}',
                           sender_email=sender_auth0_user_data['email'],
                           organization=org_name,
                           ticket_url=url)
    subject = f'[{API_NAME}] You have been invited to join {org_name}'
    sender = (API_NAME, os.environ[ENV_NOTIFICATION_EMAIL])
    msg = Message(sender=sender, recipients=[recipient_email], subject=subject, html=html)
    with current_app.app_context():
        try:
            mail.send(msg)
        except Exception:
            raise UnprocessableRequestError('Failed to send invitation. Make sure you entered a valid email')


class UserInviteView(_UserInviteView):

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS], description='Note: only available to admins and maintainers')
    @use_kwargs(UserRequestSchema, location='json')
    @roles_required(roles=[ADMIN_ROLE, MAINTAINER_ROLE], require_all=False)
    def post(self, organization_id: str, resources: List[Resource], **kwargs):
        """
        Invites a new user to join the organization. Only available to admins and maintainers.

        Args:
            organization_id (str): The ID of the organization.
            resources (List[Resource]): List of resource objects.
            **kwargs: Keyword arguments containing user details from the request.

        Returns:
            Response: The response with appropriate status code.
        """
        org: Organization = resources[0]
        org_db_obj = resources[0].db_object()

        org.check_quota_usage(name='users', description='Maximum number of users')

        if kwargs['email'].split('@')[-1] != org_db_obj.domain:
            raise UnprocessableRequestError("All members must belong to the organization's domain. "
                                            'You can add users outside your organization as collaborators.')

        session_user = agent_from_token()
        assert isinstance(session_user, UserDB)

        has_invitation = InvitationDB.query().filter_by(email=kwargs['email'], status=InviteStatus.PENDING).first()
        if has_invitation is not None:
            raise DuplicateResourceError(f'A pending invitation already exists for the user with the email: '
                                         f'{kwargs["email"]}.')

        # TODO: This try should be deleted and `download_auth0_user_data` should not raise exception
        try:
            inv_user_auth0_data = User.download_auth0_user_data(auth0_id_or_email=kwargs['email'])
        except ResourceNotFoundError:
            inv_user_auth0_data = None

        if inv_user_auth0_data:
            inv_user_db_obj = UserDB.query().filter_by(auth0_id=inv_user_auth0_data['auth0_id']).first()
            if inv_user_db_obj is not None:
                raise DuplicateResourceError((f'User with email "{kwargs["email"]}" is already a member of '
                                              f'an organization'))
            else:
                raise InvalidDataError(f'There seems to be a problem with the user "{kwargs["email"]}". '
                                       f'Please, contact {os.environ[ENV_SUPPORT_EMAIL]}')

        _invite_user.delay(
            recipient_email=kwargs['email'],
            org_name=org_db_obj.name,
            sender_user_auth0_id=session_user.auth0_id,
        )

        invitation = InvitationDB(email=kwargs['email'], organization_id=org_db_obj.organization_id)
        save_to_db(invitation)

        return Response(status=204)


#########
# Roles #
#########


class RolesView(_RoleView):

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS])
    @marshal_with(RoleResponseSchema(many=True))
    def get(self, organization_id: str, resources: List[Resource]):
        """
        Retrieves a list of roles in an organization.

        Args:
            organization_id (str): The ID of the organization.
            resources (List[Resource]): List of resource objects.

        Returns:
            Response: The response with the list of roles.
        """
        db_objects = RoleDB.query().filter_by(organization_id=resources[-1].db_object().organization_id).all()
        roles = [Role.get(agent=agent_from_token(), db_object_or_id=x, parents=resources) for x in db_objects]
        return jsonify(dump(roles))

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS], description='Note: only available to admins and maintainers')
    @use_kwargs(RoleRequestSchema, location='json')
    @marshal_with(RoleResponseSchema)
    @roles_required(roles=[ADMIN_ROLE, MAINTAINER_ROLE], require_all=False)
    def post(self, organization_id: str, resources: List[Resource], **kwargs):
        """
        Creates a new role in an organization. Only available to admins and maintainers.

        Args:
            organization_id (str): The ID of the organization.
            resources (List[Resource]): List of resource objects.
            **kwargs: Keyword arguments containing role details from the request.

        Returns:
            Response: The response with the created role details.
        """
        return process_post_or_put_request(agent=agent_from_token(),
                                           resource_or_model=Role,
                                           parents=resources,
                                           json=kwargs)


class RoleView(_RoleView):

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS], description='Note: only available to admins and maintainers')
    @roles_required(roles=[ADMIN_ROLE, MAINTAINER_ROLE], require_all=False)
    def delete(self, organization_id: str, role_id: str, resources: List[Resource]):
        """
        Deletes a role from an organization. Only available to admins and maintainers.

        Args:
            organization_id (str): The ID of the organization.
            role_id (str): The ID of the role to delete.
            resources (List[Resource]): List of resource objects.

        Returns:
            Response: The response with appropriate status code.
        """
        return process_delete_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS])
    @marshal_with(RoleResponseSchema)
    def get(self, organization_id: str, role_id: str, resources: List[Resource]):
        """
        Retrieves role details.

        Args:
            organization_id (str): The ID of the organization.
            role_id (str): The ID of the role to retrieve.
            resources (List[Resource]): List of resource objects.

        Returns:
            Response: The response with role details.
        """
        return process_get_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS], description='Note: only available to admins and maintainers')
    @use_kwargs(RoleRequestSchema, location='json')
    @marshal_with(RoleResponseSchema)
    @roles_required(roles=[ADMIN_ROLE, MAINTAINER_ROLE], require_all=False)
    def put(self, organization_id: str, role_id: str, resources: List[Resource], **kwargs):
        """
        Updates a role in an organization. Only available to admins and maintainers.

        Args:
            organization_id (str): The ID of the organization.
            role_id (str): The ID of the role to update.
            resources (List[Resource]): List of resource objects.
            **kwargs: Keyword arguments containing role details from the request.

        Returns:
            Response: The response with the updated role details.
        """
        return process_post_or_put_request(agent=agent_from_token(), resource_or_model=resources[-1], json=kwargs)


class RoleUsersView(_RoleView):
    _url_params = paging_url_params(collection_name='users')

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS], description='Returns the users with the specified role')
    @use_kwargs(_url_params, location='query')
    @marshal_with(UsersPage)
    def get(self, organization_id: str, role_id: str, resources: List[Resource], **kwargs):
        """
        Retrieves a list of users with a specified role in an organization.

        Args:
            organization_id (str): The ID of the organization.
            role_id (str): The ID of the role.
            resources (List[Resource]): List of resource objects.
            **kwargs: Keyword arguments for pagination (page number, per page, total count).

        Returns:
            Response: The response with paginated list of users.
        """
        organization = resources[0]
        role = resources[1].db_object()
        role_users = get_page_resources(query=role.users,
                                        page_number=kwargs['page'],
                                        per_page=kwargs['per_page'],
                                        total_count=kwargs['total_count'],
                                        resource_type=User,
                                        parents=[organization])
        return jsonify(role_users)


class RolePermissionsView(_RoleView, PermissionAssignmentView):
    db_model = RolePermission

    def delete(self, organization_id: str, role_id: str, resources: List[Resource], **kwargs):
        """
        Deletes a role's permissions.

        Args:
            organization_id (str): The ID of the organization.
            role_id (str): The ID of the role whose permissions are to be deleted.
            resources (List[Resource]): List of resource objects.
            **kwargs: Additional keyword arguments.

        Returns:
            Response: The response with appropriate status code.
        """
        return super().delete(organization_id=organization_id, agent_id=role_id, resources=resources, **kwargs)

    def get(self, organization_id: str, role_id: str, resources: List[Resource], **kwargs):
        """
        Retrieves a role's permissions.

        Args:
            organization_id (str): The ID of the organization.
            role_id (str): The ID of the role whose permissions are to be retrieved.
            resources (List[Resource]): List of resource objects.
            **kwargs: Additional keyword arguments.

        Returns:
            Response: The response with role's permissions.
        """
        return super().get(organization_id=organization_id, agent_id=role_id, resources=resources, **kwargs)

    def post(self, organization_id: str, role_id: str, resources: List[Resource], **kwargs):
        """
        Assigns permissions to a role.

        Args:
            organization_id (str): The ID of the organization.
            role_id (str): The ID of the role to assign permissions to.
            resources (List[Resource]): List of resource objects.
            **kwargs: Additional keyword arguments.

        Returns:
            Response: The response with role's assigned permissions.
        """
        return super().post(organization_id=organization_id, agent_id=role_id, resources=resources, **kwargs)


#################
# Collaborators #
#################


class CollaboratorsView(_CollaboratorView):
    _url_params = paging_url_params(collection_name='collaborators')

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS])
    @use_kwargs(_url_params, location='query')
    @marshal_with(CollaboratorsPage)
    def get(self, organization_id: str, resources: List[Resource], **kwargs):
        """
        Retrieves a list of collaborators in an organization.

        Args:
            organization_id (str): The ID of the organization.
            resources (List[Resource]): List of resource objects.
            **kwargs: Keyword arguments for pagination (page number, per page, total count).

        Returns:
            Response: The response with paginated list of collaborators.
        """
        query = CollaboratorDB.query().filter_by(organization_id=resources[-1].db_object().organization_id)
        collaborators = get_page_resources(query=query,
                                           page_number=kwargs['page'],
                                           per_page=kwargs['per_page'],
                                           total_count=kwargs['total_count'],
                                           resource_type=Collaborator,
                                           parents=resources)
        return jsonify(collaborators)

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS], description='Note: only available to admins and maintainers')
    @use_kwargs(CollaboratorRequestSchema, location='json')
    @marshal_with(CollaboratorResponseSchema)
    @roles_required(roles=[ADMIN_ROLE, MAINTAINER_ROLE], require_all=False)
    def post(self, organization_id: str, resources: List[Resource], **kwargs):
        """
        Creates a new collaborator in an organization. Only available to admins and maintainers.

        Args:
            organization_id (str): The ID of the organization.
            resources (List[Resource]): List of resource objects.
            **kwargs: Keyword arguments containing collaborator details from the request.

        Returns:
            Response: The response with the created collaborator details.
        """
        return process_post_or_put_request(agent=agent_from_token(),
                                           resource_or_model=Collaborator,
                                           parents=resources,
                                           json=kwargs)


class CollaboratorView(_CollaboratorView):

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS], description='Note: only available to admins and maintainers')
    @roles_required(roles=[ADMIN_ROLE, MAINTAINER_ROLE], require_all=False)
    def delete(self, organization_id: str, collaborator_id: str, resources: List[Resource]):
        """
        Deletes a collaborator from an organization. Only available to admins and maintainers.

        Args:
            organization_id (str): The ID of the organization.
            collaborator_id (str): The ID of the collaborator to delete.
            resources (List[Resource]): List of resource objects.

        Returns:
            Response: The response with appropriate status code.
        """
        return process_delete_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS])
    @marshal_with(CollaboratorResponseSchema)
    def get(self, organization_id: str, collaborator_id: str, resources: List[Resource]):
        """
        Retrieves collaborator details.

        Args:
            organization_id (str): The ID of the organization.
            collaborator_id (str): The ID of the collaborator to retrieve.
            resources (List[Resource]): List of resource objects.

        Returns:
            Response: The response with collaborator details.
        """
        return process_get_request(resource=resources[-1])


class CollaboratorPermissionsView(_CollaboratorView, PermissionAssignmentView):
    db_model = UserPermission

    def delete(self, organization_id: str, collaborator_id: str, resources: List[Resource], **kwargs):
        """
        Deletes a collaborator's permissions.

        Args:
            organization_id (str): The ID of the organization.
            collaborator_id (str): The ID of the collaborator whose permissions are to be deleted.
            resources (List[Resource]): List of resource objects.
            **kwargs: Additional keyword arguments.

        Returns:
            Response: The response with appropriate status code.
        """
        user = self._get_collaborator_user(resources)
        resources[-1] = user
        return super().delete(organization_id=organization_id, agent_id=user.public_id(), resources=resources, **kwargs)

    def get(self, organization_id: str, collaborator_id: str, resources: List[Resource], **kwargs):
        """
        Retrieves a collaborator's permissions.

        Args:
            organization_id (str): The ID of the organization.
            collaborator_id (str): The ID of the collaborator whose permissions are to be retrieved.
            resources (List[Resource]): List of resource objects.
            **kwargs: Additional keyword arguments.

        Returns:
            Response: The response with collaborator's permissions.
        """
        user = self._get_collaborator_user(resources)
        resources[-1] = user
        return super().get(organization_id=organization_id, agent_id=user.public_id(), resources=resources, **kwargs)

    def post(self, organization_id: str, collaborator_id: str, resources: List[Resource], **kwargs):
        """
        Assigns permissions to a collaborator.

        Args:
            organization_id (str): The ID of the organization.
            collaborator_id (str): The ID of the collaborator to assign permissions to.
            resources (List[Resource]): List of resource objects.
            **kwargs: Additional keyword arguments.

        Returns:
            Response: The response with collaborator's assigned permissions.
        """
        user = self._get_collaborator_user(resources)
        resources[-1] = user
        return super().post(organization_id=organization_id, agent_id=user.public_id(), resources=resources, **kwargs)

    @staticmethod
    def _get_collaborator_user(resources: List[Resource]) -> User:
        """
        Retrieves the User object for a collaborator.

        Args:
            resources (List[Resource]): List of resource objects.

        Returns:
            User: The User object associated with the collaborator.
        """
        collaborator = resources[-1]
        return User.get(
            agent=collaborator.user(),
            db_object_or_id=User.db_model().get(user_id=collaborator.db_object().user_id),
            parents=collaborator.parents(),
            check_permissions=False,  # session user may not have access to collaborator's organization
            check_parents=False)


##################
# Clients (apps) #
##################


class ClientsView(_ClientView):

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS], description='Note: only available to admins and maintainers')
    @marshal_with(AppResponseSchema(many=True))
    @roles_required(roles=[ADMIN_ROLE, MAINTAINER_ROLE], require_all=False)
    def get(self, organization_id: str, resources: List[Resource]):
        """
        Retrieves a list of clients (apps) in an organization. Only available to admins and maintainers.

        Args:
            organization_id (str): The ID of the organization.
            resources (List[Resource]): List of resource objects.

        Returns:
            Response: The response with the list of clients.
        """
        # Get all the clients registered in the organization
        org_clients_db = ClientDB.query().filter_by(organization_id=resources[-1].db_object().organization_id).all()
        clients = []
        for client_db in org_clients_db:
            # Ignore reserved clients
            if client_db.client_id <= NUM_RESERVED_CLIENTS:
                continue
            # Ignore service clients
            if Service.query().filter_by(client_id=client_db.client_id).first() is not None:
                continue
            # Return client's JSON
            clients.append(Client.get(agent=agent_from_token(), db_object_or_id=client_db, parents=resources))
        return jsonify(dump(clients))

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS], description='Note: only available to admins and maintainers')
    @use_kwargs(AppRequestSchema, location='json')
    @marshal_with(AppResponseSchema)
    @roles_required(roles=[ADMIN_ROLE, MAINTAINER_ROLE], require_all=False)
    def post(self, organization_id: str, resources: List[Resource], **kwargs):
        """
        Creates a new client (app) in an organization. Only available to admins and maintainers.

        Args:
            organization_id (str): The ID of the organization.
            resources (List[Resource]): List of resource objects.
            **kwargs: Keyword arguments containing client (app) details from the request.

        Returns:
            Response: The response with the created client details.
        """
        return process_post_or_put_request(agent=agent_from_token(),
                                           resource_or_model=Client,
                                           parents=resources,
                                           json=kwargs)


class ClientView(_ClientView):

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS], description='Note: only available to admins and maintainers')
    @roles_required(roles=[ADMIN_ROLE, MAINTAINER_ROLE], require_all=False)
    def delete(self, organization_id: str, client_id: str, resources: List[Resource]):
        """
        Deletes a client (app) from an organization. Only available to admins and maintainers.

        Args:
            organization_id (str): The ID of the organization.
            client_id (str): The ID of the client to delete.
            resources (List[Resource]): List of resource objects.

        Returns:
            Response: The response with appropriate status code.
        """
        return process_delete_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS], description='Note: only available to admins and maintainers')
    @marshal_with(AppResponseSchema)
    @roles_required(roles=[ADMIN_ROLE, MAINTAINER_ROLE], require_all=False)
    def get(self, organization_id: str, client_id: str, resources: List[Resource]):
        """
        Retrieves client (app) details.

        Args:
            organization_id (str): The ID of the organization.
            client_id (str): The ID of the client to retrieve.
            resources (List[Resource]): List of resource objects.

        Returns:
            Response: The response with client details.
        """
        return process_get_request(resource=resources[-1])

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS], description='Note: only available to admins and maintainers')
    @use_kwargs(AppRequestSchema, location='json')
    @marshal_with(AppResponseSchema)
    @roles_required(roles=[ADMIN_ROLE, MAINTAINER_ROLE], require_all=False)
    def put(self, organization_id: str, client_id: str, resources: List[Resource], **kwargs):
        """
        Updates a client (app) in an organization. Only available to admins and maintainers.

        Args:
            organization_id (str): The ID of the organization.
            client_id (str): The ID of the client to update.
            resources (List[Resource]): List of resource objects.
            **kwargs: Keyword arguments containing client (app) details from the request.

        Returns:
            Response: The response with the updated client details.
        """
        return process_post_or_put_request(agent=agent_from_token(), resource_or_model=resources[-1], json=kwargs)


class ClientAPIKeyView(_ClientView):

    @staticmethod
    def _return_api_key(client: Client, status_code: int) -> Response:
        """
        Returns an API key for a client.

        Args:
            client (Client): The client object.
            status_code (int): The HTTP status code for the response.

        Returns:
            Response: The response with the API key details and appropriate status code.
        """
        dec_token = decode_api_key(api_key=client.db_object().api_key, verify=False)
        expire_at = datetime.utcfromtimestamp(dec_token['exp']) if 'exp' in dec_token else None
        api_key = {
            'token': client.db_object().api_key,
            'scopes': sorted(dec_token['scope'].split(' ')),
            'expire_at': (expire_at.strftime(DATETIME_FORMAT) + 'Z') if expire_at else None,
        }
        response = jsonify(api_key)
        response.status_code = status_code
        response.headers['Location'] = client.url() + '/' + ENDPOINT_CLIENT_API_KEY.split('/')[-1]
        return response

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS], description='Note: only available to admins and maintainers')
    @marshal_with(APIKeyResponseSchema)
    @roles_required(roles=[ADMIN_ROLE, MAINTAINER_ROLE], require_all=False)
    def get(self, organization_id: str, client_id: str, resources: List[Resource]):
        """
        Retrieves an API key for a client. Only available to admins and maintainers.

        Args:
            organization_id (str): The ID of the organization.
            client_id (str): The ID of the client whose API key is to be retrieved.
            resources (List[Resource]): List of resource objects.

        Returns:
            Response: The response with the API key details.
        """
        return ClientAPIKeyView._return_api_key(client=resources[-1], status_code=HTTP_GET_STATUS_CODE)

    @doc(tags=[SWAGGER_TAG_ORGANIZATIONS], description='Note: only available to admins and maintainers')
    @use_kwargs(APIKeyRequestSchema, location='json')
    @marshal_with(APIKeyResponseSchema)
    @roles_required(roles=[ADMIN_ROLE, MAINTAINER_ROLE], require_all=False)
    def put(self, organization_id: str, client_id: str, resources: List[Resource], **kwargs):
        """
        Updates an API key for a client. Only available to admins and maintainers.

        Args:
            organization_id (str): The ID of the organization.
            client_id (str): The ID of the client whose API key is to be updated.
            resources (List[Resource]): List of resource objects.
            **kwargs: Keyword arguments containing API key details from the request.

        Returns:
            Response: The response with the updated API key details.
        """
        client = resources[-1]
        if 'expire_at' in kwargs:
            never_expire = kwargs['expire_at'] is None  # if `None`, the API key will never expire
            expire_at = kwargs['expire_at']
        else:
            expire_at = None  # `update_api_key()` will set the default expiration
            never_expire = False
        client.db_object().update_api_key(scopes=kwargs.get('scopes'), expire_at=expire_at, never_expire=never_expire)
        return ClientAPIKeyView._return_api_key(client=client, status_code=HTTP_PUT_STATUS_CODE)
