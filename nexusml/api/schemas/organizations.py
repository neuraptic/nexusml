import copy
from datetime import datetime

from marshmallow import fields
from marshmallow import post_dump
from marshmallow import pre_dump
from marshmallow import validate
from marshmallow import validates
from marshmallow import validates_schema
from marshmallow import ValidationError
from marshmallow_enum import EnumField

from nexusml.api.schemas.base import BaseSchema
from nexusml.api.schemas.base import PageSchema
from nexusml.api.schemas.base import ResourceRequestSchema
from nexusml.api.schemas.base import ResourceResponseSchema
from nexusml.api.schemas.files import OrganizationFileResponse
from nexusml.constants import AL_SERVICE_NAME
from nexusml.constants import API_NAME
from nexusml.constants import CL_SERVICE_NAME
from nexusml.constants import DATETIME_FORMAT
from nexusml.constants import INFERENCE_SERVICE_NAME
from nexusml.constants import MONITORING_SERVICE_NAME
from nexusml.constants import NULL_UUID
from nexusml.database.organizations import client_scopes
from nexusml.enums import BillingCycle
from nexusml.enums import Currency
from nexusml.enums import ResourceAction
from nexusml.enums import ResourceType

resource_types_str = ' | '.join([f'"{x.name.lower()}"' for x in ResourceType])
actions_str = ' | '.join([f'"{x.name.lower()}"' for x in ResourceAction])
"""
Organizations
"""


class OrganizationSchema(BaseSchema):
    name = fields.String(required=True, description='Name of the organization')
    address = fields.String(required=True, description='Business legal address')
    trn = fields.String(required=True,
                        description=('Tax Registration Number (TRN). It might also be known as the Tax Identification '
                                     'Number (TIN), Taxpayer Identification Number (TIN), Value-Added Tax (VAT) number,'
                                     ' VAT ID, VAT registration number, or Business Registration Number.'))


class OrganizationPOSTRequestSchema(OrganizationSchema, ResourceRequestSchema):
    domain = fields.String(required=True,
                           description='Domain (e.g. "neuraptic.ai")',
                           validate=validate.Regexp(regex='.+\\..+', error='Invalid domain'))


class OrganizationPUTRequestSchema(OrganizationSchema, ResourceRequestSchema):
    logo = fields.String(allow_none=True,
                         description=f'File containing the organization logo image. '
                         f'WARNING: the file must be previously uploaded to {API_NAME}')


class OrganizationResponseSchema(OrganizationSchema, ResourceResponseSchema):
    domain = fields.String(required=True, description='Domain of the organization')
    logo = fields.Nested(OrganizationFileResponse,
                         allow_none=True,
                         description='File containing the organization logo image')


#################
# Subscriptions #
#################


class SubscriptionSchema(BaseSchema):

    class Plan(BaseSchema):
        _plan_currencies = ' | '.join([f'"{x.name.lower()}"' for x in Currency])
        _plan_billings = ' | '.join([f'"{x.name.lower()}"' for x in BillingCycle])

        name = fields.String(required=True, description='Name of the plan to which the organization is subscribed')
        price = fields.Float(required=True, description='Price')
        currency = EnumField(Currency, required=True, description=f'Currency: {_plan_currencies}')
        billing_cycle = EnumField(BillingCycle, required=True, description=f'Billing cycle: {_plan_billings}')
        max_tasks = fields.Integer(required=True, description='Maximum number of tasks')
        max_deployments = fields.Integer(required=True, description='Maximum number of deployments')
        space_limit = fields.Integer(required=True, description='Space quota limit (in bytes)')
        max_users = fields.Integer(required=True, description='Maximum number of users in the organization')
        max_roles = fields.Integer(required=True, description='Maximum number of roles in the organization')
        max_collaborators = fields.Integer(required=True,
                                           description='Maximum number of collaborators in the organization')
        max_apps = fields.Integer(required=True, description='Maximum number of apps in the organization')

        @pre_dump
        def _rename_db_fields(self, data, **kwargs) -> dict:
            data = copy.deepcopy(data)
            if 'max_clients' in data:
                data['max_apps'] = data.pop('max_clients')
            return data

    class Usage(BaseSchema):
        num_tasks = fields.Integer(required=True, description='Current number of tasks')
        num_deployments = fields.Integer(required=True, description='Current number of deployments')
        space_usage = fields.Integer(description='Space usage (in bytes)')
        num_users = fields.Integer(required=True, description='Current number of users in the organization')
        num_roles = fields.Integer(required=True, description='Current number of roles in the organization')
        num_collaborators = fields.Integer(required=True,
                                           description='Current number of collaborators in the organization')
        num_apps = fields.Integer(required=True, description='Current number of apps in the organization')

        @pre_dump
        def _rename_db_fields(self, data, **kwargs) -> dict:
            data = copy.deepcopy(data)
            if 'num_clients' in data:
                data['num_apps'] = data.pop('num_clients')
            return data

    class Extras(BaseSchema):
        price = fields.Float(required=True, description='Price')
        start_at = fields.DateTime(format=DATETIME_FORMAT, required=True, description='Start datetime')
        end_at = fields.DateTime(format=DATETIME_FORMAT, description='End datetime')
        cancel_at = fields.DateTime(format=DATETIME_FORMAT, description='Cancellation datetime')
        extra_tasks = fields.Integer(description='Additional tasks')
        extra_deployments = fields.Integer(description='Additional deployments')
        extra_space = fields.Integer(description='Additional space (in bytes)')
        extra_users = fields.Integer(description='Additional users')
        extra_roles = fields.Integer(description='Additional roles')
        extra_collaborators = fields.Integer(description='Additional collaborators')
        extra_apps = fields.Integer(description='Additional apps')

        @pre_dump
        def _rename_db_fields(self, data, **kwargs) -> dict:
            data = copy.deepcopy(data)
            if 'extra_clients' in data:
                data['extra_apps'] = data.pop('extra_clients')
            return data

    class Discount(BaseSchema):
        percentage = fields.Integer(description='Discount percentage')
        start_at = fields.DateTime(format=DATETIME_FORMAT, required=True, description='Start datetime')
        end_at = fields.DateTime(format=DATETIME_FORMAT, description='End datetime')
        cancel_at = fields.DateTime(format=DATETIME_FORMAT, description='Cancellation datetime')

    plan = fields.Nested(Plan, required=True, description='Subscription plan')
    start_at = fields.DateTime(format=DATETIME_FORMAT, required=True, description='Start datetime')
    end_at = fields.DateTime(format=DATETIME_FORMAT, description='End datetime')
    cancel_at = fields.DateTime(format=DATETIME_FORMAT, description='Cancellation datetime')
    usage = fields.Nested(Usage, required=True, description='Usage')
    extras = fields.List(fields.Nested(Extras), description='Purchased additional quota')
    discounts = fields.List(fields.Nested(Discount, description='Discounts'))


#########
# Roles #
#########


class RoleSchema(BaseSchema):
    name = fields.String(required=True, description='Name of the role')
    description = fields.String(required=True, description='Description of the role')


class RoleRequestSchema(RoleSchema, ResourceRequestSchema):
    pass


class RoleResponseSchema(RoleSchema, ResourceResponseSchema):
    pass


#########
# Users #
#########


class UserRequestSchema(ResourceRequestSchema):
    email = fields.String(required=True,
                          description='Email address of the associated user',
                          validate=validate.Email(error='Invalid email address'))


class UserResponseSchema(ResourceResponseSchema):
    email = fields.String(required=True, description='Email address')
    first_name = fields.String(required=True, description='First name')
    last_name = fields.String(required=True, description='Last name')


class UserUpdateSchema(BaseSchema):
    first_name = fields.String(required=True, description='First name')
    last_name = fields.String(required=True, description='Last name')


class UsersPage(PageSchema):
    data = fields.List(fields.Nested(ResourceResponseSchema),
                       required=True,
                       description='Users in the requested page (or in the first page, if not specified)')


class UserRolesRequestSchema(BaseSchema):
    roles = fields.List(fields.String, required=True, description='Roles to be assigned to the user')


class UserRolesResponseSchema(BaseSchema):
    user = fields.String(required=True, description='User identifier')
    roles = fields.List(fields.Nested(RoleResponseSchema),
                        required=True,
                        allow_none=True,
                        description='Roles the user have')


#################
# Collaborators #
#################


class CollaboratorRequestSchema(ResourceRequestSchema):
    email = fields.String(required=True,
                          description='Email address',
                          validate=validate.Email(error='Invalid email address'))


class CollaboratorResponseSchema(ResourceResponseSchema):
    email = fields.String(description='User email address. Only visible to admins and maintainers')
    organization = fields.String(required=True, description='Name of the organization to which the user belongs')
    first_name = fields.String(required=True, description='First name')
    last_name = fields.String(required=True, description='Last name')


class CollaboratorsPage(PageSchema):
    data = fields.List(fields.Nested(ResourceResponseSchema),
                       required=True,
                       description='Collaborators in the requested page (or in the first page, if not specified)')


##################
# Clients (apps) #
##################


class AppSchema(BaseSchema):
    name = fields.String(required=True, description='Name of the app')
    description = fields.String(allow_none=True, description='Description of the app')


class AppRequestSchema(AppSchema, ResourceRequestSchema):
    icon = fields.String(allow_none=True,
                         description=f'File containing the app icon.'
                         f'WARNING: the file must be previously uploaded to {API_NAME}')

    @validates('name')
    def validate_name(self, name):
        if name in (INFERENCE_SERVICE_NAME, CL_SERVICE_NAME, AL_SERVICE_NAME, MONITORING_SERVICE_NAME):
            raise ValidationError(f'"{name}" is a reserved name')


class AppResponseSchema(AppSchema, ResourceResponseSchema):
    icon = fields.Nested(OrganizationFileResponse, allow_none=True, description='File containing the app icon')


class APIKeyRequestSchema(BaseSchema):
    _sc = ', '.join(f'"{x}"' for x in client_scopes)
    scopes = fields.List(fields.String,
                         description=(f'List of scopes.\nSupported scopes: {_sc}'
                                      '\nIf not provided, all scopes will be included.'))
    expire_at = fields.DateTime(format=DATETIME_FORMAT,
                                allow_none=True,
                                description=('Expiration datetime (UTC).'
                                             '\nIf not provided, the default expiration datetime will be used.'
                                             '\nIf `null`, the API key will never expire (not recommended).'))

    @validates('scopes')
    def validate_scopes(self, scopes):
        invalid_scopes = [x for x in scopes if x and x not in client_scopes]
        if invalid_scopes:
            raise ValidationError('Invalid scopes: ' + ', '.join(f'"{x}"' for x in invalid_scopes))

    @validates('expire_at')
    def validate_expire_at(self, expire_at):
        if expire_at is None:
            return
        if expire_at <= datetime.utcnow():
            raise ValidationError('Expiration datetime must be after current datetime')


class APIKeyResponseSchema(BaseSchema):
    token = fields.String(required=True,
                          description='JSON Web Token (JWT) bearer token to include in API requests made by the client')
    scopes = fields.List(fields.String, required=True, description='List of scopes')
    expire_at = fields.DateTime(format=DATETIME_FORMAT,
                                required=True,
                                allow_none=True,
                                description='Expiration datetime (UTC)')


###############
# Permissions #
###############


class PermissionSchema(BaseSchema):
    resource_uuid = fields.String(allow_none=True,
                                  description='Identifier of the resource to which the permission refers '
                                  '(only for resource-level permissions).'
                                  '\nWARNING: due to performance purposes, resource-level permissions '
                                  'are not allowed for *"resource_type=file"* '
                                  'and *"resource_type=example"*.')
    resource_type = EnumField(ResourceType, required=True, description='Resource type: ' + resource_types_str)
    action = EnumField(ResourceAction, required=True, description='Action: ' + actions_str)
    allow = fields.Boolean(required=True, description='true (grant) | false (deny)')

    @validates_schema
    def validate_semantics(self, data, **kwargs):
        """ Verifies if the specified permission makes sense. """
        rsrc_uuid = data.get('resource_uuid')
        if rsrc_uuid is not None:
            # Don't allow assigning creation permissions at resource level
            if data['action'] == ResourceAction.CREATE:
                raise ValidationError('Cannot assign creation permissions at resource level')
            # Disable resource-level permissions except for organizations and tasks to avoid performance issues
            if data['resource_type'] not in [ResourceType.ORGANIZATION, ResourceType.TASK]:
                raise ValidationError(f'Resource-level permissions not supported for resource type'
                                      f'"{data["resource_type"].name.lower()}"')

    @post_dump
    def _remove_null_uuid(self, data, **kwargs) -> dict:
        data = copy.deepcopy(data)
        if data.get('resource_uuid') == NULL_UUID:
            data.pop('resource_uuid')
        return data


class OrganizationPermissionSchema(PermissionSchema):
    organization = fields.String(required=True,
                                 description='Identifier of the organization in which the permission has effect')


class PermissionsPage(PageSchema):
    data = fields.List(fields.Nested(PermissionSchema),
                       required=True,
                       description='Permissions in the requested page (or in the first page, if not specified)')


class OrganizationPermissionsPage(PageSchema):
    data = fields.List(fields.Nested(OrganizationPermissionSchema),
                       required=True,
                       description='Permissions in the requested page (or in the first page, if not specified)')


class ResourceLevelPermissionsSchema(BaseSchema):

    class UserPermissions(BaseSchema):
        user = fields.String(required=True, description='User identifier')
        permissions = fields.List(fields.Nested(PermissionSchema),
                                  required=True,
                                  description='Permissions assigned to the user')

    class RolePermissions(BaseSchema):
        role = fields.String(required=True, description='Role identifier')
        permissions = fields.List(fields.Nested(PermissionSchema),
                                  required=True,
                                  description='Permissions assigned to the role')

    class CollaboratorPermissions(BaseSchema):
        collaborator = fields.String(required=True, description='Collaborator identifier')
        permissions = fields.List(fields.Nested(PermissionSchema),
                                  required=True,
                                  description='Permissions assigned to the collaborator')

    users = fields.List(fields.Nested(UserPermissions), required=True, allow_none=True)
    roles = fields.List(fields.Nested(RolePermissions), required=True, allow_none=True)
    collaborators = fields.List(fields.Nested(CollaboratorPermissions), required=True, allow_none=True)
