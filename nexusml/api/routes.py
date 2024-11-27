import sys
from typing import Dict, Type

from flask import Blueprint
from flask import Flask
from flask_apispec import MethodResource

from nexusml.api.endpoints import ENDPOINT_AI_DEPLOYMENT
from nexusml.api.endpoints import ENDPOINT_AI_INFERENCE
from nexusml.api.endpoints import ENDPOINT_AI_MODEL
from nexusml.api.endpoints import ENDPOINT_AI_MODELS
from nexusml.api.endpoints import ENDPOINT_AI_PREDICTION_LOG
from nexusml.api.endpoints import ENDPOINT_AI_PREDICTION_LOGS
from nexusml.api.endpoints import ENDPOINT_AI_TESTING
from nexusml.api.endpoints import ENDPOINT_AI_TRAINING
from nexusml.api.endpoints import ENDPOINT_AL_SERVICE
from nexusml.api.endpoints import ENDPOINT_AL_SERVICE_NOTIFICATIONS
from nexusml.api.endpoints import ENDPOINT_AL_SERVICE_STATUS
from nexusml.api.endpoints import ENDPOINT_CL_SERVICE
from nexusml.api.endpoints import ENDPOINT_CL_SERVICE_NOTIFICATIONS
from nexusml.api.endpoints import ENDPOINT_CL_SERVICE_STATUS
from nexusml.api.endpoints import ENDPOINT_CLIENT
from nexusml.api.endpoints import ENDPOINT_CLIENT_API_KEY
from nexusml.api.endpoints import ENDPOINT_CLIENTS
from nexusml.api.endpoints import ENDPOINT_COLLABORATOR
from nexusml.api.endpoints import ENDPOINT_COLLABORATOR_PERMISSIONS
from nexusml.api.endpoints import ENDPOINT_COLLABORATORS
from nexusml.api.endpoints import ENDPOINT_EXAMPLE
from nexusml.api.endpoints import ENDPOINT_EXAMPLE_COMMENTS
from nexusml.api.endpoints import ENDPOINT_EXAMPLE_SHAPE
from nexusml.api.endpoints import ENDPOINT_EXAMPLE_SHAPES
from nexusml.api.endpoints import ENDPOINT_EXAMPLE_SLICE
from nexusml.api.endpoints import ENDPOINT_EXAMPLE_SLICES
from nexusml.api.endpoints import ENDPOINT_EXAMPLES
from nexusml.api.endpoints import ENDPOINT_INFERENCE_SERVICE
from nexusml.api.endpoints import ENDPOINT_INFERENCE_SERVICE_NOTIFICATIONS
from nexusml.api.endpoints import ENDPOINT_INFERENCE_SERVICE_STATUS
from nexusml.api.endpoints import ENDPOINT_INPUT_CATEGORIES
from nexusml.api.endpoints import ENDPOINT_INPUT_CATEGORY
from nexusml.api.endpoints import ENDPOINT_INPUT_ELEMENT
from nexusml.api.endpoints import ENDPOINT_INPUT_ELEMENTS
from nexusml.api.endpoints import ENDPOINT_METADATA_CATEGORIES
from nexusml.api.endpoints import ENDPOINT_METADATA_CATEGORY
from nexusml.api.endpoints import ENDPOINT_METADATA_ELEMENT
from nexusml.api.endpoints import ENDPOINT_METADATA_ELEMENTS
from nexusml.api.endpoints import ENDPOINT_MONITORING_SERVICE
from nexusml.api.endpoints import ENDPOINT_MONITORING_SERVICE_NOTIFICATIONS
from nexusml.api.endpoints import ENDPOINT_MONITORING_SERVICE_STATUS
from nexusml.api.endpoints import ENDPOINT_MONITORING_SERVICE_TEMPLATES
from nexusml.api.endpoints import ENDPOINT_MYACCOUNT
from nexusml.api.endpoints import ENDPOINT_MYACCOUNT_CLIENT_SETTINGS
from nexusml.api.endpoints import ENDPOINT_MYACCOUNT_NOTIFICATION
from nexusml.api.endpoints import ENDPOINT_MYACCOUNT_NOTIFICATIONS
from nexusml.api.endpoints import ENDPOINT_MYACCOUNT_ORGANIZATION
from nexusml.api.endpoints import ENDPOINT_MYACCOUNT_PERMISSIONS
from nexusml.api.endpoints import ENDPOINT_MYACCOUNT_ROLES
from nexusml.api.endpoints import ENDPOINT_MYACCOUNT_SETTINGS
from nexusml.api.endpoints import ENDPOINT_ORG_FILE
from nexusml.api.endpoints import ENDPOINT_ORG_FILE_PARTS
from nexusml.api.endpoints import ENDPOINT_ORG_FILE_PARTS_COMPLETION
from nexusml.api.endpoints import ENDPOINT_ORG_FILES
from nexusml.api.endpoints import ENDPOINT_ORG_LOCAL_FILE_STORE_DOWNLOAD
from nexusml.api.endpoints import ENDPOINT_ORG_LOCAL_FILE_STORE_MULTIPART_UPLOAD
from nexusml.api.endpoints import ENDPOINT_ORG_LOCAL_FILE_STORE_UPLOAD
from nexusml.api.endpoints import ENDPOINT_ORGANIZATION
from nexusml.api.endpoints import ENDPOINT_ORGANIZATIONS
from nexusml.api.endpoints import ENDPOINT_OUTPUT_CATEGORIES
from nexusml.api.endpoints import ENDPOINT_OUTPUT_CATEGORY
from nexusml.api.endpoints import ENDPOINT_OUTPUT_ELEMENT
from nexusml.api.endpoints import ENDPOINT_OUTPUT_ELEMENTS
from nexusml.api.endpoints import ENDPOINT_ROLE
from nexusml.api.endpoints import ENDPOINT_ROLE_PERMISSIONS
from nexusml.api.endpoints import ENDPOINT_ROLE_USERS
from nexusml.api.endpoints import ENDPOINT_ROLES
from nexusml.api.endpoints import ENDPOINT_SERVICES
from nexusml.api.endpoints import ENDPOINT_SUBSCRIPTION
from nexusml.api.endpoints import ENDPOINT_TAG
from nexusml.api.endpoints import ENDPOINT_TAGS
from nexusml.api.endpoints import ENDPOINT_TASK
from nexusml.api.endpoints import ENDPOINT_TASK_FILE
from nexusml.api.endpoints import ENDPOINT_TASK_FILE_PARTS
from nexusml.api.endpoints import ENDPOINT_TASK_FILE_PARTS_COMPLETION
from nexusml.api.endpoints import ENDPOINT_TASK_FILES
from nexusml.api.endpoints import ENDPOINT_TASK_LOCAL_FILE_STORE_DOWNLOAD
from nexusml.api.endpoints import ENDPOINT_TASK_LOCAL_FILE_STORE_MULTIPART_UPLOAD
from nexusml.api.endpoints import ENDPOINT_TASK_LOCAL_FILE_STORE_UPLOAD
from nexusml.api.endpoints import ENDPOINT_TASK_QUOTA_USAGE
from nexusml.api.endpoints import ENDPOINT_TASK_SCHEMA
from nexusml.api.endpoints import ENDPOINT_TASK_SETTINGS
from nexusml.api.endpoints import ENDPOINT_TASK_STATUS
from nexusml.api.endpoints import ENDPOINT_TASKS
from nexusml.api.endpoints import ENDPOINT_TESTING_SERVICE
from nexusml.api.endpoints import ENDPOINT_TESTING_SERVICE_NOTIFICATIONS
from nexusml.api.endpoints import ENDPOINT_TESTING_SERVICE_STATUS
from nexusml.api.endpoints import ENDPOINT_USER
from nexusml.api.endpoints import ENDPOINT_USER_INVITE
from nexusml.api.endpoints import ENDPOINT_USER_PERMISSIONS
from nexusml.api.endpoints import ENDPOINT_USER_ROLE
from nexusml.api.endpoints import ENDPOINT_USER_ROLES
from nexusml.api.endpoints import ENDPOINT_USERS
from nexusml.api.utils import config
from nexusml.api.utils import get_file_storage_backend
from nexusml.api.views import ai
from nexusml.api.views import examples
from nexusml.api.views import files
from nexusml.api.views import myaccount
from nexusml.api.views import organizations
from nexusml.api.views import services
from nexusml.api.views import tags
from nexusml.api.views import tasks
from nexusml.api.views.core import register_endpoints
# TODO: Check if we need to import all
# pylint: disable-next=wildcard-import, unused-wildcard-import
from nexusml.enums import FileStorageBackend


def _register_blueprint_endpoints(app: Flask, endpoint_urls: Dict[Type[MethodResource], str]):

    modules = set(sys.modules[view_class.__module__] for view_class in endpoint_urls)

    # Create blueprints
    blueprints = []

    for module in modules:
        module.blueprint = Blueprint(name=module.__name__.split('.')[-1], import_name=module.__name__)
        blueprints.append(module.blueprint)

    # Register endpoints
    register_endpoints(api_url=config.get('server')['api_url'], endpoint_urls=endpoint_urls)

    # Register blueprints
    for blueprint in blueprints:
        if blueprint.name not in app.blueprints:
            app.register_blueprint(blueprint)


def register_myaccount_endpoints(app: Flask):
    _register_blueprint_endpoints(app=app,
                                  endpoint_urls={
                                      myaccount.MyAccountView: ENDPOINT_MYACCOUNT,
                                      myaccount.SettingsView: ENDPOINT_MYACCOUNT_SETTINGS,
                                      myaccount.ClientSettingsView: ENDPOINT_MYACCOUNT_CLIENT_SETTINGS,
                                      myaccount.NotificationsView: ENDPOINT_MYACCOUNT_NOTIFICATIONS,
                                      myaccount.NotificationView: ENDPOINT_MYACCOUNT_NOTIFICATION,
                                      myaccount.OrganizationView: ENDPOINT_MYACCOUNT_ORGANIZATION,
                                      myaccount.RolesView: ENDPOINT_MYACCOUNT_ROLES,
                                      myaccount.PermissionsView: ENDPOINT_MYACCOUNT_PERMISSIONS
                                  })


def register_organizations_endpoints(app: Flask):
    _register_blueprint_endpoints(app=app,
                                  endpoint_urls={
                                      organizations.OrganizationsView: ENDPOINT_ORGANIZATIONS,
                                      organizations.OrganizationView: ENDPOINT_ORGANIZATION,
                                      organizations.SubscriptionView: ENDPOINT_SUBSCRIPTION,
                                      organizations.UsersView: ENDPOINT_USERS,
                                      organizations.UserView: ENDPOINT_USER,
                                      organizations.UserRolesView: ENDPOINT_USER_ROLES,
                                      organizations.UserRoleView: ENDPOINT_USER_ROLE,
                                      organizations.UserPermissionsView: ENDPOINT_USER_PERMISSIONS,
                                      organizations.UserInviteView: ENDPOINT_USER_INVITE,
                                      organizations.RolesView: ENDPOINT_ROLES,
                                      organizations.RoleView: ENDPOINT_ROLE,
                                      organizations.RoleUsersView: ENDPOINT_ROLE_USERS,
                                      organizations.RolePermissionsView: ENDPOINT_ROLE_PERMISSIONS,
                                      organizations.CollaboratorsView: ENDPOINT_COLLABORATORS,
                                      organizations.CollaboratorView: ENDPOINT_COLLABORATOR,
                                      organizations.CollaboratorPermissionsView: ENDPOINT_COLLABORATOR_PERMISSIONS,
                                      organizations.ClientsView: ENDPOINT_CLIENTS,
                                      organizations.ClientView: ENDPOINT_CLIENT,
                                      organizations.ClientAPIKeyView: ENDPOINT_CLIENT_API_KEY
                                  })


def register_files_endpoints(app: Flask):
    # Set local file store endpoints if local file store is enabled
    if get_file_storage_backend() == FileStorageBackend.LOCAL:
        local_file_store_endpoints = {
            # Organization files
            files.OrgLocalStoreDownloadView: ENDPOINT_ORG_LOCAL_FILE_STORE_DOWNLOAD,
            files.OrgLocalStoreUploadView: ENDPOINT_ORG_LOCAL_FILE_STORE_UPLOAD,
            files.OrgLocalStoreMultipartUploadView: ENDPOINT_ORG_LOCAL_FILE_STORE_MULTIPART_UPLOAD,
            # Task files
            files.TaskLocalStoreDownloadView: ENDPOINT_TASK_LOCAL_FILE_STORE_DOWNLOAD,
            files.TaskLocalStoreUploadView: ENDPOINT_TASK_LOCAL_FILE_STORE_UPLOAD,
            files.TaskLocalStoreMultipartUploadView: ENDPOINT_TASK_LOCAL_FILE_STORE_MULTIPART_UPLOAD,
        }
    else:
        local_file_store_endpoints = dict()

    # Register endpoints
    _register_blueprint_endpoints(app=app,
                                  endpoint_urls={
                                      files.OrgFilesView: ENDPOINT_ORG_FILES,
                                      files.OrgFileView: ENDPOINT_ORG_FILE,
                                      files.OrgFilePartsView: ENDPOINT_ORG_FILE_PARTS,
                                      files.OrgFilePartsCompletionView: ENDPOINT_ORG_FILE_PARTS_COMPLETION,
                                      files.TaskFilesView: ENDPOINT_TASK_FILES,
                                      files.TaskFileView: ENDPOINT_TASK_FILE,
                                      files.TaskFilePartsView: ENDPOINT_TASK_FILE_PARTS,
                                      files.TaskFilePartsCompletionView: ENDPOINT_TASK_FILE_PARTS_COMPLETION,
                                      **local_file_store_endpoints
                                  })


def register_tasks_endpoints(app: Flask):
    _register_blueprint_endpoints(app=app,
                                  endpoint_urls={
                                      tasks.TasksView: ENDPOINT_TASKS,
                                      tasks.TaskView: ENDPOINT_TASK,
                                      tasks.TaskSchemaView: ENDPOINT_TASK_SCHEMA,
                                      tasks.TaskSettingsView: ENDPOINT_TASK_SETTINGS,
                                      tasks.TaskStatusView: ENDPOINT_TASK_STATUS,
                                      tasks.TaskQuotaUsageView: ENDPOINT_TASK_QUOTA_USAGE,
                                      tasks.InputElementsView: ENDPOINT_INPUT_ELEMENTS,
                                      tasks.InputElementView: ENDPOINT_INPUT_ELEMENT,
                                      tasks.OutputElementsView: ENDPOINT_OUTPUT_ELEMENTS,
                                      tasks.OutputElementView: ENDPOINT_OUTPUT_ELEMENT,
                                      tasks.MetadataElementsView: ENDPOINT_METADATA_ELEMENTS,
                                      tasks.MetadataElementView: ENDPOINT_METADATA_ELEMENT,
                                      tasks.InputCategoriesView: ENDPOINT_INPUT_CATEGORIES,
                                      tasks.InputCategoryView: ENDPOINT_INPUT_CATEGORY,
                                      tasks.OutputCategoriesView: ENDPOINT_OUTPUT_CATEGORIES,
                                      tasks.OutputCategoryView: ENDPOINT_OUTPUT_CATEGORY,
                                      tasks.MetadataCategoriesView: ENDPOINT_METADATA_CATEGORIES,
                                      tasks.MetadataCategoryView: ENDPOINT_METADATA_CATEGORY
                                  })


def register_ai_endpoints(app: Flask):
    _register_blueprint_endpoints(app=app,
                                  endpoint_urls={
                                      ai.TrainingView: ENDPOINT_AI_TRAINING,
                                      ai.InferenceView: ENDPOINT_AI_INFERENCE,
                                      ai.TestingView: ENDPOINT_AI_TESTING,
                                      ai.AIModelsView: ENDPOINT_AI_MODELS,
                                      ai.AIModelView: ENDPOINT_AI_MODEL,
                                      ai.DeploymentView: ENDPOINT_AI_DEPLOYMENT,
                                      ai.PredictionLogsView: ENDPOINT_AI_PREDICTION_LOGS,
                                      ai.PredictionLogView: ENDPOINT_AI_PREDICTION_LOG,
                                      ai.PredictionLoggingView: ENDPOINT_AI_PREDICTION_LOGS
                                  })


def register_services_endpoints(app: Flask):
    _register_blueprint_endpoints(
        app=app,
        endpoint_urls={
            services.ServicesView: ENDPOINT_SERVICES,
            # Service views
            services.InferenceServiceView: ENDPOINT_INFERENCE_SERVICE,
            services.CLServiceView: ENDPOINT_CL_SERVICE,
            services.ALServiceView: ENDPOINT_AL_SERVICE,
            services.MonitoringServiceView: ENDPOINT_MONITORING_SERVICE,
            services.TestingServiceView: ENDPOINT_TESTING_SERVICE,
            # Status views
            services.InferenceServiceStatusView: ENDPOINT_INFERENCE_SERVICE_STATUS,
            services.CLServiceStatusView: ENDPOINT_CL_SERVICE_STATUS,
            services.ALServiceStatusView: ENDPOINT_AL_SERVICE_STATUS,
            services.MonitoringServiceStatusView: ENDPOINT_MONITORING_SERVICE_STATUS,
            services.TestingServiceStatusView: ENDPOINT_TESTING_SERVICE_STATUS,
            # Notifications views
            services.InferenceServiceNotificationsView: ENDPOINT_INFERENCE_SERVICE_NOTIFICATIONS,
            services.CLServiceNotificationsView: ENDPOINT_CL_SERVICE_NOTIFICATIONS,
            services.ALServiceNotificationsView: ENDPOINT_AL_SERVICE_NOTIFICATIONS,
            services.MonitoringServiceNotificationsView: ENDPOINT_MONITORING_SERVICE_NOTIFICATIONS,
            services.TestingServiceNotificationsView: ENDPOINT_TESTING_SERVICE_NOTIFICATIONS,
            # Service-specific views
            services.MonitoringServiceTemplatesView: ENDPOINT_MONITORING_SERVICE_TEMPLATES
        })


def register_examples_endpoints(app: Flask):
    _register_blueprint_endpoints(app=app,
                                  endpoint_urls={
                                      examples.ExamplesView: ENDPOINT_EXAMPLES,
                                      examples.ExampleView: ENDPOINT_EXAMPLE,
                                      examples.CommentsView: ENDPOINT_EXAMPLE_COMMENTS,
                                      examples.ShapesView: ENDPOINT_EXAMPLE_SHAPES,
                                      examples.ShapeView: ENDPOINT_EXAMPLE_SHAPE,
                                      examples.SlicesView: ENDPOINT_EXAMPLE_SLICES,
                                      examples.SliceView: ENDPOINT_EXAMPLE_SLICE
                                  })


def register_tags_endpoints(app: Flask):
    _register_blueprint_endpoints(app=app, endpoint_urls={tags.TagsView: ENDPOINT_TAGS, tags.TagView: ENDPOINT_TAG})
