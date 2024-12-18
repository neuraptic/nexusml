# System Information (config, feature flags, health or status, etc.)
ENDPOINT_SYS_CONFIG = '/config'

# My Account
ENDPOINT_MYACCOUNT = '/myaccount'
ENDPOINT_MYACCOUNT_ORGANIZATION = ENDPOINT_MYACCOUNT + '/organization'
ENDPOINT_MYACCOUNT_ROLES = ENDPOINT_MYACCOUNT + '/roles'
ENDPOINT_MYACCOUNT_PERMISSIONS = ENDPOINT_MYACCOUNT + '/permissions'
ENDPOINT_MYACCOUNT_NOTIFICATIONS = ENDPOINT_MYACCOUNT + '/notifications'
ENDPOINT_MYACCOUNT_NOTIFICATION = ENDPOINT_MYACCOUNT_NOTIFICATIONS + '/<notification_id>'
ENDPOINT_MYACCOUNT_SETTINGS = ENDPOINT_MYACCOUNT + '/settings'
ENDPOINT_MYACCOUNT_CLIENTS_SETTINGS = ENDPOINT_MYACCOUNT_SETTINGS + '/apps'
ENDPOINT_MYACCOUNT_CLIENT_SETTINGS = ENDPOINT_MYACCOUNT_CLIENTS_SETTINGS + '/<client_id>'

# Organizations
ENDPOINT_ORGANIZATIONS = '/organizations'
ENDPOINT_ORGANIZATION = ENDPOINT_ORGANIZATIONS + '/<organization_id>'
ENDPOINT_SUBSCRIPTION = ENDPOINT_ORGANIZATION + '/subscription'
ENDPOINT_USERS = ENDPOINT_ORGANIZATION + '/users'
ENDPOINT_USER = ENDPOINT_USERS + '/<user_id>'
ENDPOINT_USER_ROLES = ENDPOINT_USER + '/roles'
ENDPOINT_USER_ROLE = ENDPOINT_USER_ROLES + '/<role_id>'
ENDPOINT_USER_PERMISSIONS = ENDPOINT_USER + '/permissions'
ENDPOINT_USER_INVITE = ENDPOINT_USERS + '/invite'
ENDPOINT_ROLES = ENDPOINT_ORGANIZATION + '/roles'
ENDPOINT_ROLE = ENDPOINT_ROLES + '/<role_id>'
ENDPOINT_ROLE_USERS = ENDPOINT_ROLE + '/users'
ENDPOINT_ROLE_PERMISSIONS = ENDPOINT_ROLE + '/permissions'
ENDPOINT_COLLABORATORS = ENDPOINT_ORGANIZATION + '/collaborators'
ENDPOINT_COLLABORATOR = ENDPOINT_COLLABORATORS + '/<collaborator_id>'
ENDPOINT_COLLABORATOR_PERMISSIONS = ENDPOINT_COLLABORATOR + '/permissions'
ENDPOINT_CLIENTS = ENDPOINT_ORGANIZATION + '/apps'
ENDPOINT_CLIENT = ENDPOINT_CLIENTS + '/<client_id>'
ENDPOINT_CLIENT_API_KEY = ENDPOINT_CLIENT + '/api-key'

# Organization Files
ENDPOINT_ORG_FILES = ENDPOINT_ORGANIZATION + '/files'
ENDPOINT_ORG_FILE = ENDPOINT_ORG_FILES + '/<file_id>'
ENDPOINT_ORG_FILE_PARTS = ENDPOINT_ORG_FILE + '/parts'
ENDPOINT_ORG_FILE_PARTS_COMPLETION = ENDPOINT_ORG_FILE_PARTS + '/complete'

# Tasks
ENDPOINT_TASKS = '/tasks'
ENDPOINT_TASK = ENDPOINT_TASKS + '/<task_id>'
ENDPOINT_TASK_SCHEMA = ENDPOINT_TASK + '/schema'
ENDPOINT_TASK_SETTINGS = ENDPOINT_TASK + '/settings'
ENDPOINT_TASK_STATUS = ENDPOINT_TASK + '/status'
ENDPOINT_TASK_QUOTA_USAGE = ENDPOINT_TASK + '/usage'
ENDPOINT_INPUT_ELEMENTS = ENDPOINT_TASK_SCHEMA + '/inputs'
ENDPOINT_INPUT_ELEMENT = ENDPOINT_INPUT_ELEMENTS + '/<input_id>'
ENDPOINT_OUTPUT_ELEMENTS = ENDPOINT_TASK_SCHEMA + '/outputs'
ENDPOINT_OUTPUT_ELEMENT = ENDPOINT_OUTPUT_ELEMENTS + '/<output_id>'
ENDPOINT_METADATA_ELEMENTS = ENDPOINT_TASK_SCHEMA + '/metadata'
ENDPOINT_METADATA_ELEMENT = ENDPOINT_METADATA_ELEMENTS + '/<metadata_id>'
ENDPOINT_INPUT_CATEGORIES = ENDPOINT_INPUT_ELEMENT + '/categories'
ENDPOINT_INPUT_CATEGORY = ENDPOINT_INPUT_CATEGORIES + '/<category_id>'
ENDPOINT_OUTPUT_CATEGORIES = ENDPOINT_OUTPUT_ELEMENT + '/categories'
ENDPOINT_OUTPUT_CATEGORY = ENDPOINT_OUTPUT_CATEGORIES + '/<category_id>'
ENDPOINT_METADATA_CATEGORIES = ENDPOINT_METADATA_ELEMENT + '/categories'
ENDPOINT_METADATA_CATEGORY = ENDPOINT_METADATA_CATEGORIES + '/<category_id>'

# Services
ENDPOINT_SERVICES = ENDPOINT_TASK + '/services'
ENDPOINT_SERVICES_API_KEYS = ENDPOINT_SERVICES + '/api-keys'
ENDPOINT_INFERENCE_SERVICE = ENDPOINT_SERVICES + '/inference'
ENDPOINT_CL_SERVICE = ENDPOINT_SERVICES + '/continual-learning'
ENDPOINT_AL_SERVICE = ENDPOINT_SERVICES + '/active-learning'
ENDPOINT_MONITORING_SERVICE = ENDPOINT_SERVICES + '/monitoring'
ENDPOINT_TESTING_SERVICE = ENDPOINT_SERVICES + '/testing'
ENDPOINT_INFERENCE_SERVICE_STATUS = ENDPOINT_INFERENCE_SERVICE + '/status'
ENDPOINT_CL_SERVICE_STATUS = ENDPOINT_CL_SERVICE + '/status'
ENDPOINT_AL_SERVICE_STATUS = ENDPOINT_AL_SERVICE + '/status'
ENDPOINT_MONITORING_SERVICE_STATUS = ENDPOINT_MONITORING_SERVICE + '/status'
ENDPOINT_TESTING_SERVICE_STATUS = ENDPOINT_TESTING_SERVICE + '/status'
ENDPOINT_INFERENCE_SERVICE_NOTIFICATIONS = ENDPOINT_INFERENCE_SERVICE + '/notifications'
ENDPOINT_CL_SERVICE_NOTIFICATIONS = ENDPOINT_CL_SERVICE + '/notifications'
ENDPOINT_AL_SERVICE_NOTIFICATIONS = ENDPOINT_AL_SERVICE + '/notifications'
ENDPOINT_MONITORING_SERVICE_NOTIFICATIONS = ENDPOINT_MONITORING_SERVICE + '/notifications'
ENDPOINT_TESTING_SERVICE_NOTIFICATIONS = ENDPOINT_TESTING_SERVICE + '/notifications'
ENDPOINT_MONITORING_SERVICE_TEMPLATES = ENDPOINT_MONITORING_SERVICE + '/templates'

# Task Files
ENDPOINT_TASK_FILES = ENDPOINT_TASK + '/files'
ENDPOINT_TASK_FILE = ENDPOINT_TASK_FILES + '/<file_id>'
ENDPOINT_TASK_FILE_PARTS = ENDPOINT_TASK_FILE + '/parts'
ENDPOINT_TASK_FILE_PARTS_COMPLETION = ENDPOINT_TASK_FILE_PARTS + '/complete'

# AI
ENDPOINT_AI_TRAINING = ENDPOINT_TASK + '/train'
ENDPOINT_AI_INFERENCE = ENDPOINT_TASK + '/predict'
ENDPOINT_AI_TESTING = ENDPOINT_TASK + '/test'
ENDPOINT_AI_MODELS = ENDPOINT_TASK + '/models'
ENDPOINT_AI_MODEL = ENDPOINT_AI_MODELS + '/<model_id>'
ENDPOINT_AI_DEPLOYMENT = ENDPOINT_TASK + '/deployment'
ENDPOINT_AI_PREDICTION_LOGS = ENDPOINT_TASK + '/prediction-logs'
ENDPOINT_AI_PREDICTION_LOG = ENDPOINT_AI_PREDICTION_LOGS + '/<prediction_id>'

# Examples
ENDPOINT_EXAMPLES = ENDPOINT_TASK + '/examples'
ENDPOINT_EXAMPLE = ENDPOINT_EXAMPLES + '/<example_id>'
ENDPOINT_EXAMPLE_COMMENTS = ENDPOINT_EXAMPLE + '/comments'
ENDPOINT_EXAMPLE_SHAPES = ENDPOINT_EXAMPLE + '/shapes'
ENDPOINT_EXAMPLE_SHAPE = ENDPOINT_EXAMPLE_SHAPES + '/<shape_id>'
ENDPOINT_EXAMPLE_SLICES = ENDPOINT_EXAMPLE + '/slices'
ENDPOINT_EXAMPLE_SLICE = ENDPOINT_EXAMPLE_SLICES + '/<slice_id>'

# Tags
ENDPOINT_TAGS = ENDPOINT_TASK + '/tags'
ENDPOINT_TAG = ENDPOINT_TAGS + '/<tag_id>'

# Local File Storage Backend
ENDPOINT_ORG_LOCAL_FILE_STORE = ENDPOINT_ORGANIZATION + '/file-store'
ENDPOINT_ORG_LOCAL_FILE_STORE_DOWNLOAD = ENDPOINT_ORG_LOCAL_FILE_STORE + '/download/<file_id>'
ENDPOINT_ORG_LOCAL_FILE_STORE_UPLOAD = ENDPOINT_ORG_LOCAL_FILE_STORE + '/upload/<file_id>'
ENDPOINT_ORG_LOCAL_FILE_STORE_MULTIPART_UPLOAD = ENDPOINT_ORG_LOCAL_FILE_STORE + '/multipart-uploads/<upload_id>'

ENDPOINT_TASK_LOCAL_FILE_STORE = ENDPOINT_TASK + '/file-store'
ENDPOINT_TASK_LOCAL_FILE_STORE_DOWNLOAD = ENDPOINT_TASK_LOCAL_FILE_STORE + '/download/<file_id>'
ENDPOINT_TASK_LOCAL_FILE_STORE_UPLOAD = ENDPOINT_TASK_LOCAL_FILE_STORE + '/upload/<file_id>'
ENDPOINT_TASK_LOCAL_FILE_STORE_MULTIPART_UPLOAD = ENDPOINT_TASK_LOCAL_FILE_STORE + '/multipart-uploads/<upload_id>'
