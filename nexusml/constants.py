############
# API INFO #
############

API_NAME = 'NexusML'
API_VERSION = 0

###########
# SWAGGER #
###########

SWAGGER_URL = '/swagger/'
SWAGGER_UI_URL = '/swagger-ui/'

SWAGGER_TAG_MYACCOUNT = 'My Account'
SWAGGER_TAG_ORGANIZATIONS = 'Organizations'
SWAGGER_TAG_TASKS = 'Tasks'
SWAGGER_TAG_SERVICES = 'Services'
SWAGGER_TAG_FILES = 'Files'
SWAGGER_TAG_AI = 'AI'
SWAGGER_TAG_EXAMPLES = 'Examples'
SWAGGER_TAG_TAGS = 'Tags'

##########
# ENGINE #
##########

# Set a maximum number of classes for not computing all metrics and the confusion matrix
# When dealing with too many classes, the computation of each value of the confusion matrix is too expensive
# So we set a threshold, and above this threshold we don't compute confusion matrix metrics and figure
ENGINE_MAX_NUM_CLASSES = 50

# Min overlap value to use for calculating metrics on object detection and object segmentation
# ODS -> Object Detection/Segmentation
ENGINE_ODS_MIN_OVERLAP = 0.5

# The classification models return the score for each class. Sometimes, there are too many classes for a response,
# so we have to limit them. This constant control the number of maximum classes score to return (as a default
# value, that can be modified as an argument of nexusml.engine.data.utils.predictions_to_example_format function)
ENGINE_LIMIT_SCORE_CLASSES = 50

# Because the model are too big to be uploaded in just one request, they are uploaded splitting them in several parts.
# This constant controls the size of each part
ENGINE_MODEL_CHUNK_SIZE = 100 * 1024 * 1024

##############
# FILE STORE #
##############

PREFIX_TASKS = 'tasks/'
PREFIX_TASK_MODELS = 'models/'
PREFIX_TASK_INPUTS = 'inputs/'
PREFIX_TASK_OUTPUTS = 'outputs/'
PREFIX_TASK_METADATA = 'metadata/'
PREFIX_TASK_PICTURES = 'pictures/'

PREFIX_ORGANIZATIONS = 'organizations/'
PREFIX_ORG_PICTURES = 'pictures/'

PREFIX_THUMBNAILS = 'thumbnails/'

MIN_FILE_PART_SIZE = 5 * 1024**2  # 5 MB

#########
# REDIS #
#########

REDIS_PREDICTION_LOG_BUFFER_KEY = 'nexusml_prediction_log_buffer'

##############################################################
# HTTP STATUS CODES                                          #
#                                                            #
# - https://datatracker.ietf.org/doc/html/rfc7231#section-6  #
# - https://developer.mozilla.org/en-US/docs/Web/HTTP/Status #
##############################################################

# Successful
HTTP_GET_STATUS_CODE = 200
HTTP_PUT_STATUS_CODE = 200
HTTP_POST_STATUS_CODE = 201
HTTP_DELETE_STATUS_CODE = 204

# Client Errors
HTTP_BAD_REQUEST_STATUS_CODE = 400
HTTP_UNAUTHORIZED_STATUS_CODE = 401
HTTP_FORBIDDEN_STATUS_CODE = 403
HTTP_NOT_FOUND_STATUS_CODE = 404
HTTP_METHOD_NOT_ALLOWED_STATUS_CODE = 405
HTTP_CONFLICT_STATUS_CODE = 409
HTTP_PAYLOAD_TOO_LARGE_STATUS_CODE = 413
HTTP_UNPROCESSABLE_ENTITY_STATUS_CODE = 422
HTTP_TOO_MANY_REQUESTS_STATUS_CODE = 429

# Server Errors
HTTP_NOT_IMPLEMENTED_STATUS_CODE = 501
HTTP_SERVICE_UNAVAILABLE = 503

############
# DATABASE #
############

DEFAULT_PLAN_ID = 1
FREE_PLAN_ID = 2

MYSQL_TINYINT_MAX_UNSIGNED = 255
MYSQL_SMALLINT_MAX_UNSIGNED = 65535
MYSQL_MEDIUMINT_MAX_UNSIGNED = 16777215
MYSQL_INT_MAX_UNSIGNED = 4294967295
MYSQL_BIGINT_MAX_UNSIGNED = 2**64 - 1

########
# MISC #
########

CONFIG_FILE = 'config.yaml'

UUID_VERSION = 4
NULL_UUID = '00000000-0000-0000-0000-000000000000'

ADMIN_ROLE = 'admin'
MAINTAINER_ROLE = 'maintainer'

NUM_RESERVED_CLIENTS = 100

DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%S'  # ISO 8601

GENERIC_DOMAINS = {'gmail', 'hotmail', 'outlook'}

INFERENCE_SERVICE_NAME = 'inference_service'
CL_SERVICE_NAME = 'cl_service'  # continual learning
AL_SERVICE_NAME = 'al_service'  # active learning
MONITORING_SERVICE_NAME = 'monitoring_service'
TESTING_SERVICE_NAME = 'testing_service'

DEFAULT_CELERY_BROKER_URL = 'redis://localhost:6379/0'
DEFAULT_CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

THUMBNAIL_SIZE = (128, 128)
