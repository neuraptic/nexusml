from enum import Enum

############
# DATABASE #
############


class DBRelationshipType(Enum):
    """
    Enumeration of different relationship types.

    More info on basic relationship patterns at: https://docs.sqlalchemy.org/en/14/orm/basic_relationships.html

    Attributes:
        PARENT (int): Represents many-to-one relationship.
        CHILD (int): Represents one-to-many relationship.
        MANY_TO_MANY (int): Represents many-to-many relationship.
        ASSOCIATION_OBJECT (int): Represents many-to-many relationship with additional fields.
    """
    PARENT = 0
    CHILD = 1
    MANY_TO_MANY = 2
    ASSOCIATION_OBJECT = 3


#####################
# API (PERMISSIONS) #
#####################


class ResourceAction(Enum):
    """
    Enumeration of possible actions that can be performed on a resource.

    Attributes:
        CREATE (int): Represents creation action.
        READ (int): Represents read action.
        UPDATE (int): Represents update action.
        DELETE (int): Represents delete action.
    """
    CREATE = 0
    READ = 1
    UPDATE = 2
    DELETE = 3


class ResourceType(Enum):
    """
    Enumeration of different resource types that permissions can be applied to.

    Attributes:
        ORGANIZATION (int): Represents organization resource.
        TASK (int): Represents task resource.
        FILE (int): Represents file resource (specific to task files currently).
        AI_MODEL (int): Represents AI model resource.
        EXAMPLE (int): Represents example resource.
        PREDICTION (int): Represents prediction resource.
    """
    ORGANIZATION = 0
    TASK = 1
    FILE = 2
    AI_MODEL = 3
    EXAMPLE = 4
    PREDICTION = 5


###############
# API (FILES) #
###############


class FileType(Enum):
    """
    Enumeration of different file types.

    Attributes:
        DOCUMENT (int): Represents a document file type.
        IMAGE (int): Represents an image file type.
        VIDEO (int): Represents a video file type.
        AUDIO (int): Represents an audio file type.
    """
    DOCUMENT = 0
    IMAGE = 1
    VIDEO = 2
    AUDIO = 3


class FileFormat(Enum):
    """
    Enumeration of different file formats.

    Attributes:
        UNKNOWN (int): Represents an unknown file format
    """
    UNKNOWN = 0
    # TODO: add more formats
    # TXT = 1
    # PDF = 2
    # JPG = 3
    # ...


class OrgFileUse(Enum):
    """
    Enumeration of different organization file uses.

    Attributes:
        PICTURE (int): Represents a file used as a picture.
    """
    PICTURE = 0


class TaskFileUse(Enum):
    """
    Enumeration of different task file uses.

    Attributes:
        AI_MODEL (int): Represents a file used as an AI model.
        INPUT (int): Represents a file used as an input.
        OUTPUT (int): Represents a file used as an output.
        METADATA (int): Represents a file used as metadata.
        PICTURE (int): Represents a file used as a picture
    """
    AI_MODEL = 0
    INPUT = 1
    OUTPUT = 2
    METADATA = 3
    PICTURE = 4


class FileStorageBackend(Enum):
    """
    Enumeration of different file storage backends.

    Attributes:
        LOCAL (int): Represents the local file storage backend.
        S3 (int): Represents the AWS S3 file storage backend.
    """
    LOCAL = 0
    S3 = 1


##################################
# API (EXAMPLES AND PREDICTIONS) #
##################################


class ElementType(Enum):
    """
    Enumeration of different task element types.

    Attributes:
        INPUT (int): Represents an input element.
        OUTPUT (int): Represents an output element.
        METADATA (int): Represents a metadata element.
    """
    INPUT = 0
    OUTPUT = 1
    METADATA = 2


class ElementValueType(Enum):
    """
    Enumeration of different element value types.

    Attributes:
        BOOLEAN (int): Represents a boolean.
        INTEGER (int): Represents an integer.
        FLOAT (int): Represents a float.
        TEXT (int): Represents a text.
        DATETIME (int): Represents a datetime.
        CATEGORY (int): Represents a category.
        DOCUMENT_FILE (int): Represents a document file.
        IMAGE_FILE (int): Represents an image file.
        VIDEO_FILE (int): Represents a video file.
        AUDIO_FILE (int): Represents an audio file.
        SHAPE (int): Represents a shape.
        SLICE (int): Represents a slice.
    """
    BOOLEAN = 0
    INTEGER = 1
    FLOAT = 2
    TEXT = 3
    DATETIME = 4
    CATEGORY = 5
    DOCUMENT_FILE = 6
    IMAGE_FILE = 7
    VIDEO_FILE = 8
    AUDIO_FILE = 9
    SHAPE = 10
    SLICE = 11


class ElementMultiValue(Enum):
    """
    Enumeration of different element multi-value types.

    Attributes:
        UNORDERED (int): Represents an unordered multi-value.
        ORDERED (int): Represents an ordered multi-value.
        TIME_SERIES (int): Represents a time-series multi-value.
    """
    UNORDERED = 0
    ORDERED = 1
    TIME_SERIES = 2


class LabelingStatus(Enum):
    """
    Enumeration of different example labeling statuses.

    Attributes:
        UNLABELED (int): Represents unlabeled example.
        PENDING_REVIEW (int): Represents example pending review.
        LABELED (int): Represents labeled example.
        REJECTED (int): Represents rejected example.
    """
    UNLABELED = 0
    PENDING_REVIEW = 1
    LABELED = 2
    REJECTED = 3


class PredictionState(Enum):
    """
    Enumeration of different prediction states.

    Attributes:
        PENDING (int): Represents pending prediction state.
        IN_PROGRESS (int): Represents in-progress prediction state.
        COMPLETE (int): Represents complete prediction state.
        FAILED (int): Represents failed prediction state
    """
    PENDING = 0
    IN_PROGRESS = 1
    COMPLETE = 2
    FAILED = 3


#######################
# API (NOTIFICATIONS) #
#######################


class NotificationSource(Enum):
    """
    Enumeration of different notification sources.

    Attributes:
        APP (int): Represents an app notification.
        ORGANIZATION (int): Represents an organization notification.
        USER (int): Represents a user notification.
        TASK (int): Represents a task notification.
        FILE (int): Represents a file notification.
        AI_MODEL (int): Represents an AI model notification.
        EXAMPLE (int): Represents an example notification.
        TAG (int): Represents a tag notification.
        COMMENT (int): Represents a comment notification.
    """
    APP = 0
    ORGANIZATION = 1
    USER = 2
    TASK = 3
    FILE = 4
    AI_MODEL = 5
    EXAMPLE = 6
    TAG = 7
    COMMENT = 8


class NotificationEvent(Enum):
    """
    Enumeration of different notification events.

    Attributes:
        CREATION (int): Represents a creation event.
        UPDATE (int): Represents an update event.
        DELETION (int): Represents a deletion event.
        MESSAGE (int): Represents a direct message.
    """
    CREATION = 0
    UPDATE = 1
    DELETION = 2
    MESSAGE = 3


class NotificationType(Enum):
    """
    Enumeration of different notification types.

    Attributes:
        POLLING (int): Represents a polling notification.
        PUSH (int): Represents a push notification.
        EMAIL (int): Represents an email notification.
    """
    POLLING = 0
    PUSH = 1
    EMAIL = 2


class NotificationStatus(Enum):
    """
    Enumeration of different notification statuses.

    Attributes:
        UNSENT (int): Represents an unsent notification.
        SENDING (int): Represents a sending notification.
        SENT (int): Represents a sent notification
    """
    UNSENT = 0
    SENDING = 1
    SENT = 2


##########
# ENGINE #
##########


class ServiceType(Enum):
    """
    Enumeration of different service types.

    Attributes:
        INFERENCE (int): Represents the Inference Service
        CONTINUAL_LEARNING (int): Represents the Continual Learning (CL) Service
        ACTIVE_LEARNING (int): Represents the Active Learning (AL) Service
        MONITORING (int): Represents the Monitoring Service
        TESTING (int): Represents the Testing Service
    """
    INFERENCE = 0
    CONTINUAL_LEARNING = 1
    ACTIVE_LEARNING = 2
    MONITORING = 3
    TESTING = 4


class AIEnvironment(Enum):
    """
    Enumeration of different AI environments.

    Attributes:
        PRODUCTION (int): Represents the production AI environment.
        TESTING (int): Represents the testing AI environment.
    """
    PRODUCTION = 0
    TESTING = 1


class TrainingDevice(Enum):
    """
    Enumeration of different AI training devices.

    Attributes:
        CPU (int): Represents CPU training device.
        GPU (int): Represents GPU training device.
    """
    CPU = 0
    GPU = 1


class TaskType(Enum):
    """
    Enumeration of different task types.

    Attributes:
        UNKNOWN (int): Represents an unknown task type.
        CLASSIFICATION (int): Represents a classification task.
        REGRESSION (int): Represents a regression task.
        OBJECT_DETECTION (int): Represents an object detection task.
        OBJECT_SEGMENTATION (int): Represents an object segmentation task.
    """
    # Generic
    UNKNOWN = 0
    CLASSIFICATION = 1
    REGRESSION = 2
    # Vision
    OBJECT_DETECTION = 3
    OBJECT_SEGMENTATION = 4


class TaskTemplate(Enum):
    """
    Enumeration of different task schema templates.

    Attributes:
        IMAGE_CLASSIFICATION (int): Represents an image classification task schema template.
        IMAGE_REGRESSION (int): Represents an image regression task schema template.
        OBJECT_DETECTION (int): Represents an object detection task schema template.
        OBJECT_SEGMENTATION (int): Represents an object segmentation task schema template.
        TEXT_CLASSIFICATION (int): Represents a text classification task schema template.
        TEXT_REGRESSION (int): Represents a text regression task schema template.
        AUDIO_CLASSIFICATION (int): Represents an audio classification task schema template.
        AUDIO_REGRESSION (int): Represents an audio regression task schema template.
        TABULAR_CLASSIFICATION (int): Represents a tabular classification task schema template.
        TABULAR_REGRESSION (int): Represents a tabular regression task schema template.
        MULTIMODAL_CLASSIFICATION (int): Represents a multimodal classification task schema template.
        MULTIMODAL_REGRESSION (int): Represents a multimodal regression task schema template.
    """
    # Image
    IMAGE_CLASSIFICATION = 0
    IMAGE_REGRESSION = 1
    OBJECT_DETECTION = 2
    OBJECT_SEGMENTATION = 3
    # NLP
    TEXT_CLASSIFICATION = 4
    TEXT_REGRESSION = 5
    # Audio
    AUDIO_CLASSIFICATION = 6
    AUDIO_REGRESSION = 7
    # Tabular
    TABULAR_CLASSIFICATION = 8
    TABULAR_REGRESSION = 9
    # Multimodal
    MULTIMODAL_CLASSIFICATION = 10
    MULTIMODAL_REGRESSION = 11


class MLProblemType(Enum):
    """
    Enumeration of different Machine Learning (ML) problem types.

    Attributes:
        REGRESSION (int): Represents a regression problem.
        BINARY_CLASSIFICATION (int): Represents a binary classification problem.
        MULTI_CLASS_CLASSIFICATION (int): Represents a multi-class classification problem.
        OBJECT_DETECTION (int): Represents an object detection problem.
        OBJECT_SEGMENTATION (int): Represents an object segmentation problem.
    """
    REGRESSION = 1
    BINARY_CLASSIFICATION = 2
    MULTI_CLASS_CLASSIFICATION = 3
    OBJECT_DETECTION = 4
    OBJECT_SEGMENTATION = 5


class EngineType(Enum):
    """
    Enumeration of different engine types.

    Attributes:
        LOCAL (int): Represents a local engine.
        CLOUD (int): Represents a cloud engine.
        EDGE (int): Represents an edge engine.
    """
    LOCAL = 0
    CLOUD = 1
    EDGE = 2


##############
# API (MISC) #
##############


class ResourceCollectionOperation(Enum):
    """
    Enumeration of different resource collection operations.

    Attributes:
        APPEND (int): Represents an append operation.
        REMOVE (int): Represents a remove operation.
        REPLACE (int): Represents a replace operation.
        CLEAR (int): Represents a clear operation.
    """
    APPEND = 1
    REMOVE = 2
    REPLACE = 3
    CLEAR = 4


class InviteStatus(Enum):
    """
    Enumeration of different invite statuses.

    Attributes:
        PENDING (int): Represents a pending invite.
        ACCEPTED (int): Represents an accepted invite.
    """
    PENDING = 0
    ACCEPTED = 1


class Currency(Enum):
    """
    Enumeration of different currencies.

    Attributes:
        DOLLAR (int): Represents the dollar currency.
        EURO (int): Represents the euro currency.
    """
    DOLLAR = 0
    EURO = 1


class BillingCycle(Enum):
    """
    Enumeration of different billing cycles.

    Attributes:
        ANNUAL (int): Represents the annual billing cycle.
        MONTHLY (int): Represents the monthly billing cycle.
    """
    ANNUAL = 0
    MONTHLY = 1
