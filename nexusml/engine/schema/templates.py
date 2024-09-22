from typing import Tuple

from nexusml.enums import PipelineType
from nexusml.enums import TaskType


def get_image_classification_regression_schema_template(num_inputs: int = 1,
                                                        task_type: TaskType = TaskType.CLASSIFICATION) -> dict:
    """
    Get the schema template for image classification or regression tasks

    Args:
        num_inputs (int): Number of input images
        task_type (TaskType): Task type, classification or regression

    Returns:
        dict: Schema template for image classification or regression tasks
    """
    if task_type not in [TaskType.CLASSIFICATION, TaskType.REGRESSION]:
        raise ValueError("'task_type' must me classification or regression")
    inputs = [{
        'name': f'image_{i}',
        'type': 'image_file',
        'nullable': False,
        'required': True
    } for i in range(1, num_inputs + 1)]
    if task_type == TaskType.CLASSIFICATION:
        outputs = [{'name': 'class', 'type': 'category', 'nullable': False, 'required': True}]
    else:
        outputs = [{'name': 'reg_value', 'type': 'float', 'nullable': False, 'required': True}]
    return {'inputs': inputs, 'outputs': outputs, 'task_type': task_type.name.lower()}


def get_object_detection_segmentation_schema_template(task_type: TaskType) -> dict:
    """
    Get the schema template for object detection or object segmentation tasks

    Args:
        task_type (TaskType): Task type, object detection or object segmentation

    Returns:
        dict: Schema template for object detection or object segmentation tasks
    """
    if task_type not in [TaskType.OBJECT_DETECTION, TaskType.OBJECT_SEGMENTATION]:
        raise ValueError("'task_type' must me object detection or object segmentation")
    inputs = [{'name': 'image', 'type': 'image_file', 'nullable': False, 'required': True}]
    outputs = [{
        'name': 'bounding_box' if task_type == TaskType.OBJECT_DETECTION else 'segment',
        'type': 'shape',
        'nullable': False,
        'required': True
    }, {
        'name': 'class',
        'type': 'category',
        'nullable': False,
        'required': False
    }]
    return {'inputs': inputs, 'outputs': outputs, 'task_type': task_type.name.lower()}


def get_nlp_classification_regression_schema_template(num_inputs: int = 1,
                                                      task_type: TaskType = TaskType.CLASSIFICATION) -> dict:
    """
    Get the schema template for NLP classification or regression tasks

    Args:
        num_inputs (int): Number of input texts
        task_type (TaskType): Task type, classification or regression

    Returns:
        dict: Schema template for NLP classification or regression tasks
    """
    if task_type not in [TaskType.CLASSIFICATION, TaskType.REGRESSION]:
        raise ValueError("'task_type' must me classification or regression")
    inputs = [{
        'name': f'text_{i}',
        'type': 'text',
        'nullable': False,
        'required': True
    } for i in range(1, num_inputs + 1)]
    if task_type == TaskType.CLASSIFICATION:
        outputs = [{'name': 'class', 'type': 'category', 'nullable': False, 'required': True}]
    else:
        outputs = [{'name': 'reg_value', 'type': 'float', 'nullable': False, 'required': True}]
    return {'inputs': inputs, 'outputs': outputs, 'task_type': task_type.name.lower()}


def get_audio_classification_regression_schema_template(num_inputs: int = 1,
                                                        task_type: TaskType = TaskType.CLASSIFICATION) -> dict:
    """
    Get the schema template for audio classification or regression tasks

    Args:
        num_inputs (int): Number of input audio files
        task_type (TaskType): Task type, classification or regression

    Returns:
        dict: Schema template for audio classification or regression tasks
    """
    if task_type not in [TaskType.CLASSIFICATION, TaskType.REGRESSION]:
        raise ValueError("'task_type' must me classification or regression")
    inputs = [{
        'name': f'audio_{i}',
        'type': 'audio_file',
        'nullable': False,
        'required': True
    } for i in range(1, num_inputs + 1)]
    if task_type == TaskType.CLASSIFICATION:
        outputs = [{'name': 'class', 'type': 'category', 'nullable': False, 'required': True}]
    else:
        outputs = [{'name': 'reg_value', 'type': 'float', 'nullable': False, 'required': True}]
    return {'inputs': inputs, 'outputs': outputs, 'task_type': task_type.name.lower()}


def get_tabular_classification_regression_schema_template(inputs: Tuple[str] = ('float', 'integer', 'category'),
                                                          task_type: TaskType = TaskType.CLASSIFICATION) -> dict:
    """
    Get the schema template for tabular classification or regression tasks

    Args:
        inputs (Tuple[str]): Tuple with the types of the input columns
        task_type (TaskType): Task type, classification or regression

    Returns:
        dict: Schema template for tabular classification or regression tasks
    """
    if len(inputs) == 0:
        raise ValueError('At least one input is needed')
    if task_type not in [TaskType.CLASSIFICATION, TaskType.REGRESSION]:
        raise ValueError("'task_type' must me classification or regression")
    if not all(list(map(lambda x: x in ['float', 'integer', 'category'], inputs))):
        raise ValueError("Valid input types are 'float', 'integer' and 'category")
    inputs = [{'name': f'col_{i}', 'type': inputs[i], 'nullable': False, 'required': True} for i in range(len(inputs))]
    if task_type == TaskType.CLASSIFICATION:
        outputs = [{'name': 'class', 'type': 'category', 'nullable': False, 'required': True}]
    else:
        outputs = [{'name': 'reg_value', 'type': 'float', 'nullable': False, 'required': True}]
    return {'inputs': inputs, 'outputs': outputs, 'task_type': task_type.name.lower()}


def get_task_schema_template(pipeline: PipelineType) -> dict:
    """
    Get the schema template for a given pipeline type

    Args:
        pipeline (PipelineType): Pipeline type

    Returns:
        dict: Schema template for the given pipeline type
    """
    if pipeline == PipelineType.IMAGE_CLASSIFICATION_REGRESSION:
        return get_image_classification_regression_schema_template()
    elif pipeline == PipelineType.OBJECT_DETECTION:
        return get_object_detection_segmentation_schema_template(TaskType.OBJECT_DETECTION)
    elif pipeline == PipelineType.OBJECT_SEGMENTATION:
        return get_object_detection_segmentation_schema_template(TaskType.OBJECT_SEGMENTATION)
    elif pipeline == PipelineType.NLP_CLASSIFICATION_REGRESSION:
        return get_nlp_classification_regression_schema_template()
    elif pipeline == PipelineType.AUDIO_CLASSIFICATION_REGRESSION:
        return get_audio_classification_regression_schema_template()
    elif pipeline == PipelineType.TABULAR_CLASSIFICATION_REGRESSION:
        return get_tabular_classification_regression_schema_template()
    else:
        raise ValueError(f'Missing template schema for pipeline {str(pipeline)}')
