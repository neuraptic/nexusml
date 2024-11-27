import copy
import json
from typing import Any, Dict, List, Optional, Union

from nexusml.engine.exceptions import SchemaError
from nexusml.enums import TaskType


class Schema(object):
    """
    Class that wraps all the information about the schema
    """

    # TODO: The task schema should be directly derived from the database instead of being initialized from a JSON.
    #       The reason for using JSONs instead of database objects is that the engine was initially developed as a
    #       standalone API. Now that the engine is integrated with the main API, there is no need to use JSONs.

    def __init__(self, inputs: List[Dict], outputs: List[Dict], task_type: Optional[TaskType] = None):
        """
        Default constructor
        Args:
            inputs: (List[Dict]): list of all input elements where each element is a dictionary with the element info
            outputs (List[Dict]): list of all output elements where each element is a dictionary with the element info
            task_type (Optional[TaskType]): the task type associated with the schema
        """
        # Save a copy of the provided inputs and outputs
        self._inputs = copy.deepcopy(inputs)
        self._outputs = copy.deepcopy(outputs)
        # ToDo: for now we do not use metadata so we initialize it as an empty list.
        self._metadata = []

        # Infer the task type
        inferred_task_type = self._infer_task_type(inputs=self._inputs, outputs=self._outputs)

        # If a task type was provided, verify it is consistent with the provided inputs and outputs
        if task_type == TaskType.UNKNOWN:
            task_type = None

        if task_type is not None:
            inf_task_types = inferred_task_type if isinstance(inferred_task_type, list) else [inferred_task_type]
            if task_type not in inf_task_types:
                raise SchemaError(f'Provided task type "{task_type}" is not consistent '
                                  f'with the inferred task type "{inferred_task_type}")')

        # Save the task type
        self._task_type = task_type if task_type is not None else inferred_task_type

    @property
    def inputs(self) -> List[Dict]:
        """
        List of all input elements.

        Returns:
            List[Dict]: list of all input elements where each element is a dictionary with the element info
        """
        return self._inputs

    @property
    def outputs(self) -> List[Dict]:
        """
        List of all output elements.

        Returns:
            List[Dict]: list of all output elements where each element is a dictionary with the element info
        """
        return self._outputs

    @property
    def metadata(self) -> List[Dict]:
        """
        List of all metadata elements.

        Returns:
            List[Dict]: list of all metadata elements where each element is a dictionary with the element info
        """
        return self._metadata

    @property
    def task_type(self) -> TaskType:
        """
        Task type associated with the schema.

        Returns:
            TaskType: the task type associated with the schema
        """
        return self._task_type

    def required_inputs(self) -> List[Dict]:
        """
        Get all required inputs

        Returns:
            List[Dict]: list of all required input elements
        """
        return list(filter(lambda x: x['required'], self.inputs))

    def required_outputs(self) -> List[Dict]:
        """
        Get all required outputs

        Returns:
            List[Dict]: list of all required output elements
        """
        return list(filter(lambda x: x['required'], self.outputs))

    def categorical_inputs(self) -> List[Dict]:
        """
        Get all categorical inputs

        Returns:
            List[Dict]: list of all categorical input elements
        """
        return list(filter(lambda x: x['type'] == 'category', self.inputs))

    def categorical_outputs(self) -> List[Dict]:
        """
        Get all categorical outputs

        Returns:
            List[Dict]: list of all categorical output elements
        """
        return list(filter(lambda x: x['type'] == 'category', self.outputs))

    def shape_type_outputs(self) -> List[Dict]:
        """
        Get all shape-type outputs

        Returns:
            List[Dict]: list of all shape-type output elements
        """
        return list(filter(lambda x: x['type'] == 'shape', self.outputs))

    @staticmethod
    def _get_match_elements(element_list: List[Dict], key_name: str, search_value: Any) -> List[Dict]:
        """
        Get all elements that match a given key with a given value
        Args:
            element_list (List[Dict]): list of elements where make the search
            key_name (str): key name to make the match
            search_value (Any): the value to be match with the key

        Returns:
            List[Dict] all elements that matches that key == search_value
        """
        return list(filter(lambda x: x[key_name] == search_value, element_list))

    def get_match_inputs(self, key_name: str, search_value: Any) -> List[Dict]:
        """
        Get all inputs that match a given key with a given value
        Args:
            key_name (str): key name to make the match
            search_value (Any): the value to be match with the key

        Returns:
            List[Dict] all inputs that matches that key == search_value
        """
        return Schema._get_match_elements(element_list=self.inputs, key_name=key_name, search_value=search_value)

    def get_match_outputs(self, key_name: str, search_value: Any) -> List[Dict]:
        """
        Get all outputs that match a given key with a given value
        Args:
            key_name (str): key name to make the match
            search_value (Any): the value to be match with the key

        Returns:
            List[Dict] all outputs that matches that key == search_value
        """
        return Schema._get_match_elements(element_list=self.outputs, key_name=key_name, search_value=search_value)

    def get_inputs_by_uuid(self, input_uuid) -> List[Dict]:
        """
        Get all inputs that have the given uuid
        Args:
            input_uuid: uuid to be search

        Returns:
            List[Dict] all inputs that matches the given uuid
        """
        return self.get_match_inputs(key_name='uuid', search_value=input_uuid)

    def get_inputs_by_id(self, input_id) -> List[Dict]:
        """
        Get all inputs that have the given id
        Args:
            input_id: id to be search

        Returns:
            List[Dict] all inputs that matches the given id
        """
        return self.get_match_inputs(key_name='id', search_value=input_id)

    def get_inputs_by_name(self, input_name) -> List[Dict]:
        """
        Get all inputs that have the given name
        Args:
            input_name: name to be search

        Returns:
            List[Dict] all inputs that matches the given name
        """
        return self.get_match_inputs(key_name='name', search_value=input_name)

    def get_inputs_by_type(self, type_name) -> List[Dict]:
        """
        Get all inputs that have the given type
        Args:
            type_name: type to be search

        Returns:
            List[Dict] all inputs that matches the given type_name
        """
        return self.get_match_inputs(key_name='type', search_value=type_name)

    def get_outputs_by_uuid(self, output_uuid) -> List[Dict]:
        """
        Get all outputs that have the given uuid
        Args:
            output_uuid: uuid to be search

        Returns:
            List[Dict] all outputs that matches the given uuid
        """
        return self.get_match_outputs(key_name='uuid', search_value=output_uuid)

    def get_outputs_by_id(self, output_id) -> List[Dict]:
        """
        Get all outputs that have the given id
        Args:
            output_id: id to be search

        Returns:
            List[Dict] all output that matches the given id
        """
        return self.get_match_outputs(key_name='id', search_value=output_id)

    def get_outputs_by_name(self, output_name) -> List[Dict]:
        """
        Get all outputs that have the given name
        Args:
            output_name: name to be search

        Returns:
            List[Dict] all outputs that matches the given name
        """
        return self.get_match_outputs(key_name='name', search_value=output_name)

    def get_outputs_by_type(self, type_name) -> List[Dict]:
        """
        Get all output that have the given type
        Args:
            type_name: type to be search

        Returns:
            List[Dict] all output that matches the given type_name
        """
        return self.get_match_outputs(key_name='type', search_value=type_name)

    @staticmethod
    def _compare(schema1, schema2) -> bool:
        """
        Static method that will compare if two schemas are the same comparing if all elements
        of the schemas are the same
        Args:
            schema1 (Schema): the first schema to compare
            schema2 (Schema): the second schema to compare

        Returns:
            True if all elements of the two schemas are the same, False otherwise
        """
        # Inputs
        if len(schema1.inputs) != len(schema2.inputs):
            return False
        for i in range(len(schema1.inputs)):
            if schema1.inputs[i]['id'] != schema2.inputs[i]['id']:
                return False
            if ('multi_value' in schema1.inputs[i]) != ('multi_value' in schema2.inputs[i]):
                return False
            if 'multi_value' in schema1.inputs[i]:
                if schema1.inputs[i]['multi_value'] != schema2.inputs[i]['multi_value']:
                    return False
            if schema1.inputs[i]['name'] != schema2.inputs[i]['name']:
                return False
            if schema1.inputs[i]['required'] != schema2.inputs[i]['required']:
                return False
            if schema1.inputs[i]['type'] != schema2.inputs[i]['type']:
                return False
            if ('nullable' in schema1.inputs[i]) != ('nullable' in schema2.inputs[i]):
                return False
            if 'nullable' in schema1.inputs[i]:
                if schema1.inputs[i]['nullable'] != schema2.inputs[i]['nullable']:
                    return False
        # Outputs
        if len(schema1.outputs) != len(schema2.outputs):
            return False
        for i in range(len(schema1.outputs)):
            if schema1.outputs[i]['id'] != schema2.outputs[i]['id']:
                return False
            if ('multi_value' in schema1.outputs[i]) != ('multi_value' in schema2.outputs[i]):
                return False
            if 'multi_value' in schema1.outputs[i]:
                if schema1.outputs[i]['multi_value'] != schema2.outputs[i]['multi_value']:
                    return False
            if schema1.outputs[i]['name'] != schema2.outputs[i]['name']:
                return False
            if schema1.outputs[i]['required'] != schema2.outputs[i]['required']:
                return False
            if schema1.outputs[i]['type'] != schema2.outputs[i]['type']:
                return False
            if ('nullable' in schema1.outputs[i]) != ('nullable' in schema2.outputs[i]):
                return False
            if 'nullable' in schema1.outputs[i]:
                if schema1.outputs[i]['nullable'] != schema2.outputs[i]['nullable']:
                    return False
        # If the code reach this line, the schemas are equal
        return True

    def compare(self, other_schema) -> bool:
        """
        Function that will compare if two schemas are the same comparing if all elements
        of the schemas are the same

        Args:
            other_schema (Schema): the other schema to compare with

        Returns:
            True if all elements of the two schemas are the same, False otherwise
        """
        return Schema._compare(self, other_schema)

    @classmethod
    def create_schema_from_dict(cls, d: Dict):
        """
        Creates a Schema object from a dictionary that defines the schema.

        Args:
            d (Dict): dictionary defining the schema

        Returns:
            Schema object created from the provided dictionary
        """
        task_type_value = d['task_type']

        if isinstance(task_type_value, TaskType):
            task_type = task_type_value
        elif isinstance(task_type_value, str):
            task_type = TaskType[task_type_value.upper()]
        else:
            raise ValueError(f'Invalid task_type value: {task_type_value}')
        # Create a Schema object with the provided inputs and outputs
        return cls(inputs=d.get('inputs', []), outputs=d.get('outputs', []), task_type=task_type)

    @classmethod
    def create_schema_from_json(cls, json_file: str):
        """
        Creates a Schema object from a JSON file that contains the schema definition.

        Args:
            json_file (str): path to the JSON file that contains the schema definition

        Returns:
            Schema object created from the provided JSON file containing the schema definition.
        """
        # Read the JSON file and create the Schema object
        with open(json_file, 'r') as f:
            return cls.create_schema_from_dict(json.load(f))

    @staticmethod
    def _infer_task_type(inputs: List[Dict], outputs: List[Dict]) -> Union[TaskType, List[TaskType]]:
        """
        Infers the task type based on the provided inputs and outputs.
    
        Cases:
            - Classification:
                - inputs: [ boolean | float | integer | category | text | audio_file | image_file ] one or more
                - outputs: [ category ]
            - Regression:
                - inputs: [ boolean | float | integer | category | text | audio_file | image_file ] one or more
                - outputs: [ float | integer ]
            - Object detection:
                - inputs: [ image_file ]
                - outputs: [ shape ] and [ NotRequired(category) ]
            - Object segmentation:
                - inputs: [ image_file ]
                - outputs: [ shape ] and [ NotRequired(category) ]
    
        Other cases are "unknown".
    
        Returns:
            Union[TaskType, List[TaskType]]: the inferred task type. If a list is returned, it means that there is no 
                                             enough information to determine the exact task type, so it could be any 
                                             of the returned types.
        """
        # If there are no inputs and there are no outputs, the task type is unknown
        if len(inputs) == 0 or len(outputs) == 0:
            return TaskType.UNKNOWN
        if len(outputs) == 1:
            # If the output is not required or there are no required inputs, return unknown
            required_inputs = False
            for i in inputs:
                if i['required']:
                    required_inputs = True
                    break
            if not outputs[0]['required'] or not required_inputs:
                return TaskType.UNKNOWN
            else:
                # Otherwise, check the type
                output_type = outputs[0]['type']
                if output_type == 'category':
                    # If the output is category it is classification
                    return TaskType.CLASSIFICATION
                elif output_type == 'float' or output_type == 'integer':
                    # If float or integer, it is regression
                    return TaskType.REGRESSION
                else:
                    # Otherwise, it is unknown
                    return TaskType.UNKNOWN
        elif len(outputs) == 2:
            # It must be a shape output and a "not required" category
            if outputs[0]['type'] == 'category':
                category_idx = 0
                shape_idx = 1
            else:
                category_idx = 1
                shape_idx = 0
            # One is category. The other one must be shape and the category must be not required.
            # And we must have one single input of type "image_file".
            # Finally, shape and image_file must be required
            # Otherwise, unknown
            is_category = outputs[category_idx]['type'] == 'category'
            cat_not_required = not outputs[category_idx]['required']
            is_shape = outputs[shape_idx]['type'] == 'shape'
            shape_required = outputs[shape_idx]['required']
            single_input = len(inputs) == 1
            input_image_and_required = (inputs[0]['type'] == 'image_file') and inputs[0]['required']

            if (is_category and cat_not_required and is_shape and shape_required and single_input and
                    input_image_and_required):
                # It could be object detection or object segmentation
                return [TaskType.OBJECT_DETECTION, TaskType.OBJECT_SEGMENTATION]
            else:
                # Some requirement is missing, so it is unknown
                return TaskType.UNKNOWN
        else:
            return TaskType.UNKNOWN
