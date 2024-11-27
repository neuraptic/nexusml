from abc import ABC
from abc import abstractmethod
from datetime import datetime
import math
import os
from typing import Dict, List, Type, TYPE_CHECKING, Union

from nexusml.database.buffers import ALBufferItemDB
from nexusml.database.buffers import BufferItemDB
from nexusml.database.buffers import MonBufferItemDB
from nexusml.database.core import delete_from_db
from nexusml.database.core import save_to_db
from nexusml.database.tasks import ElementDB
from nexusml.enums import ElementType
from nexusml.enums import ElementValueType
from nexusml.utils import FILE_TYPES

# Read for more info: https://peps.python.org/pep-0484/#runtime-or-type-checking
if TYPE_CHECKING:
    from nexusml.api.resources.ai import PredictionLog
    from nexusml.database.tasks import TaskDB

################
# BASE BUFFERS #
################


class BufferIO(ABC):
    """
    Class that performs the I/O operations related to reading, writing, and deleting buffer items.
    It operates on a specific type of buffer item (`BufferItemDB`) and is agnostic to the state or logic of the buffer.
    The class expects that the concrete buffer class will provide methods to process and manage buffer data.

    While `Buffer` manages buffer state, this class is only responsible for I/O operations.

    Args:
        task (TaskDB): TaskDB instance associated with this buffer I/O object.
        item_type (Type[BufferItemDB]): Type of buffer item the class operates on.
    """

    def __init__(self, task: 'TaskDB', item_type: Type[BufferItemDB]):
        self._task = task
        self._item_type = item_type

    def task(self) -> 'TaskDB':
        """Returns the task associated with this buffer."""
        return self._task

    def read_items(self) -> List[BufferItemDB]:
        """Returns items from the buffer sorted by relevance in descending order."""
        return self._item_type.get_items(task_id=self.task().task_id)

    def write_items(self, items: List[dict]):
        """
        Writes items to the buffer.

        Args:
            items (List[dict]): A list of dictionaries, each containing:
                                - "prediction_id": ID related to the prediction log in the database.
                                - "relevance": The relevance of the item.
                                - "size": The size of the item.
        """
        if not items:
            return
        db_objects = [
            self._item_type(task_id=self.task().task_id,
                            prediction_id=x['prediction_id'],
                            relevance=x['relevance'],
                            size=x['size']) for x in items
        ]
        save_to_db(objects=db_objects)

    def delete_items(self, items: List[BufferItemDB]):
        """
        Deletes items from the buffer.

        Args:
            items (List[BufferItemDB]): A list of buffer items to be deleted.
        """
        if not items:
            return

        # Delete items from database
        delete_from_db(objects=items)

    def delete_all_items(self):
        """Deletes all items in the buffer associated with the current task."""
        self.delete_items(items=self._item_type.query().filter_by(task_id=self.task().task_id).all())

    @abstractmethod
    def prepare_items(self, items: List[dict]) -> List[dict]:
        """
        Transforms items before they are written to the buffer.
        Must be implemented in concrete subclasses.

        Args:
            items (List[dict]): List of items to be transformed.

        Returns:
            List[dict]: Transformed items.
        """
        raise NotImplementedError()

    def item_size(self, item: dict) -> int:
        """
        Calculates the size of an item by including the size of its data (as JSON) and any referenced files.

        Args:
            item (dict): The item whose size is to be calculated.

        Returns:
            int: The size of the item in bytes.
        """
        files_by_use = self.item_files(item=item)
        files = files_by_use['inputs'] + files_by_use['outputs'] + files_by_use['metadata']
        return len(str(item)) + sum(self.file_size(file=x) for x in files)

    @abstractmethod
    def item_relevance(self, item: dict) -> Union[int, float]:
        """
        Abstract method to compute the relevance of an item.
        Must be implemented in subclasses.

        Args:
            item (dict): The item whose relevance is to be calculated.

        Returns:
            Union[int, float]: The relevance score of the item.
        """
        raise NotImplementedError()

    def item_files(self, item: dict) -> Dict[str, List[str]]:
        """
        Extracts the filenames referenced in the item data, grouped by usage type (inputs, outputs, metadata).

        Args:
            item (dict): The item containing file references.

        Returns:
            Dict[str, List[str]]: A dictionary with filenames categorized by "inputs", "outputs", and "metadata".
        """

        def _files(use: str) -> List[str]:
            """
            Returns the values of file-type elements for a given use (inputs, outputs, metadata).

            Args:
                use (str): The usage type of the files (inputs, outputs, metadata).

            Returns:
                List[str]: A list of file paths.
            """
            assert use in ['inputs', 'outputs', 'metadata']
            files = []
            for elem_value in item[use]:
                elem = elem_value['element']
                value = elem_value['value']
                element: ElementDB = ElementDB.get_from_id(id_value=elem, parent=self.task())
                if value and element.value_type in FILE_TYPES:
                    files.append(value)
            return files

        return {'inputs': _files('inputs'), 'outputs': _files('outputs'), 'metadata': _files('metadata')}

    @abstractmethod
    def file_size(self, file: str) -> int:
        """
        Abstract method to compute the size of a file.
        Must be implemented by subclasses.

        Args:
            file (str): The path to the file.

        Returns:
            int: Size of the file in bytes.
        """
        raise NotImplementedError()

    @abstractmethod
    def read_file(self, file: str) -> bytes:
        """
        Abstract method to read a file's contents.
        Must be implemented by subclasses.

        Args:
            file (str): The path to the file.

        Returns:
            bytes: The content of the file in bytes.
        """
        raise NotImplementedError()

    @abstractmethod
    def delete_files(self, files: List[str]):
        """
        Abstract method to delete a list of files.
        Must be implemented by subclasses.

        Args:
            files (List[str]): List of file paths to be deleted.
        """
        raise NotImplementedError()


class Buffer(ABC):
    """
    Base class representing a buffer for a task. This class manages buffer operations such as reading, writing, and
    clearing items, as well as handling buffer limits (maximum size and item count). Subclasses must define the specific
    database fields for counting buffer items and bytes.

    Args:
        buffer_io (BufferIO): The I/O handler responsible for performing operations on the buffer's items.
    """

    MAX_BUFFER_BYTES = 5 * 1024**3  # 5 GB TODO: Move this to conf
    MAX_BUFFER_ITEMS = 10**6  # 1 million: maximum number of items in the buffer TODO: Move this to conf

    def __init__(self, buffer_io: BufferIO):
        self._buffer_io = buffer_io

    @abstractmethod
    def num_items_db_field(self) -> str:
        """
        Abstract method to specify the database field used for counting buffer items.
        Must be implemented by subclasses.

        Returns:
            str: Name of the database field for item count.
        """
        raise NotImplementedError()

    @abstractmethod
    def num_bytes_db_field(self) -> str:
        """
        Abstract method to specify the database field used for counting buffer size in bytes.
        Must be implemented by subclasses.

        Returns:
            str: Name of the database field for buffer size.
        """
        raise NotImplementedError()

    def buffer_io(self) -> BufferIO:
        """Returns the BufferIO object responsible for performing I/O operations."""
        return self._buffer_io

    def task(self) -> 'TaskDB':
        """Returns the task associated with this buffer."""
        return self.buffer_io().task()

    def read(self) -> List[BufferItemDB]:
        """Reads all items currently in the buffer."""
        return self.buffer_io().read_items()

    def write(self, items: List['PredictionLog']):
        """
        Writes new items to the buffer, ensuring the buffer size and item count limits are respected.
        Items are sorted by relevance, and if limits are exceeded, the least relevant items are discarded.

        Args:
            items (List[PredictionLog]): A list of PredictionLog objects to write to the buffer.
        """
        if not items:
            return
        new_items = []
        for item in items:
            prepared_item = self.buffer_io().prepare_items(items=[item.dump(serialize=False)])[0]
            new_item = {
                'prediction_id': item.db_object().prediction_id,
                'relevance': self.buffer_io().item_relevance(item=prepared_item),
                'size': item.size()
            }
            new_items.append(new_item)
        # Check if buffer size limit is exceed.
        # In such a case, delete or discard the least relevant items
        buffer_items = getattr(self.task(), self.num_items_db_field()) + len(new_items)
        buffer_bytes = getattr(self.task(), self.num_bytes_db_field()) + sum(x['size'] for x in new_items)

        if buffer_items > self.MAX_BUFFER_ITEMS or buffer_bytes > self.MAX_BUFFER_BYTES:
            # Order both current and new items by relevance (descending)
            all_items = sorted(self.read() + new_items,
                               key=(lambda x: x['relevance'] if isinstance(x, dict) else x.relevance),
                               reverse=True)
            # Discard/Delete new/current items with the lowest relevance
            items_to_discard = []
            items_to_delete = []
            while buffer_items > self.MAX_BUFFER_ITEMS or buffer_bytes > self.MAX_BUFFER_BYTES:
                item = all_items.pop()
                if isinstance(item, dict):
                    items_to_discard.append(item)
                    buffer_bytes -= item['size']
                else:
                    items_to_delete.append(item)
                    buffer_bytes -= item.size
                buffer_items -= 1

            if items_to_discard:
                new_items = [x for x in new_items if x not in items_to_discard]
            if items_to_delete:
                self.buffer_io().delete_items(items=items_to_delete)
        # Save new items
        self.buffer_io().write_items(items=new_items)
        self._set_buffer_usage(num_items=buffer_items, num_bytes=buffer_bytes)

    def clear(self):
        """Clears all items from the buffer."""
        self.buffer_io().delete_all_items()
        self._set_buffer_usage(num_items=0, num_bytes=0)

    def _set_buffer_usage(self, num_items: int, num_bytes: int):
        """
        Updates the buffer usage statistics (number of items and size in bytes) in the database.

        Args:
            num_items (int): The number of items in the buffer.
            num_bytes (int): The size of the buffer in bytes.
        """
        setattr(self.task(), self.num_items_db_field(), num_items)
        setattr(self.task(), self.num_bytes_db_field(), num_bytes)
        save_to_db(objects=self.task())


##############################
# CONTINUAL LEARNING BUFFERS #
##############################


class CLBufferIO(BufferIO, ABC):
    """
    BufferIO implementation for Continual Learning (CL) Service.
    Prepares items and manages relevance scores specifically for Continual Learning Service processes.

    Note: The Continual Learning Service doesn't utilize buffers yet, so this is a placeholder.


    Args:
        task (TaskDB): TaskDB instance associated with the Continual Learning Service buffer.
    """

    def __init__(self, task: 'TaskDB'):
        super().__init__(task=task, item_type=BufferItemDB)

    def prepare_items(self, items: List[dict]) -> List[dict]:
        """
        Prepares items for the Continual Learning Service buffer without transformation.

        Args:
            items (List[dict]): List of items to be prepared.

        Returns:
            List[dict]: The unmodified list of items.
        """
        return items

    def item_relevance(self, item: dict) -> Union[int, float]:
        """
        Returns a relevance score of 0 for all items in the Continual Learning Service buffer.
        Since this service doesn't use buffers, the relevance is static.

        Args:
            item (dict): Item for which relevance is calculated.

        Returns:
            Union[int, float]: A constant relevance score of 0.
        """
        return 0


class CLBuffer(Buffer):
    """
    Buffer implementation for Continual Learning (CL) Service.
    Since the CL Service does not currently use a buffer, this class serves as a placeholder.
    The write operation is disabled, and buffer usage tracking is omitted.

    Args:
        buffer_io (CLBufferIO): The buffer I/O handler specific to Continual Learning Service.
    """

    def __init__(self, buffer_io: CLBufferIO):
        super().__init__(buffer_io=buffer_io)

    def write(self, items: List['PredictionLog']):
        """No-op: The Continual Learning Service doesn't write items to the buffer."""
        pass

    def num_items_db_field(self) -> str:
        """
        Returns an empty string since Continual Learning Service does not track item counts in the buffer.

        Returns:
            str: An empty string.
        """
        return ''

    def num_bytes_db_field(self) -> str:
        """
        Returns an empty string since Continual Learning Service does not track buffer size.

        Returns:
            str: An empty string.
        """
        return ''

    def _set_buffer_usage(self, num_items: int, num_bytes: int):
        """No-op: The Continual Learning Service does not track buffer usage."""
        pass


###########################
# ACTIVE LEARNING BUFFERS #
###########################


class ALBufferIO(BufferIO):
    """
    BufferIO implementation for Active Learning (AL) Service.
    Handles preparation of items and calculation of their relevance based on entropy of predicted values.

    Args:
        task (TaskDB): TaskDB instance associated with the Active Learning Service buffer.
    """

    def __init__(self, task: 'TaskDB'):
        super().__init__(task=task, item_type=ALBufferItemDB)

    def prepare_items(self, items: List[dict]) -> List[dict]:
        """
        Prepares items by retaining only necessary fields: 'inputs', 'outputs', and 'metadata'.

        Args:
            items (List[dict]): List of items to be prepared.

        Returns:
            List[dict]: The prepared list of items containing only 'inputs', 'outputs', and 'metadata'.
        """
        return [{k: v for k, v in x.items() if k in ['inputs', 'outputs', 'metadata']} for x in items]

    def item_relevance(self, item: dict) -> Union[int, float]:
        """
        Calculates the relevance of an item based on the entropy of its predicted values.

        Args:
            item (dict): The item for which relevance is calculated.

        Returns:
            Union[int, float]: The relevance score, based on entropy.
        """
        return self._entropy(predicted_values=item['outputs'])

    def _entropy(self, predicted_values: List[dict]) -> float:
        """
        Computes the average entropy of predicted values for each output, which is used to determine item relevance.

        Args:
            predicted_values (List[dict]): List of predicted values containing output UUIDs and associated scores.

        Returns:
            float: The average entropy across all outputs.
        """

        def _output_entropy(probs: list) -> float:
            """
            Calculates the entropy for a given list of probability scores.

            Args:
                probs (list): A list of probabilities for a single output.

            Returns:
                float: The entropy for the given probabilities.
            """
            try:
                total_prob = sum(probs)
                if total_prob != 1:
                    probs = [p / total_prob for p in probs]
                return -sum(p * math.log(p) for p in probs)
            except Exception:
                return 0.0

        # Get the scores for each output
        scores: list = list()
        for predicted_value in predicted_values:
            element: ElementDB = ElementDB.get_from_id(id_value=predicted_value['element'], parent=self.task())

            if (element and element.element_type == ElementType.OUTPUT and
                    element.value_type == ElementValueType.CATEGORY):
                scores.append(list(predicted_value['value']['scores'].values()))
        if not scores:
            return 0.0

        # Compute the entropy for each output
        total_entropy = 0
        num_outputs = 0
        for output_scores in scores:
            total_entropy += _output_entropy(output_scores)
            num_outputs += 1

        # Return the average entropy
        return total_entropy / num_outputs

    def read_file(self, file: str) -> bytes:
        """Placeholder for reading a file's contents. Currently not implemented."""
        pass

    def delete_files(self, files: List[str]):
        """Placeholder for deleting files. Currently not implemented."""
        pass

    def file_size(self, file: str) -> int:
        """Returns the size of a file in bytes by using the `os.stat` method."""
        return os.stat(file).st_size


class ALBuffer(Buffer):
    """
    Buffer implementation for Active Learning (AL) Service.
    Tracks the number of items and bytes in the buffer, ensuring that the buffer respects size and count limits.

    Args:
        buffer_io (BufferIO): The buffer I/O handler specific to Active Learning.
    """

    def __init__(self, buffer_io: BufferIO):
        super().__init__(buffer_io=buffer_io)

    def num_items_db_field(self) -> str:
        """
        Returns the database field name for tracking the number of items in the Active Learning Service buffer.

        Returns:
            str: The name of the field 'al_buffer_items'.
        """
        return 'al_buffer_items'

    def num_bytes_db_field(self) -> str:
        """
        Returns the database field name for tracking the size of the Active Learning Service buffer in bytes.

        Returns:
            str: The name of the field 'al_buffer_bytes'.
        """
        return 'al_buffer_bytes'


######################
# MONITORING BUFFERS #
######################


class MonBufferIO(BufferIO):
    """
    BufferIO implementation for Monitoring Service.
    Prepares items for the monitoring buffer and calculates relevance based on the current UTC timestamp.

    Args:
        task (TaskDB): TaskDB instance associated with the monitoring buffer.
    """

    def __init__(self, task: 'TaskDB'):
        super().__init__(task=task, item_type=MonBufferItemDB)

    def prepare_items(self, items: List[dict]) -> List[dict]:
        """
        Prepares monitoring items by retaining only necessary fields: 'inputs', 'outputs', and 'metadata'.

        Args:
            items (List[dict]): List of items to be prepared.

        Returns:
            List[dict]: The prepared list of items containing only 'inputs', 'outputs', and 'metadata'.
        """
        return [{k: v for k, v in x.items() if k in ['inputs', 'outputs', 'metadata']} for x in items]

    def item_relevance(self, item: dict) -> Union[int, float]:
        """
        Calculates the relevance of an item based on the current UTC timestamp.

        Args:
            item (dict): The item for which relevance is calculated.

        Returns:
            Union[int, float]: The UTC timestamp representing the item's relevance.
        """
        return datetime.utcnow().timestamp()

    def file_size(self, file: str) -> int:
        """Placeholder for calculating the file size. Currently not implemented."""
        pass

    def read_file(self, file: str) -> bytes:
        """Placeholder for reading a file's contents. Currently not implemented."""
        pass

    def delete_files(self, files: List[str]):
        """Placeholder for deleting files. Currently not implemented."""
        pass


class MonBuffer(Buffer):
    """
    Buffer implementation for Monitoring service.
    Tracks the number of items and bytes in the buffer and handles buffer size limits for monitoring operations.

    Args:
        buffer_io (MonBufferIO): The buffer I/O handler specific to monitoring.
    """

    def __init__(self, buffer_io: MonBufferIO):
        super().__init__(buffer_io=buffer_io)

    def num_items_db_field(self) -> str:
        """
        Returns the database field name for tracking the number of items in the monitoring buffer.

        Returns:
            str: The name of the field 'mon_buffer_items'.
        """
        return 'mon_buffer_items'

    def num_bytes_db_field(self) -> str:
        """
        Returns the database field name for tracking the size of the monitoring buffer in bytes.

        Returns:
            str: The name of the field 'mon_buffer_bytes'.
        """
        return 'mon_buffer_bytes'
