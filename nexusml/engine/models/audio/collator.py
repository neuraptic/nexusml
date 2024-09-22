from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data.dataloader import default_collate
from torchaudio.transforms import Resample
from transformers import Wav2Vec2Processor
from transformers.file_utils import PaddingStrategy

from nexusml.engine.data.transforms.base import ElementTransforms


@dataclass
class DataCollatorWithPadding:
    """
    Data collator used for training.
    """

    input_transforms: ElementTransforms
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, batch: List[Dict[str, Any]]) -> Union[Tuple[Dict[str, Any], Any], Dict[str, Any]]:
        """
        Args:
            batch (list): List of samples from a batch.

        Returns:
            tuple: A tuple for data collation with the input and target values
        """
        inputs = {}
        input_features = list(map(lambda x: x[0] if isinstance(x, tuple) else x, batch))
        for k, v in self.input_transforms.element_transform_map.items():
            input_batch = list(map(lambda x: x[k][0], input_features))
            input_batch = v.processor_transform.processor.pad(input_batch,
                                                              padding=self.padding,
                                                              max_length=self.max_length,
                                                              pad_to_multiple_of=self.pad_multiple_of,
                                                              return_tensors=self.return_tensors)
            inputs[k] = input_batch

        if isinstance(batch[0], tuple):
            output_features = list(map(lambda x: x[1], batch))
            outputs = default_collate(output_features)
            return inputs, outputs
        else:
            return inputs


@dataclass
class DataCollatorProcessorWithPadding:
    """
    Data collator used for asr.
    """

    input_transforms: ElementTransforms
    processor: Wav2Vec2Processor
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, batch: List[Dict[str, Any]]) -> Tuple[Any, Any]:
        """
        Args:
            batch (list): List of samples from a batch.

        Returns:
            tuple: A tuple for data collation with the input and target values
        """
        inputs = {}
        if isinstance(batch[0], tuple):
            input_features = list(map(lambda x: x[0], batch))
        else:
            input_features = batch

        for k, v in self.input_transforms.element_transform_map.items():
            input_batch = list(map(lambda x: x[k][0], input_features))
            input_batch = list(
                map(lambda x: Resample(x[1], self.processor.current_processor.sampling_rate)(x[0]), input_batch))
            input_batch = list(map(lambda x: torch.squeeze(x).numpy(), input_batch))
            input_batch = self.processor(input_batch,
                                         sampling_rate=self.processor.current_processor.sampling_rate,
                                         return_tensors=self.return_tensors,
                                         padding=self.padding)
            inputs[k] = input_batch

        if isinstance(batch[0], tuple):
            output_features = list(map(lambda x: x[1], batch))

            outputs = []
            for output in output_features:
                for k, v in output.items():
                    outputs.append(v)

            return inputs, outputs
        else:
            return inputs, None
