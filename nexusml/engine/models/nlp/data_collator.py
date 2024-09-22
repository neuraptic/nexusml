from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from torch.utils.data.dataloader import default_collate
from transformers.file_utils import PaddingStrategy

from nexusml.engine.data.transforms.base import ElementTransforms


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that dynamically pads input sequences based on the given strategy, ensuring that all sequences
    in a batch have the same length for efficient processing. The padding strategy can be customized to fit the
    model's requirements, and the collator supports batching with various padding options.

    The class relies on the tokenizer associated with each input type to handle the padding logic. It can pad
    to the longest sequence in a batch, a maximum specified length, or not pad at all. This functionality
    is particularly useful when batching sequences of different lengths for model training or inference.

    Attributes:
        input_transforms (ElementTransforms): Defines the transformations for each input type, including tokenization.
        padding (Union[bool, str, PaddingStrategy]): The strategy to use for padding. Default is `True`
        (pad to the longest).
        max_length (Optional[int]): The maximum length to which sequences will be padded. If not set,
        padding will be dynamic.
        pad_to_multiple_of (Optional[int]): If set, pads sequences to a multiple of this value. This is useful for
                                             optimizing hardware performance on specific platforms.
        return_tensors (str): The type of tensors to return ("pt", "np", "tf"). Defaults to "pt" for PyTorch tensors.
    """
    input_transforms: ElementTransforms
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Tuple[Any, Any]:
        """
        Prepares a batch of input features by applying the appropriate tokenizer transformations and padding
        strategies. For each feature, the method pads the input sequences according to the specified strategy,
        either padding to the longest sequence in the batch, a maximum length, or based on other criteria.

        This method processes a batch of feature dictionaries, where each dictionary contains multiple inputs
        (e.g., text, labels). It pads the inputs using the tokenizer, applies input transformations, and
        constructs a tensor batch for model training or inference.

        Steps:
        1. Extracts the input features from the batch.
        2. Applies the relevant tokenizer transformation for each input.
        3. Pads the input sequences based on the collator's padding settings.
        4. If the features contain a tuple (input, label), batches the labels as well using default_collate.

        Args:
            features (List[Dict[str, Any]]): A list of dictionaries where each dictionary represents an example from
                                             the dataset. Each dictionary may contain inputs and optionally labels.

        Returns:
            Tuple[Any, Any]: A tuple containing the padded inputs and, if available, the batched labels. If the features
                             are not tuples, only the padded inputs are returned.
        """

        # Receives a list of length batch size where each element is an element of the dataset
        # Create batches with each input of the examples
        inputs = {}
        input_features = list(map(lambda x: x[0] if isinstance(x, tuple) else x, features))
        for k, v in self.input_transforms.element_transform_map.items():
            input_batch = list(map(lambda x: x[k][0],
                                   input_features))  # Get first element because it is a list of one element
            input_batch = v.tokenizer_transform.tokenizer.pad(input_batch,
                                                              padding=self.padding,
                                                              max_length=self.max_length,
                                                              pad_to_multiple_of=self.pad_to_multiple_of,
                                                              return_tensors=self.return_tensors)
            inputs[k] = input_batch

        if isinstance(features[0], tuple):
            outputs = default_collate(list(map(lambda x: x[1], features)))

            return inputs, outputs
        else:
            return inputs
