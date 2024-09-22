from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from torch.utils.data.dataloader import default_collate
from transformers.file_utils import PaddingStrategy

from nexusml.engine.data.transforms.nlp.text import BasicNLPTransform


@dataclass
class MultimodalDataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """
    # diccionario inputs transforms
    text_transform: BasicNLPTransform
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = 'pt'

    def __call__(self, features: List[Tuple[Dict[str, List[Dict[str, Any]]]]]) -> Tuple[Any, Any]:
        """

        Args:
            features (Dict[List[Dict[str, Any]]]): features to collate

        Returns:
            Collated inputs and outputs
        """

        # Receives a list of length batch size where each element is an element of the dataset
        # Create batchs with each input of the examples
        inputs = {}
        input_features = list(map(lambda x: x[0] if isinstance(x, tuple) else x, features))

        if 'text' in input_features[0]:
            input_batch = list(map(lambda x: x['text'],
                                   input_features))  # Get first element because it is a list of one element
            input_batch = self.text_transform.tokenizer_transform.tokenizer.pad(
                input_batch,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors)
            input_batch.data['text_tokens'] = input_batch.data['input_ids']
            input_batch.data['attn_mask'] = input_batch.data['attention_mask']
            del input_batch.data['input_ids']
            del input_batch.data['attention_mask']
            inputs['text'] = input_batch

        inputs['image'] = default_collate(list(map(lambda x: x['image'], input_features)))
        inputs['tabular'] = default_collate(list(map(lambda x: x['tabular'], input_features)))

        if isinstance(features[0], tuple):
            outputs = default_collate(list(map(lambda x: x[1], features)))

            return inputs, outputs
        else:
            return inputs
