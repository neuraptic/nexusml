from typing import Any, Dict, List, Tuple, Union

from torch.utils.data.dataloader import default_collate
from transformers import PreTrainedTokenizer
from transformers import PreTrainedTokenizerFast


class MultimodalDataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    """

    def __init__(self, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]):
        """

        Args:
            tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
                The tokenizer used for encoding the data.
        """
        self.tokenizer = tokenizer

    def __call__(self, features: List[Tuple[Dict[str, List[Dict[str, Any]]]]]) -> Union[Tuple[Any, Any], Any]:
        """

        Args:
            features (Dict[List[Dict[str, Any]]]): features to collate

        Returns:
            Collated inputs and outputs
        """

        # Receives a list of length batch size where each element is an element of the dataset
        # Create batches with each input of the examples
        inputs = {}
        input_features = list(map(lambda x: x[0] if isinstance(x, tuple) else x, features))

        if 'text' in input_features[0]:
            input_batch = list(map(lambda x: x['text'], input_features))
            input_batch = self.tokenizer.pad(input_batch,
                                             padding=True,
                                             max_length=None,
                                             pad_to_multiple_of=None,
                                             return_tensors='pt')
            input_batch.data['text_tokens'] = input_batch.data['input_ids']
            input_batch.data['attn_mask'] = input_batch.data['attention_mask']
            del input_batch.data['input_ids']
            del input_batch.data['attention_mask']
            inputs['text'] = input_batch

        if 'image' in input_features[0]:
            inputs['image'] = default_collate(list(map(lambda x: x['image'], input_features)))

        if 'tabular' in input_features[0]:
            inputs['tabular'] = default_collate(list(map(lambda x: x['tabular'], input_features)))

        if isinstance(features[0], tuple):
            outputs = default_collate(list(map(lambda x: x[1], features)))

            return inputs, outputs
        else:
            return inputs
