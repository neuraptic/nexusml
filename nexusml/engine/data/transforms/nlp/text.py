from typing import List

import numpy as np

from nexusml.engine.data.transforms.base import Transform
from nexusml.engine.data.transforms.nlp.common import CleanerTransform
from nexusml.engine.data.transforms.nlp.common import TokenizerTransform


class BasicNLPTransform(Transform):
    """
    Basic NLP transform.
    Applies the cleaner and tokenizer transforms to the input data without modifying it
    """

    def __init__(self, path: str, train_tokenizer: bool = False, **kwargs):
        """
        Default constructor
        Args:
            path: path to the pretrained tokenizer
            train_tokenizer: train tokenizer or use a pretrained one
            **kwargs: other arguments
        """
        super().__init__(**kwargs)
        self.cleaner_transform = CleanerTransform()
        self.tokenizer_transform = TokenizerTransform(path=path, train_tokenizer=train_tokenizer)

    def fit(self, x: np.ndarray):
        """
        Fit the cleaner and tokenizer transforms
        Args:
            x (ndarray): data to fit

        Returns:

        """
        self.cleaner_transform.fit(x)
        self.tokenizer_transform.fit(x)

    def transform(self, x: np.ndarray) -> List:
        """
        Transform given data
        Args:
            x (ndarray): data to transform

        Returns:
            List with transformed data
        """
        x = self.cleaner_transform.transform(x)
        x = self.tokenizer_transform.transform(x)

        return x

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply inverse transform to given data
        Args:
            x (ndarray): data to transform (inverse)

        Returns:
            ndarray after apply inverse transform
        """
        pass
