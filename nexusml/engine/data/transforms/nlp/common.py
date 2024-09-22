import re
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer
from unidecode import unidecode

from nexusml.engine.data.transforms.base import Transform


class CleanerTransform(Transform):
    """
    NLP cleaner transform.
    Applies a text cleaner to the input data including lowercase and remove accents, symbols and special characters
    """

    def __init__(self, **kwargs):
        """
        Default constructor
        Args:
            **kwargs: other arguments
        """
        super().__init__(**kwargs)

        self.REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        self.BAD_SYMBOLS_RE = re.compile('[^0-9a-z \n]')

    def fit(self, x: np.ndarray):
        """
        Fit the cleaner transform
        Args:
            x (np.ndarray): data to fit

        Returns:

        """
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform given data
        Args:
            x (np.ndarray): data to transform

        Returns:
            np.ndarray with transformed data
        """
        return np.array(list(map(self._transform, x)))

    def _transform(self, x: str) -> str:
        """
        Transform given text
        Args:
            x (str): text to clean

        Returns:
            str with cleaned text
        """

        x = x.lower()
        x = unidecode(x)
        x = self.REPLACE_BY_SPACE_RE.sub(' ', x)
        x = self.BAD_SYMBOLS_RE.sub('', x)
        return x

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply inverse transform to given data
        Args:
            x (np.ndarray): data to transform (inverse)

        Returns:
            np.ndarray after apply inverse transform
        """
        pass


class TokenizerTransform(Transform):
    """
    Basic tokenizer transform.
    Applies the tokenizer transform to the input data encoding it
    """

    def __init__(self, path: str, train_tokenizer: bool = False, **kwargs):
        """
        Default constructor
        Args:
            path: path to the pretrained tokenizer
            train_tokenizer: train the tokenizer or use a pretrained one
            **kwargs: other arguments
        """
        super().__init__(**kwargs)

        self.train_tokenizer = train_tokenizer

        if not train_tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
        else:
            raise NotImplementedError()

    def fit(self, x: np.ndarray):
        """
        Fit the tokenizer transform
        Args:
            x (np.ndarray): data to fit

        Returns:

        """
        pass

    def transform(self, x: np.ndarray) -> List:
        """
        Transform given data
        Args:
            x (np.ndarray): data to transform

        Returns:
            List with transformed data
        """
        return list(map(self._transform, x))

    def _transform(self, x: str) -> torch.Tensor:
        """
        Transform given text by encoding it with the tokenizer
        Args:
            x (str): text to transform

        Returns:
            torch.Tensor with transformed data
        """
        x = self.tokenizer.encode_plus(x, truncation=True, padding=True)
        return x

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply inverse transform to given data
        Args:
            x (np.ndarray): data to transform (inverse)

        Returns:
            np.ndarray after apply inverse transform
        """
        pass
