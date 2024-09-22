from typing import List, Tuple

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor

from nexusml.engine.data.transforms.base import Transform


class DefaultSpeechTransform(Transform):
    """
    Default speech transform.
    """

    def __init__(self, path: str, target_sr: int, **kwargs):
        super().__init__(**kwargs)
        self.read_transform = ReadSpeechFileTransform()
        self.resample_transform = ResampleSpeechTransform(target_sr)
        self.processor_transform = ProcessorTransform(path=path, target_sr=target_sr)

    def fit(self, x: np.ndarray):
        """ Fit the transform """
        return

    def transform(self, x: str) -> List[torch.Tensor]:
        """ Transform the input """
        x = self.read_transform.transform(x)
        x = self.resample_transform.transform(x)
        x = self.processor_transform.transform(x)
        return x


class ReadSpeechFileTransform(Transform):
    """
    Read speech transform
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, x: np.ndarray):
        return

    def transform(self, x: str) -> List[Tuple[torch.Tensor, int]]:
        return list(map(self._transform, x))

    @staticmethod
    def _transform(x: str) -> Tuple[torch.Tensor, int]:
        x, sr = torchaudio.load(x)
        return x, sr


class ResampleSpeechTransform(Transform):
    """
    Resample speech transform
    """

    def __init__(self, target_sr: int, **kwargs):
        super().__init__(**kwargs)
        self.target_sr = target_sr

    def fit(self, x: np.ndarray):
        return

    def transform(self, x: np.ndarray) -> List[torch.Tensor]:
        return list(map(self._transform, x))

    def _transform(self, x: Tuple) -> torch.Tensor:
        x = torchaudio.transforms.Resample(orig_freq=x[1], new_freq=self.target_sr)(x[0])
        return x


class ProcessorTransform(Transform):
    """
    Processor transform
    """

    def __init__(self, path: str, target_sr: int, **kwargs):
        super().__init__(**kwargs)

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(path)
        self.target_sr = target_sr

    def fit(self, x: np.ndarray):
        pass

    def transform(self, x: np.ndarray) -> List[torch.Tensor]:
        return list(map(self._transform, x))

    def _transform(self, x: np.ndarray) -> torch.Tensor:
        x = self.processor(x, sampling_rate=self.target_sr, return_tensors='pt', padding=True)
        x['input_values'] = torch.squeeze(x['input_values'])
        x['attention_mask'] = torch.ones_like(x['input_values'])
        return x
