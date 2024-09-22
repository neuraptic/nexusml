import pandas as pd
from torch.utils.data import Dataset

from nexusml.engine.data.transforms.base import ElementTransforms


class SpeechDataset(Dataset):
    """ PyTorch Dataset for speech data. """

    def __init__(self,
                 df: pd.DataFrame,
                 input_transforms: ElementTransforms,
                 output_transforms: ElementTransforms,
                 train: bool = True):
        self.df = df
        self.input_transforms = input_transforms
        self.output_transforms = output_transforms
        self.train = train

    def __getitem__(self, item):
        row = self.df.iloc[item:(item + 1), :]
        x = self.input_transforms.transform(row)
        if self.train:
            y = self.output_transforms.transform(row)
            return x, y
        else:
            return x

    def __len__(self):
        return self.df.shape[0]
