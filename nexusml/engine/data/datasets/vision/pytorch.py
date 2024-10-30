'''

1. Metadata + Fold ==> TrainMetadata, TestMetadata
2. Generate Datasets
3. Generate DataLoaders
4. Train the model
5. Evaluate the model

'''
from typing import Dict

import pandas as pd
from torch.utils.data import Dataset

from nexusml.engine.data.transforms.base import Transform


class ImageDataset(Dataset):
    """
    Dataset for image classification. General class for one or various outputs network
    """

    def __init__(self,
                 df: pd.DataFrame,
                 input_transform_functions: Dict[str, Transform],
                 output_transform_functions: Dict[str, Transform],
                 train: bool = True):
        """
        Default constructor

        Args:
            df (pd.DataFrame): metadata DataFrame
            input_transform_functions (Dict[str, Transform]): sequence of transformations to be applied to each image
            output_transform_functions (Dict[str, Transform]): sequence of transformations to be applied to each target
            train (bool): if is for training (True) of for testing (False)
        """
        self.df = df
        self.input_transform_functions = input_transform_functions
        self.output_transform_functions = output_transform_functions
        self.train = train

    def __len__(self):
        """
        Return the number of examples in the dataset.

        Returns:
            int with the number of examples in the dataset, in this case, the number of rows in the DataFrame
        """
        return self.df.shape[0]

    def __getitem__(self, item):
        """
        Function to get a single element (example).

        Args:
            item: index of the example to get

        Returns:
            If training, it returns a tuple with the example input (x) and output/target (y)
            If not training, it returns only the example input (x)

        """
        img = {}
        for k, v in self.input_transform_functions.items():
            im = v.transform(self.df.iloc[item][k])
            img[k] = im

        # If training
        if self.train:
            out = {}
            for k, v in self.output_transform_functions.items():
                o = v.transform(self.df.iloc[item:(item + 1)][k].to_numpy())
                out[k] = o

            return img, out
        else:
            return img
