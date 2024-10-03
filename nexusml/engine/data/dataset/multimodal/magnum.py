from typing import Dict

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from nexusml.engine.data.transforms.base import Transform
from nexusml.engine.schema.base import Schema


class MagnumDataset(Dataset):
    """
    Dataset for multimodal classification. General class for one or various outputs network
    """

    def __init__(self,
                 schema: Schema,
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
        self.schema = schema
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
        data = {}
        x_tabular = {'x_num': None, 'x_cat': None}
        for k, v in self.input_transform_functions.items():
            element = self.schema.get_inputs_by_name(input_name=k)[0]
            if element['type'] == 'image_file':
                data['image'] = v.transform(self.df.iloc[item][k])
            elif element['type'] == 'text':
                data['text'] = v.transform(self.df.iloc[item:(item + 1)][k].to_numpy())[0]
            elif element['type'] in ['float', 'integer']:
                x = v.transform(self.df.iloc[item:(item + 1)][k].to_numpy())
                x_tabular['x_num'] = x if x_tabular['x_num'] is None else np.concatenate((x_tabular['x_num'], x))
            elif element['type'] == 'category':
                x = v.transform(self.df.iloc[item:(item + 1)][k].to_numpy())
                x_tabular['x_cat'] = x if x_tabular['x_cat'] is None else np.concatenate((x_tabular['x_cat'], x))

        if any(value is not None for value in x_tabular.values()):
            data['tabular'] = x_tabular

        # If training
        if self.train:
            out = {}
            for k, v in self.output_transform_functions.items():
                o = v.transform(self.df.iloc[item:(item + 1)][k].to_numpy())
                out[k] = o

            return data, out
        else:
            return data
