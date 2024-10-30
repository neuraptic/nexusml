import pandas as pd
from torch.utils.data import Dataset

from nexusml.engine.data.transforms.base import ElementTransforms


class NLPDataset(Dataset):
    """
    Generic Dataset class for NLP
    """

    def __init__(self,
                 df: pd.DataFrame,
                 input_transforms: ElementTransforms,
                 output_transforms: ElementTransforms,
                 train: bool = True):
        """
        Constructor
        Args:
            df (DataFrame): dataframe with the data
            input_transforms (ElementTransforms): transformations that are applied to each input element
            output_transforms (ElementTransforms): transformations that are applied to each output element
            train (bool): if is for training (True) of for testing (False)
        """
        self.df = df
        self.input_transforms = input_transforms
        self.output_transforms = output_transforms
        self.train = train

    def __getitem__(self, item):
        """
        Function to get a single element (example)

        Args:
            item: index of the example to get

        Returns:
            If training, it returns a tuple with the example input (x) and output/target (y)
            If not training, it returns only the example input (x)

        """
        sub_df = self.df.iloc[item:(item + 1), :]
        x = self.input_transforms.transform(sub_df)

        # If training
        if self.train:
            y = self.output_transforms.transform(sub_df)
            return x, y
        else:
            return x

    def __len__(self):
        """
        Return the number of examples in the dataset

        Returns:
            int with the number of examples in the dataset, in this case, the number of rows in the DataFrame
        """
        return self.df.shape[0]
