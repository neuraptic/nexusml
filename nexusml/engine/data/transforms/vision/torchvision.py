import numpy as np
from PIL import Image
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torch
from torchvision.transforms import transforms as T

from nexusml.engine.data.transforms.base import Transform


class BasicImageTransform(Transform):
    """
    Transform for images
    Train transforms:
        - Resize
        - RandomHorizontalFlip
        - timm.data.auto_augment.RandAugment
        - ToTensor
        - Normalize
        - timm.data.random_erasing.RandomErasing

    Test transforms:
        - Resize
        - ToTensor
        - ConvertImageDtype
        - Normalize

    """

    def __init__(self, **kwargs):
        """
        Constructor
        Args:
            path (str): path where images are stored
            **kwargs:
        """
        super().__init__(**kwargs)

        self.mean_aspect_ratio = None
        self.training = True
        self.train_transform = None
        self.test_transform = None

    def fit(self, x: np.ndarray):
        """
        Fit method
        Computes mean size of the largest side of all images to do the resize and avoid
        errors when images have different sizes
        Args:
            x (np.ndarray): paths of images to fit with

        Returns:

        """

        # Compute mean size of the largest side of all images
        self.mean_aspect_ratio = 0
        for i in range(x.shape[0]):
            img = Image.open(x[i])
            self.mean_aspect_ratio += img.size[1] / img.size[0]

        self.mean_aspect_ratio = self.mean_aspect_ratio / x.shape[0]

        # Train transforms
        self.train_transform = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.2,
            re_mode='pixel',
            re_count=1,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )

        # Remove RandomResizedCropAndInterpolation and put Resize
        self.train_transform = T.Compose([T.Resize([224, int(self.mean_aspect_ratio * 224)])] +
                                         self.train_transform.transforms[1:])

        # Test transforms
        self.test_transform = T.Compose([
            T.Resize([224, int(self.mean_aspect_ratio * 224)]),
            # T.CenterCrop(224),
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])

    def transform(self, x: str) -> torch.Tensor:
        """
        Transform method
        Args:
            x (str): path to the image to get transformed

        Returns:
            torch.Tensor: transformed image
        """
        x = Image.open(x)
        if x.mode != 'RGB':
            x = x.convert('RGB')

        if self.training:
            return self.train_transform(x)
        else:
            return self.test_transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        pass

    def train(self):
        """
        Train method. Sets training to True
        """
        self.training = True

    def eval(self):
        """
        Test method. Sets training to False
        """
        self.training = False
