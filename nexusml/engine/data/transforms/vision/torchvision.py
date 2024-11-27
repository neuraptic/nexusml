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
    Transform class for images, providing different transformations for training and testing modes.

    The training transforms include:
        - Resizing the image based on mean aspect ratio
        - Random horizontal flipping for data augmentation
        - Auto augment using timm's RandAugment
        - Converting images to tensors
        - Normalization using ImageNet's default mean and standard deviation
        - Random erasing for augmentation

    The test transforms include:
        - Resizing the image based on mean aspect ratio
        - Converting images to tensors
        - Normalization using ImageNet's default mean and standard deviation
    """

    def __init__(self, **kwargs):
        """
        Initializes the BasicImageTransform with default settings for training and testing transformations.

        Args:
            path (str): Path where images are stored (if applicable).
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

        self.mean_aspect_ratio = None
        self.training = True
        self.train_transform = None
        self.test_transform = None

    def fit(self, x: np.ndarray):
        """
        Computes the mean aspect ratio of the largest side of the images provided in order to
        resize images during training and testing. Also sets up the train and test transformations
        based on the computed mean aspect ratio.

        Args:
            x (np.ndarray): Array of image paths to compute the mean aspect ratio.

        Returns:
            None
        """
        # Compute mean aspect ratio of all images
        self.mean_aspect_ratio = 0
        for i in range(x.shape[0]):
            img = Image.open(x[i])
            self.mean_aspect_ratio += img.size[1] / img.size[0]

        self.mean_aspect_ratio = self.mean_aspect_ratio / x.shape[0]

        # Train transforms setup
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

        # Adjust train transform to include resizing
        self.train_transform = T.Compose([T.Resize([224, int(self.mean_aspect_ratio * 224)])] +
                                         self.train_transform.transforms[1:])

        # Test transforms setup
        self.test_transform = T.Compose([
            T.Resize([224, int(self.mean_aspect_ratio * 224)]),
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])

    def transform(self, x: str) -> torch.Tensor:
        """
        Transforms the input image based on the current mode (training or testing).
        In training mode, it applies random augmentations; in test mode, it applies basic preprocessing.

        Args:
            x (str): Path to the image to be transformed.

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        x = Image.open(x)
        if x.mode != 'RGB':
            x = x.convert('RGB')

        if self.training:
            return self.train_transform(x)
        else:
            return self.test_transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Inverse transform method placeholder.
        This method can be implemented to reverse the applied transformations if needed.

        Args:
            x (np.ndarray): The transformed image data.

        Returns:
            np.ndarray: The original image data (currently not implemented).
        """
        pass

    def train(self):
        """
        Switches the transformer to training mode, where data augmentations are applied.
        """
        self.training = True

    def eval(self):
        """
        Switches the transformer to evaluation mode, where only basic preprocessing is applied.
        """
        self.training = False


class SquareImageTransform(BasicImageTransform):
    """
    A specific image transformation class where the input images are transformed to square format
    (224x224) for both training and testing. Inherits from BasicImageTransform.
    """

    def __init__(self, **kwargs):
        """
        Initializes the SquareImageTransform class.

        Args:
            **kwargs: Additional keyword arguments passed to the base class.
        """
        super().__init__(**kwargs)

    def fit(self, x: np.ndarray):
        """
       Sets up the training and testing transformations for square images. This overrides the aspect
       ratio resizing logic from the BasicImageTransform class and instead applies a fixed square crop.

       Args:
           x (np.ndarray): Array of image paths (not used in this method, as images are cropped to squares).

       Returns:
           None
       """

        # Train transforms setup for square cropping
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

        # Test transforms setup for square cropping
        self.test_transform = T.Compose([
            T.CenterCrop(224),
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
