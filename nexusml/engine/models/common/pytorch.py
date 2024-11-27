import functools
import importlib
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nexusml.engine.data.transforms.base import ElementTransforms
from nexusml.engine.data.transforms.sklearn import LabelEncoderTransform
from nexusml.engine.exceptions import ConfigFileError


class SimpleClassificationOutputLayer(nn.Module):
    """
    Simple PyTorch module to make the output of a classification
    It simply contains a linear that match the input features to the number of classes
    And, in case of being in evaluation model, it applies the Softmax activation so the output is viewed as probability
    """

    def __init__(self, input_features: int, num_classes: int):
        """
        Default constructor
        Args:
            input_features (int): number of features that will be input
            num_classes (int): the number of output classes
        """
        super().__init__()
        # Create a linear to transform the input features in the output number of classes
        self.out = nn.Linear(input_features, num_classes)
        # Create a softmax activation to transform the output to a "probability distribution"
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        """
        Forward call of the module
        Args:
            x: input tensor to the module

        Returns:
            tensor after apply the linear layer and the softmax activation (if is in eval mode)
        """
        # Call linear layer
        x = self.out(x)
        # If not training, apply softmax
        if not self.training:
            x = self.softmax(x)
        return x


def _setup_simple_output_layers(last_num_feat: int, outputs_info: List[Dict],
                                output_transforms: ElementTransforms) -> Tuple[nn.ModuleDict, dict]:
    """
    Function that set up a simple output layer for each output of the schema
    Args:
        last_num_feat (int): how many features does the backbone output
        outputs_info (List[Dict]): list with the infor of all outputs of the schema
        output_transforms (ElementTransforms): the transformations that are applied to output elements

    Returns:
        Tuple:
            - nn.ModuleDict where the key is the output id and the value a PyTorch module
            - dict with the mapping (modified output name => original output name)
    """
    # Dict to store layers
    output_layers = {}
    # Name mapping (points "." are not allowed on nn.ModuleDict, so we replace them)
    name_mapping = {}
    # For each output
    for i in outputs_info:
        # Get the ID
        output_id = i['name']
        # Replace "."
        output_id = output_id.replace('.', '#_:_#')
        # Add to mapping
        name_mapping[output_id] = i['name']
        # Get the output transform information
        tfm_out_info = output_transforms.get_transform(i['name']).get_transform_output_info()
        # If output is float or int, it is a regression problem. Set a simple Linear
        if tfm_out_info.output_type in ['float', 'int']:
            output_layers[output_id] = nn.Linear(last_num_feat, tfm_out_info.num_features)
        elif tfm_out_info.output_type == 'category':
            # If the output is categorical, add a SimpleClassificationOutputLayer module
            output_layers[output_id] = SimpleClassificationOutputLayer(input_features=last_num_feat,
                                                                       num_classes=len(tfm_out_info.choices))
        else:
            raise ValueError(f'Output type "{i["type"]}" not supported')

    # Create the ModuleDict from dict
    output_layers = nn.ModuleDict(output_layers)
    return output_layers, name_mapping


def basic_train_loop(model: nn.Module,
                     loss_function: nn.Module,
                     optimizer: optim.Optimizer,
                     scheduler: optim.lr_scheduler._LRScheduler,
                     epochs: int,
                     train_dl: DataLoader,
                     device: str,
                     verbose: bool = False) -> Dict:
    """
    Perform basic training of the given model
    Args:
        model (nn.Module): PyTorch model to be trained
        loss_function (nn.Module): loss function to use for training
        optimizer (optim.Optimizer): optimizer to use for training
        scheduler: the learning rate scheduler to use for training
        epochs (int): number of epochs to train the model
        train_dl (DataLoader): loader with training data
        device (str): device to use for training
        verbose (bool): to plot loss of each iteration or not

    Returns:
        Dict with the loss history of each output for each iteration
    """
    # Move model to the given device and set it to train mode
    model.to(device)
    model.train()
    # Init history as None
    loss_hist = None
    # For each epoch
    for epoch in range(1, epochs + 1):
        print(f'[+] Starting epoch {epoch}')
        # For each batch
        for x, y in tqdm(train_dl):
            # Reset gradients
            optimizer.zero_grad()
            # Move data to device
            x = {k: v.to(device) for k, v in x.items()}
            y = {k: v.to(device) for k, v in y.items()}
            # Make the prediction
            prediction = model(x)
            # Get the loss
            loss_by_output, loss = loss_function(prediction, y)
            if verbose:
                print(loss_by_output, loss)
            # Update loss history
            loss_hist = _join_torch_dict(gd=loss_hist, pd=loss_by_output)
            # Compute gradients
            loss.backward()
            # Update model
            optimizer.step()
            # Get next lr calling the scheduler
            if scheduler is not None:
                scheduler.step()
    return loss_hist


def _join_torch_dict(gd: Union[None, Dict], pd: Dict):
    """
    Function that joins the torch values of the two given dicts
    First convert to numpy the torch tensor of the second dictionary
    Then, concatenate by axis 0 all the second values to the first dictionary values
    In case that the first dictionary is None, the second one is returned after converting data to numpy
    Args:
        gd (Dict): dict to where concatenate data
        pd (Dict): dict from where get data to be concatenated

    Returns:
        The first dictionary after data is concatenated
    """
    # Convert data to numpy
    pd = {k: v.detach().cpu().numpy() for k, v in pd.items()}
    pd = {k: v[np.newaxis] if v.ndim == 0 else v for k, v in pd.items()}
    # If the first dictionary is None, there is nothing to concatenate, so return the converted dict
    if gd is None:
        return pd
    else:
        # For each item, make the concatenation on axis 0
        for k, v in pd.items():
            gd[k] = np.concatenate((gd[k], v), axis=0)
    return gd


def basic_predict_loop_with_targets(model: nn.Module, dl: DataLoader,
                                    device: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Perform basic prediction loop with the given model and return both, predictions and target
    Args:
        model (nn.Module): PyTorch model to be trained
        dl (DataLoader): loader with data to be predicted. It includes the true target
        device (str): device to use for training
    Returns:
        Tuple:
            Dict with the predictions of each schema output
            Dict with the targets of each schema output, get from given DataLoader
    """
    # Move model to device and set as eval mode
    model.to(device)
    model.eval()
    # Initialize predictions and targets
    n_examples = len(dl.dataset)
    predictions = {}
    targets = {}
    # Predict one batch to get shapes and types
    for x, y in dl:
        # Move to device
        x = {k: v.to(device) for k, v in x.items()}
        # Predict and concatenate
        prediction = model(x)

        for k, v in prediction.items():
            v = v.detach().cpu().numpy()
            shape = (n_examples,) if v.ndim == 1 else (n_examples, v.shape[1])
            dtype = v.dtype
            predictions[k] = np.empty(shape, dtype=dtype)

        for k, v in y.items():
            v = v.detach().cpu().numpy()
            shape = (n_examples,) if v.ndim == 1 else (n_examples, v.shape[1])
            dtype = v.dtype
            targets[k] = np.empty(shape, dtype=dtype)

        break

    idx = 0
    # To no compute gradients
    with torch.no_grad():
        print('[+] Making predictions')
        # For each batch
        for x, y in tqdm(dl):
            # Move data to device
            x = {k: v.to(device) for k, v in x.items()}
            # Make prediction
            prediction = model(x)
            # Concatenate both, predictions and targets
            next_idx = None
            for k, v in prediction.items():
                if next_idx is None:
                    next_idx = idx + v.shape[0]
                predictions[k][idx:next_idx] = v.detach().cpu().numpy()

            for k, v in y.items():
                if next_idx is None:
                    next_idx = idx + v.shape[0]
                targets[k][idx:next_idx] = v.detach().cpu().numpy()

            idx = next_idx

    return predictions, targets


def basic_predict_loop(model: nn.Module, dl: DataLoader, device: str) -> Dict[str, np.ndarray]:
    """
    Perform basic prediction loop with the given model and return only the predictions
    Args:
        model (nn.Module): PyTorch model to be trained
        dl (DataLoader): loader with data to be predicted. It does not include the true target
        device (str): device to use for training
    Returns:
        Tuple:
            Dict with the predictions of each schema output
            Dict with the targets of each schema output, get from given DataLoader
    """
    # Move model to device and set as eval mode
    model.to(device)
    model.eval()
    # Initialize predictions
    predictions = {}
    n_examples = len(dl.dataset)
    # Predict one batch to get shapes and types
    for x in dl:
        # Move to device
        x = {k: v.to(device) for k, v in x.items()}
        # Predict and concatenate
        prediction = model(x)

        for k, v in prediction.items():
            v = v.detach().cpu().numpy()
            shape = (n_examples,) if v.ndim == 1 else (n_examples, v.shape[1])
            dtype = v.dtype
            predictions[k] = np.empty(shape, dtype=dtype)
        break

    idx = 0
    # To no compute gradients
    with torch.no_grad():
        print('[+] Making predictions')
        # For each batch
        for x in tqdm(dl):
            # Move to device
            x = {k: v.to(device) for k, v in x.items()}
            # Predict and concatenate
            prediction = model(x)
            next_idx = None
            for k, v in prediction.items():
                if next_idx is None:
                    next_idx = idx + v.shape[0]
                predictions[k][idx:next_idx] = v.detach().cpu().numpy()
            idx = next_idx

    return predictions


def _get_class_weight(tfm: LabelEncoderTransform) -> np.ndarray:
    """
    Function that returns the class weight of the given classification output for making cost-sensitive learning
    as balance data
    Note: the code is get from
    https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/utils/class_weight.py#L8
    when class_weight=='balance'
    Args:
        tfm (LabelEncoderTransform): transform used for encode the classification output. Used for getting the
                                    number of elements for each class

    Returns:
        np.ndarray with the weight to be applied for each class
    """
    # Get the number of elements for each class
    class_count = tfm.get_transform_output_info().choice_counter
    num_classes = len(class_count)
    num_examples = sum(class_count.values())
    # Initialize an empty dict to store the freq of each class
    recip_freq = {}
    for k, v in class_count.items():
        recip_freq[k] = num_examples / (v * num_classes)
    # Transform dict to array in the same order as the transformation has encoded the classes
    class_weight = [recip_freq[i] for i in tfm.sklearn_transform.classes_]
    # Return array as numpy and float32
    return np.array(class_weight).astype(np.float32)


class BasicLossFunction(nn.Module):
    """
    Class that creates the basic loss function for each output
    In case of regression, nn.MSELoss() is used
    In case of classification, nn.CrossEntropyLoss() is used. And it can be cost-sensitive or not
    """

    def __init__(self,
                 outputs_info: List[Dict],
                 output_transforms: ElementTransforms,
                 classification_cost_sensitive: bool = False,
                 device: str = 'cpu'):
        """
        Default constructor
        Args:
            outputs_info (List[Dict]): the information of each output element
            output_transforms (ElementTransforms): transformations applied to output elements
            classification_cost_sensitive (bool): apply or not cost-sensitive to CrossEntropyLoss when classification
            device (str): device to use for training
        """
        super().__init__()
        # Store params
        self.classification_cost_sensitive = classification_cost_sensitive
        self.device = device
        # Name mapping (points "." are not allowed on nn.ModuleDict, so we replace them)
        self.name_mapping = {}
        # Initialize empty dict to store the loss functions (one for each output)
        loss_functions = {}

        # For each output
        for i in outputs_info:
            # Get the ID
            output_id = i['name'].replace('.', '#_:_#')
            self.name_mapping[output_id] = i['name']
            # Get the output transformation
            tfm_out_info = output_transforms.get_transform(i['name']).get_transform_output_info()
            # If the output is float or int, use MSELoss
            if tfm_out_info.output_type in ['float', 'int']:
                loss_functions[output_id] = nn.MSELoss()
            elif tfm_out_info.output_type == 'category':
                # If the output is categorical, use CrossEntropy
                # Initialize class weight as None
                cw = None
                # If 'classification_cost_sensitive' is True, get class weight and transform to tensor
                if classification_cost_sensitive:
                    cw = torch.tensor(_get_class_weight(tfm=output_transforms.get_transform(i['name'])),
                                      device=self.device)
                # Create the loss function
                loss_functions[output_id] = nn.CrossEntropyLoss(cw)
            else:
                raise ValueError(f'Output type {i["type"]} not supported')
        # Convert dict to nn.ModuleDict
        self.loss_functions = nn.ModuleDict(loss_functions)

    def forward(self, prediction, target):
        """
        Forward called to compute losses
        Args:
            prediction: tensor with predictions
            target: tensor with targets

        Returns:
            Tuple of:
                Dict with the loss obtained on each output
                Tensor with the sum of the loss of each output (where .backward() call is applied)
        """
        # Initialize total loss as 0
        total_loss = torch.tensor(0., device=self.device)
        # To store the loss for each output
        loss_by_output = {}
        # For each loss function
        for k, v in self.loss_functions.items():
            k = self.name_mapping[k]
            if isinstance(v, nn.CrossEntropyLoss):
                # If the loss is CrossEntropy, call to it but getting only the first column
                k_loss = v(prediction[k], target[k][:, 0])
            else:
                # Call to loss
                k_loss = v(prediction[k], target[k])
            # Store current output loss
            loss_by_output[k] = k_loss
            # Add to total loss
            total_loss += k_loss
        return loss_by_output, total_loss


def _get_optimizer_from_config(optimizer_config: Dict):
    """
    Function that will create the optimizer from configuration
    Args:
        optimizer_config (Dict): dict that contains the optimizer configuration

    Returns:
        Optimizer created with given configuration
    """
    raise NotImplementedError()


def _get_scheduler_from_config(scheduler_config: Dict):
    """
    Function that will create the learning rate scheduler from configuration
    Args:
        scheduler_config (Dict): dict that contains the learning rate scheduler configuration

    Returns:
        Learning rate scheduler created with given configuration
    """
    raise NotImplementedError()


def _get_loss_function_from_config(loss_function_args: Dict, outputs_info: List[Dict],
                                   output_transforms: ElementTransforms, device: str):
    """
    Function that returns the loss function module already built from config
    Args:
        loss_function_args (Dict): loss function configuration.
                                It is created calling to 'class' argument with 'args' arguments
        outputs_info (List[Dict]): the information of all output elements
        output_transforms (ElementTransforms): the transformations that are applied to the output elements
        device (str): the device to be used for training

    Returns:
        PyTorch loss function valid for all outputs
    """
    # The 'class' key is required
    if 'class' not in loss_function_args:
        raise ConfigFileError("Missing 'class' key for loss function")
    # Create class constructor callable
    loss_function_constructor = _from_class_name_to_constructor(loss_function_args['class'])
    # If custom args are given, modify the constructor with them
    if 'args' in loss_function_args and loss_function_args['args'] is not None:
        loss_function_constructor = functools.partial(loss_function_constructor, **loss_function_args['args'])
    # Create loss function given the output elements info, output elements transformation and the device
    loss_function = loss_function_constructor(outputs_info=outputs_info,
                                              output_transforms=output_transforms,
                                              device=device)
    return loss_function


def _from_class_name_to_constructor(module_and_class: str) -> Callable:
    """
    Function that returns a callable object creating it from the given module and class
    Args:
        module_and_class (str): the string that contains the module and class to be converted to callable

    Returns:
        Constructor of the given class after import it from the given module
    """
    # Split by '.'
    module_and_class = module_and_class.split('.')
    # All but the last part is the module
    module = '.'.join(module_and_class[:-1])
    # The las part is the class name
    class_name = module_and_class[-1]
    # Import module
    module = importlib.import_module(module)
    # Get class constructor from module
    return getattr(module, class_name)


def freeze_all_but_bn(m: nn.Module):
    """
    Function that set a module as no trainable (not required grad) only if it is not a BatchNorm module
    Args:
        m(nn.Module): PyTorch Module
    """
    if not isinstance(m, nn.modules.batchnorm._BatchNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)
