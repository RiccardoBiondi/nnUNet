"""
"""
import os
import logging
from torch import nn
from copy import deepcopy
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from typing import Union, Tuple, List, Type, Callable, NoReturn
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from functools import lru_cache
import nnunetv2
from nnunetv2.training.loss.loss_wrapper import LossWrapper

from nnunetv2.training.logging.base_logger import Logger
__author__ = ["Riccardo Biondi"]
__email__ = ["riccardo.biondi7@unibo.it"]

class TrainConfigurationManager(object):
    """
    Facility class that taken the configuration dictionary takes the responmsability to
    correctly initialize all the classes and parameters for the specific training 
    configuration.
    """

    def __init__(self, configuration_dict: dict):
        """
        Initializer of the specific train configuration .
        
        Parameters
        ----------
        configuration_dict: dict
            Dictionary that contains the full configuration retrieved from the .json file
        """
        self._configuration = configuration_dict

        keys = configuration_dict.keys() # list of available configuration parameters

        self._initial_lr = configuration_dict["initial_lr"] if "initial_lr" in keys else 1e-2
        self._weight_decay = configuration_dict["weight_decay"] if "weight_decay" in keys else 3e-5
        self._oversample_foreground_percent = configuration_dict["oversample_foreground_percent"] if "oversample_foreground_percent" in keys else 0.33
        self._probabilistic_oversampling = configuration_dict["probabilistic_oversampling"] if "probabilistic_oversampling" in keys else False
        self._num_iterations_per_epoch = configuration_dict["num_iterations_per_epoch"] if "num_iterations_per_epoch" in keys else 250
        self._num_val_iterations_per_epoch = configuration_dict["num_val_iterations_per_epoch"] if "num_val_iterations_per_epoch" in keys else 50
        self._num_epochs = configuration_dict["num_epochs"] if "num_epochs" in keys else 1000
        self._enable_deep_supervision = configuration_dict["enable_deep_supervision"] if "enable_deep_supervision" in keys else True
        self._loss = None # TODO Add here the default loss initialization, so the loss initializer do not need to provide the dummy initialization. Maybe it is a good idea

        _ = self._loss_initializer()

        # TODO Add optimizer initializer
        # TODO Add lr scheduler initializer

    def _loss_initializer(self) -> NoReturn:
        """
        Initialize the loss for the given confuguration.
        Each loss class should be implemented in nnunetv2..trainin.loss

        Finally, each loss is wrapped using the LossWrapper class.

        This function initilize the loss property of  this class.
        If no loss is provided in the configuration, then a Dice Loss + BCE loss is returned (Default loss of nnUNet)
        """

        if "losses" not in self._configuration.keys():
            return # here add the return of the default loss for nnUNet
        
        loss_names = self._configuration["losses"].keys()
        losses = [] # list of all the losses specified in the configuration file
        weigths = [] # list of all the weights for each specified loss 
        for loss_name in loss_names:
            
            loss_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "loss"), loss_name, current_module="nnunetv2.training.loss")

            loss_kwargs = self._configuration["losses"][loss_name]["kwargs"] if "kwargs" in self._configuration["losses"][loss_name].keys() else {}
            loss_weight = self._configuration["losses"][loss_name]["weight"] if "weight" in self._configuration["losses"][loss_name].keys() else 1.

            losses.append(loss_class(**loss_kwargs))
            weigths.append(loss_weight)

        self._loss = LossWrapper(losses=losses, weights=weigths)

    @property
    def num_epochs(self) -> int:
        return self._num_epochs

    @property
    def initial_lr(self) -> float:
        return self._initial_lr

    @property
    def weight_decay(self) -> float:
        return self._weight_decay

    @property
    def oversample_foreground_percent(self) -> float:
        return self._oversample_foreground_percent

    @property
    def probabilistic_oversampling(self) -> float:
        return self._probabilistic_oversampling

    @property
    def num_iterations_per_epoch(self) -> int:
        return self._num_iterations_per_epoch

    @property
    def num_val_iterations_per_epoch(self) -> int:
        return self._num_val_iterations_per_epoch

    @property
    def loss(self) -> nn.Module:
        return self._loss



class TrainPlansManager(object):
    """
    Just a class inspired by PlansManager to handle the TrainPlans file for the AdptiveTraines,
    allowing initializing the training from a configuration file and not by subclassing every time the
    Trainer class.

    Why to we need a TrainPlansManager on top of the TrainConfigurationManager?
    For the same reasons we need the PlansManager:
        1) resolve inheritance in configurations
        2) expose otherwise annoying stuff like getting the label manager or IO class from a string
        3) clearly expose the things that are in the plans instead of hiding them in a dict
        4) cache shit

    TODO: Think about make this class inherit form the PlansManager (is it a good idea?)
    """

    def __init__(self, plans_file_or_dict: Union[str, dict]):
        """
        Initialize the PlansManager class.

        Parameters
        ----------
        plans_file_or_dict: Union[str, dict]
            path to a .json file or directly the dictionary containing the plans to parse.
        """
        self.plans = plans_file_or_dict if isinstance(plans_file_or_dict, dict) else load_json(plans_file_or_dict)

        # here initialize the general properties of the training configuration (i.e. the logger, which should be alwais the same since
        # its not strictly a train parameter)
        _ = self._init_logger()

    def __repr__(self):
        return self.plans.__repr__()
    
    def _init_logger(self):
        """
        Convenience function to initialize the training logger.
        If the logger is not specified in the configuration file, then the dafault logger is used.
        """
        #config = load_json(join(os.environ["nnUNet_preprocessed"], self.plans_manager.dataset_name, "nnUNetTrainer.json"))

        if "logger" in self.plans.keys():
            logger_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "logging"),
                self.plans["logger"]["name"],
                current_module="nnunetv2.training.logging")
            if "kwargs" in self.plans["logger"].keys():
                self._logger = logger_class(**self.plans["logger"]["kwargs"])
            else:
                self._logger = logger_class()
        else:
            self._logger = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "logging"),
                "nnUNetLogger",
                current_module="nnunetv2.training.logging")

    def _internal_resolve_configuration_inheritance(self, configuration_name: str,
                                                    visited: Tuple[str, ...] = None) -> dict:
        """
        """
        if configuration_name not in self.plans['configurations'].keys():
            raise ValueError(f'The configuration {configuration_name} does not exist in the plans I have. Valid '
                            f'configuration names are {list(self.plans["configurations"].keys())}.')
        configuration = deepcopy(self.plans['configurations'][configuration_name])
        if 'inherits_from' in configuration:
            parent_config_name = configuration['inherits_from']

            if visited is None:
                visited = (configuration_name,)
            else:
                if parent_config_name in visited:
                    raise RuntimeError(
                        f"Circular dependency detected. The following configurations were visited "
                        f"while solving inheritance (in that order!): {visited}. "
                        f"Current configuration: {configuration_name}. Its parent configuration "
                        f"is {parent_config_name}.")
                visited = (*visited, configuration_name)

            base_config = self._internal_resolve_configuration_inheritance(parent_config_name, visited)
            base_config.update(configuration)
            configuration = base_config
        return configuration

    @lru_cache(maxsize=10)
    def get_configuration(self, configuration_name: str):
        if configuration_name not in self.plans['configurations'].keys():
            raise RuntimeError(f"Requested configuration {configuration_name} not found in plans. "
                                f"Available configurations: {list(self.plans['configurations'].keys())}")

        configuration_dict = self._internal_resolve_configuration_inheritance(configuration_name)

        return TrainConfigurationManager(configuration_dict)
    
    @property
    def available_configurations(self) -> List[str]:
        return list(self.plans['configurations'].keys())

    @property
    def logger(self) -> Logger:
        return self._logger