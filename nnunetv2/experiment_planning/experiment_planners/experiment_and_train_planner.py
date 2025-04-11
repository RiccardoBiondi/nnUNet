import shutil
from copy import deepcopy
from typing import List, Union, Tuple

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json, isfile, maybe_mkdir_p
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm

from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.preprocessing.normalization.map_channel_name_to_normalization import get_normalization_scheme
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape, compute_new_shape
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
# Just an utility class to allow the generation of the training planner file when generating 
# the plans for the experiments.
# This class Inherit from the default experiment planner.

__author__ = ["Riccardo Biondi"]
__email__ = ["riccardo.biondi7@unibo.it"]


class TrainExperimentPlanner(ExperimentPlanner):

    def save_plans(self, plans):
        print("Ciaoooooo")
        
        #call the paln saver of the base class 
        super().save_plans(plans)

        # now the custom part. Here the default training plan is saved.
        train_file = join(nnUNet_preprocessed, self.dataset_name,  "TrainPlans.json")

        train_plans = {
            "configurations": {
                "default": {
                    "initial_lr": 1e-2,
                    "weight_decay": 3e-5,
                    "oversample_foreground_percent": .33,
                    "probabilistic_oversampling": False,
                    "num_iterations_per_epoch": 250,
                    "num_val_iterations_per_epoc": 50,
                    "num_epochs": 1000,
                    "enable_deep_supervision": True
                }
            }
        }
        if isfile(train_file):
            old_plans = load_json(train_file)
            old_configurations = old_plans['configurations']
            for c in train_plans['configurations'].keys():
                if c in old_configurations.keys():
                    del (old_configurations[c])
            train_plans['configurations'].update(old_configurations)

        maybe_mkdir_p(join(nnUNet_preprocessed, self.dataset_name))
        save_json(train_plans, train_file, sort_keys=False)
