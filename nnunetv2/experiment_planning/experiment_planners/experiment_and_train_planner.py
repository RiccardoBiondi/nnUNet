from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json, isfile, maybe_mkdir_p
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner

# Just an utility class to allow the generation of the training planner file when generating 
# the plans for the experiments.
# This class Inherit from the default experiment planner.

__author__ = ["Riccardo Biondi"]
__email__ = ["riccardo.biondi7@unibo.it"]


class TrainExperimentPlanner(ExperimentPlanner):

    def save_plans(self, plans):
        
        # here just add the train plan filename and the 
        # dafault specification of the train configuration to use.
        # that is made in order to not change any parameter in the 
        # cli. 
        # I know that is not efficient but that is the best I came up.... for now!

        plans.update({
            "train_configs_filename": "TrainPlans",
            "train_config_to_use": "default" 
        })
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
