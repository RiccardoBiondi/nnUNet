
import os
import logging
import functools
import operator
import statistics
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional, Union
from argparse import Namespace

import numpy as np
from PIL import Image
from typing_extensions import override
from torch import Tensor
from torch.nn import Module
from dataclasses import asdict, is_dataclass
from time import time

from nnunetv2.training.logging.base_logger import Logger

try: 
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME


    _MLFLOW_EXISTS = True

except ImportError:
    _MLFLOW_EXISTS = False


if _MLFLOW_EXISTS:


    def _get_resolve_tags() -> Callable:

        # before v1.1.0
        if hasattr(mlflow.tracking.context, "resolve_tags"):
            from mlflow.tracking.context import resolve_tags
        # since v1.1.0
        elif hasattr(mlflow.tracking.context, "registry"):
            from mlflow.tracking.context.registry import resolve_tags
        else:
            resolve_tags = lambda tags: tags

        return resolve_tags


    class MLFlowLogger(Logger):
        """
        """
        LOGGER_JOIN_CHAR = "-"
        def __init__(
                    self,
                    experiment_name: str = "diffusion_experiment",
                    run_name: Optional[str] = None,
                    tracking_uri: Optional[str] = os.getenv("MLFLOW_TRACKING_URI"),
                    tags: Optional[dict[str, Any]] = None,
                    save_dir: Optional[str] = "./mlruns",
                    log_model:  [True, False, "all"] = False,
                    checkpoint_path_prefix: str = "",
                    prefix: str = "",
                    artifact_location: Optional[str] = None,
                    run_id: Optional[str] = None,
                    synchronous: Optional[bool] = None,
                    verbose=True):
            """
            """

            # just to keep some compatibility
            self.my_fantastic_logging = {
            'mean_fg_dice': list(),
            'ema_fg_dice': list(),
            'dice_per_class_or_region': list(),
            'train_losses': list(),
            'val_losses': list(),
            'lrs': list(),
            'epoch_start_timestamps': list(),
            'epoch_end_timestamps': list()
             }

            self._experiment_name = experiment_name
            self._experiment_id: Optional[str] = None
            self._tracking_uri = tracking_uri
            self._run_name = run_name
            self._run_id = run_id
            self.tags = tags
            self._log_model = log_model
            self._logged_model_time: dict[str, float] = {}
            self._checkpoint_callback = None
            self._prefix = prefix
            self._artifact_location = artifact_location
            self._log_batch_kwargs = {} if synchronous is None else {"synchronous": synchronous}
            self._initialized = False
            self._checkpoint_path_prefix = checkpoint_path_prefix
            self._mlflow_client = MlflowClient(tracking_uri)
            self._synchronous = synchronous

        @property
        def experiment(self) -> "MlflowClient":
            r"""Actual MLflow object. To use MLflow features in your :class:`~lightning.pytorch.core.LightningModule` do the
            following.

            Example::

                self.logger.experiment.some_mlflow_function()

            """

            if self._initialized:
                return self._mlflow_client

            mlflow.set_tracking_uri(self._tracking_uri)

            if self._run_id is not None:
                run = self._mlflow_client.get_run(self._run_id)
                self._experiment_id = run.info.experiment_id
                self._initialized = True
                return self._mlflow_client

            if self._experiment_id is None:
                expt = self._mlflow_client.get_experiment_by_name(self._experiment_name)
                if expt is not None:
                    self._experiment_id = expt.experiment_id
                else:
                    logging.warning(f"Experiment with name {self._experiment_name} not found. Creating it.")
                    self._experiment_id = self._mlflow_client.create_experiment(
                        name=self._experiment_name, artifact_location=self._artifact_location
                    )

            if self._run_id is None:
                if self._run_name is not None:
                    self.tags = self.tags or {}


                    if MLFLOW_RUN_NAME in self.tags:
                        logging.warning(
                            f"The tag {MLFLOW_RUN_NAME} is found in tags. The value will be overridden by {self._run_name}."
                        )
                    self.tags[MLFLOW_RUN_NAME] = self._run_name

                resolve_tags = _get_resolve_tags()
                run = self._mlflow_client.create_run(experiment_id=self._experiment_id, tags=resolve_tags(self.tags))
                self._run_id = run.info.run_id
            self._initialized = True
            return self._mlflow_client

        @property
        def run_id(self) -> Optional[str]:
            """Create the experiment if it does not exist to get the run id.

            Returns
            -------
            str: The run id.
            """
            _  = self.experiment
            return self._run_id

        @property
        def experiment_id(self) -> Optional[str]:
            """Create the experiment if it does not exist to get the experiment id.

            Returns
            -------
            str: The experiment id.
            """
            _ = self.experiment
            return self._experiment_id


    
        @property
        def experiment_name(self) -> Optional[str]:
            """
            Return the experiment name.

            Returns
            -------
            str: experiment name
            """
            return self._experiment_name
            
    
        @property
        def version(self) -> Optional[Union[int, str]]:
            """
            Return the experiment version.
            """
            ...
    
        @property
        def root_dir(self) -> Optional[str]:
            """Return the root directory where all versions of an experiment get saved, or `None` if the logger does not
            save data locally."""
            return None
    
        @property
        def log_dir(self) -> Optional[str]:
            """Return directory the current version of the experiment gets saved, or `None` if the logger does not save
            data locally."""
            return None
    
        @property
        def group_separator(self) -> str:
            """Return the default separator used by the logger to group the data into subfolders."""
            return "/"

        @override
        def log(self, key, value, epoch: int):
            
            assert key in self.my_fantastic_logging.keys() and isinstance(self.my_fantastic_logging[key], list), \
            'This function is only intended to log stuff to lists and to have one entry per epoch'

            if len(self.my_fantastic_logging[key]) < (epoch + 1):
                self.my_fantastic_logging[key].append(value)
            else:
                assert len(self.my_fantastic_logging[key]) == (epoch + 1), 'something went horribly wrong. My logging ' \
                                                                        'lists length is off by more than 1'
                print(f'maybe some logging issue!? logging {key} and {value}')
                self.my_fantastic_logging[key][epoch] = value

            # handle the ema_fg_dice special case! It is automatically logged when we add a new mean_fg_dice
            if key == 'mean_fg_dice':
                new_ema_pseudo_dice = self.my_fantastic_logging['ema_fg_dice'][epoch - 1] * 0.9 + 0.1 * value \
                    if len(self.my_fantastic_logging['ema_fg_dice']) > 0 else value
                self.log('ema_fg_dice', new_ema_pseudo_dice, epoch)


            assert isinstance(key, str)

            if isinstance(value, list):



                # sometimes a list of metrics (corresponding to the same key)
                # is porvided (when the metric value for each different class is logged)
                # here the dictionary (with a number indicating the logged class)
                # is provided, logging a single metric for each class.
                metrics = {f"{key}_{i}": v for i, v in enumerate(value)}
            else:
                metrics = {key: value}

            self._log_metrics(metrics, epoch)

        def get_checkpoint(self):
            print("Called get checkpoint")

        def load_checkpoint(self, checkpoint: dict):
            print("called load checkpoint")
        
        def _log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
            """Records metrics. This method logs metrics as soon as it received them.
            Args:
                metrics: Dictionary with metric names as keys and measured quantities as values
                step: Step number at which the metrics should be recorded
    
            """
                
            
            from mlflow.entities import Metric

            metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
            metrics_list: list[Metric] = []

            timestamp_ms = int(time() * 1000)
            for k, v in metrics.items():
                if isinstance(v, str):
                    log.warning(f"Discarding metric with string value {k}={v}.")
                    continue
                metrics_list.append(Metric(key=k, value=v, timestamp=timestamp_ms, step=step or 0))

            self.experiment.log_batch(run_id=self.run_id, metrics=metrics_list, **self._log_batch_kwargs)

        def log_hyperparams(self, params: Union[dict[str, Any], Namespace], *args: Any, **kwargs: Any) -> None:
            """Record hyperparameters.
            Args:
                params: :class:`~argparse.Namespace` or `Dict` containing the hyperparameters
                args: Optional positional arguments, depends on the specific logger being used
                kwargs: Optional keyword arguments, depends on the specif
                ic logger being used
    
            """
            params = _convert_params(params)
            params = _flatten_dict(params)

            from mlflow.entities import Param
    
            # Truncate parameter values to 250 characters.
            # TODO: MLflow 1.28 allows up to 500 characters: https://github.com/mlflow/mlflow/releases/tag/v1.28.0
            params_list = [Param(key=k, value=str(v)[:250]) for k, v in params.items()]
    
            # Log in chunks of 100 parameters (the maximum allowed by MLflow).
            for idx in range(0, len(params_list), 100):
                self.experiment.log_batch(run_id=self.run_id, params=params_list[idx : idx + 100], **self._log_batch_kwargs)
    
        def log_graph(self, model: Module, input_array: Optional[Tensor] = None) -> None:
            """
            Record model graph.
    
            Parameters
            ----------
            model: the model with an implementation of ``forward``.
            input_array: input passes to `model.forward`
    
            """
            ...
            #if artifact_file is not None:
            #    self._mlflow_client.log_image(image=image, run_id=self.run_id, artifact_file=artifact_file, synchronous=self._synchronous)
            #else:
            #    
            #    self._mlflow_client.log_image(image=image, run_id=self.run_id, key=key, step=step, timestamp=timestamp_ms, synchronous=self._synchronous
        def save_dir(self) -> Optional[str]:
            """The root file directory in which MLflow experiments are saved.
    
            Return:
                Local path to the root experiment directory if the tracking uri is local.
                Otherwise returns `None`.
    
            """
            if self._tracking_uri.startswith(LOCAL_FILE_URI_PREFIX):
                return self._tracking_uri[len(LOCAL_FILE_URI_PREFIX) :]
            return None

    

        def finalize(self, status: str = "success") -> None:
            if not self._initialized:
                return
            if status == "success":
                status = "FINISHED"
            elif status == "failed":
                status = "FAILED"
            elif status == "finished":
                status = "FINISHED"
    
            # log checkpoints as artifacts
            if self._checkpoint_callback:
                self._scan_and_log_checkpoints(self._checkpoint_callback)
    
            if self.experiment.get_run(self.run_id):
                self.experiment.set_terminated(self.run_id, status)


        def after_save_checkpoint(self, checkpoint_callback) -> None:
            # log checkpoints as artifacts
            if self._log_model == "all" or self._log_model is True and checkpoint_callback.save_top_k == -1:
                self._scan_and_log_checkpoints(checkpoint_callback)
            elif self._log_model is True:
                self._checkpoint_callback = checkpoint_callback



        def _scan_and_log_checkpoints(self, checkpoint_callback) -> None:
            # get checkpoints to be saved with associated score
            checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)
    
            # log iteratively all new checkpoints
            for t, p, s, tag in checkpoints:
                metadata = {
                    # Ensure .item() is called to store Tensor contents
                    "score": s.item() if isinstance(s, Tensor) else s,
                    "original_filename": Path(p).name,
                    "Checkpoint": {
                        k: getattr(checkpoint_callback, k)
                        for k in [
                            "monitor",
                            "mode",
                            "save_last",
                            "save_top_k",
                            "save_weights_only",
                            "_every_n_train_steps",
                            "_every_n_val_epochs",
                        ]
                        # ensure it does not break if `Checkpoint` args change
                        if hasattr(checkpoint_callback, k)
                    },
                }
                aliases = ["latest", "best"] if p == checkpoint_callback.best_model_path else ["latest"]
    
                # Artifact path on mlflow
                artifact_path = Path(self._checkpoint_path_prefix) / Path(p).stem
    
                # Log the checkpoint
                self.experiment.log_artifact(self._run_id, p, artifact_path)
    
                # Create a temporary directory to log on mlflow
                with tempfile.TemporaryDirectory() as tmp_dir:
                    # Log the metadata
                    with open(f"{tmp_dir}/metadata.yaml", "w") as tmp_file_metadata:
                        yaml.dump(metadata, tmp_file_metadata, default_flow_style=False)
    
                    # Log the aliases
                    with open(f"{tmp_dir}/aliases.txt", "w") as tmp_file_aliases:
                        tmp_file_aliases.write(str(aliases))
    
                    # Log the metadata and aliases
                    self.experiment.log_artifacts(self._run_id, tmp_dir, artifact_path)
    
                # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
                self._logged_model_time[p] = t

        def _init_logger(self):
            ...





def _convert_params(params: Optional[Union[dict[str, Any], Namespace]]) -> dict[str, Any]:
    """Ensure parameters are a dict or convert to dict if necessary.

    Args:
        params: Target to be converted to a dictionary

    Returns:
        params as a dictionary

    """
    # in case converting from namespace
    if isinstance(params, Namespace):
        params = vars(params)

    if params is None:
        params = {}

    return params

def _flatten_dict(params, delimiter: str = "/", parent_key: str = "") -> dict[str, Any]:
    """Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.

    Args:
        params: Dictionary containing the hyperparameters
        delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.

    Returns:
        Flattened dict.

    Examples:
        >>> _flatten_dict({'a': {'b': 'c'}})
        {'a/b': 'c'}
        >>> _flatten_dict({'a': {'b': 123}})
        {'a/b': 123}
        >>> _flatten_dict({5: {'a': 123}})
        {'5/a': 123}
        >>> _flatten_dict({"dl": [{"a": 1, "c": 3}, {"b": 2, "d": 5}], "l": [1, 2, 3, 4]})
        {'dl/0/a': 1, 'dl/0/c': 3, 'dl/1/b': 2, 'dl/1/d': 5, 'l': [1, 2, 3, 4]}

    """
    result: dict[str, Any] = {}
    for k, v in params.items():
        new_key = parent_key + delimiter + str(k) if parent_key else str(k)
        if is_dataclass(v) and not isinstance(v, type):
            v = asdict(v)
        elif isinstance(v, Namespace):
            v = vars(v)

        if isinstance(v, MutableMapping):
            result = {**result, **_flatten_dict(v, parent_key=new_key, delimiter=delimiter)}
        # Also handle the case where v is a list of dictionaries
        elif isinstance(v, list) and all(isinstance(item, MutableMapping) for item in v):
            for i, item in enumerate(v):
                result = {**result, **_flatten_dict(item, parent_key=f"{new_key}/{i}", delimiter=delimiter)}
        else:
            result[new_key] = v
    return result


def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Returns params with non-primitvies converted to strings for logging.

    >>> import torch
    >>> params = {"float": 0.3,
    ...           "int": 1,
    ...           "string": "abc",
    ...           "bool": True,
    ...           "list": [1, 2, 3],
    ...           "namespace": Namespace(foo=3),
    ...           "layer": torch.nn.BatchNorm1d}
    >>> import pprint
    >>> pprint.pprint(_sanitize_params(params))  # doctest: +NORMALIZE_WHITESPACE
    {'bool': True,
        'float': 0.3,
        'int': 1,
        'layer': "<class 'torch.nn.modules.batchnorm.BatchNorm1d'>",
        'list': '[1, 2, 3]',
        'namespace': 'Namespace(foo=3)',
        'string': 'abc'}

    """
    for k in params:
        if isinstance(params[k], (np.bool_, np.integer, np.floating)):
            params[k] = params[k].item()
        if type(params[k]) not in [bool, int, float, str, Tensor]:
            params[k] = str(params[k])
    return params


def _add_prefix(
    metrics: Mapping[str, Union[Tensor, float]], prefix: str, separator: str
) -> Mapping[str, Union[Tensor, float]]:
    """Insert prefix before each key in a dict, separated by the separator.

    Args:
        metrics: Dictionary with metric names as keys and measured quantities as values
        prefix: Prefix to insert before each key
        separator: Separates prefix and original key name

    Returns:
        Dictionary with prefix and separator inserted before each key

    """
    if not prefix:
        return metrics
    return {f"{prefix}{separator}{k}": v for k, v in metrics.items()}