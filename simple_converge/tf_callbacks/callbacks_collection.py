from simple_converge.tf_callbacks.CheckpointCallback import CheckpointCallback
from simple_converge.tf_callbacks.CsvLoggerCallback import CsvLoggerCallback
from simple_converge.tf_callbacks.EarlyStoppingCallback import EarlyStoppingCallback
from simple_converge.tf_callbacks.ReduceLrOnPlateauCallback import ReduceLrOnPlateauCallback
from simple_converge.tf_callbacks.TensorBoardCallback import TensorBoardCallback

callbacks_collection = {

    "checkpoint_callback": CheckpointCallback,
    "csv_logger_callback": CsvLoggerCallback,
    "early_stopping_callback": EarlyStoppingCallback,
    "reduce_lr_on_plateau": ReduceLrOnPlateauCallback,
    "tensorboard": TensorBoardCallback
}
