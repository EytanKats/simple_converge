from tf_callbacks.CheckpointCallback import CheckpointCallback
from tf_callbacks.CsvLoggerCallback import CsvLoggerCallback
from tf_callbacks.EarlyStoppingCallback import EarlyStoppingCallback
from tf_callbacks.ReduceLrOnPlateauCallback import ReduceLrOnPlateauCallback

callbacks_collection = {

    "checkpoint_callback": CheckpointCallback,
    "csv_logger_callback": CsvLoggerCallback,
    "early_stopping_callback": EarlyStoppingCallback,
    "reduce_lr_on_plateau": ReduceLrOnPlateauCallback
}
