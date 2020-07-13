from enum import Enum


class CallbacksEnum(Enum):
    CHECKPOINT = 0
    LOGGER = 1
    EARLY_STOPPING = 2
    REDUCE_LR_ON_PLATEAU = 3
