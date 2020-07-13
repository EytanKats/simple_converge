from tf_optimizers.SgdOptimizer import SgdOptimizer
from tf_optimizers.AdamOptimizer import AdamOptimizer

optimizers_collection = {

    "sgd_optimizer": SgdOptimizer,
    "adam_optimizer": AdamOptimizer
}