import numpy as np

from utils.RunMode import RunMode
from tf_sequences.Sequence import Sequence
from generator import Generator


class ClassesGenerator(Generator.Generator):

    """
    This class expands data generator for categorical classification problems
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(ClassesGenerator, self).__init__()

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(ClassesGenerator, self).parse_args(**kwargs)

    def get_sequence(self, run_mode, fold=0):

        """
        This method creates instance of keras.utils Sequence class that can be passed to model fit method
        The Sequence object created for current fold and according to run mode (training or validation)
        :param run_mode: run mode (can be training or validation)
        :param fold: current fold
        :return: Sequence object
        """

        self.sequence_args["dataset"] = self.dataset
        if run_mode == RunMode.TRAINING:
            self.sequence_args["data_info"] = self.train_info[fold]
        else:
            self.sequence_args["data_info"] = self.val_info[fold]

            # Balance dataset during validation to get more insightful metrics (not sure that it's a right decision)
            self.sequence_args["oversample"] = True
            self.sequence_args["subsample"] = False

        sequence = Sequence()
        sequence.parse_args(params=self.sequence_args)
        sequence.initialize()

        return sequence

    def split_data(self):

        if self.set_info:
            self.set_data_info()
        else:
            self.train_info = [None] * self.folds_num
            self.val_info = [None] * self.folds_num
            self.test_info = [None] * self.folds_num

            np.random.seed(self.data_random_seed)

            self._calculate_leave_out_folds_values(self.dataset.original_info)

            data_info = self.dataset.filtered_info
            for cls_idx, cls_info in enumerate(data_info):
                self._fill_info(cls_info, cls_idx)

            # If set_test_data_info is True _fill_info method will duplicate test_info classes number times
            # To prevent duplication we will replace test info by settings it one another time
            if self.set_test_data_info:
                self.set_test_info()
