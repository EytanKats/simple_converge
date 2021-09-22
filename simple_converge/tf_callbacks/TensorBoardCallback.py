import tensorflow as tf
from simple_converge.tf_callbacks.BaseCallback import BaseCallback


class TensorBoardCallback(BaseCallback):

    """
    This class encapsulates tensorflow TensorBoard callback
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(TensorBoardCallback, self).__init__()

        self.log_dir = ""
        self.histogram_freq = 0
        self.write_graph = False
        self.write_images = False
        self.update_freq = "epoch"
        self.profile_batch = 2
        self.embeddings_freq = 0
        self.embeddings_metadata = None

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(TensorBoardCallback, self).parse_args(**kwargs)

        if "log_dir" in self.params.keys():
            self.log_dir = self.params["log_dir"]

        if "histogram_freq" in self.params.keys():
            self.histogram_freq = self.params["histogram_freq"]

        if "write_graph" in self.params.keys():
            self.write_graph = self.params["write_graph"]

        if "write_images" in self.params.keys():
            self.write_images = self.params["write_images"]

        if "update_freq" in self.params.keys():
            self.update_freq = self.params["update_freq"]

        if "profile_batch" in self.params.keys():
            self.profile_batch = self.params["profile_batch"]

        if "embeddings_freq" in self.params.keys():
            self.embeddings_freq = self.params["embeddings_freq"]

        if "embeddings_metadata" in self.params.keys():
            self.embeddings_metadata = self.params["embeddings_metadata"]

    def get_callback(self):

        """
        This method returns callback
        :return: callback
        """

        callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                                  histogram_freq=self.histogram_freq,
                                                  write_graph=self.write_graph,
                                                  write_images=self.write_images,
                                                  update_freq=self.update_freq,
                                                  profile_batch=self.profile_batch,
                                                  embeddings_freq=self.embeddings_freq,
                                                  embeddings_metadata=self.embeddings_metadata)

        return callback
