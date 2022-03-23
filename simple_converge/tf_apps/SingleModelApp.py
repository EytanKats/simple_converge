import numpy as np
import tensorflow as tf
from loguru import logger
from apps.BaseSingleModelApp import BaseSingleModelApp

default_settings = {
    'epochs': None,
    'monitor': 'not_defined',
    'monitor_regime': 'not_defined',
    'ckpt_freq': 1,
    'ckpt_save_best_only': True,
    'use_early_stopping': False,
    'early_stopping_patience': 10,
    'use_reduce_lr_on_plateau': False,
    'reduce_lr_on_plateau_patience': 3,
    'reduce_lr_on_plateau_factor': 0.8,
    'reduce_lr_on_plateau_min': 1e-6
}


class SingleModelApp(BaseSingleModelApp):

    """
    This class defines common methods to all Tensorflow models
    """

    def __init__(
            self,
            settings,
            model,
            optimizer,
            scheduler,
            loss_fns,
            loss_weights,
            loss_names,
            metric_fns,
            metric_names,
            metric_num
            ):
        
        """
        This method initializes parameters
        :return: None 
        """
        
        super(SingleModelApp, self).__init__(
            settings,
            model,
            optimizer,
            scheduler,
            loss_fns,
            loss_weights,
            loss_names,
            metric_fns,
            metric_names,
            metric_num
        )

        self.ckpt = tf.train.Checkpoint(model=self.model)

    def _get_latest_ckpt(self):
        return tf.train.latest_checkpoint

    def _restore_ckpt(self, ckpt_path):
        self.ckpt.restore(ckpt_path)

    def _save_ckpt(self, ckpt_path):
        self.ckpt.save(ckpt_path)

    def _get_current_lr(self):
        return self.optimizer.learning_rate.numpy()

    def _step(self, data, labels, training):

        with tf.GradientTape() as gradient_tape:

            # Forward pass
            model_output = self.model(data, training=training)

            # Calculate loss
            batch_loss = 0
            batch_loss_list = list()

            if len(self.loss_fns) == 1:
                loss = self.loss_fns[0](tf.cast(tf.convert_to_tensor(labels), dtype=tf.float32), model_output)
                batch_loss_list.append(loss.numpy())
                batch_loss += loss * self.loss_weights[0]
            else:
                for int_loss_idx in range(len(self.loss_fns)):
                    loss = self.loss_fns[int_loss_idx](labels[int_loss_idx], model_output[int_loss_idx])
                    batch_loss_list.append(loss.numpy())
                    batch_loss += loss * self.loss_weights[int_loss_idx]

            # Calculate metrics
            batch_metric_list = list()

            if len(self.metric_fns) == 1:
                for metric_fn in self.metric_fns[0]:
                    metric = metric_fn(tf.cast(tf.convert_to_tensor(labels), dtype=tf.float32), model_output)
                    batch_metric_list.append(metric.numpy())

            else:
                for int_output_idx, metric_fns_for_output in enumerate(self.metric_fns):
                    for metric_fn in metric_fns_for_output:
                        metric = metric_fn(tf.cast(tf.convert_to_tensor(labels[int_output_idx]), dtype=tf.float32), model_output[int_output_idx])
                        batch_metric_list.append(metric.numpy())

        # Backward pass
        if training:
            gradients = gradient_tape.gradient(batch_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return batch_loss_list, batch_metric_list

    def predict(self, data):

        predictions = self.model.predict(np.array(data), verbose=1)
        return predictions

    def summary(self):
        logger.info("Model architecture:")
        self.model.summary(print_fn=lambda x: logger.info(x))
