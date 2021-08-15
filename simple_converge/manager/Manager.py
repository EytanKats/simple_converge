"""
This file contains methods to train, evaluate and inference models
"""

import os
import glob
import shutil
import numpy as np
import pandas as pd

from clearml import Task

from simple_converge.utils.RunMode import RunMode
from simple_converge.utils.dataset_utils import load_dataset_file
from simple_converge.utils import plots_matplotlib

from simple_converge.logs.Logger import Logger
from simple_converge.data.DatasetSplitter import DatasetSplitter
from simple_converge.tf_sequences.Sequence import Sequence

from simple_converge.tf_models.models_collection import models_collection as sc_models_collection
from simple_converge.tf_metrics.metrics_collection import metrics_collection as sc_metrics_collection
from simple_converge.tf_callbacks.callbacks_collection import callbacks_collection as sc_callbacks_collection
from simple_converge.tf_optimizers.optimizers_collection import optimizers_collection as sc_optimizers_collection
from simple_converge.tf_regularizers.regularizers_collection import regularizers_collection as sc_regularizers_collection


class Manager(object):

    """
    This class instantiates and connects building blocks of pipeline (logger, dataset, model, etc.) and
    manages training / testing / inference flow.
    Main responsibilities of this class:
    -
    """

    def __init__(self, settings, dataset):

        """
        This method initializes parameters
        :return: None
        """

        self.settings = settings
        self.dataset = dataset

        self.collection = dict()
        self.collection["models"] = sc_models_collection
        self.collection["metrics"] = sc_metrics_collection
        self.collection["callbacks"] = sc_callbacks_collection
        self.collection["optimizers"] = sc_optimizers_collection
        self.collection["regularizers"] = sc_regularizers_collection

        self.logger = None
        self.dataset_splitter = None
        self.model = None

    def update_collection(self, collection_id, custom_collection):

        if custom_collection is not None:
            self.collection[collection_id].update(custom_collection)

    def initialize_logger(self):

        self.logger = Logger()
        self.logger.parse_args(params=self.settings.logger_args)

    def initialize_dataset(self):

        self.dataset.parse_args(params=self.settings.dataset_args)
        self.dataset.set_logger(self.logger)

    def initialize_data_splitter(self):

        self.dataset_splitter = DatasetSplitter()
        self.dataset_splitter.parse_args(params=self.settings.data_splitter_args)
        self.dataset_splitter.set_logger(self.logger)

    def initialize_model(self, train_sequence=None, val_sequence=None):

        # Get model from collection
        model_name = self.settings.model_args["model_name"]
        model_fn = self.collection["models"][model_name]
        self.model = model_fn()

        self.model.parse_args(params=self.settings.model_args)
        self.model.set_logger(self.logger)

        if train_sequence is not None:
            self.model.set_train_sequence(train_sequence)

        if val_sequence is not None:
            self.model.set_val_sequence(val_sequence)

        self.model.build()

    def initialize_sequence(self, settings, dataset_df):

        sequence = Sequence()
        sequence.parse_args(params=settings.sequence_args)
        sequence.set_logger(self.logger)
        sequence.set_dataset_df(dataset_df)
        sequence.set_dataset(self.dataset)

        sequence.initialize()

        return sequence

    def train(self,
              settings,
              custom_models_collection=None,
              custom_metrics_collection=None,
              custom_callbacks_collection=None,
              custom_optimizers_collection=None,
              custom_regularizers_collection=None):

        # Initialize ClearML task
        # TODO improve ClearML integration
        task = None
        if settings.clear_ml:
            task = Task.init(project_name=settings.clear_ml_project_name,
                             task_name=settings.clear_ml_task_name,
                             auto_connect_frameworks=settings.clear_ml_connect_frameworks)

            task.connect(settings.dataset_args, name="DatasetArgs")
            task.connect(settings.data_splitter_args, name="DataSplitterArgs")
            task.connect(settings.sequence_args, name="SequenceArgs")
            task.connect(settings.model_args, name="ModelArgs")

        # Create simulations directory
        if not os.path.exists(settings.simulation_folder):
            os.makedirs(settings.simulation_folder)

        # Copy settings file to simulation directory
        shutil.copyfile(settings.settings_file_name,
                        os.path.join(settings.simulation_folder, os.path.basename(settings.settings_file_name)))

        # Update collections
        self.update_collection("models", custom_models_collection)
        self.update_collection("metrics", custom_metrics_collection)
        self.update_collection("callbacks", custom_callbacks_collection)
        self.update_collection("optimizers", custom_optimizers_collection)
        self.update_collection("regularizers", custom_regularizers_collection)

        # Initialize logger, dataset and data splitter
        self.initialize_logger(settings)
        self.initialize_dataset(settings)  # TODO: add initialize method to dataset
        data_splitter = self.initialize_data_splitter(settings)

        # Split dataset to folds and farther to training, validation and test partitions
        # TODO: Set data or initialize and split it, and do it in a right place (initialization method)
        data_splitter.initialize(run_mode=RunMode.TRAINING)
        data_splitter.split_dataset()

        # Train model for each fold
        for fold in settings.training_folds:

            self.logger.log("Training model for fold: {0}".format(fold))

            # Update simulation directory for current fold
            fold_simulation_folder = os.path.join(settings.simulation_folder, str(fold))
            if not os.path.exists(fold_simulation_folder):
                os.makedirs(fold_simulation_folder)

            # Update model arguments for current fold
            if settings.load_model:
                settings.model_args["load_model_path"] = settings.load_model_path[fold]

            settings.model_args["save_model_path"] = os.path.join(fold_simulation_folder, settings.saved_model_name)
            settings.model_args["load_weights_path"] = os.path.join(fold_simulation_folder, settings.saved_model_name)

            for callback_args in settings.model_args["callbacks_args"]:

                if callback_args["callback_name"] == "checkpoint_callback":
                    callback_args["checkpoint_path"] = os.path.join(fold_simulation_folder, settings.saved_model_name)

                if callback_args["callback_name"] == "csv_logger_callback":
                    callback_args["training_log_path"] = os.path.join(fold_simulation_folder, settings.training_log_name)

                if callback_args["callback_name"] == "tensorboard":
                    callback_args["log_dir"] = os.path.join(fold_simulation_folder, settings.tensorboard_log_dir)

            # Save training, validation and test dataframes for current fold
            data_splitter.save_dataframes_for_fold(fold_simulation_folder, fold)

            # Initialize training and validation sequences for current fold
            # TODO divide sequence settings to 'training' sequence settings and 'validation' sequence settings
            train_sequence = self.initialize_sequence(settings, data_splitter.train_df_list[fold])

            # Balance dataset during validation by oversampling to get more insightful metrics (not sure that it's a right decision)
            if settings.sequence_args["subsample"]:
                settings.sequence_args["oversample"] = True
                settings.sequence_args["subsample"] = False
            val_sequence = self.initialize_sequence(settings, data_splitter.val_df_list[fold])

            # Build model
            # TODO check loading model and its compilation
            if settings.load_model:

                temp_model_name = settings.model_args["model_name"]  # workaround to support secondary model initialization during saving (need to be initialized with original settings)
                settings.model_args["model_name"] = "base_model"  # change settings to load base model
                model = self.initialize_model(settings,
                                              custom_metrics_collection,
                                              custom_callbacks_collection,
                                              custom_optimizers_collection,
                                              custom_regularizers_collection,
                                              train_sequence,
                                              val_sequence)
                settings.model_args["model_name"] = temp_model_name

                model.load_model()
                model.compile()

            else:
                model = self.initialize_model(settings,
                                              custom_models_collection,
                                              custom_metrics_collection,
                                              custom_callbacks_collection,
                                              custom_optimizers_collection,
                                              custom_regularizers_collection,
                                              train_sequence,
                                              val_sequence)
                model.compile()

            # Train model
            model.fit()

            # Load best weights from checkpoint and save model in 'SavedModel' format
            model = self.initialize_model(settings,  # workaround to save model without compiling
                                          custom_models_collection)

            model.load_weights()
            model.save_model()

            if task:
                task.update_output_model(model_uri="file://" + settings.model_args["save_model_path"])

            # Visualize training metrics
            plots_matplotlib.training_plot(training_log_path=os.path.join(fold_simulation_folder,
                                                                          settings.training_log_name),
                                           metrics_to_plot=settings.plot_metrics,
                                           output_folder=fold_simulation_folder,
                                           clear_ml_task=task)

            # Get test data
            test_data = self.dataset.get_data_batch(batch_df=data_splitter.test_df_list[fold],
                                                    get_data=True,
                                                    get_label=True,
                                                    augment=False,
                                                    preprocess=True,
                                                    run_mode=RunMode.TEST)

            original_test_data = None
            if settings.get_original_data:
                original_test_data = self.dataset.get_data_batch(batch_df=data_splitter.test_df_list[fold],
                                                                 get_data=True,
                                                                 get_label=True,
                                                                 augment=False,
                                                                 preprocess=False,
                                                                 run_mode=RunMode.TEST)

            # Calculate predictions
            test_predictions = model.predict(test_data[0])

            # Apply postprocessing
            postprocessed_test_predictions = self.dataset.apply_postprocessing_on_predictions_batch(predictions=test_predictions,
                                                                                                    preprocessed_data_and_labels=test_data,
                                                                                                    not_preprocessed_data_and_labels=original_test_data,
                                                                                                    batch_df=data_splitter.test_df_list[fold],
                                                                                                    batch_id=fold,
                                                                                                    run_mode=RunMode.TEST)

            # Calculate metrics
            self.dataset.calculate_batch_metrics(postprocessed_predictions=postprocessed_test_predictions,
                                                 preprocessed_data_and_labels=test_data,
                                                 not_preprocessed_data_and_labels=original_test_data,
                                                 batch_df=data_splitter.test_df_list[fold],
                                                 batch_id=fold,
                                                 output_dir=fold_simulation_folder)

            # Save tested data
            if settings.save_test_data:
                self.dataset.save_data_batch(postprocessed_predictions=postprocessed_test_predictions,
                                             output_dir=fold_simulation_folder,
                                             not_postprocessed_predictions=test_predictions,
                                             preprocessed_data_and_labels=test_data,
                                             not_preprocessed_data_and_labels=original_test_data,
                                             batch_df=data_splitter.test_df_list[fold],
                                             batch_id=fold)

        self.dataset.aggregate_metrics_for_all_batches(settings.simulation_folder)
        self.logger.end()

    def test(self, settings):

        # Create simulations directory
        if not os.path.exists(settings.simulation_folder):
            os.makedirs(settings.simulation_folder)

        # Copy settings file to simulation directory
        shutil.copyfile(settings.settings_file_name,
                        os.path.join(settings.simulation_folder, os.path.basename(settings.settings_file_name)))

        # Initialize logger, dataset and generator
        self.initialize_logger(settings)
        self.initialize_dataset(settings)
        data_splitter = self.initialize_data_splitter(settings)

        # TODO: initialize data splitter in initialization method
        data_splitter.initialize(run_mode=RunMode.TEST)

        if settings.test_simulation:  # Set test data that were generated during inference
            test_dataset_files = [os.path.join(settings.simulation_folder, str(fold), data_splitter.test_df_file_name)
                                  for fold in settings.training_folds]
            data_splitter.test_df_list = [load_dataset_file(test_dataset_file)
                                          for test_dataset_file in test_dataset_files]

        else:  # Set filtered data info
            data_splitter.test_df_list = [data_splitter.dataset_df]
            settings.training_folds = [0]

        # Test model for each fold
        for fold_idx, fold in enumerate(settings.training_folds):

            self.logger.log("Test model for fold: {0}".format(fold))
            self.logger.log("Number of samples to test: {0}".format(data_splitter.test_df_list[fold_idx].shape[0]))

            # Update simulation directory for current fold
            fold_simulation_folder = os.path.join(settings.simulation_folder, str(fold))
            if not os.path.exists(fold_simulation_folder):
                os.makedirs(fold_simulation_folder)

            # Update path to model to be loaded for test for current fold
            # TODO define settings parameters specific to test, for example 'test_load_model_path'
            if settings.test_simulation:  # Load model saved during simulation
                settings.model_args["load_model_path"] = os.path.join(fold_simulation_folder, settings.saved_model_name)
            else:  # Load model
                settings.model_args["load_model_path"] = settings.load_model_path[fold_idx]

            # Visualize training metrics
            if settings.test_simulation:
                plots_matplotlib.training_plot(training_log_path=os.path.join(fold_simulation_folder,
                                                                              settings.training_log_name),
                                               metrics_to_plot=settings.plot_metrics,
                                               output_folder=fold_simulation_folder)

            # Build model
            settings.model_args["model_name"] = "base_model"  # change settings to initialize base model
            model = self.initialize_model(settings)
            model.load_model()

            # Get test data
            # TODO split test data to batches
            test_data = self.dataset.get_data_batch(batch_df=data_splitter.test_df_list[fold],
                                                    get_data=True,
                                                    get_label=True,
                                                    augment=False,
                                                    preprocess=True,
                                                    run_mode=RunMode.TEST)

            original_test_data = None
            if settings.get_original_data:
                original_test_data = self.dataset.get_data_batch(batch_df=data_splitter.test_df_list[fold],
                                                                 get_data=True,
                                                                 get_label=True,
                                                                 augment=False,
                                                                 preprocess=False,
                                                                 run_mode=RunMode.TEST)

            # Calculate predictions
            test_predictions = model.predict(test_data[0])

            # Apply postprocessing
            postprocessed_test_predictions = self.dataset.apply_postprocessing_on_predictions_batch(predictions=test_predictions,
                                                                                                    preprocessed_data_and_labels=test_data,
                                                                                                    not_preprocessed_data_and_labels=original_test_data,
                                                                                                    batch_df=data_splitter.test_df_list[fold],
                                                                                                    batch_id=fold,
                                                                                                    run_mode=RunMode.TEST)

            # Calculate metrics
            self.dataset.calculate_batch_metrics(postprocessed_predictions=postprocessed_test_predictions,
                                                 preprocessed_data_and_labels=test_data,
                                                 not_preprocessed_data_and_labels=original_test_data,
                                                 batch_df=data_splitter.test_df_list[fold],
                                                 batch_id=fold,
                                                 output_dir=fold_simulation_folder)

            # Save tested data
            if settings.save_test_data:
                self.dataset.save_data_batch(postprocessed_predictions=postprocessed_test_predictions,
                                             output_dir=fold_simulation_folder,
                                             not_postprocessed_predictions=test_predictions,
                                             preprocessed_data_and_labels=test_data,
                                             not_preprocessed_data_and_labels=original_test_data,
                                             batch_df=data_splitter.test_df_list[fold],
                                             batch_id=fold)

        self.dataset.aggregate_metrics_for_all_batches(settings.simulation_folder)
        self.logger.end()

    def inference(self, settings):

        # Create simulations directory
        if not os.path.exists(settings.simulation_folder):
            os.makedirs(settings.simulation_folder)

        # Copy settings file to simulation directory
        shutil.copyfile(settings.settings_file_name,
                        os.path.join(settings.simulation_folder, os.path.basename(settings.settings_file_name)))

        # Create data definition file if it is not defined
        if not settings.inference_args["inference_data_definition_file_path"]:  # empty string results in True

            paths = glob.glob(settings.inference_args["inference_data_pattern"])
            inference_df = pd.DataFrame()
            inference_df[settings.inference_args["data_column"]] = paths

            inference_df_file_path = os.path.join(settings.simulation_folder,
                                                  settings.inference_args["inference_df_file_name"])
            inference_df.to_csv(inference_df_file_path, index=False)

        else:
            inference_df = load_dataset_file(settings.inference_args["inference_data_definition_file_path"])

        # Initialize logger, dataset and generator
        self.initialize_logger(settings)
        self.initialize_dataset(settings)

        # Split inference data to batches
        inference_df_list = np.array_split(inference_df, settings.inference_args["inference_batch_size"])

        # Update model arguments with model path to load
        settings.model_args["load_model_path"] = settings.inference_args["inference_load_model_path"]

        # Copy model files to simulation folder
        shutil.copytree(settings.inference_args["inference_load_model_path"],
                        os.path.join(settings.simulation_folder, "model"))

        # Build model
        settings.model_args["model_name"] = "base_model"  # change settings to initialize base model
        model = self.initialize_model(settings)

        model.load_model()

        for batch_idx in range(len(inference_df_list)):

            inference_data = self.dataset.get_data_batch(batch_df=inference_df_list[batch_idx],
                                                         get_data=True,
                                                         get_label=False,
                                                         augment=False,
                                                         preprocess=True,
                                                         run_mode=RunMode.INFERENCE)

            original_inference_data = None
            if settings.inference_args["get_original_inference_data"]:
                original_inference_data = self.dataset.get_data_batch(batch_df=inference_df_list[batch_idx],
                                                                      get_data=True,
                                                                      get_label=False,
                                                                      augment=False,
                                                                      preprocess=True,
                                                                      run_mode=RunMode.INFERENCE)

            batch_predictions = model.predict(inference_data[0])

            # Apply postprocessing
            postprocessed_batch_predictions = self.dataset.apply_postprocessing_on_predictions_batch(predictions=batch_predictions,
                                                                                                     preprocessed_data_and_labels=inference_data,
                                                                                                     not_preprocessed_data_and_labels=original_inference_data,
                                                                                                     batch_df=inference_df_list[batch_idx],
                                                                                                     batch_id=batch_idx,
                                                                                                     run_mode=RunMode.INFERENCE)

            # Save tested data
            # TODO add to 'save_data_batch_method' 'run_mode' argument
            # TODO add 'settings.inference_args["inference_simulation_folder"]' settings entry
            if settings.inference_args["save_inference_data"]:
                self.dataset.save_data_batch(postprocessed_predictions=postprocessed_batch_predictions,
                                             output_dir=settings.simulation_folder,
                                             not_postprocessed_predictions=batch_predictions,
                                             preprocessed_data_and_labels=inference_data,
                                             not_preprocessed_data_and_labels=original_inference_data,
                                             batch_df=inference_df_list[batch_idx],
                                             batch_id=batch_idx)
