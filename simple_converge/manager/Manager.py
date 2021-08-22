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

from simple_converge.tf_models.models_collection import models_collection
from simple_converge.tf_metrics.metrics_collection import metrics_collection
from simple_converge.tf_callbacks.callbacks_collection import callbacks_collection
from simple_converge.tf_optimizers.optimizers_collection import optimizers_collection
from simple_converge.tf_regularizers.regularizers_collection import regularizers_collection


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
        self.collection["models"] = models_collection
        self.collection["metrics"] = metrics_collection
        self.collection["callbacks"] = callbacks_collection
        self.collection["optimizers"] = optimizers_collection
        self.collection["regularizers"] = regularizers_collection

        self.logger = self.create_logger()
        self.mlops_task = self.create_mlops_task()
        self.dataset_splitter = self.create_data_splitter()

        self.initialize_dataset()

    def update_collection(self, collection_id, custom_collection):

        if custom_collection is not None:
            self.logger.log(f'Updating collection of {collection_id}')
            self.collection[collection_id].update(custom_collection)

    def create_logger(self):

        logger = Logger()
        logger.parse_args(params=self.settings.logger_args)
        return logger

    def create_mlops_task(self):

        if not self.settings.mlops_args.use_mlops:
            return

        mlops_task = Task.init(project_name=self.settings.mlops_args.project_name,
                               task_name=self.settings.mlops_args.task_name,
                               auto_connect_frameworks=self.settings.mlops_args.connect_frameworks)

        mlops_task.connect(self.settings.dataset_args, name="DatasetArgs")
        mlops_task.connect(self.settings.data_splitter_args, name="DataSplitterArgs")

        # TODO: split model args to different components
        mlops_task.connect(self.settings.model_args, name="ModelArgs")
        mlops_task.connect(self.settings.model_args["train_sequence_args"], name="TrainSequenceArgs")
        mlops_task.connect(self.settings.model_args["val_sequence_args"], name="ValSequenceArgs")

        return mlops_task

    def create_data_splitter(self):

        if not self.settings.data_splitter_args.use_data_splitter:
            return

        dataset_splitter = DatasetSplitter()
        dataset_splitter.parse_args(params=self.settings.data_splitter_args)
        dataset_splitter.set_logger(self.logger)

        return dataset_splitter

    def create_model(self):

        # Create model instance
        model_name = self.settings.model_args["model_name"]
        model_fn = self.collection["models"][model_name]
        model = model_fn()

        # Initialize model instance
        model.parse_args(params=self.settings.model_args)
        model.set_logger(self.logger)

        return model

    def initialize_dataset(self):

        self.dataset.parse_args(params=self.settings.dataset_args)
        self.dataset.set_logger(self.logger)

    def fit(self,
            custom_models_collection=None,
            custom_metrics_collection=None,
            custom_callbacks_collection=None,
            custom_optimizers_collection=None,
            custom_regularizers_collection=None):

        # Check that simulation folder doesn't exist and create it
        if not os.path.exists(self.settings.manager_args.simulation_folder):
            self.logger.log(f'Creating simulation folder {self.settings.manager_args.simulation_folder}')
            os.makedirs(self.settings.manager_args.settings.simulation_folder)
        else:
            self.logger.log(f'Simulation folder {self.settings.manager_args.simulation_folder} already exists')
            self.logger.log(f'Specify new simulation folder')
            return

        # TODO: Save experiment settings in simulation folder

        # Update collections with custom models, metrics, callbacks, optimizers, regularizers
        self.update_collection("models", custom_models_collection)
        self.update_collection("metrics", custom_metrics_collection)
        self.update_collection("callbacks", custom_callbacks_collection)
        self.update_collection("optimizers", custom_optimizers_collection)
        self.update_collection("regularizers", custom_regularizers_collection)

        # Set custom split of dataset or load entire dataset file and split it
        # to folds and farther to training, validation and test partitions
        if self.settings.manager_args.set_custom_data_split:
            self.dataset_splitter.set_custom_data_split()
        else:
            self.dataset_splitter.load_dataset_file()
            self.dataset_splitter.split_dataset()

        # Train model for each fold
        self.logger.log(f'The model will be trained for {self.settings.manager_args.active_folds} folds')
        for fold in self.settings.manager_args.active_folds:

            # Create simulation directory for current fold
            self.logger.log(f'Creating simulation folder for fold {fold}')
            fold_simulation_folder = os.path.join(self.settings.manager_args.simulation_folder, str(fold))
            os.makedirs(fold_simulation_folder)

            # Save training, validation and test dataframes for current fold
            self.dataset_splitter.save_dataframes_for_fold(fold_simulation_folder, fold)

            # Create model
            self.logger.log(f"Creating model for fold: {fold}")
            model = self.create_model()
            model.create_train_sequence(self.dataset_splitter.train_df_list[fold], self.dataset)
            model.create_val_sequence(self.dataset_splitter.val_df_list[fold], self.dataset)

            # Load start point model or build model from scratch and compile it
            if self.settings.model_args["start_point_model"]:

                self.logger.log(f"Loading start point model and compiling it (if needed)")
                model.load_model(self.settings.model_args["start_point_model_path"])
                if self.settings.model_args["compile_start_point_model"]:
                    model.compile()
            else:
                self.logger.log(f"Building model and compiling it")
                model.build()
                model.compile()

            # Train model
            self.logger.log(f"Training model")
            model.fit()

            # Load best model
            best_model_path = os.path.join(fold_simulation_folder,
                                           self.settings.model_args["callbacks_args"]["output_model_name"])
            model.load_model(best_model_path)

            # TODO: upload model to dedicated storage and write it path to ClearML

            # TODO: add option to test using batches and not all test data at the same time
            # Get test data
            test_data = self.dataset.get_data_batch(batch_df=self.dataset_splitter.test_df_list[fold],
                                                    get_data=True,
                                                    get_label=True,
                                                    augment=False,
                                                    preprocess=True,
                                                    run_mode=RunMode.TEST)

            original_test_data = None
            if self.settings.manager_args["get_original_data"]:
                original_test_data = self.dataset.get_data_batch(batch_df=self.dataset_splitter.test_df_list[fold],
                                                                 get_data=True,
                                                                 get_label=True,
                                                                 augment=False,
                                                                 preprocess=False,
                                                                 run_mode=RunMode.TEST)

            # Calculate predictions
            test_predictions = model.predict(test_data[0])

            # Apply postprocessing
            postprocessed_test_predictions = self.dataset.\
                apply_postprocessing_on_predictions_batch(predictions=test_predictions,
                                                          preprocessed_data_and_labels=test_data,
                                                          not_preprocessed_data_and_labels=original_test_data,
                                                          batch_df=self.dataset_splitter.test_df_list[fold],
                                                          batch_id=fold,
                                                          run_mode=RunMode.TEST)

            # Calculate metrics
            self.dataset.calculate_batch_metrics(postprocessed_predictions=postprocessed_test_predictions,
                                                 preprocessed_data_and_labels=test_data,
                                                 not_preprocessed_data_and_labels=original_test_data,
                                                 batch_df=self.dataset_splitter.test_df_list[fold],
                                                 batch_id=fold,
                                                 output_dir=fold_simulation_folder)

            # Save tested data
            if self.settings.manager_args["settings.save_test_data"]:
                self.dataset.save_data_batch(postprocessed_predictions=postprocessed_test_predictions,
                                             output_dir=fold_simulation_folder,
                                             not_postprocessed_predictions=test_predictions,
                                             preprocessed_data_and_labels=test_data,
                                             not_preprocessed_data_and_labels=original_test_data,
                                             batch_df=self.dataset_splitter.test_df_list[fold],
                                             batch_id=fold)

        self.dataset.aggregate_metrics_for_all_batches(self.settings.manager_args["simulation_folder"])
        self.logger.end()




    # def evaluate(self, settings):
    #
    #     # Create simulations directory
    #     if not os.path.exists(settings.simulation_folder):
    #         os.makedirs(settings.simulation_folder)
    #
    #     # Copy settings file to simulation directory
    #     shutil.copyfile(settings.settings_file_name,
    #                     os.path.join(settings.simulation_folder, os.path.basename(settings.settings_file_name)))
    #
    #     # Initialize logger, dataset and generator
    #     self.initialize_logger(settings)
    #     self.initialize_dataset(settings)
    #     data_splitter = self.initialize_data_splitter(settings)
    #
    #     # TODO: initialize data splitter in initialization method
    #     data_splitter.initialize(run_mode=RunMode.TEST)
    #
    #     if settings.test_simulation:  # Set test data that were generated during inference
    #         test_dataset_files = [os.path.join(settings.simulation_folder, str(fold), data_splitter.test_df_file_name)
    #                               for fold in settings.training_folds]
    #         data_splitter.test_df_list = [load_dataset_file(test_dataset_file)
    #                                       for test_dataset_file in test_dataset_files]
    #
    #     else:  # Set filtered data info
    #         data_splitter.test_df_list = [data_splitter.dataset_df]
    #         settings.training_folds = [0]
    #
    #     # Test model for each fold
    #     for fold_idx, fold in enumerate(settings.training_folds):
    #
    #         self.logger.log("Test model for fold: {0}".format(fold))
    #         self.logger.log("Number of samples to test: {0}".format(data_splitter.test_df_list[fold_idx].shape[0]))
    #
    #         # Update simulation directory for current fold
    #         fold_simulation_folder = os.path.join(settings.simulation_folder, str(fold))
    #         if not os.path.exists(fold_simulation_folder):
    #             os.makedirs(fold_simulation_folder)
    #
    #         # Update path to model to be loaded for test for current fold
    #         # TODO define settings parameters specific to test, for example 'test_load_model_path'
    #         if settings.test_simulation:  # Load model saved during simulation
    #             settings.model_args["load_model_path"] = os.path.join(fold_simulation_folder, settings.saved_model_name)
    #         else:  # Load model
    #             settings.model_args["load_model_path"] = settings.load_model_path[fold_idx]
    #
    #         # Visualize training metrics
    #         if settings.test_simulation:
    #             plots_matplotlib.training_plot(training_log_path=os.path.join(fold_simulation_folder,
    #                                                                           settings.training_log_name),
    #                                            metrics_to_plot=settings.plot_metrics,
    #                                            output_folder=fold_simulation_folder)
    #
    #         # Build model
    #         settings.model_args["model_name"] = "base_model"  # change settings to initialize base model
    #         model = self.initialize_model(settings)
    #         model.load_model()
    #
    #         # Get test data
    #         # TODO split test data to batches
    #         test_data = self.dataset.get_data_batch(batch_df=data_splitter.test_df_list[fold],
    #                                                 get_data=True,
    #                                                 get_label=True,
    #                                                 augment=False,
    #                                                 preprocess=True,
    #                                                 run_mode=RunMode.TEST)
    #
    #         original_test_data = None
    #         if settings.get_original_data:
    #             original_test_data = self.dataset.get_data_batch(batch_df=data_splitter.test_df_list[fold],
    #                                                              get_data=True,
    #                                                              get_label=True,
    #                                                              augment=False,
    #                                                              preprocess=False,
    #                                                              run_mode=RunMode.TEST)
    #
    #         # Calculate predictions
    #         test_predictions = model.predict(test_data[0])
    #
    #         # Apply postprocessing
    #         postprocessed_test_predictions = self.dataset.apply_postprocessing_on_predictions_batch(predictions=test_predictions,
    #                                                                                                 preprocessed_data_and_labels=test_data,
    #                                                                                                 not_preprocessed_data_and_labels=original_test_data,
    #                                                                                                 batch_df=data_splitter.test_df_list[fold],
    #                                                                                                 batch_id=fold,
    #                                                                                                 run_mode=RunMode.TEST)
    #
    #         # Calculate metrics
    #         self.dataset.calculate_batch_metrics(postprocessed_predictions=postprocessed_test_predictions,
    #                                              preprocessed_data_and_labels=test_data,
    #                                              not_preprocessed_data_and_labels=original_test_data,
    #                                              batch_df=data_splitter.test_df_list[fold],
    #                                              batch_id=fold,
    #                                              output_dir=fold_simulation_folder)
    #
    #         # Save tested data
    #         if settings.save_test_data:
    #             self.dataset.save_data_batch(postprocessed_predictions=postprocessed_test_predictions,
    #                                          output_dir=fold_simulation_folder,
    #                                          not_postprocessed_predictions=test_predictions,
    #                                          preprocessed_data_and_labels=test_data,
    #                                          not_preprocessed_data_and_labels=original_test_data,
    #                                          batch_df=data_splitter.test_df_list[fold],
    #                                          batch_id=fold)
    #
    #     self.dataset.aggregate_metrics_for_all_batches(settings.simulation_folder)
    #     self.logger.end()
    #
    # def predict(self, settings):
    #
    #     # Create simulations directory
    #     if not os.path.exists(settings.simulation_folder):
    #         os.makedirs(settings.simulation_folder)
    #
    #     # Copy settings file to simulation directory
    #     shutil.copyfile(settings.settings_file_name,
    #                     os.path.join(settings.simulation_folder, os.path.basename(settings.settings_file_name)))
    #
    #     # Create data definition file if it is not defined
    #     if not settings.inference_args["inference_data_definition_file_path"]:  # empty string results in True
    #
    #         paths = glob.glob(settings.inference_args["inference_data_pattern"])
    #         inference_df = pd.DataFrame()
    #         inference_df[settings.inference_args["data_column"]] = paths
    #
    #         inference_df_file_path = os.path.join(settings.simulation_folder,
    #                                               settings.inference_args["inference_df_file_name"])
    #         inference_df.to_csv(inference_df_file_path, index=False)
    #
    #     else:
    #         inference_df = load_dataset_file(settings.inference_args["inference_data_definition_file_path"])
    #
    #     # Initialize logger, dataset and generator
    #     self.initialize_logger(settings)
    #     self.initialize_dataset(settings)
    #
    #     # Split inference data to batches
    #     inference_df_list = np.array_split(inference_df, settings.inference_args["inference_batch_size"])
    #
    #     # Update model arguments with model path to load
    #     settings.model_args["load_model_path"] = settings.inference_args["inference_load_model_path"]
    #
    #     # Copy model files to simulation folder
    #     shutil.copytree(settings.inference_args["inference_load_model_path"],
    #                     os.path.join(settings.simulation_folder, "model"))
    #
    #     # Build model
    #     settings.model_args["model_name"] = "base_model"  # change settings to initialize base model
    #     model = self.initialize_model(settings)
    #
    #     model.load_model()
    #
    #     for batch_idx in range(len(inference_df_list)):
    #
    #         inference_data = self.dataset.get_data_batch(batch_df=inference_df_list[batch_idx],
    #                                                      get_data=True,
    #                                                      get_label=False,
    #                                                      augment=False,
    #                                                      preprocess=True,
    #                                                      run_mode=RunMode.INFERENCE)
    #
    #         original_inference_data = None
    #         if settings.inference_args["get_original_inference_data"]:
    #             original_inference_data = self.dataset.get_data_batch(batch_df=inference_df_list[batch_idx],
    #                                                                   get_data=True,
    #                                                                   get_label=False,
    #                                                                   augment=False,
    #                                                                   preprocess=True,
    #                                                                   run_mode=RunMode.INFERENCE)
    #
    #         batch_predictions = model.predict(inference_data[0])
    #
    #         # Apply postprocessing
    #         postprocessed_batch_predictions = self.dataset.apply_postprocessing_on_predictions_batch(predictions=batch_predictions,
    #                                                                                                  preprocessed_data_and_labels=inference_data,
    #                                                                                                  not_preprocessed_data_and_labels=original_inference_data,
    #                                                                                                  batch_df=inference_df_list[batch_idx],
    #                                                                                                  batch_id=batch_idx,
    #                                                                                                  run_mode=RunMode.INFERENCE)
    #
    #         # Save tested data
    #         # TODO add to 'save_data_batch_method' 'run_mode' argument
    #         # TODO add 'settings.inference_args["inference_simulation_folder"]' settings entry
    #         if settings.inference_args["save_inference_data"]:
    #             self.dataset.save_data_batch(postprocessed_predictions=postprocessed_batch_predictions,
    #                                          output_dir=settings.simulation_folder,
    #                                          not_postprocessed_predictions=batch_predictions,
    #                                          preprocessed_data_and_labels=inference_data,
    #                                          not_preprocessed_data_and_labels=original_inference_data,
    #                                          batch_df=inference_df_list[batch_idx],
    #                                          batch_id=batch_idx)
