"""
This file contains methods to train, evaluate and inference models
"""

import os
import glob
import numpy as np
import pandas as pd

from clearml import Task

from simple_converge.utils.RunMode import RunMode
from simple_converge.utils.dataset_utils import load_dataset_file

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

        self.output_checkpoint_path = ""

        if not self.create_output_folder():
            exit(0)

        self.logger = self.create_logger()
        self.mlops_task = self.create_mlops_task()
        self.dataset_splitter = self.create_data_splitter()

        self.initialize_dataset()

    def create_output_folder(self):

        # Check that simulation folder doesn't exist and create it
        if not os.path.exists(self.settings.manager_args["output_folder"]):
            print(f'Creating simulation folder {self.settings.manager_args["output_folder"]}')
            os.makedirs(self.settings.manager_args["output_folder"])
            return True

        else:
            print(f'Simulation folder {self.settings.manager_args["output_folder"]} already exists')
            print(f'Specify new simulation folder')
            return False

    def create_logger(self):

        logger = Logger()
        logger.parse_args(params=self.settings.logger_args)
        logger.start(folder=self.settings.manager_args["output_folder"])
        return logger

    def create_mlops_task(self):

        if not self.settings.manager_args["use_mlops"]:
            return

        mlops_task = Task.init(project_name=self.settings.mlops_args["project_name"],
                               task_name=self.settings.mlops_args["task_name"],
                               auto_connect_frameworks=self.settings.mlops_args["connect_frameworks"])

        self.settings.log_settings(task=mlops_task, output_folder=self.settings.manager_args["output_folder"])

        return mlops_task

    def create_data_splitter(self):

        dataset_splitter = DatasetSplitter()
        dataset_splitter.parse_args(params=self.settings.data_splitter_args)
        dataset_splitter.set_logger(self.logger)

        return dataset_splitter

    def initialize_dataset(self):

        self.dataset.parse_args(params=self.settings.dataset_args)
        self.dataset.set_logger(self.logger)

    def update_collection(self, collection_id, custom_collection):

        if custom_collection is not None:
            self.logger.log(f'Updating collection of {collection_id}')
            self.collection[collection_id].update(custom_collection)

    def update_callbacks_settings(self, fold_output_folder):

        for callback_args in self.settings.model_args["callbacks_args"]:

            if callback_args["callback_name"] == "checkpoint_callback":
                self.output_checkpoint_path = os.path.join(fold_output_folder, "checkpoint",
                                                           callback_args["output_weights_name"])
                callback_args["checkpoint_path"] = os.path.join(self.output_checkpoint_path)

            if callback_args["callback_name"] == "tensorboard_callback":
                callback_args["log_dir"] = os.path.join(fold_output_folder, callback_args["log_dir_name"])

    def create_model(self):

        # Create model instance
        model_name = self.settings.model_args["model_name"]
        model_fn = self.collection["models"][model_name]
        model = model_fn()

        # Initialize model instance
        model.parse_args(params=self.settings.model_args)
        model.set_logger(self.logger)

        return model

    def predict_fold(self, fold, fold_output_folder, model, fold_df,
                     split_to_batches=False, batch_size=1,
                     get_original_data=False, evaluate=False, save=False, run_mode=RunMode.TEST):

        # Split data to batches
        if split_to_batches:
            batch_df_list = np.array_split(fold_df, fold_df.shape[0] // batch_size)
        else:
            batch_df_list = [fold_df]

        for batch_df in batch_df_list:

            # Get test data
            test_data = self.dataset.get_data_batch(batch_df=batch_df,
                                                    get_data=True,
                                                    get_label=evaluate,
                                                    augment=False,
                                                    preprocess=True,
                                                    run_mode=run_mode)

            original_test_data = None
            if get_original_data:
                original_test_data = self.dataset.get_data_batch(batch_df=batch_df,
                                                                 get_data=True,
                                                                 get_label=evaluate,
                                                                 augment=False,
                                                                 preprocess=False,
                                                                 run_mode=run_mode)

            # Calculate predictions
            test_predictions = model.predict(test_data[0])

            # Apply postprocessing
            postprocessed_test_predictions = self.dataset. \
                apply_postprocessing_on_predictions_batch(predictions=test_predictions,
                                                          preprocessed_data_and_labels=test_data,
                                                          not_preprocessed_data_and_labels=original_test_data,
                                                          batch_df=batch_df,
                                                          fold=fold,
                                                          run_mode=RunMode.TEST)

            # Calculate metrics
            if evaluate:
                self.dataset.calculate_batch_metrics(postprocessed_predictions=postprocessed_test_predictions,
                                                     preprocessed_data_and_labels=test_data,
                                                     not_preprocessed_data_and_labels=original_test_data,
                                                     batch_df=batch_df,
                                                     fold=fold,
                                                     output_dir=fold_output_folder,
                                                     task=self.mlops_task)

            # Save data
            if save:
                self.dataset.save_data_batch(postprocessed_predictions=postprocessed_test_predictions,
                                             output_dir=fold_output_folder,
                                             not_postprocessed_predictions=test_predictions,
                                             preprocessed_data_and_labels=test_data,
                                             not_preprocessed_data_and_labels=original_test_data,
                                             batch_df=batch_df,
                                             fold=fold,
                                             run_mode=run_mode,
                                             task=self.mlops_task)

        self.dataset.aggregate_predictions_for_all_batches(fold=fold,
                                                           output_dir=fold_output_folder,
                                                           task=self.mlops_task)

    def fit(self,
            custom_models_collection=None,
            custom_metrics_collection=None,
            custom_callbacks_collection=None,
            custom_optimizers_collection=None,
            custom_regularizers_collection=None):

        # Update collections with custom models, metrics, callbacks, optimizers, regularizers
        self.update_collection("models", custom_models_collection)
        self.update_collection("metrics", custom_metrics_collection)
        self.update_collection("callbacks", custom_callbacks_collection)
        self.update_collection("optimizers", custom_optimizers_collection)
        self.update_collection("regularizers", custom_regularizers_collection)

        # Set custom split of dataset or load entire dataset file and split it
        # to folds and farther to training, validation and test partitions
        if self.settings.manager_args["set_custom_data_split"]:
            self.dataset_splitter.set_custom_data_split(self.settings.manager_args["train_data_files"],
                                                        self.settings.manager_args["val_data_files"],
                                                        self.settings.manager_args["test_data_files"])
        else:
            self.dataset_splitter.load_dataset_file()
            self.dataset_splitter.split_dataset()

        # Train model for each fold
        self.logger.log(f'The model will be trained for {self.settings.manager_args["active_folds"]} folds')
        for fold in self.settings.manager_args["active_folds"]:

            # Create simulation directory for current fold
            self.logger.log(f'Creating simulation folder for fold {fold}')
            fold_output_folder = os.path.join(self.settings.manager_args["output_folder"], str(fold))
            os.makedirs(fold_output_folder)

            # Save training, validation and test dataframes for current fold
            self.dataset_splitter.save_dataframes_for_fold(fold_output_folder, fold)

            # Update callbacks output directories for current fold
            self.update_callbacks_settings(fold_output_folder)

            # Create model
            self.logger.log(f"Creating model for fold: {fold}")
            model = self.create_model()
            model.create_train_sequence(self.dataset_splitter.train_df_list[fold], self.dataset)
            model.create_val_sequence(self.dataset_splitter.val_df_list[fold], self.dataset)

            # Set collections to model
            model.set_metrics_collection(self.collection["metrics"])
            model.set_callbacks_collection(self.collection["callbacks"])
            model.set_optimizers_collection(self.collection["optimizers"])
            model.set_regularizers_collection(self.collection["regularizers"])

            # Load start point model or build model from scratch and compile it
            if self.settings.manager_args["start_point_model"]:

                self.logger.log(f"Loading start point model and compiling it (if needed)")
                model.load_model(self.settings.manager_args["start_point_model_path"])
                if self.settings.manager_args["compile_start_point_model"]:
                    model.compile()
            else:
                self.logger.log(f"Building model and compiling it")
                model.build()
                model.compile()

            # Train model
            self.logger.log(f"Training model")
            model.fit()

            # Load best weights and save the entire model
            model.load_weights(self.output_checkpoint_path)
            model.save_model(os.path.join(fold_output_folder, "model"))

            # TODO: upload model to dedicated storage and write it path to ClearML

            # Test model
            if self.settings.manager_args["evaluate_at_the_end_of_training"]:

                self.predict_fold(fold, fold_output_folder,
                                  model, self.dataset_splitter.test_df_list[fold],
                                  split_to_batches=self.settings.manager_args["split_test_data_to_batches"],
                                  batch_size=self.settings.manager_args["test_batch_size"],
                                  get_original_data=self.settings.manager_args["get_original_data_during_test"],
                                  evaluate=True,
                                  save=self.settings.manager_args["save_test_data"],
                                  run_mode=RunMode.TEST)

        if self.settings.manager_args["evaluate_at_the_end_of_training"]:
            self.dataset.aggregate_predictions_for_all_folds(self.settings.manager_args["output_folder"],
                                                             task=self.mlops_task)
        self.logger.end()

    def evaluate(self):

        # If 'test_simulation' flag is true load test partition from simulation in 'output_folder'
        # Else create new 'output_folder' and 'load' custom test data files
        if self.settings.manager_args["test_simulation"]:
            self.logger.log(f'Loading test data files from simulation'
                            f' {self.settings.manager_args["simulation_folder"]}')

            test_data_files_paths = [os.path.join(self.settings.manager_args["simulation_folder"],
                                                  str(fold), self.settings.data_splitter_args["test_df_file_name"])
                                     for fold in self.settings.manager_args["active_folds"]]
            test_data_files = [load_dataset_file(file_path) for file_path in test_data_files_paths]

            models_paths = [os.path.join(self.settings.manager_args["simulation_folder"], str(fold), "checkpoint", "weights")
                            for fold in self.settings.manager_args["active_folds"]]

        else:
            self.logger.log(f'Loading custom test data files {self.settings.manager_args["test_data_files"]}')

            test_data_files_paths = self.settings.manager_args["test_data_files"]
            test_data_files = [load_dataset_file(file_path) for file_path in test_data_files_paths]
            self.dataset_splitter.set_custom_data_split([], [], test_data_files)

            models_paths = self.settings.manager_args["test_models_paths"]

        # Test model for each fold
        for fold_idx, fold in enumerate(self.settings.manager_args["active_folds"]):

            self.logger.log(f'Test model {models_paths[fold_idx]} for file: {test_data_files_paths[fold_idx]}')
            self.logger.log(f'Number of samples to test: {test_data_files[fold_idx].shape[0]}')

            # Update simulation directory for current fold
            fold_output_folder = os.path.join(self.settings.manager_args['output_folder'], str(fold))
            os.makedirs(fold_output_folder)

            # Load model
            self.logger.log(f"Creating model and loading weights")
            model = self.create_model()
            model.build()
            model.load_weights(models_paths[fold_idx])

            # Evaluate
            self.predict_fold(fold, fold_output_folder,
                              model, test_data_files[fold],
                              split_to_batches=self.settings.manager_args["split_test_data_to_batches"],
                              batch_size=self.settings.manager_args["test_batch_size"],
                              get_original_data=self.settings.manager_args["get_original_data_during_test"],
                              evaluate=True, save=self.settings.manager_args["save_test_data"],
                              run_mode=RunMode.TEST)

        self.dataset.aggregate_predictions_for_all_folds(self.settings.manager_args["output_folder"],
                                                         task=self.mlops_task)
        self.logger.end()

    def predict(self):

        # Create data definition file if it is not defined
        if not self.settings.manager_args["inference_data_definition_file_path"]:

            paths = glob.glob(self.settings.manager_args["inference_data_pattern"])
            inference_df = pd.DataFrame()
            inference_df[self.settings.manager_args["inference_data_column"]] = paths

            inference_df_file_path = os.path.join(self.settings.manager_args["output_folder"],
                                                  self.settings.manager_args["inference_df_file_name"])
            inference_df.to_csv(inference_df_file_path, index=False)

        else:
            inference_df = load_dataset_file(self.settings.manager_args["inference_data_definition_file_path"])

        # Load model

        # TODO: change loading weights to loading entire model (weights, architecture, training conditions), probably bug in TF

        self.logger.log(f"Creating model and loading weights")
        model = self.create_model()
        model.build()
        model.load_weights(self.settings.manager_args["inference_model_path"])

        self.predict_fold(0, self.settings.manager_args["output_folder"],
                          model, inference_df,
                          split_to_batches=self.settings.manager_args["split_inference_data_to_batches"],
                          batch_size=self.settings.manager_args["inference_batch_size"],
                          get_original_data=self.settings.manager_args["get_original_data_during_inference"],
                          evaluate=False, save=self.settings.manager_args["save_inference_data"],
                          run_mode=RunMode.INFERENCE)

