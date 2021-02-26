"""
This file contains methods to train, evaluate and inference models
"""

import os
import glob
import shutil
import pandas as pd

from utils.RunMode import RunMode
from utils.dataset_utils import load_dataset_file
from plots import plots

from logs.Logger import Logger
from data.DatasetSplitter import DatasetSplitter
from tf_sequences.Sequence import Sequence


def initialize_logger(settings):

    logger = Logger()
    logger.parse_args(params=settings.logger_args)
    logger.start(settings.simulation_folder)
    logger.log(settings.log_message)
    return logger


def initialize_dataset(settings, dataset, logger):

    dataset.parse_args(params=settings.dataset_args)
    dataset.set_logger(logger)

    return dataset


def initialize_data_splitter(settings, logger):

    dataset_splitter = DatasetSplitter()
    dataset_splitter.parse_args(params=settings.dataset_splitter_args)
    dataset_splitter.set_logger(logger)

    return dataset_splitter


def initialize_model(settings, models_collection, logger, train_sequence=None, val_sequence=None):

    model_name = settings.model_args["model_name"]
    model_fn = models_collection[model_name]
    model = model_fn()

    model.parse_args(params=settings.model_args)
    model.set_logger(logger)

    if train_sequence is not None:
        model.set_train_sequence()

    if val_sequence is not None:
        model.set_val_sequence()

    model.build()

    return model


def initialize_sequence(settings, logger, dataset_df, dataset):

    sequence = Sequence()
    sequence.parse_args(params=settings.sequence_args)
    sequence.set_logger(logger)
    sequence.set_dataset_df(dataset_df)
    sequence.set_dataset(dataset)

    return sequence


def train(settings, dataset, models_collection):

    # Create simulations directory
    if not os.path.exists(settings.simulation_folder):
        os.makedirs(settings.simulation_folder)

    # Copy settings file to simulation directory
    shutil.copyfile(settings.settings_file_name, os.path.join(settings.simulation_folder, os.path.basename(settings.settings_file_name)))

    # Initialize logger, dataset and data splitter
    logger = initialize_logger(settings)
    dataset = initialize_dataset(settings, dataset, logger)
    data_splitter = initialize_data_splitter(settings, logger)

    # Split dataset to folds and farther to training, validation and test partitions
    # TODO: Set data instead splitting it
    data_splitter.split_dataset()

    # Train model for each fold
    for fold in settings.training_folds:

        logger.log("Training model for fold: {0}".format(fold))

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

        # Save training, validation and test data for current fold
        data_splitter.save_dataframes_for_fold(fold_simulation_folder, fold)

        # Initialize training and validation sequences for current fold
        train_sequence = initialize_sequence(data_splitter.train_df_list[fold], dataset)
        val_sequence = initialize_sequence(data_splitter.val_df_list[fold], dataset)

        # Build model
        # TODO check loading model and its compilation
        if settings.load_model:
            settings.model_args["model_name"] = "base_model"  # change settings to load base model
            model = initialize_model(settings, models_collection, logger, train_sequence, val_sequence)
            model.load_model()

        else:
            model = initialize_model(settings, models_collection, logger, train_sequence, val_sequence)
            model.compile()

        # Train model
        model.fit(fold=fold)

        # Load best weights from checkpoint and save model in 'SavedModel' format
        model = initialize_model(settings, models_collection, logger)  # workaround to save model without compiling
        model.load_weights()
        model.save_model()

        # Visualize training metrics
        plots.training_plot(training_log_path=os.path.join(fold_simulation_folder, settings.training_log_name),
                            plot_metrics=settings.plot_metrics,
                            output_dir=fold_simulation_folder)

        # Change 'load_model_path' of the model to the best model saved during training
        model.load_model_path = os.path.join(fold_simulation_folder, settings.saved_model_name)

        # Calculate predictions
        test_predictions = model.predict(run_mode=RunMode.TEST, fold=fold)

        # Get test data
        test_data = generator.get_pair(run_mode=RunMode.TEST, preprocess=True, augment=False, get_label=True, get_data=True, fold=fold)
        original_test_data = generator.get_pair(run_mode=RunMode.TEST, preprocess=False, augment=False, get_label=True, get_data=True, fold=fold)

        # Apply postprocessing
        postprocessed_test_predictions = dataset.apply_postprocessing(test_predictions, test_data, original_test_data,
                                                                      generator.test_info[fold], fold, run_mode=RunMode.TRAINING)

        # Calculate metrics
        dataset.calculate_fold_metrics(postprocessed_test_predictions, test_data, original_test_data,
                                       generator.test_info[fold], fold, fold_simulation_folder)

        # Save tested data
        if settings.save_tested_data:
            dataset.save_tested_data(postprocessed_test_predictions, test_data, original_test_data,
                                     generator.test_info[fold], fold, fold_simulation_folder)

    dataset.log_metrics(settings.simulation_folder)
    logger.end()


def test(settings,
         dataset,
         generator,
         models_collection):

    # Create simulations directory
    if not os.path.exists(settings.simulation_folder):
        os.makedirs(settings.simulation_folder)

    # Copy settings file to simulation directory
    # Need to be deleted when settings will be serialized to .json file
    shutil.copyfile(settings.settings_file_name, os.path.join(settings.simulation_folder, os.path.basename(settings.settings_file_name)))

    # Initialize logger, dataset and generator
    settings, logger = _initialize_logger(settings)
    settings, dataset = _initialize_dataset(settings, dataset, run_mode=RunMode.TEST)
    settings, generator = _initialize_generator(settings, generator)

    if settings.test_simulation:  # Set test data that were generated during inference
        test_files = [os.path.join(settings.simulation_folder, str(fold), generator.test_data_file_name) for fold in settings.training_folds]
        generator.test_info = [load_dataset_file(test_file) for test_file in test_files]

    else:  # Set filtered data info
        generator.test_info = [dataset.filtered_info]
        settings.training_folds = [0]

    # Test model for each fold
    for fold_idx, fold in enumerate(settings.training_folds):

        logger.log("Test model for fold: {0}".format(fold))
        logger.log("Number of samples to test: {0}".format(generator.test_info[fold_idx].shape[0]))

        # Update simulation directory for current fold
        fold_simulation_folder = os.path.join(settings.simulation_folder, str(fold))
        if not os.path.exists(fold_simulation_folder):
            os.makedirs(fold_simulation_folder)

        # Update path to model to be loaded for test for current fold
        if settings.test_simulation:  # Load model saved during simulation
            settings.model_args["load_model_path"] = os.path.join(fold_simulation_folder, settings.saved_model_name)
        else:  # Load model
            settings.model_args["load_model_path"] = settings.load_model_path[fold_idx]

        # Visualize training metrics
        if settings.test_simulation:
            plots.training_plot(training_log_path=os.path.join(fold_simulation_folder, settings.training_log_name),
                                plot_metrics=settings.plot_metrics,
                                output_dir=fold_simulation_folder)

        # Build model
        settings.model_args["model_name"] = "base_model"  # change settings to initialize base model
        model = _initialize_model(settings, models_collection)

        # Calculate predictions
        test_predictions = model.predict(run_mode=RunMode.TEST, fold=fold_idx)

        # Get data
        test_data = generator.get_pair(run_mode=RunMode.TEST, preprocess=True, augment=False, get_label=True, get_data=True, fold=fold_idx)
        original_test_data = generator.get_pair(run_mode=RunMode.TEST, preprocess=False, augment=False, get_label=True, get_data=True, fold=fold_idx)

        # Apply postprocessing
        postprocessed_test_predictions = dataset.apply_postprocessing(test_predictions, test_data, original_test_data,
                                                                      generator.test_info[fold_idx], fold, run_mode=RunMode.TEST)

        # Calculate metrics
        dataset.calculate_fold_metrics(postprocessed_test_predictions, test_data, original_test_data,
                                       generator.test_info[fold_idx], fold, fold_simulation_folder)

        # Save tested data
        if settings.save_tested_data:
            dataset.save_tested_data(postprocessed_test_predictions, test_data, original_test_data,
                                     generator.test_info[fold_idx], fold, fold_simulation_folder)

    dataset.log_metrics(settings.simulation_folder)
    logger.end()


def inference(settings,
              dataset,
              generator,
              models_collection):

    # Create simulations directory
    if not os.path.exists(settings.simulation_folder):
        os.makedirs(settings.simulation_folder)

    # Copy settings file to simulation directory
    # Need to be deleted when settings will be serialized to .json file
    shutil.copyfile(settings.settings_file_name, os.path.join(settings.simulation_folder, os.path.basename(settings.settings_file_name)))

    # Create data definition file if it is not defined
    if not settings.dataset_args["data_definition_file"]:  # empty string results in True

        paths = glob.glob(settings.inference_data_pattern)
        inference_info = pd.DataFrame()
        inference_info[settings.dataset_args["data_path_column"]] = paths

        inference_info_path = os.path.join(settings.simulation_folder, "inference_info.csv")
        inference_info.to_csv(inference_info_path, index=False)

        settings.dataset_args["data_definition_file"] = inference_info_path

    # Initialize logger, dataset and generator
    settings, logger = _initialize_logger(settings)
    settings, dataset = _initialize_dataset(settings, dataset, run_mode=RunMode.INFERENCE)
    settings, generator = _initialize_generator(settings, generator)

    # Split inference data to batches
    dataset.split_inference_data()

    # Update model arguments with model path to load
    settings.model_args["load_model_path"] = settings.load_model_path

    # Copy model files to simulation folder
    shutil.copytree(settings.load_model_path, os.path.join(settings.simulation_folder, "model"))

    # Build model
    settings.model_args["model_name"] = "base_model"  # change settings to build base model
    model = _initialize_model(settings, models_collection)

    for batch_idx in range(len(dataset.inference_info)):
        batch_predictions = model.predict(run_mode=RunMode.INFERENCE, fold=batch_idx)

        inference_info = dataset.inference_info[batch_idx]
        inference_data = generator.get_pair(run_mode=RunMode.INFERENCE, preprocess=True, augment=False, get_label=False, get_data=True, fold=batch_idx)
        original_inference_data = generator.get_pair(run_mode=RunMode.INFERENCE, preprocess=False, augment=False, get_label=False, get_data=True, fold=batch_idx)

        batch_predictions = dataset.apply_postprocessing(batch_predictions, inference_data, original_inference_data, inference_info, batch_idx, run_mode=RunMode.INFERENCE)
        dataset.save_inferenced_data(batch_predictions, inference_data, original_inference_data, inference_info, batch_idx, settings.simulation_folder)

    def get_sequence(self, run_mode, fold=0):

        """
        This method creates instance of Sequence class that can be passed to model fit method
        The Sequence object created for current fold and according to run mode (training or validation)
        :param run_mode: run mode (can be training or validation)
        :param fold: current fold
        :return: Sequence object
        """

        self.sequence_args["dataset"] = self.dataset
        if run_mode == RunMode.TRAINING:
            self.sequence_args["data_info"] = self.train_info[fold]
            self.sequence_args["steps_per_epoch"] = self.train_steps_per_epoch
        else:
            self.sequence_args["data_info"] = self.val_info[fold]
            self.sequence_args["steps_per_epoch"] = self.val_steps_per_epoch

        sequence = Sequence()
        sequence.parse_args(params=self.sequence_args)
        sequence.initialize()

        return sequence

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