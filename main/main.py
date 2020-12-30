"""
This file contains methods to train, evaluate and inference models
"""

import os
import glob
import shutil
import pandas as pd

from plots import plots
from logger.Logger import Logger
from utils.RunMode import RunMode
from utils.dataset_utils import load_dataset_file


def _initialize_logger(settings):

    # Initialize logger
    logger = Logger()
    logger.parse_args(params=settings.logger_args)
    logger.start(settings.simulation_folder)
    logger.log(settings.log_message)

    # Set arguments with logger
    settings.model_args["logger"] = logger
    settings.dataset_args["logger"] = logger
    settings.generator_args["logger"] = logger

    return settings, logger


def _initialize_dataset(settings, dataset, run_mode):

    # Initialize dataset
    dataset.parse_args(params=settings.dataset_args)
    dataset.initialize_dataset(run_mode)

    # Set generator arguments with dataset
    settings.generator_args["dataset"] = dataset

    return settings, dataset


def _initialize_generator(settings, generator):

    # Initialize generator
    generator.parse_args(params=settings.generator_args)

    # Set model arguments with generator
    settings.model_args["generator"] = generator

    return settings, generator


def _initialize_model(settings, models_collection):

    # Build model
    model_name = settings.model_args["model_name"]  # get model from models collection
    model_fn = models_collection[model_name]
    model = model_fn()
    model.parse_args(params=settings.model_args)
    model.build()

    return model


def train(settings,
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
    settings, dataset = _initialize_dataset(settings, dataset, run_mode=RunMode.TRAINING)
    settings, generator = _initialize_generator(settings, generator)

    # Split data to training/validation/test and folds
    generator.split_data()

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

        settings.model_args["saved_model_path"] = os.path.join(fold_simulation_folder, settings.saved_model_name)
        settings.model_args["load_weights_path"] = os.path.join(fold_simulation_folder, settings.saved_model_name)

        for callback_args in settings.model_args["callbacks_args"]:

            if callback_args["callback_name"] == "checkpoint_callback":
                callback_args["checkpoint_path"] = os.path.join(fold_simulation_folder, settings.saved_model_name)

            if callback_args["callback_name"] == "csv_logger_callback":
                callback_args["training_log_path"] = os.path.join(fold_simulation_folder, settings.training_log_name)

            if callback_args["callback_name"] == "tensorboard":
                callback_args["log_dir"] = os.path.join(fold_simulation_folder, settings.tensorboard_log_dir)

        # Save training, validation and test data
        generator.save_split_data(fold_simulation_folder, fold)

        # Build model
        if settings.load_model:
            settings.model_args["model_name"] = "base_model"  # change settings to load base model
            model = _initialize_model(settings, models_collection)
            model.load_model()

        else:
            model = _initialize_model(settings, models_collection)
            model.compile()

        # Train model
        model.fit(fold=fold)

        # Load best weights from checkpoint and save model in 'SavedModel' format
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
    for fold in settings.training_folds:

        logger.log("Test model for fold: {0}".format(fold))
        logger.log("Number of samples to test: {0}".format(generator.test_info[fold].shape[0]))

        # Update simulation directory for current fold
        fold_simulation_folder = os.path.join(settings.simulation_folder, str(fold))
        if not os.path.exists(fold_simulation_folder):
            os.makedirs(fold_simulation_folder)

        # Update path to model to be loaded for test for current fold
        if settings.test_simulation:  # Load model saved during simulation
            settings.model_args["load_model_path"] = os.path.join(fold_simulation_folder, settings.saved_model_folder_name)
        else:  # Load model
            settings.model_args["load_model_path"] = settings.load_model_path[fold]

        # Visualize training metrics
        if settings.test_simulation:
            plots.training_plot(training_log_path=os.path.join(fold_simulation_folder, settings.training_log_name),
                                plot_metrics=settings.plot_metrics,
                                output_dir=fold_simulation_folder)

        # Build model
        settings.model_args["model_name"] = "base_model"  # change settings to initialize base model
        model = _initialize_model(settings, models_collection)

        # Calculate predictions
        test_predictions = model.predict(run_mode=RunMode.TEST, fold=fold)

        # Get data
        test_data = generator.get_pair(run_mode=RunMode.TEST, preprocess=True, augment=False, get_label=True, get_data=True, fold=fold)
        original_test_data = generator.get_pair(run_mode=RunMode.TEST, preprocess=False, augment=False, get_label=True, get_data=True, fold=fold)

        # Apply postprocessing
        postprocessed_test_predictions = dataset.apply_postprocessing(test_predictions, test_data, original_test_data,
                                                                      generator.test_info[fold], fold, run_mode=RunMode.TEST)

        # Calculate metrics
        dataset.calculate_fold_metrics(postprocessed_test_predictions, test_data, original_test_data,
                                       generator.test_info[fold], fold, fold_simulation_folder)

        # Save tested data
        if settings.save_tested_data:
            dataset.save_tested_data(postprocessed_test_predictions, test_data, original_test_data,
                                     generator.test_info[fold], fold, fold_simulation_folder)

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
    shutil.copytree(settings.load_model_path, settings.simulation_folder)

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

