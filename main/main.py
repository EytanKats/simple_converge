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


def _initialize_dataset(settings, dataset):

    # Initialize dataset
    dataset.parse_args(params=settings.dataset_args)
    dataset.initialize_dataset()

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
    settings, dataset = _initialize_dataset(settings, dataset)
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
        if settings.load_weights:
            settings.model_args["load_weights_path"] = settings.load_weights_path[fold]

        settings.model_args["model_architecture_file_path"] = os.path.join(fold_simulation_folder, settings.model_architecture_file_name)
        settings.model_args["saved_model_folder_path"] = os.path.join(fold_simulation_folder, settings.saved_model_folder_name)

        for callback_args in settings.model_args["callbacks_args"]:

            if callback_args["callback_name"] == "checkpoint_callback":
                callback_args["checkpoint_weights_path"] = os.path.join(fold_simulation_folder, settings.weights_name)

            if callback_args["callback_name"] == "csv_logger_callback":
                callback_args["training_log_path"] = os.path.join(fold_simulation_folder, settings.training_log_name)

        # Save training, validation and test data
        generator.save_split_data(fold_simulation_folder, fold)

        # Build and compile model
        model = _initialize_model(settings, models_collection)
        model.compile()
        if settings.load_weights:
            model.load_weights()

        # Train model
        model.fit(fold=fold)

        # Visualize training metrics
        plots.training_plot(training_log_path=os.path.join(fold_simulation_folder, settings.training_log_name),
                            plot_metrics=settings.plot_metrics,
                            output_dir=fold_simulation_folder)

        # Change 'load_weights_path' of the model to the best weights saved during training
        model.load_weights_path = os.path.join(fold_simulation_folder, settings.weights_name)

        # Save full model and model architecture
        model.save_model()
        model.to_json()

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
    settings, dataset = _initialize_dataset(settings, dataset)
    settings, generator = _initialize_generator(settings, generator)

    if settings.test_simulation:  # Split data to training/validation/test and folds just as before training (not sufficiently robust because depends on specific hardware random)
        generator.split_data()
    else:  # Set prepared data info that contains pairs of samples and annotations
        generator.test_info = [pd.read_json(test_file) for test_file in settings.test_data_info]

    # Test model for each fold
    for fold in settings.training_folds:

        logger.log("Test model for fold: {0}".format(fold))

        # Update simulation directory for current fold
        fold_simulation_folder = os.path.join(settings.simulation_folder, str(fold))
        if not os.path.exists(fold_simulation_folder):
            os.makedirs(fold_simulation_folder)

        # Update path to weights to be loaded for test for current fold
        if settings.test_simulation:  # Use weights generated by simulation
            settings.model_args["load_weights_path"] = os.path.join(fold_simulation_folder, settings.weights_name)
        else:  # Set weights
            settings.model_args["load_weights_path"] = settings.load_weights_path[fold]

        # Visualize training metrics
        if settings.test_simulation:
            plots.training_plot(training_log_path=os.path.join(fold_simulation_folder, settings.training_log_name),
                                plot_metrics=settings.plot_metrics,
                                output_dir=fold_simulation_folder)

        # Build model
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

        inference_info_path = os.path.join(settings.simulation_folder, "inference_info.json")
        inference_info.to_json(inference_info_path)

        settings.dataset_args["data_definition_file"] = inference_info_path

    # Initialize logger, dataset and generator
    settings, logger = _initialize_logger(settings)
    settings, dataset = _initialize_dataset(settings, dataset)
    settings, generator = _initialize_generator(settings, generator)

    # Split inference data to batches
    dataset.split_inference_data()

    # Update model arguments with weights path to load
    settings.model_args["load_weights_path"] = settings.load_weights_path

    # Copy weights file to simulation folder
    shutil.copyfile(settings.load_weights_path, os.path.join(settings.simulation_folder, os.path.basename(settings.load_weights_path)))

    # Build model
    model = _initialize_model(settings, models_collection)

    for batch_idx in range(len(dataset.inference_info)):
        batch_predictions = model.predict(run_mode=RunMode.INFERENCE, fold=batch_idx)

        inference_info = dataset.inference_info[batch_idx]
        inference_data = generator.get_pair(run_mode=RunMode.INFERENCE, preprocess=True, augment=False, get_label=False, get_data=True, fold=batch_idx)
        original_inference_data = generator.get_pair(run_mode=RunMode.INFERENCE, preprocess=False, augment=False, get_label=False, get_data=True, fold=batch_idx)

        batch_predictions = dataset.apply_postprocessing(batch_predictions, inference_data, original_inference_data, inference_info, batch_idx, run_mode=RunMode.INFERENCE)
        dataset.save_inferenced_data(batch_predictions, inference_data, original_inference_data, inference_info, batch_idx, settings.simulation_folder)

