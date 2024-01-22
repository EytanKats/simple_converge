"""
This file contains methods to fit and evaluate model and run inference
"""

import os
from loguru import logger
from clearml import TaskTypes

import simple_converge as sc
from simple_converge.trainer import Trainer
from simple_converge.utils.RunMode import RunMode
from simple_converge.mlops.MLOpsTask import MLOpsTask


def _create_output_folder(settings):
    """
    This method checks that simulation folder doesn't exist and create it
    """

    if not os.path.exists(settings["output_folder"]):
        logger.info(f'Create output folder {settings["output_folder"]}.')
        os.makedirs(settings["output_folder"])
        return True

    else:
        logger.info(f'Output folder {settings["output_folder"]} already exists.'
                    f'\nSpecify new output folder.')
        return False


def _get_app(
    settings,
    mlops_task=None,
    architecture=None,
    loss_function=None,
    metric=None,
    scheduler=None,
    optimizer=None,
    app=None
):

    # If 'app' input parameter is None get application from registry
    # else call the custom 'app' method
    if app is None:
        _fold_app = sc.apps.Registry[settings['app']['registry_name']](
            settings,
            mlops_task=mlops_task,
            architecture=architecture,
            loss_function=loss_function,
            metric=metric,
            scheduler=scheduler,
            optimizer=optimizer,
        )
    else:
        _fold_app = app(
                settings,
                mlops_task=mlops_task,
                architecture=architecture,
                loss_function=loss_function,
                metric=metric,
                scheduler=scheduler,
                optimizer=optimizer,
        )

    return _fold_app


def _get_postprocessor(
    settings,
    dataframe,
    mlops_task=None,
    postprocessor=None

):

    # If 'postprocessor' input parameter is None get postprocessor from registry
    # else call the custom 'app' method
    if postprocessor is None:
        _fold_postprocessor = sc.postprocessors.Registry[settings['postprocessor']['registry_name']](
            settings['postprocessor'],
            mlops_task=mlops_task,
            dataframe=dataframe
        )
    else:
        _fold_postprocessor = postprocessor(
            settings['postprocessor'],
            mlops_task=mlops_task,
            dataframe=dataframe
        )

    return _fold_postprocessor


def _predict_fold(
    mlops_task,
    app,
    postprocessor,
    test_data_loader,
    output_folder,
    fold,
    run_mode
):

    for test_data in test_data_loader:

        # Calculate predictions
        test_predictions = app.predict(test_data)

        # Apply postprocessing
        postprocessor.postprocess_predictions(
            predictions=test_predictions,
            data=test_data,
            run_mode=run_mode,
        )

        # Calculate metrics
        if run_mode == RunMode.TEST:
            postprocessor.calculate_metrics(
                output_folder=output_folder,
                fold=fold,
                task=mlops_task
            )

        # Save data
        postprocessor.save_predictions(
            output_folder=output_folder,
            fold=fold,
            task=mlops_task,
            run_mode=run_mode
        )

    # Calculate total metrics
    if run_mode == RunMode.TEST or run_mode == RunMode.TRAINING:
        postprocessor.calculate_total_metrics(
            output_folder=output_folder,
            fold=fold,
            task=mlops_task
        )


def fit(
    settings,
    mlops_task=None,
    architecture=None,
    loss_function=None,
    metric=None,
    scheduler=None,
    optimizer=None,
    app=None,
    train_dataset=None,
    train_loader=None,
    val_dataset=None,
    val_loader=None,
    test_dataset=None,
    test_loader=None,
    postprocessor=None
):

    # Create training MLOps task
    if mlops_task is None:
        mlops_task = MLOpsTask(settings=settings)

    # Create output folder
    if not _create_output_folder(settings['manager']):
        exit(0)

    # Create trainer
    trainer = Trainer(settings['trainer'])

    # Train model for each fold
    logger.info(f'The model will be trained for {settings["manager"]["active_folds"]} folds.')
    for fold in settings["manager"]["active_folds"]:

        logger.info(f'Fold: {fold}.')

        # Create simulation directory for current fold
        logger.info(f'Create simulation folder for current fold.')
        fold_output_folder = os.path.join(settings['manager']['output_folder'], str(fold))
        os.makedirs(fold_output_folder)

        # TODO: find generic method to save data log
        # logger.info(f'Save data log for current fold.')
        # train_loader[fold].dataset.save_dataframe(os.path.join(fold_output_folder, 'train_data'))
        # val_loader[fold].dataset.save_dataframe(os.path.join(fold_output_folder, 'val_data'))
        # if test_loader is not None:
        #     test_loader[fold].dataset.save_dataframe(os.path.join(fold_output_folder, 'test_data'))

        # Get app
        logger.info(f'Get application.')
        fold_app = _get_app(
            settings=settings,
            mlops_task=mlops_task,
            architecture=architecture,
            loss_function=loss_function,
            metric=metric,
            scheduler=scheduler,
            optimizer=optimizer,
            app=app
        )

        # Load start point checkpoint
        if settings['manager']['restore_checkpoint']:
            logger.info(f'Restore checkpoint: {settings["manager"]["restore_checkpoint_path"]}.')
            fold_app.restore_ckpt(settings['manager']['restore_checkpoint_path'])

        # Create checkpoint directory
        logger.info(f'Create checkpoints` folder.')
        fold_ckpt_folder = os.path.join(settings['manager']['output_folder'], str(fold), 'checkpoint')
        os.makedirs(fold_ckpt_folder)

        # Train model
        logger.info(f'Train model.')
        trainer.fit(
            app=fold_app,
            train_data_loader=train_loader[fold],
            val_data_loader=val_loader[fold],
            ckpt_path=os.path.join(fold_ckpt_folder, 'ckpt'),
            fold=fold,
            mlops_task=mlops_task
        )

        # Test model
        if test_loader is not None:
            logger.info(f'Evaluate model.')
            fold_app.restore_ckpt()

            # Create test directory for current fold
            fold_test_output_folder = os.path.join(fold_output_folder, 'test')
            os.makedirs(fold_test_output_folder)

            # Get postprocessor
            fold_postprocessor = _get_postprocessor(
                settings=settings,
                mlops_task=mlops_task,
                postprocessor=postprocessor,
                dataframe=test_loader[fold].dataset.dataframe
            )

            _predict_fold(
                mlops_task,
                fold_app,
                fold_postprocessor,
                test_loader[fold],
                fold_test_output_folder,
                fold,
                run_mode=RunMode.TRAINING)


def predict(
    settings,
    mlops_task=None,
    architecture=None,
    app=None,
    test_dataset=None,
    test_loader=None,
    postprocessor=None,
    run_mode=RunMode.INFERENCE
):

    # Create test MLOps task
    if mlops_task is None:
        mlops_task = MLOpsTask(settings=settings)

    # Test model for each of active fold
    for fold in settings["manager"]["active_folds"]:

        # Get app
        logger.info(f'Get application.')
        fold_app = _get_app(
            settings=settings,
            mlops_task=mlops_task,
            architecture=architecture,
            app=app
        )

        # Get postprocessor
        fold_postprocessor = _get_postprocessor(
            settings=settings,
            mlops_task=mlops_task,
            postprocessor=postprocessor,
            dataframe=test_loader[fold].dataset.dataframe
        )

        # Update simulation directory for current fold
        output_folder = os.path.join(settings['manager']['output_folder'], 'test', str(fold))
        os.makedirs(output_folder)

        # Restore checkpoint
        fold_app.restore_ckpt(settings['test']['checkpoints'][fold])

        # Predict
        logger.info(f'Predict.')
        _predict_fold(
            mlops_task,
            fold_app,
            fold_postprocessor,
            test_loader[fold],
            output_folder,
            fold,
            run_mode=run_mode
        )
