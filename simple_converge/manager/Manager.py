"""
This file contains methods to train, evaluate and inference models
"""

import os
from loguru import logger
from utils.RunMode import RunMode


default_manager_settings = {
    'output_folder': './default_output_folder',
    'use_mlops': False,
    'active_folds': [0],
    'restore_checkpoint': False,
    'restore_checkpoint_path': ''
}


class Manager(object):

    """
    This class instantiates and connects building blocks of pipeline (logger, dataset, model, etc.) and
    manages training / testing / inference flow.
    Main responsibilities of this class:
    -
    """

    def __init__(
            self,
            settings,
            mlops_task
    ):

        """
        This method initializes parameters
        :return: None
        """

        self.settings = settings
        self.mlops_task = mlops_task

        if not self.create_output_folder():
            exit(0)

    def create_output_folder(self):

        # Check that simulation folder doesn't exist and create it
        if not os.path.exists(self.settings["output_folder"]):
            print(f'Create output folder {self.settings["output_folder"]}.')
            os.makedirs(self.settings["output_folder"])
            return True

        else:
            print(f'Output folder {self.settings["output_folder"]} already exists.'
                  f'\nSpecify new output folder.')
            return False

    def predict_fold(
            self,
            app,
            postprocessor,
            test_data_loader,
            output_folder,
            fold,
            run_mode
    ):

        for test_data in test_data_loader:

            # Calculate predictions
            test_predictions = app.predict(test_data[0])  # There is an assumption that test_data[0] contains model input data

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
                    task=self.mlops_task
                )

            # Save data
            postprocessor.save_predictions(
                output_folder=output_folder,
                fold=fold,
                task=self.mlops_task,
                run_mode=run_mode
            )

        # Calculate total metrics
        if run_mode == RunMode.TEST:
            postprocessor.calculate_total_metrics(
                output_folder=output_folder,
                fold=fold,
                task=self.mlops_task
            )

    def fit(
            self,
            get_app_fns,
            train_data_loaders,
            val_data_loaders,
            test_data_loaders=None,
            postprocessor=None
    ):

        # Train model for each fold
        logger.info(f'The model will be trained for {self.settings["active_folds"]} folds')
        for fold in self.settings["active_folds"]:

            # Create app
            app = get_app_fns[fold]()

            logger.info(f'Fold: {fold}.')

            # Create simulation directory for current fold
            logger.info(f'Create simulation folder for current fold.')
            fold_output_folder = os.path.join(self.settings["output_folder"], str(fold))
            os.makedirs(fold_output_folder)

            # Save data log
            logger.info(f'Save data log for current fold.')
            train_data_loaders[fold].dataset.save_dataframe(os.path.join(fold_output_folder, 'train_data'))
            val_data_loaders[fold].dataset.save_dataframe(os.path.join(fold_output_folder, 'val_data'))
            if test_data_loaders is not None:
                test_data_loaders[fold].dataset.save_dataframe(os.path.join(fold_output_folder, 'test_data'))

            # Load start point checkpoint
            if self.settings['restore_checkpoint']:
                logger.info(f'Restore checkpoint: {self.settings["restore_checkpoint_path"]}.')
                app.restore(self.settings["restore_checkpoint_path"])

            # Create checkpoint directory
            fold_ckpt_folder = os.path.join(self.settings["output_folder"], str(fold), 'checkpoint')
            os.makedirs(fold_ckpt_folder)

            # Train model
            logger.info(f'Train model.')
            app.fit(
                train_data_loader=train_data_loaders[fold],
                val_data_loader=val_data_loaders[fold],
                ckpt_path=fold_ckpt_folder + 'ckpt',
                fold=fold,
                mlops_task=self.mlops_task
            )

            # Test model
            if postprocessor is not None and test_data_loaders is not None:

                logger.info(f'Evaluate model.')
                app.restore(latest=True)
                self.predict_fold(
                    app,
                    postprocessor,
                    test_data_loaders[fold],
                    fold_output_folder,
                    fold,
                    run_mode=RunMode.TRAINING)

        if postprocessor is not None and test_data_loaders is not None:
            postprocessor.aggregate_predictions_for_all_folds(
                output_folder=self.settings["output_folder"],
                task=self.mlops_task,
                run_mode=RunMode.TRAINING
            )

    def predict(
            self,
            get_app_fns,
            get_postprocessor_fns,
            data_loaders,
            checkpoint_paths,
            run_mode,
    ):

        # Test model for each fold
        for idx in range(len(data_loaders)):

            # Create application and postprocessor
            app = get_app_fns[idx]()
            postprocessor = get_postprocessor_fns[idx]()

            # Update simulation directory for current fold
            output_folder = os.path.join(self.settings['output_folder'], str(idx))
            os.makedirs(output_folder)

            # Restore checkpoint
            logger.info(f'Restore checkpoint {checkpoint_paths[idx]}.')
            app.restore(checkpoint_paths[idx])

            # Predict
            logger.info(f'Predict.')
            self.predict_fold(
                app,
                postprocessor,
                data_loaders[idx],
                output_folder,
                idx,
                run_mode=run_mode
            )
