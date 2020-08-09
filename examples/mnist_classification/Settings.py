
class Settings(object):

    def __init__(self):

        # Dataset arguments ##############################################
        self.dataset_args = dict()

        # BaseDataset obligatory fields
        self.dataset_args["data_definition_file"] = "../../../../Datasets/MNIST/annotations/train_data.csv"
        self.dataset_args["data_path_column"] = "image"
        self.dataset_args["filters"] = dict()

        self.dataset_args["preload_labels"] = False
        self.dataset_args["preload_data"] = False

        self.dataset_args["inference_batch_size"] = 64

        # BaseClassesDataset obligatory fields
        self.dataset_args["label_column"] = "label"
        self.dataset_args["apply_class_filters"] = True
        # self.dataset_args["apply_class_filters"] = False  # need to be set to False during inference
        self.dataset_args["class_filters"] = [{"label": {"equal": 0}},
                                              {"label": {"equal": 1}},
                                              {"label": {"equal": 2}},
                                              {"label": {"equal": 3}},
                                              {"label": {"equal": 4}},
                                              {"label": {"equal": 5}},
                                              {"label": {"equal": 6}},
                                              {"label": {"equal": 7}},
                                              {"label": {"equal": 8}},
                                              {"label": {"equal": 9}}]
        self.dataset_args["class_labels"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.dataset_args["class_names"] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        # MNIST dataset specific fields
        self.dataset_args["true_predictions_folder"] = "true_predictions"
        self.dataset_args["false_predictions_folder"] = "false_predictions"
        self.dataset_args["save_true_predictions"] = False
        self.dataset_args["save_false_predictions"] = True

        # Generator arguments ##############################################
        self.generator_args = dict()

        # Generator obligatory fields
        self.generator_args["data_random_seed"] = 1

        self.generator_args["folds_num"] = 1

        self.generator_args["data_info_folder"] = ""  # simulation folder (not fold folder)
        self.generator_args["train_data_file_name"] = "train_data.json"
        self.generator_args["val_data_file_name"] = "val_data.json"
        self.generator_args["test_data_file_name"] = "test_data.json"

        self.generator_args["sample_training_info"] = False  # randomly choose number of rows from train info that was set
        self.generator_args["training_data_rows"] = 0  # number of rows to randomly choose from train info that was set

        self.generator_args["train_split"] = 0.8
        self.generator_args["test_split"] = 0.2  # useful only in 1-fold setting without leave out setting

        self.generator_args["leave_out"] = False  # allows to choose for test data with unique values of 'self.leave_out_param'
        self.generator_args["leave_out_param"] = ""
        self.generator_args["leave_out_values"] = list()

        self.generator_args["set_info"] = False  # set training, validation and test data info
        self.generator_args["set_test_data_info"] = False  # set only test data info
        self.generator_args["set_test_data_param"] = ""  # parameter based on which training-validation data will be cleaned from test samples

        # Sequence arguments obligatory fields
        self.generator_args["sequence_args"] = dict()

        self.generator_args["sequence_args"]["batch_size"] = 32
        self.generator_args["sequence_args"]["apply_augmentations"] = True

        self.generator_args["sequence_args"]["multi_input"] = False
        self.generator_args["sequence_args"]["multi_output"] = False
        self.generator_args["sequence_args"]["inputs_num"] = 1
        self.generator_args["sequence_args"]["outputs_num"] = 1

        # ClassesSequence obligatory fields
        self.generator_args["sequence_args"]["subsample"] = False
        self.generator_args["sequence_args"]["subsampling_param"] = "label"
        self.generator_args["sequence_args"]["oversample"] = True
        self.generator_args["sequence_args"]["oversampling_param"] = "label"

        # Model arguments
        self.model_args = dict()

        # Base model arguments
        self.model_args["model_name"] = "toy_classification_net"
        self.model_args["load_weights_path"] = ""  # will be set by training script
        self.model_args["epochs"] = 3000
        self.model_args["steps_per_epoch"] = 0  # training script set this parameter according to samples number
        self.model_args["val_steps"] = 0  # training script set this parameter according to samples number
        self.model_args["prediction_batch_size"] = 64

        # Toy classification model arguments
        self.model_args["input_shape"] = (28, 28, 1)

        # Losses arguments
        self.model_args["regularizer_args"] = dict()
        self.model_args["regularizer_args"]["regularizer_name"] = "l1_l2_regularizer"
        self.model_args["regularizer_args"]["l1_reg_factor"] = 1e-2
        self.model_args["regularizer_args"]["l2_reg_factor"] = 1e-2

        # Losses arguments
        self.model_args["losses_args"] = list()

        self.model_args["losses_args"].append(dict())
        self.model_args["losses_args"][0]["metric_name"] = "cross_entropy_metric"
        self.model_args["losses_args"][0]["categorical_cross_entropy"] = True
        self.model_args["losses_args"][0]["loss_weight"] = 1

        # Optimizer arguments
        self.model_args["optimizer_args"] = dict()
        self.model_args["optimizer_args"]["optimizer_name"] = "adam_optimizer"
        self.model_args["optimizer_args"]["learning_rate"] = 1e-2

        # Metrics arguments
        self.model_args["metrics_args"] = list()

        # Callback arguments
        self.model_args["callbacks_args"] = list()

        # Checkpoint callback arguments
        self.model_args["callbacks_args"].append(dict())
        self.model_args["callbacks_args"][0]["callback_name"] = "checkpoint_callback"
        self.model_args["callbacks_args"][0]["checkpoint_weights_path"] = ""  # will be set by training script
        self.model_args["callbacks_args"][0]["save_best_only"] = True
        self.model_args["callbacks_args"][0]["monitor"] = "val_loss"

        # CSV Logger callback arguments
        self.model_args["callbacks_args"].append(dict())
        self.model_args["callbacks_args"][1]["callback_name"] = "csv_logger_callback"
        self.model_args["callbacks_args"][1]["training_log_path"] = ""  # will be set by training script

        # Early stopping callback arguments
        self.model_args["callbacks_args"].append(dict())
        self.model_args["callbacks_args"][2]["callback_name"] = "early_stopping_callback"
        self.model_args["callbacks_args"][2]["patience"] = 20
        self.model_args["callbacks_args"][2]["monitor"] = "val_loss"

        # Reduce learning rate callback arguments
        self.model_args["callbacks_args"].append(dict())
        self.model_args["callbacks_args"][3]["callback_name"] = "reduce_lr_on_plateau"
        self.model_args["callbacks_args"][3]["reduce_factor"] = 0.8
        self.model_args["callbacks_args"][3]["patience"] = 4
        self.model_args["callbacks_args"][3]["min_lr"] = 1e-4
        self.model_args["callbacks_args"][3]["monitor"] = "val_loss"

        # Logger arguments ##############################################
        self.logger_args = dict()

        self.logger_args["message_format"] = "%(message)s"
        self.logger_args["file_name"] = "results.log"

        # Output settings
        self.simulation_folder = "../../../../Simulations/MNIST/test"
        self.save_tested_data = True
        self.weights_name = "weights.h5"
        self.training_log_name = "metrics.log"
        self.settings_file_name = "Settings.py"
        self.model_architecture_file_name = "architecture.json"
        self.saved_model_folder_name = "model"

        # Test settings
        self.test_simulation = True
        self.test_data_info = [""]

        # Inference settings
        self.inference_data_pattern = ""

        # Training settings
        self.training_folds = [0]
        self.load_weights = False
        self.load_weights_path = ""  # list for train/test, string for inference

        self.logs_dir = self.simulation_folder
        self.log_message = "MNIST classification\n" \
                           "Training dataset: Datasets/MNIST/annotations/train_data.csv\n"

        self.plot_metrics = ["loss"]

