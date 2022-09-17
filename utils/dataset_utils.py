import os
import glob
import numpy as np
import pandas as pd
from loguru import logger
from utils.plots_matplotlib import bar_plot, grouped_bar_plot


def load_dataset_file(
        dataset_file_path: str) -> pd.DataFrame:

    """
    This method loads dataset file as pandas DataFrame\n
    Supported formats are CSV and JSON

    :param dataset_file_path: path of the dataset file
    :return: pandas `DataFrame` that contains dataset information
    """

    logger.info(f'Loading dataset file {dataset_file_path}')

    _, file_extension = os.path.splitext(dataset_file_path)
    if file_extension == '.csv':
        data_info = pd.read_csv(dataset_file_path)
    elif file_extension == '.json':
        data_info = pd.read_json(dataset_file_path)
    else:
        raise ValueError(f'Unknown file type: {file_extension}')

    return data_info


def create_dataset(
        data_template: str,
        mask_template: str = '',
        classification: bool = False):

    """
    This method creates pandas DataFrame from the specified files with following columns:
     - data_name
     - data_path
     - mask_path (optional)
     - data_class (optional)
    In segmentation case this method assumes that corresponding data and mask files will have the same index after sorting\n
    In classification case data_class column will contain name of the parent directory of the data file\n
    :param data_template: data template readable by glob
    :param mask_template: mask template readable by glob; if empty mask_path column won't be created
    :param classification: if True data_class column is created
    :return: pandas DataFrame that contains dataset information
    """

    logger.info(f'Creating dataset file')
    data_info = pd.DataFrame()

    # Create data_path column
    data_paths = glob.glob(data_template)
    if len(data_paths) == 0:
        raise FileNotFoundError(f'No files found for data template: {data_template}')

    logger.info(f'Found {len(data_paths)} data files')
    data_paths.sort()
    data_info['data_path'] = data_paths

    # Create data_name column
    data_names = [os.path.basename(path) for path in data_paths]
    data_info["data_name"] = data_names

    # Create mask_path column
    if mask_template:
        mask_paths = glob.glob(mask_template)
        if len(mask_paths) == 0:
            raise FileNotFoundError(f'No files found for data template: {mask_template}')

        logger.info(f'Found {len(mask_paths)} mask files')
        assert len(mask_paths) == len(data_paths), f'Number of data files {len(data_paths)} is NOT equal to number of mask files {len(mask_paths)}'

        mask_paths.sort()
        data_info['mask_path'] = mask_paths

    # Create data_class column
    if classification:
        data_classes = [os.path.basename(os.path.dirname(path)) for path in data_paths]
        data_info['data_class'] = data_classes

    return data_info


def analyze_dataset(dataset, save_plots=False, output_plots_dir=""):

    """
    This method performs analysis of the dataset while assuming that dataset contains 'group' and optionally 'class'
    columns.
    :param dataset: pandas dataframe that describes dataset
    :param save_plots: if True the plots of dataset analysis will be saved
                       if False the plots of dataset analysis will be shown
    :param output_plots_dir: directory to save plots
    :return: None
    """

    groups = dataset["group"].value_counts()
    output_path = os.path.join(output_plots_dir, "groups.png")
    bar_plot(groups, save_plots, output_path)

    # If dataset doesn't contain the class column - return
    if "class" not in dataset.columns:
        return

    classes = dataset["class"].value_counts()
    output_path = os.path.join(output_plots_dir, "classes.png")
    bar_plot(classes, save_plots, output_path)

    group_values = groups.index.values.tolist()
    for group_value in group_values:

        group_value_df = dataset[dataset["group"] == group_value]
        classes_for_group = group_value_df["class"].value_counts()
        output_path = os.path.join(output_plots_dir, "group_" + str(group_value) + ".png")
        bar_plot(classes_for_group, save_plots, output_path)

    class_values = classes.index.values.tolist()
    for class_value in class_values:
        class_value_df = dataset[dataset["class"] == class_value]
        groups_for_class = class_value_df["group"].value_counts()
        output_path = os.path.join(output_plots_dir, "class_" + str(class_value) + ".png")
        bar_plot(groups_for_class, save_plots, output_path)


def analyze_predictions(dataset, save_plots=False, output_dir=""):

    """
    This method performs analysis of classification model predictions while assuming that dataset contains
    'group', 'class', and 'predicted_class' columns.
    :param dataset: pandas dataframe that describes dataset and contains predictions
    :param save_plots: if True the plots of predictions analysis will be saved
                       if False the plots of predictions analysis will be shown
    :param output_dir: directory to save output files
    :return: None
    """

    dataset["result"] = ["true" if row["class"] == row["predicted_class"] else "false" for _, row in dataset.iterrows()]

    groups = dataset["group"].unique()
    for group in groups:
        group_dataset = dataset[dataset["group"] == group]
        results_for_group = group_dataset["result"].value_counts()
        output_path = os.path.join(output_dir, "group_" + str(group) + ".png")
        bar_plot(results_for_group, save_plot=save_plots, output_plot_path=output_path)


def calculate_folds_statistic(metrics_dfs):

    df_concat = pd.concat(metrics_dfs)
    df_groupby_index = df_concat.groupby(df_concat.index)

    df_mean = df_groupby_index.mean()
    df_mean = df_mean.add_suffix('_mean')

    df_std = df_groupby_index.std()
    df_std = df_std.add_suffix('_std')

    df_statistic = pd.concat((df_mean, df_std), axis=1)

    return df_statistic


def plot_folds_metrics(metrics_dfs, groups_names=None, save_plot=False, output_dir=""):

    if groups_names is None:
        groups_names = [f'{idx}' for idx in range(len(metrics_dfs))]

    metrics_transposed_dfs = [metrics_df.transpose() for metrics_df in metrics_dfs]
    metrics_names = metrics_transposed_dfs[0].columns
    metrics_transposed_dfs = [metrics_df.reset_index() for metrics_df in metrics_transposed_dfs]
    [metrics_df.insert(0, 'group', groups_names[fold]) for fold, metrics_df in enumerate(metrics_transposed_dfs)]
    df_concat = pd.concat(metrics_transposed_dfs)

    for metric_name in metrics_names:
        output_path = os.path.join(output_dir, f'{metric_name}.png')
        grouped_bar_plot(
            data=df_concat,
            x_column_name='index',
            y_column_name=metric_name,
            hue_column_name='group',
            save_plot=save_plot,
            output_plot_path=output_path
        )


def get_sample_weights(dataset, labels, label_column):

    # Calculate not normalized label weights
    total_samples_num = dataset.shape[0]
    label_weights_list = []
    for label in labels:
        label_weights_list.append(total_samples_num / dataset[dataset[label_column] == label].shape[0])

    # Normalize label weights
    norm = np.linalg.norm(label_weights_list)
    label_weights_list = label_weights_list / norm

    # Assign normalized weight for each label
    label_weights_dict = {}
    for label_idx, label in enumerate(labels):
        label_weights_dict[label] = label_weights_list[label_idx]

    # Assign normalized weight for each data sample
    sample_weights = [label_weights_dict[row[label_column]] for _, row in dataset.iterrows()]

    return sample_weights


def subsample(dataframe, sampling_column):

    """
    Subsample dataframe so that each unique value of the sampling_column will have
    a number of occurrences equal to the number of occurrences of the smallest one in the dataframe.
    :param dataframe: dataframe.
    :param sampling_column: column of the dataframe that contains unique values according to which the dataframe
    will be sampled.
    :return: subsampled dataframe
    """

    # Find number of samples in the smallest class
    unique_values = dataframe[sampling_column].unique()
    rows_cnt = []
    for value in unique_values:
        value_info = dataframe.loc[dataframe[sampling_column] == value]
        rows_cnt.append(value_info.shape[0])
    min_count = np.min(rows_cnt)

    # Subsample equal number of samples from all the classes
    subsampled_dataset_df = pd.DataFrame()
    for value in unique_values:
        value_info = dataframe.loc[dataframe[sampling_column] == value]
        value_info_subsampled = value_info.sample(n=min_count)
        subsampled_dataset_df = subsampled_dataset_df.append(value_info_subsampled)

    return subsampled_dataset_df


def oversample(dataframe, sampling_column):

    """
    Oversample dataframe so that each unique value of the sampling_column will have
    a number of occurrences equal to the number of occurrences of the largest one in the dataframe.
    :param dataframe: dataframe.
    :param sampling_column: column of the dataframe that contains unique values according to which the dataframe
    will be sampled.
    :return: oversampled dataframe.
    """

    # Find number of samples in the largest class
    unique_values = dataframe[sampling_column].unique()
    rows_cnt = []
    for value in unique_values:
        value_info = dataframe.loc[dataframe[sampling_column] == value]
        rows_cnt.append(value_info.shape[0])
    max_count = np.max(rows_cnt)

    # Oversample equal number of samples from all the classes
    oversampled_dataset_df = pd.DataFrame()
    for value in unique_values:
        value_info = dataframe.loc[dataframe[sampling_column] == value]
        value_info_oversampled = value_info.sample(n=max_count, replace=True)
        oversampled_dataset_df = oversampled_dataset_df.append(value_info_oversampled)

    oversampled_dataset_df = oversampled_dataset_df.reset_index(drop=True)  # prevent duplicate indexes

    return oversampled_dataset_df


def balance(dataframe, sampling_column, samples_num, replace=True):

    """
    Balance dataframe so that each unique value of the sampling_column will have a number of occurrences
    equal to samples_num.
    :param dataframe: dataframe.
    :param sampling_column: column of the dataframe that contains unique values according to which the dataframe
    will be sampled.
    :param samples_num: number of occurrences to sample for each unique value in sampling_column.
    :param replace: if True sampling will be done with replacement, else without replacement.
    :return: balanced dataframe.
    """

    unique_values = dataframe[sampling_column].unique()

    # Sample equal number of samples from all the classes
    balanced_dataset_df = pd.DataFrame()
    for value in unique_values:
        value_info = dataframe.loc[samples_num[sampling_column] == value]
        value_info_balanced = value_info.sample(n=samples_num, replace=replace)
        balanced_dataset_df = balanced_dataset_df.append(value_info_balanced)

    balanced_dataset_df = balanced_dataset_df.reset_index(drop=True)  # prevent duplicate indexes

    return balanced_dataset_df
