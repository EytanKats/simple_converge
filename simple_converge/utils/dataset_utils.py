import os
import glob
import numpy as np
import pandas as pd
from utils.plots_matplotlib import bar_plot


def load_dataset_file(dataset_file_path):

    """
    This method loads dataset file.
    Currently supported formats are .csv and .json.
    :param dataset_file_path: path of the dataset file
    :return: pandas dataframe that contains dataset information
    """

    _, file_extension = os.path.splitext(dataset_file_path)
    if file_extension == ".csv":
        data_info = pd.read_csv(dataset_file_path)
    elif file_extension == ".json":
        data_info = pd.read_json(dataset_file_path)
    else:
        raise ValueError("Unknown file type: {0}".format(file_extension))

    return data_info


def create_dataset(data_template, mask_template=None, classification=False, save_dataset_file=False, output_dataset_file_path=""):

    """
    This method creates dataset file that contains columns with paths of the data files, group names and optionally paths of mask files and class names
    under the assumption that corresponding files will have the same index after sorting.
    Data paths column will have name: "image".
    Mask paths column will have name: "mask".
    Basename column ("image_basename") will be created.
    Group column with name "group" will be created to distinguish between different groups of data and will contain name of the parent directory
    (in classification case second degree parent directory) of the data file.
    In classification case class column with name "class" will be created and will contain name of the parent directory of the data file
    :param data_template: data template that are readable by glob
    :param mask_template: mask template that are readable by glob
    :param classification: if True classification dataset file will be created ("class" column will be added), if False segmentation dataset will be created
    :param save_dataset_file: if True dataset file will be saved in .csv format
    :param output_dataset_file_path: path of the output dataset file
    :return: pandas dataframe that contains paths of the files in directories
    """

    data_info = pd.DataFrame()

    data_paths = glob.glob(data_template)
    if len(data_paths) == 0:
        print("There is no data for data template: {0}".format(data_template))
        return data_info

    data_paths.sort()
    data_names = [os.path.basename(path) for path in data_paths]

    if classification:
        data_groups = [os.path.basename(os.path.dirname(os.path.dirname(path))) for path in data_paths]
        data_classes = [os.path.basename(os.path.dirname(path)) for path in data_paths]
        data_info["class"] = data_classes
    else:
        data_groups = [os.path.basename(os.path.dirname(path)) for path in data_paths]

    data_info["image_basename"] = data_names
    data_info["image"] = data_paths
    data_info["group"] = data_groups

    if mask_template:
        mask_paths = glob.glob(mask_template)
        if len(mask_paths) == 0:
            print("There is no data for mask template: {0}".format(mask_template))
            return data_info

        mask_paths.sort()
        data_info["mask"] = mask_paths

    if save_dataset_file:

        if not output_dataset_file_path:
            print("Output path is empty")
            return data_info

        data_info.to_csv(output_dataset_file_path, index=False)

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
