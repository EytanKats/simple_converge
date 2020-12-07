import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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


def apply_filters(dataset, filters):

    """
    This method applies filters on dataset.
    The filters have to be of the following form: {"feature_1": {"action_1_1": value_1_1, "action_1_2": value_1_2}, "feature_2": {...}}:
     - feature is a column name of pandas dataframe
     - actions and values define range of values to filter; can be one of the following: "max_open", "max_close", "min_open", "min_close", "equal"
     - "func" is a specific action that defines mapping from feature values and applied before the filtering; filtering will be applied on mapped feature values
     - filters are applied as 'AND' logic
    :param dataset: pandas dataframe that contains dataset information
    :param filters: dictionary of the filters
    :return: pandas dataframe that contains filtered dataset information
    """

    filtered_info = dataset
    for feature in filters.keys():

        if "func" in filters[feature].keys():
            calculated_feature = filtered_info[feature].apply(filters[feature]["func"])
        else:
            calculated_feature = filtered_info[feature]

        if "max_open" in filters[feature].keys():
            filtered_info = filtered_info.loc[calculated_feature < filters[feature]["max_open"]]

        if "max_close" in filters[feature].keys():
            filtered_info = filtered_info.loc[calculated_feature <= filters[feature]["max_close"]]

        if "min_open" in filters[feature].keys():
            filtered_info = filtered_info.loc[calculated_feature > filters[feature]["min_open"]]

        if "min_close" in filters[feature].keys():
            filtered_info = filtered_info.loc[calculated_feature >= filters[feature]["min_close"]]

        if "equal" in filters[feature].keys():
            filtered_info = filtered_info.loc[calculated_feature == filters[feature]["equal"]]

    return filtered_info


def create_dataset(data_template, mask_template, classification=False, save_dataset_file=False, output_dataset_file_path=""):

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
    _bar_plot(groups, save_plots, output_path)

    # If dataset doesn't contain the class column - return
    if "class" not in dataset.columns:
        return

    classes = dataset["class"].value_counts()
    output_path = os.path.join(output_plots_dir, "classes.png")
    _bar_plot(classes, save_plots, output_path)

    group_values = groups.index.values.tolist()
    for group_value in group_values:

        group_value_df = dataset[dataset["group"] == group_value]
        classes_for_group = group_value_df["class"].value_counts()
        output_path = os.path.join(output_plots_dir, "group_" + str(group_value) + ".png")
        _bar_plot(classes_for_group, save_plots, output_path)

    class_values = classes.index.values.tolist()
    for class_value in class_values:
        class_value_df = dataset[dataset["class"] == class_value]
        groups_for_class = class_value_df["group"].value_counts()
        output_path = os.path.join(output_plots_dir, "class_" + str(class_value) + ".png")
        _bar_plot(groups_for_class, save_plots, output_path)


def _bar_plot(groups, save_plot=False, output_plot_path=""):

    """
    This method build bar plot from the pandas series.
    :param groups: pandas series
                   index of the series contains x-axis names, values of the series defines height of the bars
    :param save_plot: if True the plot will be saved
                       if False the plot will be shown
    :param output_plot_path: path to save plot
    :return: None
    """

    x = groups.index.values.tolist()
    y = groups.values.tolist()

    plt.figure()
    groups_plot = sns.barplot(x=x, y=y)
    for idx, count in enumerate(y):
        groups_plot.text(idx, count, count, color="black", ha="center")

    plt.xticks(range(0, len(x)), x, rotation="vertical")
    plt.tight_layout()  # this prevents clipping of bar names
    if save_plot:
        plt.savefig(output_plot_path)
    else:
        plt.show()

    plt.close()


def rle_to_mask(rle_encoding, shape):

    """
    This method converts RLE encoding to binary mask
    :param rle_encoding: string RLE encoding
    :param shape: shape of the mask
    :return: binary mask
    """

    segments = rle_encoding.split()
    flatten_mask = np.zeros(shape[1] * shape[0], dtype=np.uint8)
    for idx in range(len(segments) // 2):
        start = int(segments[2 * idx]) - 1
        length = int(segments[2 * idx + 1])
        flatten_mask[start: start + length] = 1

    mask = flatten_mask.reshape((shape[1], shape[0]))
    return mask.T


def mask_to_rle(mask):

    """
    This method converts RLE encoding to binary mask
    :param mask: binary mask
    :return: string RLE encoding
    """

    flatten_mask = mask.T.flatten()
    binary_flatten_mask = (flatten_mask > 0).astype(np.int8)
    padded_flatten_mask = np.concatenate([[0], binary_flatten_mask, [0]])

    start_end_indices = np.where(padded_flatten_mask[1:] != padded_flatten_mask[:-1])[0] + 1
    start_indices = start_end_indices[::2]
    lengths = start_end_indices[1::2] - start_end_indices[::2]

    rle_encoding = ""
    for start_idx, length in zip(start_indices, lengths):
        rle_encoding = " ".join([rle_encoding, str(start_idx)])
        rle_encoding = " ".join([rle_encoding, str(length)])

    return rle_encoding
