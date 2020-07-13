import os
import glob
import pandas as pd


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
