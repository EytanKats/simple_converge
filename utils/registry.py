from simple_converge.apps import Registry as AppsRegistry
from simple_converge.datasets import Registry as DatasetsRegistry
from simple_converge.postprocessors import Registry as PostrpocessorsRegistry


def register_app(app_name, app_class):
    """
    This method adds new application to applications registry

    :param app_name: name of the application that can be used in settings
    :param app_class: class of the application that will be created and used
    :return: None
    """

    AppsRegistry[app_name] = app_class


def register_dataset(dataset_name, dataset_class):
    """
    This method adds new application to applications registry

    :param dataset_name: name of the dataset that can be used in settings
    :param dataset_class: class of the dataset that will be created and used
    :return: None
    """

    DatasetsRegistry[dataset_name] = dataset_class


def register_postprocessor(postprocessor_name, postprocessor_class):
    """
    This method adds new application to applications registry

    :param postprocessor_name: name of the postprocessor that can be used in settings
    :param postprocessor_class: class of the postprocessor that will be created and used
    :return: None
    """

    PostrpocessorsRegistry[postprocessor_name] = postprocessor_class
