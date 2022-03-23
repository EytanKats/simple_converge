import pydicom
from pydicom import pixel_data_handlers


color_space_tag = pydicom.tag.Tag(0x0028, 0x0004)


def load(path):

    """
    This method loads DICOM dataset and extracts pixel data in RGB format from it
    :param path: path of DICOM dataset
    :return: numpy array of pixel data if succeeded to load, else - None
    """

    try:
        dataset = pydicom.dcmread(path)
    except OSError as e:
        print("Failed to read Dicom for {0}: {1}".format(path, str(e)))
        return

    try:
        pixel_data = dataset.pixel_array
    except RuntimeError as e:
        print("Failed to read Dicom pixel data for {0}: {1}".format(path, str(e)))
        return

    # Currently only one color space conversion is supported
    if color_space_tag in dataset:

        if dataset[color_space_tag].value == "YBR_FULL_422":

            pixel_data = pixel_data_handlers.convert_color_space(pixel_data, "YBR_FULL_422", "RGB")

        if dataset[color_space_tag].value not in ["YBR_FULL_422", "RGB", "MONOCHROME2"]:
            print("Unknown color space of pixel data for {0}: '{1}' ".format(path, dataset[color_space_tag].value))
            return

    return pixel_data
