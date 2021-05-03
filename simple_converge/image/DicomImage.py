import pydicom
from pydicom import pixel_data_handlers

from simple_converge.image.Image import Image


class DicomImage(Image):

    """
    This class defines methods specific for DICOM images
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(DicomImage, self).__init__()

        self.dataset = None

        self._color_space_tag = pydicom.tag.Tag(0x0028, 0x0004)

    def load(self, path):

        """
        This method loads DICOM dataset and extracts pixel data in RGB format from it
        :return: numpy array of pixel data if succeeded to load, else - None
        """

        self.path = path

        try:
            self.dataset = pydicom.dcmread(path)
        except OSError as e:
            print("Failed to read Dicom for {0}: {1}".format(self.path, str(e)))
            return None

        try:
            self.pixel_data = self.dataset.pixel_array
        except RuntimeError as e:
            print("Failed to read Dicom pixel data for {0}: {1}".format(self.path, str(e)))
            return None

        # Currently we support only one color space conversion
        if self._color_space_tag in self.dataset:

            if self.dataset[self._color_space_tag].value == "YBR_FULL_422":

                self.pixel_data = pixel_data_handlers.convert_color_space(self.pixel_data, "YBR_FULL_422", "RGB")

            if self.dataset[self._color_space_tag].value not in ["YBR_FULL_422", "RGB", "MONOCHROME2"]:
                print("Unknown color space of pixel data for {0}: '{1}' ".format(self.path, self.dataset[self._color_space_tag].value))
                self.pixel_data = None
                return None

        return self.pixel_data
