import cv2
import pydicom
import numpy as np
from image.Image import Image


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
        :return: None
        """

        self.path = path
        self.dataset = pydicom.dcmread(path)
        self.pixel_data = self.dataset.pixel_array

        # Currently we support only one color space conversion
        if self._color_space_tag in self.dataset:

            if self.dataset[self._color_space_tag].value == "YBR_FULL_422":

                if len(self.pixel_data.shape) == 4:  # multi_frame
                    converted_data = np.zeros_like(self.pixel_data)
                    for frame in range(self.pixel_data.shape[0]):
                        converted_data[frame] = cv2.cvtColor(self.pixel_data[frame], cv2.COLOR_YCrCb2RGB)
                    self.pixel_data = converted_data

                else:  # single_frame
                    self.pixel_data = cv2.cvtColor(self.pixel_data, cv2.COLOR_YCrCb2RGB)

            if self.dataset[self._color_space_tag].value not in ["YBR_FULL_422", "RGB"]:
                print("Unknown color space of pixel data: '{0}'".format(self.dataset[self._color_space_tag].value))
