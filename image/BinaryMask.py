import cv2
from image.Image import Image


class BinaryMask(Image):

    """
    This class defines methods specific for binary masks
    """

    def resize(self, shape, interpolation=cv2.INTER_NEAREST):

        """
        This method resizes mask
        :param shape: new shape of the mask: (width, height)
        :param interpolation: interpolation method (default: nearest)
        :return: None
        """

        cv2.resize(self.pixel_data, shape, interpolation)
