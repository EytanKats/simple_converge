import cv2
import numpy as np


class Image(object):

    """
    This class defines common methods to manipulate single channel or 3-channel images.
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        self.path = ""
        self.pixel_data = None

    def load(self, path):

        """
        This method loads image
        :param path: path to image
        :return: None
        """

        self.path = path
        self.pixel_data = cv2.imread(path)

    def resize(self, shape, interpolation=cv2.INTER_LINEAR):

        """
        This method resizes image
        :param shape: new shape of the image: (width, height)
        :param interpolation: interpolation method (default: linear)
        :return: None
        """

        self.pixel_data = cv2.resize(self.pixel_data, shape, interpolation=interpolation)

    def flip_lr(self):

        """
        This method flips image horizontally
        :return: None
        """

        self.pixel_data = cv2.flip(self.pixel_data, flipCode=1)

    def flip_ud(self):

        """
        This method flips image vertically
        :return: None
        """

        self.pixel_data = cv2.flip(self.pixel_data, flipCode=0)

    def rotate(self, angle, interpolation=cv2.INTER_LINEAR):

        """
        This method rotates the image by defined angle
        :param angle: angle to rotate image
        :return: None
        """

        rot_mat = cv2.getRotationMatrix2D((self.pixel_data.shape[1] / 2, self.pixel_data.shape[0] / 2),
                                          angle=angle,
                                          scale=1)

        self.pixel_data = cv2.warpAffine(self.pixel_data,
                                         rot_mat,
                                         (self.pixel_data.shape[1], self.pixel_data.shape[0]),
                                         flags=interpolation)

    def crop(self, top, left, height, width):

        """
        This method crops the image according to parameters
        :param top: top coordinate of the cropped image
        :param left: left coordinate of the cropped image
        :param height: height of the cropped image
        :param width: width of the cropped image
        :return: None
        """

        cur_shape = self.pixel_data.shape
        if top + height > cur_shape[0] or left + width > cur_shape[1]:
            print("Requested crop exceeds the boundaries of the current image")
            return

        self.pixel_data = self.pixel_data[top:top + height, left: left + width, ...]
