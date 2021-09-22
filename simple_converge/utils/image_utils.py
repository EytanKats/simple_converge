import cv2


def resize(image, shape, interpolation=cv2.INTER_LINEAR):

    """
    This method resizes image
    :param image: image to process
    :param shape: new shape of the image: (width, height)
    :param interpolation: interpolation method (default: linear)
    :return: resized image
    """

    processed_image = cv2.resize(image, shape, interpolation=interpolation)
    return processed_image


def flip_lr(image):

    """
    This method flips image horizontally
    :param image: image to process
    :return: flipped image
    """

    processed_image = cv2.flip(image, flipCode=1)
    return processed_image


def flip_ud(image):

    """
    This method flips image vertically
    :param image: image to process
    :return: flipped image
    """

    processed_image = cv2.flip(image, flipCode=0)
    return processed_image


def rotate(image, angle, interpolation=cv2.INTER_LINEAR):

    """
    This method rotates the image by defined angle
    :param image: image to process
    :param angle: angle to rotate image
    :param interpolation: interpolation method (default: linear)
    :return: rotated image
    """

    rot_mat = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2),
                                      angle=angle,
                                      scale=1)

    processed_image = cv2.warpAffine(image,
                                     rot_mat,
                                     (image.shape[1], image.shape[0]),
                                     flags=interpolation)

    return processed_image


def crop(image, top, left, height, width):

    """
    This method crops the image according to parameters
    :param image: image to process
    :param top: top coordinate of the cropped image
    :param left: left coordinate of the cropped image
    :param height: height of the cropped image
    :param width: width of the cropped image
    :return: cropped image if succeeded, or None if requested crop exceeds the boundaries of the current image
    """

    cur_shape = image.shape
    if top + height > cur_shape[0] or left + width > cur_shape[1]:
        print("Requested crop exceeds the boundaries of the current image")
        return

    processed_image = image[top:top + height, left: left + width, ...]
    return processed_image
