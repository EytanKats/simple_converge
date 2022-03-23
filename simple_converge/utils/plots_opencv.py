import cv2


def contours_plot(image,
                  masks,
                  colors,
                  thickness=1,
                  show=False,
                  output_path=""):

    """
    This method draws external contours for provided masks on the image
    :param image: RGB image
    :param masks: list of binary masks
    :param colors: list of RGB colors of the contours
    :param thickness: thickness of the contour
    :param show: if True image with contours is shown, else image with contours is saved
    :param output_path: path to save the image
    :return: None
    """

    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    bgr_colors = [(color[2], color[1], color[0]) for color in colors]

    for mask, bgr_color in zip(masks, bgr_colors):
        contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(bgr_image, contours, contourIdx=-1, color=bgr_color, thickness=thickness)

    if show:
        cv2.imshow("win", bgr_image)
        cv2.waitKey(0)
        cv2.destroyWindow("win")
    else:
        cv2.imwrite(output_path, bgr_image)
