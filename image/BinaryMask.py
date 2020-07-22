import cv2
import numpy as np
from image.Image import Image


class BinaryMask(Image):

    """
    This class defines methods specific for single channel binary masks
    """

    def get_bounding_box(self):

        """
        This method calculates bounding box for the binary mask
        :return: bounding box coordinates (x1, x2, y1, y2) when x is a row and y is a column
                 if mask doesn't contain any non zero pixels (0, shape[0], 0, shape[1]) will be returned
        """

        h, w = self.pixel_data.shape[:2]
        mask0, mask1 = self.pixel_data.any(0), self.pixel_data.any(1)
        col_start, col_end = mask0.argmax(), w - mask0[::-1].argmax()
        row_start, row_end = mask1.argmax(), h - mask1[::-1].argmax()

        return row_start, row_end, col_start, col_end

    def remove_holes(self):

        """
        This method removes holes inside the the binary mask by assigning '1' to all regions
        that are not reachable from the mask corners
        :return: None
        """

        height, width = self.pixel_data.shape
        postprocessed_mask = np.copy(self.pixel_data)
        im_floodfill = np.copy(postprocessed_mask)

        h, w = postprocessed_mask.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        cv2.floodFill(im_floodfill, mask, (0, 0), 2)
        cv2.floodFill(im_floodfill, mask, (0, height - 1), 2)
        cv2.floodFill(im_floodfill, mask, (width - 1, 0), 2)
        cv2.floodFill(im_floodfill, mask, (width - 1, height - 1), 2)

        postprocessed_mask[im_floodfill != 2] = 1

        self.pixel_data = postprocessed_mask

    def keep_biggest_connected_component(self):

        """
        This method removes all small connected components from the binary mask but keep the biggest one
        :return: None
        """

        postprocessed_mask = np.copy(self.pixel_data)
        cc_with_stats = cv2.connectedComponentsWithStats(postprocessed_mask.astype(np.uint8), 8)

        if len(cc_with_stats[2]) > 2:  # Check that there is at least 2 connected components not including background

            cc_max_area_idx = np.argmax([cc_with_stats[2][1:, 4]]) + 1
            for idx in range(1, cc_with_stats[0]):

                if idx == cc_max_area_idx:
                    continue

                postprocessed_mask[cc_with_stats[1] == idx] = postprocessed_mask[cc_with_stats[1] == idx] - 1

        self.pixel_data = postprocessed_mask

    def remove_small_connected_components(self, size_thr):

        """
        This method removes connected components that are smaller than 'size_thr' from the binary mask
        :param size_thr: threshold that defines size for excluding connected components from binary mask
        :return: None
        """

        postprocessed_mask = np.copy(self.pixel_data)
        cc_with_stats = cv2.connectedComponentsWithStats(postprocessed_mask.astype(np.uint8), 8)

        for idx in range(1, cc_with_stats[0]):

            if cc_with_stats[2][idx, 4] < size_thr:
                postprocessed_mask[cc_with_stats[1] == idx] = postprocessed_mask[cc_with_stats[1] == idx] - 1

        self.pixel_data = postprocessed_mask