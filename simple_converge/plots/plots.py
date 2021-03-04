import os
import cv2
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import color as ski_color

from simple_converge.metrics import metrics


def line_plot(output_path,
              x_points,
              y_points,
              x_lims=None,
              y_lims=None,
              title="",
              x_label="",
              y_label="",
              legend=None):

    ax_acc = plt.figure(1)
    ax_acc.set_facecolor("gainsboro")
    plt.title(title)
    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if x_lims is not None:
        plt.xlim(x_lims)
    if y_lims is not None:
        plt.ylim(y_lims)

    colors = ['b', 'r', 'g']
    for points_idx, points in enumerate(y_points):

        if legend is not None:
            legend_label = legend[points_idx]
        else:
            legend_label = None

        plt.plot(x_points, points, colors[points_idx], label=legend_label)

    if legend is not None:
        plt.legend(loc="best")

    plt.savefig(output_path)
    plt.close()


def training_plot(training_log_path,
                  plot_metrics,
                  output_dir,
                  x_lims=None,
                  y_lims=None,
                  clear_ml_task=None):

    df = pd.read_csv(training_log_path)

    epochs = df["epoch"]
    for idx, metric in enumerate(plot_metrics):
        train_metrics = df[metric]
        val_metrics = df["val_" + metric]

        ax_acc = plt.figure(1)
        ax_acc.set_facecolor("gainsboro")
        plt.title(metric)
        plt.grid(True)
        plt.xlabel("epochs")
        plt.ylabel(metric)

        if x_lims is not None:
            plt.xlim(x_lims[idx])
        if y_lims is not None:
            plt.ylim(y_lims[idx])

        plt.plot(epochs, train_metrics, 'b', label="training")
        plt.plot(epochs, val_metrics, 'r', label="validation")
        plt.legend(loc="best")

        if clear_ml_task:
            clear_ml_task.logger.report_matplotlib_figure(title=metric, series="", iteration=0, figure=plt)

        plt.savefig(os.path.join(output_dir, metric + ".png"))
        plt.close()


def roc_plot(output_path,
             fpr,
             tpr,
             thr):

    opt_idx = np.argmax(np.add(1 - fpr, tpr))

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot(fpr, thr, color='navy', lw=2, label='Threshold')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    plt.scatter(fpr[opt_idx], tpr[opt_idx], color='blue', marker='o', label='Optimal threshold')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - specificity')
    plt.ylabel('sensitivity')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close()


def confusion_matrix_plot(output_path,
                          confusion_matrix,
                          classes_names,
                          normalize=False,
                          cmap=plt.cm.Blues):

    if normalize:
        confusion_matrix = metrics.normalized_confusion_matrix(confusion_matrix)

    plt.figure()
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes_names))
    plt.xticks(tick_marks, classes_names, rotation=45)
    plt.yticks(tick_marks, classes_names)

    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()


def image_plot(image,
               title="",
               colors_map="gray",
               show=False,
               output_path="",
               fig_size=(10, 5.5)):

    plt.figure(figsize=fig_size)
    plt.imshow(image, cmap=colors_map)
    plt.title(title)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(output_path)
    plt.close()


def side_by_side_plot(images,
                      titles,
                      colors_map="gray",
                      show=False,
                      output_path=""):

    f, axarr = plt.subplots(1, len(images))
    for img_idx in range(len(images)):
        axarr[img_idx].imshow(images[img_idx], cmap=colors_map)
        axarr[img_idx].set_title(titles[img_idx])

    if show:
        plt.show()
    else:
        plt.savefig(output_path)
    plt.close()


def center_slices_plot(image,
                       colors_map="gray",
                       show=False,
                       output_path=""):

    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(image[int(image.shape[0] / 2), :, :], cmap=colors_map)
    axarr[0].set_title('Axial')
    axarr[1].imshow(image[:, int(image.shape[1] / 2), :], cmap=colors_map)
    axarr[1].set_title('Coronal')
    axarr[2].imshow(image[:, :, int(image.shape[2] / 2)], cmap=colors_map)
    axarr[2].set_title('Saggital')

    if show:
        plt.show()
    else:
        plt.savefig(output_path)
    plt.close()


def overlay_plot(image,
                 overlays,
                 colors,
                 opacity=0.6,
                 show=False,
                 outputs_path=""):

    """
    This method put overlays on grayscale image
    :param image: grayscale image; values have to be between 0 and 255
    :param overlays: binary masks; values have to be 0 or 255
    :param colors: array of color indexes; 0 for red, 1 for green and 2 for blue
    :param opacity: overlay opacity; value between 0 and 1
    :param show: if True image with overlays is shown, else image with overlay is saved
    :param outputs_path: path to save the image
    :return: None
    """

    color_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for overlay_idx, overlay in enumerate(overlays):
        color_mask[:, :, colors[overlay_idx]] = overlay

    color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    color_image[:, :, 0] = image
    color_image[:, :, 1] = image
    color_image[:, :, 2] = image

    color_img_hsv = ski_color.rgb2hsv(color_image)
    color_mask_hsv = ski_color.rgb2hsv(color_mask)

    color_img_hsv[..., 0] = color_mask_hsv[..., 0]
    color_img_hsv[..., 1] = color_mask_hsv[..., 1] * opacity

    masked_image = ski_color.hsv2rgb(color_img_hsv)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(masked_image)

    if show:
        plt.show()
    else:
        plt.savefig(outputs_path)
    plt.close()

def contours_plot(image,
                  masks,
                  colors,
                  thickness=1,
                  show=False,
                  outputs_path=""):

    """
    This method draws all external contours of the mask on the image
    :param image: RGB image
    :param masks: list of binary masks
    :param color: list of RGB colors of the contours
    :param thickness: thickness of the contour
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
        cv2.imwrite(outputs_path, bgr_image)
