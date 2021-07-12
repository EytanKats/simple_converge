import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import color as ski_color

from simple_converge.utils import metrics


def line_plot(x_points,
              y_points,
              x_lim=None,
              y_lim=None,
              title="",
              x_label="",
              y_label="",
              legend=None,
              show=False,
              output_path="",
              fig_size=(10, 5.5)):

    """
    This method draws line plot
    :param x_points: 1D array of x axis values
    :param y_points: list of 1D arrays; each array contains y axis values of separate line
    :param x_lim: maximum x value to show on plot
    :param y_lim: maximum y value to show on plot
    :param title: title of the plot
    :param x_label: x axis name
    :param y_label: y axis name
    :param legend: legend of the plot
    :param show: if True plot will be shown, if False plot will be saved
    :param output_path: path to save the plot
    :param fig_size: size of the plot
    :return: None
    """

    ax_acc = plt.figure(figsize=fig_size)
    ax_acc.set_facecolor("gainsboro")
    plt.title(title)
    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)

    colors = ['b', 'r', 'g']
    for points_idx, points in enumerate(y_points):

        if legend is not None:
            legend_label = legend[points_idx]
        else:
            legend_label = None

        plt.plot(x_points, points, colors[points_idx], label=legend_label)

    if legend is not None:
        plt.legend(loc="best")

    if show:
        plt.show()
    else:
        plt.savefig(output_path)
    plt.close()


def training_plot(training_log_path,
                  metrics_to_plot,
                  x_lim=None,
                  y_lim=None,
                  show=False,
                  output_folder="",
                  fig_size=(10, 5.5),
                  clear_ml_task=None):

    """
    This method draws line plots for each metric specified in training log file
    :param training_log_path: path to CSV log file that contains training metrics
    :param metrics_to_plot: list of training metrics to plot; for training metric with name 'metric'
     corresponding validation metric expected to has name 'val_metric'
    :param x_lim: maximum x value to show on plot
    :param y_lim: maximum y value to show on plot
    :param show: if True plot will be shown, if False plot will be saved
    :param output_folder: folder path to save the plot
    :param fig_size: size of the plot
    :param clear_ml_task: ClearML Task object that is used to save the plot into experiments management framework
    :return: None
    """

    df = pd.read_csv(training_log_path)

    epochs = df["epoch"]
    for idx, metric in enumerate(metrics_to_plot):
        train_metrics = df[metric]
        val_metrics = df["val_" + metric]

        ax_acc = plt.figure(figsize=fig_size)
        ax_acc.set_facecolor("gainsboro")
        plt.title(metric)
        plt.grid(True)
        plt.xlabel("epochs")
        plt.ylabel(metric)

        if x_lim is not None:
            plt.xlim(x_lim[idx])
        if y_lim is not None:
            plt.ylim(y_lim[idx])

        plt.plot(epochs, train_metrics, 'b', label="training")
        plt.plot(epochs, val_metrics, 'r', label="validation")
        plt.legend(loc="best")

        if clear_ml_task:
            clear_ml_task.logger.report_matplotlib_figure(title=metric, series="", iteration=0, figure=plt)

        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(output_folder, metric + ".png"))
        plt.close()


def roc_plot(fpr,
             tpr,
             thr,
             show=False,
             output_path="",
             fig_size=(10, 5.5)):

    """
    This method draws ROC plot  (https://en.wikipedia.org/wiki/Receiver_operating_characteristic#:~:text=A%20receiver%20operating%20characteristic%20curve,which%20led%20to%20its%20name.)
    :param fpr: false positive rate; 1D array of x axis values
    :param tpr: true positive rate; 1D array of y axis values
    :param thr: threshold values; 1D array if y axis values
    :param show: if True plot will be shown, if False plot will be saved
    :param output_path: path to save the plot
    :param fig_size: size of the plot
    :return: None
    """

    opt_idx = np.argmax(np.add(1 - fpr, tpr))

    plt.figure(figsize=fig_size)
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

    if show:
        plt.show()
    else:
        plt.savefig(output_path)
    plt.close()


def confusion_matrix_plot(confusion_matrix,
                          class_names,
                          normalize=False,
                          cmap=plt.cm.Blues,
                          show=False,
                          output_path="",
                          fig_size=(10, 5.5)):

    """
    This method draws confusion matrix plot (https://en.wikipedia.org/wiki/Confusion_matrix)
    :param confusion_matrix: confusion matrix; 2D array
    :param class_names: list of class names that presented in confusion matrix
    :param normalize: if True confusion matrix will be normalized, if False confusion matrix will be plotted as is
    :param cmap: color map for confusion matrix plot
    :param show: if True plot will be shown, if False plot will be saved
    :param output_path: path to save the plot
    :param fig_size: size of the plot
    :return: None
    """

    if normalize:
        confusion_matrix = metrics.normalized_confusion_matrix(confusion_matrix)

    plt.figure(figsize=fig_size)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(output_path)
    plt.close()


def image_plot(image,
               title="",
               cmap="gray",
               show=False,
               output_path="",
               fig_size=(10, 5.5)):

    """
    This method plots the image
    :param image: image to plot
    :param title: name of the plot
    :param cmap: color map to plot the image with
    :param show: if True plot will be shown, if False plot will be saved
    :param output_path: path to save the plot
    :param fig_size: size of the plot
    :return: None
    """

    plt.figure(figsize=fig_size)
    plt.imshow(image, cmap=cmap)
    plt.title(title)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(output_path)
    plt.close()


def side_by_side_images_plot(images,
                             titles,
                             cmap="gray",
                             show=False,
                             output_path="",
                             fig_size=(10, 5.5)):

    """
    This method plots the images side by side
    :param images: list of images to plot
    :param titles: list of names for each image
    :param cmap: color map to plot the images with
    :param show: if True plot will be shown, if False plot will be saved
    :param output_path: path to save the plot
    :param fig_size: size of the plot
    :return: None
    """

    f, axarr = plt.subplots(1, len(images), figsize=fig_size)
    for img_idx in range(len(images)):
        axarr[img_idx].imshow(images[img_idx], cmap=cmap)
        axarr[img_idx].set_title(titles[img_idx])

    if show:
        plt.show()
    else:
        plt.savefig(output_path)
    plt.close()


def center_slices_volume_plot(volume,
                              cmap="gray",
                              show=False,
                              output_path="",
                              fig_size=(10, 5.5)):

    """
    This method plots the axial, coronal and sagittal central slices of the volume side by side
    :param volume: volume from which slices will be extracted and plotted
    :param cmap: color map to plot the slices with
    :param show: if True plot will be shown, if False plot will be saved
    :param output_path: path to save the plot
    :param fig_size: size of the plot
    :return: None
    """

    f, axarr = plt.subplots(1, 3, figsize=fig_size)
    axarr[0].imshow(volume[int(volume.shape[0] / 2), :, :], cmap=cmap)
    axarr[0].set_title('Axial')
    axarr[1].imshow(volume[:, int(volume.shape[1] / 2), :], cmap=cmap)
    axarr[1].set_title('Coronal')
    axarr[2].imshow(volume[:, :, int(volume.shape[2] / 2)], cmap=cmap)
    axarr[2].set_title('Sagittal')

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
                 output_path="",
                 fig_size=(10, 5.5)):

    """
    This method put overlays on grayscale image
    :param image: grayscale image; values have to be between 0 and 255
    :param overlays: binary masks; values have to be 0 or 255
    :param colors: array of color indexes; 0 for red, 1 for green and 2 for blue
    :param opacity: overlay opacity; value between 0 and 1
    :param show: if True image with overlays is shown, else image with overlay is saved
    :param output_path: path to save the image
    :param fig_size: size of the plot
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

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    ax.imshow(masked_image)

    if show:
        plt.show()
    else:
        plt.savefig(output_path)
    plt.close()

