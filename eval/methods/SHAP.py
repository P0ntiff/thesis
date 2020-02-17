import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

import shap
from shap.plots.colors import red_transparent_blue

from eval.util.image_util import ImageHandler
from eval.util.imagenet import get_classification_classes


RAW_CLASSES = get_classification_classes()


# Modified version of SHAP plotter function (see shap.plots.image)
def image_plot(shap_values, x, labels=None, show=True, width=20, aspect=0.2, hspace=0.2, labelpad=None,
               output_img_path=''):
    """ Plots SHAP values for image inputs.
    """
    multi_output = True
    if type(shap_values) != list:
        multi_output = False
        shap_values = [shap_values]

    # make sure labels
    if labels is not None:
        assert labels.shape[0] == shap_values[0].shape[0], "Labels must have same row count as shap_values arrays!"
        if multi_output:
            assert labels.shape[1] == len(shap_values), "Labels must have a column for each output in shap_values!"
        else:
            assert len(labels.shape) == 1, "Labels must be a vector for single output shap_values."

    label_kwargs = {} if labelpad is None else {'pad': labelpad}

    # plot our explanations
    fig_size = np.array([3 * (len(shap_values) + 1), 2.5 * (x.shape[0] + 1)])
    if fig_size[0] > width:
        fig_size *= width / fig_size[0]
    fig, axes = plt.subplots(nrows=x.shape[0], ncols=len(shap_values) + 1, figsize=fig_size)
    if len(axes.shape) == 1:
        axes = axes.reshape(1, axes.size)
    for row in range(x.shape[0]):
        x_curr = x[row].copy()

        # make sure
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
            x_curr = x_curr.reshape(x_curr.shape[:2])
        if x_curr.max() > 1:
            x_curr /= 255.

        # get a grayscale version of the image
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
            x_curr_gray = (
                        0.2989 * x_curr[:, :, 0] + 0.5870 * x_curr[:, :, 1] + 0.1140 * x_curr[:, :, 2])  # rgb to gray
        else:
            x_curr_gray = x_curr

        axes[row, 0].imshow(x_curr, cmap=plt.get_cmap('gray'))
        axes[row, 0].axis('off')
        if len(shap_values[0][row].shape) == 2:
            abs_vals = np.stack([np.abs(shap_values[i]) for i in range(len(shap_values))], 0).flatten()
        else:
            abs_vals = np.stack([np.abs(shap_values[i].sum(-1)) for i in range(len(shap_values))], 0).flatten()
        max_val = np.nanpercentile(abs_vals, 99.9)
        for i in range(len(shap_values)):
            if labels is not None:
                axes[row, i + 1].set_title(labels[row, i], **label_kwargs)
            sv = shap_values[i][row] if len(shap_values[i][row].shape) == 2 else shap_values[i][row].sum(-1)
            axes[row, i + 1].imshow(x_curr_gray, cmap=plt.get_cmap('gray'), alpha=0.15,
                                    extent=(-1, sv.shape[0], sv.shape[1], -1))
            im = axes[row, i + 1].imshow(sv, cmap=red_transparent_blue, vmin=-max_val, vmax=max_val)
            # quick workaround to get subplot as export
            extent = axes[row, i + 1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # fig.savefig(output_img_path, bbox_inches=extent.expanded(1.3, 1))
            axes[row, i + 1].axis('off')
    if hspace == 'auto':
        fig.tight_layout()
    else:
        fig.subplots_adjust(hspace=hspace)
    cb = fig.colorbar(im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal",
                      aspect=fig_size[0] / aspect)
    cb.outline.set_visible(False)
    if show:
        plt.show()


def attribute(model, layer_n: int, ih: ImageHandler):
    background_data, Y = shap.datasets.imagenet50()

    # explain how the input to a layer of the model explains the top class
    def map2layer(img, layer):
        feed_dict = dict(zip([model.layers[0].input], [img]))
        return K.get_session().run(model.layers[layer].input, feed_dict)

    # combines expectation with sampling values from whole background data set
    e = shap.GradientExplainer(
        (model.layers[layer_n].input, model.layers[-1].output),
        map2layer(background_data, layer_n),
        local_smoothing=0  # std dev of smoothing noise
    )

    # get outputs for top prediction count "ranked_outputs"
    shap_values, indexes = e.shap_values(map2layer(ih.get_processed_img(), layer_n), ranked_outputs=1)

    # get the names for the classes
    # index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

    # using the top indexes (i.e out of 1:1000)
    img_labels = np.vectorize(lambda x: RAW_CLASSES[str(x)][1])(indexes)

    # plot the explanations
    image_plot(shap_values, ih.get_expanded_img(), img_labels, output_img_path=ih.get_output_path('shap'))

    # modified_SHAP_plot(shap_values, input_image, imgLabels)

