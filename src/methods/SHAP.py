import sys
sys.path.append('src/')
import os

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import keras.backend as K
import numpy as np
import json

# helper functions
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from preprocessing.keras_util import getPreprocessForModel
from preprocessing.imagenet import getClassificationClasses

import shap
from methods.visualiser import modified_SHAP_plot


RAW_CLASSES = getClassificationClasses()


IMG_BASE_PATH = 'data/imagenet_val_subset/ILSVRC2012_val_000000'


# Modified version of SHAP plotter function (see shap.plots.image)
def image_plot(shap_values, x, labels=None, show=True, width=20, aspect=0.2, hspace=0.2, labelpad=None, output_img_path=''):
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
        axes = axes.reshape(1,axes.size)
    for row in range(x.shape[0]):
        x_curr = x[row].copy()

        # make sure
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
            x_curr = x_curr.reshape(x_curr.shape[:2])
        if x_curr.max() > 1:
            x_curr /= 255.

        # get a grayscale version of the image
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
            x_curr_gray = (0.2989 * x_curr[:,:,0] + 0.5870 * x_curr[:,:,1] + 0.1140 * x_curr[:,:,2]) # rgb to gray
        else:
            x_curr_gray = x_curr

        axes[row,0].imshow(x_curr, cmap=plt.get_cmap('gray'))
        axes[row,0].axis('off')
        if len(shap_values[0][row].shape) == 2:
            abs_vals = np.stack([np.abs(shap_values[i]) for i in range(len(shap_values))], 0).flatten()
        else:
            abs_vals = np.stack([np.abs(shap_values[i].sum(-1)) for i in range(len(shap_values))], 0).flatten()
        max_val = np.nanpercentile(abs_vals, 99.9)
        for i in range(len(shap_values)):
            if labels is not None:
                axes[row,i+1].set_title(labels[row,i], **label_kwargs)
            sv = shap_values[i][row] if len(shap_values[i][row].shape) == 2 else shap_values[i][row].sum(-1)
            axes[row,i+1].imshow(x_curr_gray, cmap=plt.get_cmap('gray'), alpha=0.15, extent=(-1, sv.shape[0], sv.shape[1], -1))
            im = axes[row,i+1].imshow(sv, cmap=colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
            extent = axes[row,i+1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig('test_subplot_shap.png', bbox_inches=extent)
            axes[row,i+1].axis('off')
    if hspace == 'auto':
        fig.tight_layout()
    else:
        fig.subplots_adjust(hspace=hspace)
    cb = fig.colorbar(im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal", aspect=fig_size[0]/aspect)
    cb.outline.set_visible(False)
    if show:
        plt.show()















preprocess = getPreprocessForModel('vgg16')

backgroundData, Y = shap.datasets.imagenet50()

imgSize = (224, 224)
inputImagePath = 'data/imagenet_val_subset/ILSVRC2012_val_00000003.JPEG'
inputImage = load_img(inputImagePath, target_size=imgSize)
inputImage = img_to_array(inputImage)
inputImage = np.expand_dims(inputImage, axis=0)
img = preprocess(inputImage)


model = VGG16(weights='imagenet', include_top=True)






# explain how the input to the 7th layer of the model explains the top two classes
def map2layer(x, layer):
    feed_dict = dict(zip([model.layers[0].input], [preprocess(x.copy())]))
    return K.get_session().run(model.layers[layer].input, feed_dict)

# combines expectation with sampling values from whole background data set
e = shap.GradientExplainer(
    (model.layers[7].input, model.layers[-1].output),
    map2layer(backgroundData, 7),
    local_smoothing=0 # std dev of smoothing noise
)

# get outputs for top prediction count "ranked_outputs"
shap_values, indexes = e.shap_values(map2layer(inputImage, 7), ranked_outputs=1)

# get the names for the classes
#index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

# using the top indexes (i.e 1:1000)
imgLabels = np.vectorize(lambda x: RAW_CLASSES[str(x)][1])(indexes)

# plot the explanations
shap.image_plot(shap_values, inputImage, imgLabels)

#modified_SHAP_plot(shap_values, inputImage, imgLabels)



