import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import innvestigate
import innvestigate.utils

from ..util.image_util import ImageHandler


# high level wrapper for DeepLIFT
# TODO: replace with direct implementation

# suppress output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def attribute(model, ih: ImageHandler):
    # strip softmax layer
    model = innvestigate.utils.model_wo_softmax(model)

    analyzer = innvestigate.analyzer.DeepLIFT(model)
    #analyzer = innvestigate.create_analyzer("deep_lift.wrapper", model)
    a = analyzer.analyze(ih.get_processed_img())

    # Aggregate along color channels and normalize to [-1, 1]
    a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
    a /= np.max(np.abs(a))

    # Plot
    # TODO : use extent to put input image down in greyscale

    ###plt.imshow(inputImg, cmap=plt.get_cmap('gray'), alpha=0.15, extent=(-1, a[0].shape[0], a[0].shape[1], -1))
    plt.imshow(ih.get_raw_img(), alpha=0.4, extent=(-1, a[0].shape[0], a[0].shape[1], -1))
    #print(np.abs(a[0]))
    maxVal = np.nanpercentile(np.abs(a[0]), 99.9)
    #plt.imshow(a[0], cmap=red_transparent_blue, vmin=-maxVal, vmax=maxVal)
    plt.imshow(a[0], cmap='seismic', clim=(-1, 1))
    plt.savefig(ih.get_output_path('deeplift'))

    #plt.show()
    plt.cla()
