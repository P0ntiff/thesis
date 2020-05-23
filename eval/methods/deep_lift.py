import matplotlib.pyplot as plt
import numpy as np

import innvestigate
import innvestigate.utils

from ..util.image_util import ImageHandler, deprocess_image

# high level wrapper for DeepLIFT
# TODO: replace with direct implementation

# suppress output
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class DeepLift:
    def __init__(self, model):
        # strip softmax layer
        self.model = innvestigate.utils.model_wo_softmax(model)
        # analyzer = innvestigate.create_analyzer("deep_lift.wrapper", model)
        self.analyzer = innvestigate.analyzer.DeepLIFT(self.model)

    def attribute(self, ih: ImageHandler, visualise: bool = False, save: bool = True):
        a = self.analyzer.analyze(ih.get_processed_img())

        # Aggregate along color channels and normalize to [-1, 1]
        a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
        a /= np.max(np.abs(a))

        if save:
            plt.imshow(a[0], cmap='seismic', clim=(-1, 1))
            plt.savefig(ih.get_output_path('deeplift'))
            plt.cla()

        if visualise:
            plt.figure(figsize=(15, 10))
            plt.title('DeepLIFT')
            plt.subplot(121)
            plt.axis('off')
            plt.imshow(ih.get_original_img())

            plt.subplot(122)
            plt.axis('off')
            plt.imshow(a[0], cmap='seismic', clim=(-1, 1))

            plt.show()
            plt.cla()