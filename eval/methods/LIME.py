import matplotlib.pyplot as plt

# LIME explanation
from lime import lime_image
from skimage.segmentation import mark_boundaries

from ..util.constants import INCEPT
from ..util.image_util import ImageHandler, deprocess_image, get_preprocess_for_model
import numpy as np


class Lime:
    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        self.explainer = lime_image.LimeImageExplainer()
        self.preprocess = get_preprocess_for_model(model_name)

    def prediction_wrapper(self, x):
        # these x are batches of 10, where each example is generated around the neighbourhood of the OG source
        return self.model.predict(self.preprocess(x))

    def attribute(self, ih: ImageHandler, visualise: bool = False, save: bool = True):
        explanation = self.explainer.explain_instance(ih.get_raw_img(),
                                                      classifier_fn=self.prediction_wrapper,
                                                      top_labels=1,
                                                      num_samples=100)

        # TODO fix support for positive and negative evidence
        top_exp = explanation.local_exp[explanation.top_labels[0]]
        output = get_attribution(exp=top_exp,
                                 segments=explanation.segments,
                                 size=ih.get_size(),
                                 num_features=5)
        # deprocess
        # output = deprocess_image(output)
        #print(np.amin(output, axis=(0, 1)))
        #print(np.amax(output, axis=(0, 1)))
        output /= np.max(np.abs(output))
        # output *= 255.0
        # output = output.astype('uint8')
        # turn into RGB array (currently 2D numpy array)
        # red = np.clip(output, min=0)
        # blue = np.abs(np.clip(output, max=0))
        # output = np.stack((red, np.zeros(output.shape), blue), axis=2)
        #output = output.astype('uint8')

        plt.imshow(output, cmap='seismic', clim=(-1, 1))

        if save:
            plt.savefig(ih.get_output_path('lime'))
        if visualise:
            plt.show()


def get_attribution(exp, segments, size, num_features=5):
    """simplification OF lime_image.py function get_image_and_mask())

    Args:'
        num_features: number of superpixels to include in explanation

    Returns:
        image, where image is a 3d numpy array
    """
    output = np.zeros(size)
    for feature, weight in exp[:num_features]:
        output[segments == feature] = weight
    return output
