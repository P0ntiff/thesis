import numpy as np

# LIME explanation
from lime import lime_image

from ..util.image_util import ImageHandler, get_preprocess_for_model, show_figure


class Lime:
    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        self.explainer = lime_image.LimeImageExplainer()
        self.preprocess = get_preprocess_for_model(model_name)

    def prediction_wrapper(self, x):
        # these x are batches of 10, where each example is generated around the neighbourhood of the OG source
        return self.model.predict(self.preprocess(x))

    def attribute(self, ih: ImageHandler):
        explanation = self.explainer.explain_instance(ih.get_raw_img(),
                                                      classifier_fn=self.prediction_wrapper,
                                                      top_labels=1,
                                                      hide_color=0,
                                                      num_samples=1000)

        # TODO fix support for positive and negative evidence
        top_exp = explanation.local_exp[explanation.top_labels[0]]
        output = get_attribution(exp=top_exp,
                                 segments=explanation.segments,
                                 size=ih.get_size(),
                                 num_features=5)

        #temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True,
        #                                            num_features=5, hide_rest=True)
        #output = mark_boundaries(deprocess_image(temp), mask)
        # deprocess
        # normalize tensor: center on 0., ensure std is 0.1
        output -= output.mean()
        output /= (output.std() + 1e-5)
        output *= 0.2
        # clip to (-1, 1)
        output = np.clip(output, -1, 1)

        return output


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
