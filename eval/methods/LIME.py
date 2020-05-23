import matplotlib.pyplot as plt

# LIME explanation
from lime import lime_image
from skimage.segmentation import mark_boundaries

from ..util.image_util import ImageHandler, deprocess_image, get_preprocess_for_model


class Lime:
    def __init__(self, model, model_name: str):
        self.model = model
        self.explainer = lime_image.LimeImageExplainer()
        self.preprocess = get_preprocess_for_model(model_name)

    def prediction_wrapper(self, x):
        return self.model.predict(self.preprocess(x))

    def attribute(self, ih: ImageHandler, visualise: bool = True, save: bool = True):
        # the function has to take a normal, unprocessed image (?)
        explanation = self.explainer.explain_instance(ih.get_raw_img(),
                                                      classifier_fn=self.prediction_wrapper,
                                                      top_labels=5,
                                                      hide_color=0,
                                                      num_samples=1000)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                    positive_only=False,
                                                    num_features=5,
                                                    hide_rest=True)

        # plt.imshow(img, cmap=plt.get_cmap('gray'), alpha=0.15, extent=(-1, temp.shape[0], temp.shape[1], -1))
        # im = axes[row,i+1].imshow(sv, cmap=colors.red_transparent_blue, vmin=-max_val, vmax=max_val)

        # for inception, remember to / 2 and + 0.5
        plt.imshow(mark_boundaries(deprocess_image(temp), mask))

        if save:
            plt.savefig(ih.get_output_path('lime'))
        if visualise:
            plt.show()
