import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K

# LIME explanation
from lime import lime_image
from skimage.segmentation import mark_boundaries

from ..util.image_util import ImageHandler, deprocess_image


def attribute(model, ih: ImageHandler):
    preprocess = ih.get_preprocess_for_model()
    explainer = lime_image.LimeImageExplainer()

    def prediction_wrapper(x):
        return model.predict(preprocess(x))

    # the function has to take a normal, unprocessed image
    explanation = explainer.explain_instance(ih.get_raw_img(), prediction_wrapper, top_labels=5, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)

    #plt.imshow(img, cmap=plt.get_cmap('gray'), alpha=0.15, extent=(-1, temp.shape[0], temp.shape[1], -1))
    #im = axes[row,i+1].imshow(sv, cmap=colors.red_transparent_blue, vmin=-max_val, vmax=max_val)

    # for inception, remember to / 2 and + 0.5
    plt.imshow(mark_boundaries(deprocess_image(temp), mask))
    plt.savefig(ih.get_output_path('lime'))
    plt.show()

