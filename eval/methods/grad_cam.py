# DISCLAIMER
# This script is modified from an original sourced from a open-source Keras implementation of Grad-CAM (MIT license)
# https://github.com/eclique/keras-gradcam
# Refinement and adaption of this original here
# https://github.com/jacobgil/keras-grad-cam

import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf

from keras import backend as K
from tensorflow.python.framework import ops

from eval.util.image_util import ImageHandler, deprocess_gradcam


def build_guided_model(build_model_fn):
    """Function returning modified model.

    Changes gradient function for all ReLu activations
    according to Guided Backpropagation.
    """
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)

    g = tf.get_default_graph()
    # TODO check potential bug here from overriden implementation
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = build_model_fn()
    return new_model


class GradCam:
    def __init__(self, model, build_model_fn, layer_no: int):
        # strip softmax layer
        self.model = model
        self.guided_model = build_guided_model(build_model_fn)
        self.layer_no = layer_no

    def reset_layer_no(self, layer_no: int):
        if layer_no is None:
            return
        self.layer_no = layer_no

    def attribute(self, ih: ImageHandler):
        """Compute saliency.
            -layer_name: layer to compute gradients;
            -cls: class number to localize (-1 for most probable class).
        """
        # get the class to localise if it's not available
        predictions = self.model.predict(ih.get_processed_img())
        cls = np.argmax(predictions)

        gradcam = self.grad_cam(ih, cls)
        gb = self.guided_backprop(ih)
        guided_gradcam = gb * gradcam[..., np.newaxis]

        # only interested in guided gradcam (the class discriminative "high-resolution" combination of guided-BP and GC.
        # # normalise along color channels and normalise to (-1, 1)
        guided_gradcam = guided_gradcam.sum(axis=np.argmax(np.asarray(guided_gradcam.shape) == 3))
        guided_gradcam /= np.max(np.abs(guided_gradcam))

        # output attribution (numpy 2D array)
        return guided_gradcam[0]

    def guided_backprop(self, ih: ImageHandler):
        """Guided Backpropagation method for visualizing input saliency."""
        input_imgs = self.guided_model.input
        layer_output = self.guided_model.layers[self.layer_no].output
        grads = K.gradients(layer_output, input_imgs)[0]
        backprop_fn = K.function([input_imgs, K.learning_phase()], [grads])
        grads_val = backprop_fn([ih.get_processed_img(), 0])[0]

        return grads_val

    def grad_cam(self, ih: ImageHandler, cls):
        """GradCAM method for visualizing input saliency."""
        y_c = self.model.output[0, cls]
        conv_output = self.model.layers[self.layer_no].output
        grads = K.gradients(y_c, conv_output)[0]
        # Normalize if necessary
        # grads = normalize(grads)
        gradient_function = K.function([self.model.input], [conv_output, grads])

        output, grads_val = gradient_function([ih.get_processed_img()])
        output, grads_val = output[0, :], grads_val[0, :, :, :]

        weights = np.mean(grads_val, axis=(0, 1))
        cam = np.dot(output, weights)

        # Process CAM
        cam = cv2.resize(cam, ih.get_size(), cv2.INTER_LINEAR)
        cam = np.maximum(cam, 0)
        cam_max = cam.max()
        if cam_max != 0:
            cam = cam / cam_max
        return cam

    # def grad_cam_batch(self, images, classes):
    #     """GradCAM method for visualizing input saliency.
    #     Same as grad_cam but processes multiple images in one run."""
    #     loss = tf.gather_nd(self.model.output, np.dstack([range(images.shape[0]), classes])[0])
    #     layer_output = self.model.get_layer(self.layer_name).output
    #     grads = K.gradients(loss, layer_output)[0]
    #     gradient_fn = K.function([self.model.input, K.learning_phase()], [layer_output, grads])
    #
    #     conv_output, grads_val = gradient_fn([images, 0])
    #     weights = np.mean(grads_val, axis=(1, 2))
    #     cams = np.einsum('ijkl,il->ijk', conv_output, weights)
    #
    #     # Process CAMs
    #     new_cams = np.empty((images.shape[0], W, H))
    #     for i in range(new_cams.shape[0]):
    #         cam_i = cams[i] - cams[i].mean()
    #         cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i, 2) + 1e-10)
    #         new_cams[i] = cv2.resize(cam_i, (H, W), cv2.INTER_LINEAR)
    #         new_cams[i] = np.maximum(new_cams[i], 0)
    #         new_cams[i] = new_cams[i] / new_cams[i].max()
    #
    #     return new_cams


