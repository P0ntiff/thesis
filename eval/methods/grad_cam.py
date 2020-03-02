# DISCLAIMER
# This script is modified from an original sourced from a open-source Keras implementation of Grad-CAM (MIT license)
# https://github.com/eclique/keras-gradcam
# Refinement and adaption of this original here
# https://github.com/jacobgil/keras-grad-cam

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions

import tensorflow as tf
from tensorflow.python.framework import ops

from eval.util.image_util import ImageHandler, deprocess_image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# Input shape, defined by the model (model.input_shape)
H, W = 224, 224


def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)


def build_guided_model(model):
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
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = model
    return new_model


def guided_backprop(input_model, images, layer_name):
    """Guided Backpropagation method for visualizing input saliency."""
    input_imgs = input_model.input
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(layer_output, input_imgs)[0]
    backprop_fn = K.function([input_imgs, K.learning_phase()], [grads])
    grads_val = backprop_fn([images, 0])[0]
    return grads_val


def grad_cam(input_model, image, cls, layer_name):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
    # Normalize if necessary
    # grads = normalize(grads)
    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam_max = cam.max()
    if cam_max != 0:
        cam = cam / cam_max
    return cam


def grad_cam_batch(input_model, images, classes, layer_name):
    """GradCAM method for visualizing input saliency.
    Same as grad_cam but processes multiple images in one run."""
    loss = tf.gather_nd(input_model.output, np.dstack([range(images.shape[0]), classes])[0])
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(loss, layer_output)[0]
    gradient_fn = K.function([input_model.input, K.learning_phase()], [layer_output, grads])

    conv_output, grads_val = gradient_fn([images, 0])
    weights = np.mean(grads_val, axis=(1, 2))
    cams = np.einsum('ijkl,il->ijk', conv_output, weights)

    # Process CAMs
    new_cams = np.empty((images.shape[0], W, H))
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i, 2) + 1e-10)
        new_cams[i] = cv2.resize(cam_i, (H, W), cv2.INTER_LINEAR)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams[i] = new_cams[i] / new_cams[i].max()

    return new_cams


def compute_saliency(model, guided_model, ih: ImageHandler, layer_name='block5_conv3', cls=-1,
                     visualize=True, save=True):
    """Compute saliency using all three approaches.
        -layer_name: layer to compute gradients;
        -cls: class number to localize (-1 for most probable class).
    """
    # get the class to localize if it's not available
    predictions = model.predict(ih.get_processed_img())
    if cls == -1:
        cls = np.argmax(predictions)

    gradcam = grad_cam(model, ih.get_processed_img(), cls, layer_name)
    gb = guided_backprop(guided_model, ih.get_processed_img(), layer_name)
    guided_gradcam = gb * gradcam[..., np.newaxis]

    # only interested in guided gradcam (the class discriminative "high-resolution" combination of guided-BP and GC.
    if save:
        # jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
        # jetcam = (np.float32(jetcam) + load_image(img_path, preprocess=False)) / 2
        # cv2.imwrite(output_img_path + 'gc.png', np.uint8(jetcam))
        # cv2.imwrite('guided_backprop.jpg', deprocess_image(gb[0]))
        # cv2.imwrite(output_img_path + 'ggc.png', deprocess_image(guided_gradcam[0]))
        #plt.imshow(ih.get_original_img())
        #plt.imshow(deprocess_image(guided_gradcam[0]))
        cv2.imwrite(ih.get_output_path('gradcam'), deprocess_image(guided_gradcam[0]))
        #plt.savefig(ih.get_output_path('gradcam'))
        #plt.cla()

    if visualize:
        plt.figure(figsize=(15, 10))
        plt.subplot(131)
        plt.title('GradCAM')
        plt.axis('off')
        plt.imshow(ih.get_original_img())
        plt.imshow(gradcam, cmap='jet', alpha=0.5)

        plt.subplot(132)
        plt.title('Guided Backprop')
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(gb[0]), -1))

        plt.subplot(133)
        plt.title('Guided GradCAM')
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(guided_gradcam[0]), -1))
        plt.show()

    return gradcam, gb, guided_gradcam


def attribute(model, ih: ImageHandler):
    guided_model = build_guided_model(model)
    gradcam, gb, guided_gradcam = compute_saliency(model, guided_model, layer_name='block5_conv3',
                                                   ih=ih, cls=-1, visualize=False, save=True)
