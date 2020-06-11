IMAGENET_CLASSES_OUTPUT = 'data/imagenet_classes.json'

IMAGENET_OBJ_DET_CLASSES_INPUT = 'data/imagenet_obj_det_classes_raw.txt'
IMAGENET_OBJ_DET_CLASSES_OUTPUT = 'data/imagenet_obj_det_classes.json'

# For image annotation (imagenet_annotator.py)
IMG_BASE_PATH = 'data/imagenet_val_subset/ILSVRC2012_val_'
XML_BASE_PATH = 'data/imagenet_bb_subset/ILSVRC2012_val_'
ANNOTATE_BASE_PATH = 'data/imagenet_annotated_subset/ILSVRC2012_val_'
MASK_BASE_PATH = 'data/imagenet_masked_ground_truth/ILSVRC2012_val_'

# methods
LIFT = 'deeplift'
LIME = 'lime'
SHAP = 'shap'
GRAD = 'gradcam'

# models
VGG = 'vgg16'
INCEPT = 'inception'
RESNET = 'resnet'

# metrics
INTERSECT = 'intersect'
INTENSITY = 'intensity'

# metric thesholds for IOU/ IOU*
INTERSECT_THRESHOLD = 1
INTENSITY_THRESHOLD = 2

METHODS = [LIME, SHAP, LIFT, GRAD]
MODELS = [VGG, INCEPT, RESNET]
METRICS = [INTERSECT, INTENSITY]

# results folder
RESULTS_EVAL_PATH = 'results/eval'
RESULTS_ADAPTED_PATH = 'results/adapted'
RESULTS_BASE_PATH = RESULTS_EVAL_PATH
# RESULTS_BASE_PATH = RESULTS_ADAPTED_PATH

# For feeding into model architectures
STD_IMG_SIZE = (224, 224)
NONSTD_IMG_SIZE = (299, 299)

# strong confidence ImageNet examples (p > 0.75) for VGG-16
GOOD_EXAMPLES = [7, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 24, 25, 27, 31, 33, 35, 36, 42, 43, 45, 47, 52, 53, 54, 56,
                 58, 63, 66, 67, 68, 69, 72, 73, 74, 75, 78, 81, 82, 86, 89, 90, 92, 93, 96, 97, 98, 99, 100, 103, 107,
                 109, 113, 116, 120, 122, 123, 124, 125, 127, 129, 130, 131, 133, 135, 138, 139, 142, 143, 144, 145,
                 149, 153, 154, 156, 157, 158, 160, 164, 165, 166, 167, 168, 169, 171, 172, 173, 177, 179, 186, 187,
                 188, 192, 194, 196, 198, 199, 200, 201, 206, 208, 209, 211, 213, 215, 216, 217, 218, 222, 225, 226,
                 227, 230, 231, 233, 234, 235, 236, 237, 238, 239, 240, 243, 246, 247, 248, 250, 251, 252, 256, 258,
                 259, 262, 264, 266, 271, 272, 275, 276, 278, 280, 281, 282, 283, 284, 287, 288, 289, 290, 293, 294,
                 299]

VGG_LAYER_MAP = {"block5_conv3": 17,
                 "block4_conv3": 13,
                 "block3_conv3": 9,
                 "block3_conv1": 7,     # shap original target
                 "block2_conv2": 5}

INCEPTION_LAYER_MAP = {"conv2d_94": 299,
                       "conv2d_188": 299,
                       "mixed9": 279,
                       "mixed10": 310}

RESNET_LAYER_MAP = {"res5c_branch2c": 171,
                    "res2c_branch2c": 35}

LAYER_TARGETS = {
    SHAP:
        {INCEPT: INCEPTION_LAYER_MAP["conv2d_188"],
         VGG: VGG_LAYER_MAP["block3_conv3"],  # block5_conv3, block3_conv1
         RESNET: RESNET_LAYER_MAP["res2c_branch2c"]},
    GRAD:
        {INCEPT: INCEPTION_LAYER_MAP["conv2d_188"],
         VGG: VGG_LAYER_MAP["block5_conv3"],
         RESNET: RESNET_LAYER_MAP["res5c_branch2c"]},
    LIFT:
        {INCEPT: None,
         VGG: None,
         RESNET: None},
    LIME:
        {INCEPT: None,
         VGG: None,
         RESNET: None}
}
