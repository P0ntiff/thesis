import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ..util.constants import ANNOTATE_BASE_PATH, IMG_BASE_PATH, XML_BASE_PATH, MASK_BASE_PATH, RESULTS_BASE_PATH
from ..util.image_util import get_image_file_name, show_figure
from ..util.image_util import get_classification_mappings


def read_annotations(xml_path: str):
    """Reads a path to an XML file containing bounding box coordinates for a single image.

        Returns a dictionary mapping WNID/class to a list of dicts of bounding box coords. The bounding box dict
        has keys 'xmin', 'xmax', 'ymin' and 'ymax' with int values (pixel coords).
        The list is because some images may have several annotations of the same class/WNID.
    """
    root = ET.parse(xml_path).getroot()
    output = {}
    for obj in root.findall('object'):
        bnd_box = {}
        for coord in obj.find('bndbox'):
            bnd_box[coord.tag] = int(coord.text)
        name = obj.find('name').text
        if name not in output:
            output[name] = []
        output[name].append(bnd_box)

    return output


def get_xml_image_and_output_paths(img_no: int):
    # file paths
    xml_path = get_image_file_name(XML_BASE_PATH, img_no) + '.xml'
    image_path = get_image_file_name(IMG_BASE_PATH, img_no) + '.JPEG'
    output_path = get_image_file_name(ANNOTATE_BASE_PATH, img_no) + '.JPEG'
    return xml_path, image_path, output_path


def draw_annotation(img_no: int, class_map: dict, save=True, visualise=False):
    """Draws annotations that are read from an XML file at 'xml_path' onto the image read from 'image_path'.
        Requires class_map object created by imagenet.get_classification_mappings()

        Optionally outputs the image to file, with an optional file name (otherwise taken from 'image_path').
    """
    xml_path, image_path, output_path = get_xml_image_and_output_paths(img_no)
    # ingest data
    wnid_map = read_annotations(xml_path)
    data = plt.imread(image_path)
    plt.imshow(data)
    ax = plt.gca()
    # for each class annotation in the image
    for wnid in wnid_map:
        # for each annotation of the same class
        for box in wnid_map[wnid]:
            # get rectangle information
            y1, x1, y2, x2 = box['ymin'], box['xmin'], box['ymax'], box['xmax']
            width, height = x2 - x1, y2 - y1
            rect = Rectangle(xy=(x1, y1), width=width, height=height, fill=False, color='white')
            # draw rectangle
            ax.add_patch(rect)
            # draw label in top left (class and wnid)
            label = "{}: '{}'".format(class_map[wnid], wnid)
            plt.text(x1, y1, label, color='white')
    # output
    if save:
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', transparent=True, pad_inches=0)
    if visualise:
        plt.show()
        plt.close()
    plt.cla()


def get_annotated_image(img_no: int):
    output_path = get_image_file_name(ANNOTATE_BASE_PATH, img_no) + '.JPEG'
    return plt.imread(output_path)


def get_mask_for_eval(img_no: int, target_size: tuple, save=False, visualise=False):
    #print('Drawing mask for img_no = \t' + str(img_no))
    xml_path, image_path, output_path = get_xml_image_and_output_paths(img_no)

    # ingest data
    wnid_map = read_annotations(xml_path)
    data = plt.imread(image_path)

    # data is a 3D array, only interested in a 2D mask, therefore use first two dimensions
    output_mask = np.zeros(data.shape[0:2])
    # for each class annotation in the image
    for wnid in wnid_map:
        # for each annotation of the same class
        for box in wnid_map[wnid]:
            # get rectangle information
            y1, x1, y2, x2 = box['ymin'], box['xmin'], box['ymax'], box['xmax']

            output_mask[y1:y2, x1:x2] = 1

    # resize for desired target shape
    output_mask = cv2.resize(output_mask, target_size, cv2.INTER_AREA)

    if visualise:
        # show OG figure (squeezed into target shape)
        show_figure(cv2.resize(data, target_size, cv2.INTER_AREA))
        # show mask over the figure
        show_figure(output_mask)
    if save:
        plt.imshow(output_mask, cmap='seismic', clim=(-1, 1))
        plt.savefig(output_path)
        plt.cla()

    return output_mask


def demo_resizer(img_no: int, target_size: tuple):
    # only used to get a BB-annotated image squeezed in to model dimensions for input size
    # modification of draw_annotation() above
    # just for visualisation / demo purposes
    output_path = get_image_file_name(ANNOTATE_BASE_PATH, img_no) + '.JPEG'
    # read from file and return
    annotated_img = cv2.imread(output_path)
    height, width = annotated_img.shape[:2]
    max_height, max_width = target_size
    x_scale = 0.0
    y_scale = 0.0
    if max_width < width:
        x_scale = max_width / float(width)
    if max_height < height:
        y_scale = max_height / float(height)
    # re-read and resize
    output = cv2.resize(annotated_img, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_AREA)
    #cv2.imshow("Scaled Annotation", output)
    cv2.imwrite('data/temp/{}_rescale.PNG'.format(img_no), output)

    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


def draw_annotations(img_no_array: list):
    class_map = get_classification_mappings()
    for img_no in img_no_array:
        print('Annotating img_no=' + str(img_no))
        draw_annotation(img_no, class_map)


def get_masks_for_eval(img_no_array: list, target_size: tuple, visualise=False, save=True):
    output = []
    for img_no in img_no_array:
        output.append(get_mask_for_eval(img_no, target_size, visualise=visualise, save=save))

    return output

# draw_annotations([i for i in range(16, 300)])
