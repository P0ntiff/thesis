import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ..util.constants import OUTPUT_BASE_PATH, IMG_BASE_PATH, XML_BASE_PATH
from ..util.image_util import get_image_file_name
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


def draw_annotation(img_no: int, class_map: dict, save_to_file=True, display=False,
                    image_base_path: str = IMG_BASE_PATH, xml_base_path: str = XML_BASE_PATH):
    """Draws annotations that are read from an XML file at 'xml_path' onto the image read from 'image_path'.
        Requires class_map object created by imagenet.get_classification_mappings()

        Optionally outputs the image to file, with an optional file name (otherwise taken from 'image_path').
    """
    # file paths
    xml_path = get_image_file_name(xml_base_path, img_no) + '.xml'
    image_path = get_image_file_name(image_base_path, img_no) + '.JPEG'
    output_path = get_image_file_name(OUTPUT_BASE_PATH, img_no) + '.JPEG'
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
    if save_to_file:
        plt.savefig(output_path)
    if display:
        plt.show()
    plt.cla()
    plt.clf()
    plt.close()


def draw_annotations(img_no_array: list):
    class_map = get_classification_mappings()
    for img_no in img_no_array:
        print('Annotating img_no=' + str(img_no))
        draw_annotation(img_no, class_map)


# draw_annotations([i for i in range(16, 300)])
