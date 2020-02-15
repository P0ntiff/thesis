import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .imagenet import get_classification_mappings

IMG_BASE_PATH = 'data/imagenet_val_subset/ILSVRC2012_val_000000'
XML_BASE_PATH = 'data/imagenet_bb_subset/ILSVRC2012_val_000000'
OUTPUT_BASE_PATH = 'data/imagenet_annotated_subset/'

CLASS_MAP = get_classification_mappings()


def get_imagenet_boxes(xml_path):
    root = ET.parse(xml_path).getroot()
    output = {}
    for obj in root.findall('object'):
        bndbox = {}
        for coord in obj.find('bndbox'):
            bndbox[coord.tag] = int(coord.text)
        name = obj.find('name').text
        if name not in output:
            output[name] = []
        output[name].append(bndbox)

    return output


def draw_annotation(image_path, xml_path, output_name=None, save_to_file=True):
    wnid_map = get_imagenet_boxes(xml_path)
    data = plt.imread(image_path)
    plt.imshow(data)
    ax = plt.gca()
    # some images might have several annotations of the same class
    for wnid in wnid_map:
        for box in wnid_map[wnid]:
            y1, x1, y2, x2 = box['ymin'], box['xmin'], box['ymax'], box['xmax']
            width, height = x2 - x1, y2 - y1
            rect = Rectangle((x1, y1), width, height,
                             fill=False, color='white')
            # draw box
            ax.add_patch(rect)
            # draw label in top left (class and wnid)
            label = "{}: '{}'".format(CLASS_MAP[wnid], wnid)
            plt.text(x1, y1, label, color='white')
    if save_to_file:
        if output_name is None:
            plt.savefig(OUTPUT_BASE_PATH + 'ILSVRC2012_val_' +
                        image_path[-13:-5] + '.JPEG')
        else:
            plt.savefig(OUTPUT_BASE_PATH + output_name + '.JPEG')
    else:
        plt.show()
    plt.cla()
    plt.clf()
    plt.close()


draw_annotation(IMG_BASE_PATH + '14.JPEG',
                XML_BASE_PATH + '14.xml', save_to_file=True)
draw_annotation(IMG_BASE_PATH + '15.JPEG',
                XML_BASE_PATH + '15.xml', save_to_file=True)

# drawAnnotation(IMG_BASE_PATH + '10.JPEG', XML_BASE_PATH + '10.xml', saveToFile=True)
# drawAnnotation(IMG_BASE_PATH + '11.JPEG', XML_BASE_PATH + '11.xml', saveToFile=True)
# drawAnnotation(IMG_BASE_PATH + '12.JPEG', XML_BASE_PATH + '12.xml', saveToFile=True)
# drawAnnotation(IMG_BASE_PATH + '13.JPEG', XML_BASE_PATH + '13.xml', saveToFile=True)
