

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from imagenet import getClassificationMappings

IMG_BASE_PATH = 'data/imagenet_val_subset/ILSVRC2012_val_000000'
XML_BASE_PATH = 'data/imagenet_bb_subset/ILSVRC2012_val_000000'
OUTPUT_BASE_PATH = 'data/imagenet_annotated_subset/'

CLASS_MAP = getClassificationMappings()


def getImagenetBoxes(xmlPath):
    root = ET.parse(xmlPath).getroot()
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

def drawAnnotation(imagePath, xmlPath, outputName=None, saveToFile=True):
    wnidMap = getImagenetBoxes(xmlPath)
    data = plt.imread(imagePath)
    plt.imshow(data)
    ax = plt.gca()
    # some images might have several annotations of the same class
    for wnid in wnidMap:
        for box in wnidMap[wnid]:
            y1, x1, y2, x2 = box['ymin'], box['xmin'], box['ymax'], box['xmax']
            width, height = x2 - x1, y2 - y1
            rect = Rectangle((x1, y1), width, height,
                             fill=False, color='white')
            # draw box
            ax.add_patch(rect)
            # draw label in top left (class and wnid)
            label = "{}: '{}'".format(CLASS_MAP[wnid], wnid)
            plt.text(x1, y1, label, color='white')
    if saveToFile:
        if outputName is None:
            plt.savefig(OUTPUT_BASE_PATH + 'ILSVRC2012_val_' +
                        imagePath[-13:-5] + '.JPEG')
        else:
            plt.savefig(OUTPUT_BASE_PATH + outputName + '.JPEG')
    else:
        plt.show()
    plt.cla()
    plt.clf()
    plt.close()


drawAnnotation(IMG_BASE_PATH + '14.JPEG',
               XML_BASE_PATH + '14.xml', saveToFile=True)
drawAnnotation(IMG_BASE_PATH + '15.JPEG',
               XML_BASE_PATH + '15.xml', saveToFile=True)
#drawAnnotation(IMG_BASE_PATH + '10.JPEG', XML_BASE_PATH + '10.xml', saveToFile=True)
#drawAnnotation(IMG_BASE_PATH + '11.JPEG', XML_BASE_PATH + '11.xml', saveToFile=True)
#drawAnnotation(IMG_BASE_PATH + '12.JPEG', XML_BASE_PATH + '12.xml', saveToFile=True)
#drawAnnotation(IMG_BASE_PATH + '13.JPEG', XML_BASE_PATH + '13.xml', saveToFile=True)
