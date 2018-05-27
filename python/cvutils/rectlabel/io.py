"""RectLabel annotation utility module"""
import xml.etree.ElementTree
from collections import namedtuple
from skimage.draw import polygon_perimeter, polygon, ellipse
from skimage.measure import regionprops
import numpy as np

Annotation = namedtuple('Annotation', ['mask', 'border', 'points', 'object_type', 'bound_type', 'properties'])


def parse_object(o, img_shape, mask_value=np.iinfo(np.uint8).max):
    """Parse RectLabel xml object representing a single annotation"""
    o_type = o.find('name').text

    if o.find('polygon'):
        # <polygon>
        #   <x1>122</x1>
        #   <y1>32</y1>
        #   <x2>106</x2>
        #   <y2>48</y2>
        #   ...
        # </polygon>
        bound = 'polygon'
        coords = o.find('polygon').getchildren()
        assert len(coords) % 2 == 0
        pts = []
        for i in range(0, len(coords), 2):
            xc, yc = coords[i], coords[ i +1]
            xi, xv = int(xc.tag.replace('x', '')), int(xc.text)
            yi, yv = int(yc.tag.replace('y', '')), int(yc.text)
            assert xi == yi == i // 2 + 1
            pts.append([xv, yv])
    elif o.find('bndbox'):
        # <bndbox>
        #   <xmin>245</xmin>
        #   <ymin>54</ymin>
        #   <xmax>300</xmax>
        #   <ymax>98</ymax>
        # </bndbox>
        bound = 'box'
        bb = o.find('bndbox')
        xmin, ymin, xmax, ymax = [int(bb.find(p).text) for p in ['xmin', 'ymin', 'xmax', 'ymax']]
        pts = [
            [xmin, ymin],
            [xmin, ymax],
            [xmax, ymax],
            [xmax, ymin]
        ]
    else:
        raise ValueError('Cound not determine mask shape')

    pts = np.array(pts)
    r, c = pts[:, 1], pts[:, 0]

    mask = np.zeros(img_shape, dtype=np.uint8)
    rr, cc = polygon(r, c, shape=img_shape)
    mask[rr, cc] = mask_value

    border = np.zeros(img_shape, dtype=np.uint8)
    rr, cc = polygon_perimeter(r, c, shape=img_shape)
    border[rr, cc] = mask_value

    props = regionprops(mask)
    assert len(props) == 1

    return Annotation(mask, border, pts, o_type, bound, props[0])


def load_annotations(path):
    """Load a RectLabel annotation file

    :param path: Path to RectLabel xml annotation file
    :return: (shape, annotations) where:
        shape = (h, w) of annotated image
        annotations = list of Annotation objects
    """
    e = xml.etree.ElementTree.parse(path).getroot()
    w = int(e.find('size').find('width').text)
    h = int(e.find('size').find('height').text)
    shape = (h, w)
    annotations = []
    for o in e.findall('object'):
        try:
            annotations.append(parse_object(o, img_shape))
        except:
            print(
                'Failed to process object {}'
                .format(xml.etree.ElementTree.tostring(o, encoding='utf8', method='xml'))
            )
            raise
    return shape, annotations


