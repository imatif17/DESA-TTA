"""Build the VOC-C benchmark used in DESA-TTA.

VOC-C applies the 15 common corruptions of Michaelis et al. (2019),
"Benchmarking Robustness in Object Detection", to the 4,952 images of the
PASCAL VOC 2007 test set, using the `imagecorruptions` package.

Output layout (the default `--out-dir` matches the VOC-C config):

    data/VOC-C/
    ├── test.json              # COCO-style annotations for VOC2007 test
    ├── gaussian_noise/000001.jpg ...
    ├── shot_noise/
    └── ...                    # one folder per corruption

Example:
    python tools/prepare_voc_c.py /path/to/VOCdevkit --out-dir data/VOC-C
"""
import argparse
import json
import os
import os.path as osp
import xml.etree.ElementTree as ET
from functools import partial
from multiprocessing import Pool

import numpy as np
from PIL import Image

VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

CORRUPTIONS = ('gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
               'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost',
               'fog', 'brightness', 'contrast', 'elastic_transform',
               'pixelate', 'jpeg_compression')


def parse_args():
    parser = argparse.ArgumentParser(description='Build VOC-C.')
    parser.add_argument('voc_root', help='VOCdevkit folder containing VOC2007')
    parser.add_argument('--out-dir', default='data/VOC-C')
    parser.add_argument(
        '--severity', type=int, default=5, choices=[1, 2, 3, 4, 5])
    parser.add_argument(
        '--corruptions',
        nargs='+',
        default=list(CORRUPTIONS),
        choices=CORRUPTIONS,
        help='subset of corruptions to generate (default: all 15)')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument(
        '--skip-images',
        action='store_true',
        help='only write test.json')
    return parser.parse_args()


def read_ids(voc2007):
    with open(osp.join(voc2007, 'ImageSets', 'Main', 'test.txt')) as f:
        return [line.strip() for line in f if line.strip()]


def build_coco_json(voc2007, img_ids):
    """VOC XML -> COCO json, following mmdet's pascal_voc.py converter:
    0-based category ids, 0-based pixel coordinates and `difficult`
    objects stored with iscrowd=1 so that COCO evaluation ignores them."""
    images, annotations = [], []
    ann_id = 0
    for image_id, name in enumerate(img_ids):
        root = ET.parse(osp.join(voc2007, 'Annotations', f'{name}.xml')).getroot()
        size = root.find('size')
        images.append(
            dict(
                id=image_id,
                file_name=f'{name}.jpg',
                width=int(size.find('width').text),
                height=int(size.find('height').text)))
        for obj in root.findall('object'):
            box = obj.find('bndbox')
            x1, y1, x2, y2 = (int(box.find(k).text) - 1
                              for k in ('xmin', 'ymin', 'xmax', 'ymax'))
            difficult = obj.find('difficult')
            difficult = int(difficult.text) if difficult is not None else 0
            w, h = x2 - x1, y2 - y1
            annotations.append(
                dict(
                    id=ann_id,
                    image_id=image_id,
                    category_id=VOC_CLASSES.index(obj.find('name').text),
                    bbox=[x1, y1, w, h],
                    area=w * h,
                    iscrowd=difficult,
                    ignore=0,
                    segmentation=[[x1, y1, x1, y2, x2, y2, x2, y1]]))
            ann_id += 1
    categories = [
        dict(id=i, name=name, supercategory='none')
        for i, name in enumerate(VOC_CLASSES)
    ]
    return dict(
        images=images,
        annotations=annotations,
        categories=categories,
        type='instance')


def corrupt_one(name, src_dir, out_dir, corruption, severity):
    from imagecorruptions import corrupt
    dst = osp.join(out_dir, corruption, f'{name}.jpg')
    if osp.exists(dst):
        return
    img = np.asarray(Image.open(osp.join(src_dir, f'{name}.jpg')).convert('RGB'))
    out = corrupt(img, corruption_name=corruption, severity=severity)
    Image.fromarray(np.uint8(out)).save(dst, quality=95)


def main():
    args = parse_args()
    voc2007 = osp.join(args.voc_root, 'VOC2007')
    img_ids = read_ids(voc2007)
    os.makedirs(args.out_dir, exist_ok=True)

    coco = build_coco_json(voc2007, img_ids)
    with open(osp.join(args.out_dir, 'test.json'), 'w') as f:
        json.dump(coco, f)
    print(f'Wrote test.json: {len(coco["images"])} images, '
          f'{len(coco["annotations"])} boxes')
    if args.skip_images:
        return

    src_dir = osp.join(voc2007, 'JPEGImages')
    for corruption in args.corruptions:
        os.makedirs(osp.join(args.out_dir, corruption), exist_ok=True)
        fn = partial(
            corrupt_one,
            src_dir=src_dir,
            out_dir=args.out_dir,
            corruption=corruption,
            severity=args.severity)
        with Pool(args.workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(fn, img_ids, 16), 1):
                if i % 500 == 0 or i == len(img_ids):
                    print(f'[{corruption}] {i}/{len(img_ids)}', flush=True)


if __name__ == '__main__':
    main()
