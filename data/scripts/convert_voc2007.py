import xml.etree.ElementTree as ET

from tqdm import tqdm
from pathlib import Path
import random

# VOC2007 train val set for learning
# download https://github.com/ultralytics/yolov5/releases/download/v1.0/VOCtrainval_06-Nov-2007.zip to ../../../datasets, unzip to VOC2007, 446MB total, 5012 images
# train set 2501 images, val set 500 images

yaml = {"names":['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
        "path": "../../../datasets/VOC2007",
        "val_num": 500}

def convert_label(path, lb_path, year, image_id):
    def convert_box(size, box):
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh

    in_file = open(path / f'VOC{year}/Annotations/{image_id}.xml')
    out_file = open(lb_path, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls in yaml['names'] and not int(obj.find('difficult').text) == 1:
            xmlbox = obj.find('bndbox')
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
            cls_id = yaml['names'].index(cls)  # class id
            out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')

# Convert
dir = Path(yaml['path']).resolve()  # dataset root dir
print(dir)
path = dir / f'VOCdevkit'
for year, image_set in  ('2007', 'train'), ('2007', 'val'):
    imgs_path = dir / 'images' / f'{image_set}{year}'
    lbs_path = dir / 'labels' / f'{image_set}{year}'
    imgs_path.mkdir(exist_ok=True, parents=True)
    lbs_path.mkdir(exist_ok=True, parents=True)

    with open(path / f'VOC{year}/ImageSets/Main/{image_set}.txt') as f:
        image_ids = f.read().strip().split()
        if image_set == 'val':
            image_ids = random.sample(image_ids, yaml['val_num']) # random sample val_num
    for id in tqdm(image_ids, desc=f'{image_set}{year}'):
        f = path / f'VOC{year}/JPEGImages/{id}.jpg'  # old img path
        lb_path = (lbs_path / f.name).with_suffix('.txt')  # new label path
        f.rename(imgs_path / f.name)  # move image
        convert_label(path, lb_path, year, id)  # convert labels to YOLO format