import pickle
import os
import cv2
import sys
import shutil

sys.path.append('..');
from constants import *

def crop_whales(ann, train, dest_path):
    """
    Crops whale locations in an image and saves it to disk
    """

    img_names = [];
    for i, img_name in enumerate(ann):
        img_path = os.path.join(DATA_DIR, 'train', img_name);
        img = cv2.imread(img_path);
        """whale correctly detected"""
        if len(ann[img_name]['bbox']) == 1:
            x1, y1, x2, y2 = ann[img_name]['bbox'][0];
            cropped = img[y1:y2, x1:x2];
            cv2.imwrite(os.path.join(dest_path, img_name), cropped);	
        """incorrect detection"""
        else:
            id_path = os.path.join(DATA_DIR, 'debug', train[img_name]);
            if not os.path.exists(id_path):
                os.makedirs(id_path);
            for j, bbox in enumerate(ann[img_name]['bbox']):
                x1, y1, x2, y2 = bbox;
                cropped = img[y1:y2, x1:x2];
                img_id = os.path.splitext(img_name)[0];
                cv2.imwrite(os.path.join(id_path, img_id+'_'+str(j)+'.jpg'), cropped);	

        if i % 1000 == 0:
            print('Completed - {}'.format(i));

    print(len(img_names), len(set(img_names)));
    return;

def debug(ann):

    names = {};
    for i, row in enumerate(ann):
        img_path, x1, y1, x2, y2, cls = row.split(',');
        img_name = os.path.split(img_path)[-1];
        if img_name not in names:
            names[img_name] = {};
            names[img_name]['object'] = cls;
            names[img_name]['bbox'] = []
        names[img_name]['bbox'].append([int(x1), int(y1), int(x2), int(y2)]);    
 
    train = {}

    with open('/home/raghul/dev/kaggle/datasets/humpback-whale-identification/train.csv')\
      as f:
        ds = f.read().split('\n');
        ds = list(filter(None, ds))
        for row in ds:
            img, _id = row.split(',');
            train[img] = _id;

    reps = {}
    for name in names:
        if len(names[name]['bbox']) > 1:
            reps[name] = names[name];

    return names, train;


if __name__ == '__main__':

    """
    Program to read whale bounding boxes for training whale images, crop
	and write them to disk
    """

    csv_path = sys.argv[1];
    dest_path = sys.argv[2];

    if not os.path.exists(dest_path):
        os.makedirs(dest_path);

    with open(csv_path) as f:
        ann = f.read().split('\n');
        ann = list(filter(None, ann));

    print('No. of rows - {}'.format(len(ann)));    
    ann, train = debug(ann);
    crop_whales(ann, train, dest_path);

    