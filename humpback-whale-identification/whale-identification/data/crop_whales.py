import pickle
import os
import cv2
import sys
import shutil

sys.path.append('..');
from constants import *

def crop_whales(ann, dest_path):
    """
    Crops whale locations in an image and saves it to disk
    """

    img_names = [];
    for i, row in enumerate(ann):
        img_path, x1, y1, x2, y2, cls = row.split(',');
        img_name = os.path.split(img_path)[-1];
        img = cv2.imread(img_path);
        cropped = img[int(y1):int(y2), int(x1):int(x2)];
        cv2.imwrite(os.path.join(dest_path, img_name), cropped);	
        img_names.append(img_name);

        if i % 1000 == 0:
            print('Completed - {}'.format(i));

    print(len(img_names), len(set(img_names)));
    return;

if __name__ == '__main__':

    """
    Program to read whale bounding boxes for training whale images, crop
	and write them to disk
    """

    csv_path = sys.argv[1];
    dest_path = sys.argv[2];

    if os.path.exists(dest_path):
        shutil.rmtree(dest_path);
    os.makedirs(dest_path);

    with open(csv_path) as f:
        ann = f.read().split('\n');
        ann = list(filter(None, ann));

    print('No. of rows - {}'.format(len(ann)));
    crop_whales(ann, dest_path);

    