import pickle
import os
import cv2
import sys
sys.path.append('..');

from constants import *


def crop_whale(img, pt):

	return;

if __name__ == '__main__':

	modes = ['train', 'test'];

	with open(os.path.join(DATA_DIR, 'bounding-box', 'bounding-box.pickle'), 'rb') as f:
		data = pickle.load(f);

	print(len(data));
	print(data.keys());
	for mode in modes:
		images = os.listdir(os.path.join(DATA_DIR, mode));
		print(len(set(images) & set(data.keys())))
		for file in images:
			img = cv2.imread(os.path.join(DATA_DIR, mode, file));
			print(file);
			print(data[file[1:]]);
			break;
		break;
