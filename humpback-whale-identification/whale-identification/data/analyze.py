import os
import sys
sys.path.append('..');

from constants import *
import pandas as pd

def readY(mode):
	data = pd.read_csv(os.path.join(DATA_DIR, mode + '.csv'));
	print(data.head());
	image = data['Image'];
	_id = data['Id'];

	cls = {};
	for i in range(len(data)):
		row = data.iloc[i];
		if row['Id'] not in cls:
			cls[row['Id']] = [];
		cls[row['Id']].append(row['Image']);

	print(cls);
	print(len(cls['new_whale']));
	print(len(cls));


def readX(mode):
	image_nms = os.listdir(os.path.join(DATA_DIR, mode));
	print('Total number of {} images -- {}'.format(mode, len(image_nms)));


if __name__ == '__main__':

	modes = ['train', 'test'];
	for mode in modes:
		print('*** {} ***'.format(mode));
		readX(mode);
		if mode == 'train':
			readY(mode);

		print('\n');