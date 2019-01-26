DATA_DIR = '/opt/infilect/dev/datasets/object_detection/whale';
import os
import cv2
import numpy as np
import shutil

def read_img(path):
	img = cv2.imread(path);
	return img;

def show_img(img):
	cv2.imshow('img', img);
	cv2.waitKey(0);

def create_csv(_type, rows, debug=False):	
	f = open(os.path.join(_type + '.csv'), 'w');	

	print('Preparing {}...'.format(_type));
	if os.path.exists(_type):
		shutil.rmtree(_type);
	os.makedirs(_type);

	for row in rows:
		row = row.split(',');
		img_nm = row[0];
		poly = np.int32(row[1:]).reshape(-1, 2);
		rect = cv2.minAreaRect(poly);
		box = np.int0(cv2.boxPoints(rect));
		minx = np.amin(poly[:,0]);
		miny = np.amin(poly[:,1]);
		maxx = np.amax(poly[:,0]);
		maxy = np.amax(poly[:,1]);

		img_path = os.path.join(DATA_DIR, _type, 'JPEGImages', img_nm);		
		os.system('cp ../dataset/train/{} {}'.format(img_nm, _type));

		if debug:
			print(img_nm);			
			print(minx, miny, maxx, maxy);
			img = read_img('../dataset/train/' + img_nm);
			print(img.shape);
			cv2.rectangle(img, (minx, miny), (maxx, maxy),(0, 255, 0), 2);
			cv2.drawContours(img,[poly],0,(0,0,255),2)
			show_img(img);
	
		# path/to/image.jpg,x1,y1,x2,y2,class_name
		f.write('{},{},{},{},{},whale\n'.format(img_path,str(minx),\
			str(miny), str(maxx), str(maxy)));

	f.close();


def read_data(path):

	with open(path) as f:
		rows = f.read().split('\n');
		rows = list(filter(None, rows));

	trn_ratio, val_ratio = int(.8 * len(rows)), int(.2 * len(rows));
	train = np.random.choice(rows, size=trn_ratio, replace=False);
	rows = list(set(rows) - set(train))
	val = rows

	common = set(train) & set(val);
	print('Common {}'.format(len(common)));	
	print('Train {}, Val {}'.format(len(train), len(val)));

	create_csv('train', train);
	create_csv('val', val);


def debug(path):

	val = os.listdir('/opt/infilect/dev/datasets/object_detection/whale/val/JPEGImages');
	with open(path) as f:
		rows = f.read().split('\n');
		rows = list(filter(None, rows));

	_type = 'val';
	f = open(os.path.join(_type + '.csv'), 'w');	
	
	for row in rows:
		row = row.split(',');
		img_nm = row[0];
		if img_nm not in val:
			continue;
		poly = np.int32(row[1:]).reshape(-1, 2);
		rect = cv2.minAreaRect(poly);
		box = np.int0(cv2.boxPoints(rect));
		minx = np.amin(poly[:,0]);
		miny = np.amin(poly[:,1]);
		maxx = np.amax(poly[:,0]);
		maxy = np.amax(poly[:,1]);

		img_path = os.path.join(DATA_DIR, _type, 'JPEGImages', img_nm);		
		# path/to/image.jpg,x1,y1,x2,y2,class_name
		f.write('{},{},{},{},{},whale\n'.format(img_path,str(minx),\
			str(miny), str(maxx), str(maxy)));

	f.close();



if __name__ == '__main__':

	read_data(os.path.join('/opt/infilect/dev/datasets/kaggle/whale-categorization-playground/cropping.txt'));

