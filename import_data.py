# utils
import os
import glob
import shutil
from itertools import izip
import re
# image processing
import cv2
from pymorph import blob,label
# numeric calculus
import numpy as np
# xml handling
from dicttoxml import dicttoxml
from xml.dom.minidom import parseString

folder = 'Newspapers_clustering_using_py-faster-rcnn'
source = {}
source['database'] = 'Newspapers_clustering'
source['annotation'] = 'documents'
source['image'] = 'flickr'
source['flickrid'] = '0'
owner = {}
owner['flickrid'] = '?'
owner['name'] = 'UCL'


main_path = '/home/kyoucef/py-faster-rcnn/data/NewsPapers/UCL'
annotation_path = main_path+'/Annotations/'
imagesets_path = main_path+'/ImageSets/'
jpegimages_path = main_path+'/JPEGImages/'

if os.path.isdir(main_path):
	os.system('rm -rf '+main_path)

try:
	os.system('mkdir -p '+main_path)
	os.system('sleep 5')
	os.mkdir(annotation_path)
	os.mkdir(jpegimages_path)
	os.system('mkdir -p '+imagesets_path+'/Main')
except:
	pass
train_set_file = open(imagesets_path+'Main/train.txt','w')
val_set_file = open(imagesets_path+'Main/val.txt','w')
trainval_set_file = open(imagesets_path+'Main/trainval.txt','w')
test_set_file = open(imagesets_path+'Main/test.txt','w')


if os.path.exists('data_papers') == False:
    os.mkdir('data_papers')
    if os.path.exists('dataset_segmentation.rar') == False:
        os.system('wget https://archive.ics.uci.edu/ml/machine-learning-databases/00306/dataset_segmentation.rar')
    os.system('unrar e dataset_segmentation.rar data_papers')

resizing_factor = 2

list_mask = glob.glob('data_papers/*_m.*')
count = 0
for file_mask in list_mask:
	file_image = glob.glob(file_mask[:-6]+'.*')[0]
	print "currently processing "+file_image
	dummy, filename = os.path.split(file_image)
	mask = cv2.imread(file_mask)
	mask = cv2.resize(mask,(mask.shape[1]/resizing_factor,mask.shape[0]/resizing_factor))
	height,width,c = mask.shape
	ground_truth_ = np.zeros((height,width))
	# get text bounding boxes
    	ground_truth_text = ground_truth_ + np.where(np.linalg.norm(mask-[255,0,0],axis=2)<10,1,0)
	label_text = label(ground_truth_text)
	bbox_text = blob(label_text, measurement = 'boundingbox', output="data")
	if bbox_text.size>0:
		anno_text_x1 = bbox_text[:,0].tolist()
		anno_text_y1 = bbox_text[:,1].tolist()
		anno_text_x2 = bbox_text[:,2].tolist()
		anno_text_y2 = bbox_text[:,3].tolist()
		anno_text_label = list('t'*len(anno_text_x1))
	else:
		anno_text_x1 = []
		anno_text_y1 = []
		anno_text_x2 = []
		anno_text_y2 = []
		anno_text_label = []
	# get illustration bounding boxes
	ground_truth_illu = ground_truth_ + np.where(np.linalg.norm(mask-[0,0,255],axis=2)<10,1,0)
	label_illu = label(ground_truth_illu)
	bbox_illu = blob(label_illu, measurement = 'boundingbox', output="data")
	if bbox_illu.size>0:
		anno_illu_x1 = bbox_illu[:,0].tolist()
		anno_illu_y1 = bbox_illu[:,1].tolist()
		anno_illu_x2 = bbox_illu[:,2].tolist()
		anno_illu_y2 = bbox_illu[:,3].tolist()
		anno_illu_label = list('i'*len(anno_illu_x1))
	else:
		anno_illu_x1 = []
		anno_illu_y1 = []
		anno_illu_x2 = []
		anno_illu_y2 = []
		anno_illu_label = []
	
	# gather text and illustration bounding boxes
	anno_x1 = anno_text_x1+anno_illu_x1
	anno_x2 = anno_text_x2+anno_illu_x2
	anno_y1 = anno_text_y1+anno_illu_y1
	anno_y2 = anno_text_y2+anno_illu_y2
	anno_label = anno_text_label+anno_illu_label
	annotation_dict = {
	'annotation':
         {
                'folder':      folder,
                'filename':    filename,
                'source':
                {
                        'database':   source['database'],
                        'annotation': source['annotation'],
                        'image':      source['image'],
                        'flickrid':   source['flickrid']
                },
                'owner':
                {
                        'flickrid':   owner['flickrid'],
                        'name':       owner['name']
                },
                'size':
                {
                        'width':      str(width),
                        'height':     str(height),
                        'depth':      str(c)
                },
                'segmented':   '0'
        }
        }
	objects = []
        for x1,y1,x2,y2,lbl in izip(anno_x1,anno_y1,anno_x2,anno_y2,anno_label):
		x1 = x1-x1*(x1<0)+1
		y1 = y1-y1*(y1<0)+1
		t = 0
		if x2>=(width):
			x2 = width-1
			t=1
		if y2>=(height):
			y2 = height-1
			t=1
		if lbl=='t':
			label_name = 'text'
		else:
			label_name = 'illustration'
		if (x1==1) | (y1==1) | (y2==(height-1)) | (x2==(width-1)):
			t = 1
		o = {'object':
			{
				'name':         label_name,
				'pose':         'unknown',
				'truncated':    str(t),
				'difficult':    '0',
				'bndbox':
				{
					'xmin':     str(x1),
					'ymin':     str(y1),
					'xmax':     str(x2),
					'ymax':     str(y2)
				}
			}
                           }
		objects.append(o)
	annotation_dict['annotation']['objects'] = objects
        annotation_xml = dicttoxml(annotation_dict,root=False,attr_type=False)
        xml_string = parseString(annotation_xml).toprettyxml()
        xml_string = re.sub('<item>','',xml_string)
        xml_string = re.sub('</item>\n','',xml_string)
        xml_string = re.sub('<objects>\n','',xml_string)
        xml_string = re.sub('</objects>','',xml_string)
        xml_file = open(annotation_path+filename[:-3]+'xml','w')
        xml_file.write(xml_string)
        xml_file.close()
	image_gray = cv2.imread(file_image)
    	image_gray = cv2.resize(image_gray,(image_gray.shape[1]/resizing_factor,image_gray.shape[0]/resizing_factor))
	cv2.imwrite(jpegimages_path+'/'+filename[:-3]+'jpg',image_gray)
        if count%3 == 0:
            train_set_file.write(filename[:-4]+'\n')
            trainval_set_file.write(filename[:-4]+'\n')
        elif count%3 == 1:
            val_set_file.write(filename[:-4]+'\n')
            trainval_set_file.write(filename[:-4]+'\n')
        else:
            test_set_file.write(filename[:-4]+'\n')
        count = count+1;

