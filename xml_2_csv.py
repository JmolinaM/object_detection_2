import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
train_percentage= 0.8 
image_path =('data/labelled/labels')
#image_path = (['train/bag/labels','train/phone/labels','train/wallet/labels','train/portatil/labels'])
xml_list= []
#for dir_lab in xrange(1,len(image_path)):
#for xml_file in glob.glob(image_path[dir_lab] + '/*.xml') :
for xml_file in glob.glob(image_path + '/*.xml') :
	tree = ET.parse(xml_file)
	root = tree.getroot()
	for member in root.findall('object'):
		value=(root.find('filename').text, int(root.find('size')[0].text), int(root.find('size')[1].text), member[0].text, int(member[4][0].text), int(member[4][1].text), int(member[4][2].text), int(member[4][3].text))
		xml_list.append(value)
column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
print round(0.8*len(xml_list))
train_list= xml_list[1:int(round(0.8*len(xml_list))) ]
evaluate_list= xml_list[(int(round(0.8*len(xml_list)))+1):]
xml_df_train = pd.DataFrame(train_list, columns=column_name)
xml_df_train.to_csv('objects_labels_train.csv', index=None)
xml_df_evaluate = pd.DataFrame(evaluate_list, columns=column_name)
xml_df_evaluate.to_csv('objects_labels_evaluate.csv', index=None)


		
		
