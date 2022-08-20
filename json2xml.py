
import os
import numpy as np
import codecs
import json
from glob import glob
import cv2
import shutil
from sklearn.model_selection import train_test_split

# 1.存放的json标签路径
labelme_path = "/Volumes/My_Passport/LBT/DATASET/anti-vibration_hammer/timdatasets/json/"

# 原始labelme标注数据路径
saved_path = "VOC2012/"
# 保存路径
isUseTest = False  # 是否创建test集

# 2.创建要求文件夹
if not os.path.exists(saved_path + "Annotations"):
    os.makedirs(saved_path + "Annotations")
if not os.path.exists(saved_path + "JPEGImages/"):
    os.makedirs(saved_path + "JPEGImages/")
if not os.path.exists(saved_path + "ImageSets/Main/"):
    os.makedirs(saved_path + "ImageSets/Main/")

# 3.获取待处理文件
files = glob(labelme_path + "*.json")
files = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in files]
print(files)

# 4.读取标注信息并写入 xml
for json_file_ in files:
    json_filename = labelme_path + json_file_ + ".json"
    json_file = json.load(open(json_filename, "r", encoding="utf-8"))
    height, width, channels = cv2.imread('/Volumes/My_Passport/LBT/DATASET/anti-vibration_hammer/timdatasets/images/' + json_file_ + ".jpg").shape
    with codecs.open(saved_path + "Annotations/" + json_file_ + ".xml", "w", "utf-8") as xml:

        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'WH_data' + '</folder>\n')
        xml.write('\t<filename>' + json_file_ + ".jpg" + '</filename>\n')
        xml.write('\t<source>\n')
        xml.write('\t\t<database>WH Data</database>\n')
        xml.write('\t\t<annotation>WH</annotation>\n')
        xml.write('\t\t<image>flickr</image>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t</source>\n')
        xml.write('\t<owner>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t\t<name>WH</name>\n')
        xml.write('\t</owner>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>' + str(width) + '</width>\n')
        xml.write('\t\t<height>' + str(height) + '</height>\n')
        xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')
        for multi in json_file["shapes"]:
            points = np.array(multi["points"])
            labelName = multi["label"]
            xmin = min(points[:, 0])
            xmax = max(points[:, 0])
            ymin = min(points[:, 1])
            ymax = max(points[:, 1])
            label = multi["label"]
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                xml.write('\t<object>\n')
                xml.write('\t\t<name>' + labelName + '</name>\n')
                xml.write('\t\t<pose>Unspecified</pose>\n')
                xml.write('\t\t<truncated>1</truncated>\n')
                xml.write('\t\t<difficult>0</difficult>\n')
                xml.write('\t\t<bndbox>\n')
                xml.write('\t\t\t<xmin>' + str(int(xmin)) + '</xmin>\n')
                xml.write('\t\t\t<ymin>' + str(int(ymin)) + '</ymin>\n')
                xml.write('\t\t\t<xmax>' + str(int(xmax)) + '</xmax>\n')
                xml.write('\t\t\t<ymax>' + str(int(ymax)) + '</ymax>\n')
                xml.write('\t\t</bndbox>\n')
                xml.write('\t</object>\n')
                print(json_filename, xmin, ymin, xmax, ymax, label)
        xml.write('</annotation>')

# 5.复制图片到 VOC2007/JPEGImages/下
image_files = glob("/Volumes/My_Passport/LBT/DATASET/anti-vibration_hammer/timdatasets/images/" + "*.jpg")
print("copy image files to VOC2012/JPEGImages/")
for image in image_files:
    shutil.copy(image, saved_path + "JPEGImages/")

# 6.拆分训练集、测试集、验证集
txtsavepath = saved_path + "ImageSets/Main/"
ftrainval = open(txtsavepath + '/trainval.txt', 'w')
ftest = open(txtsavepath + '/test.txt', 'w')
ftrain = open(txtsavepath + '/train.txt', 'w')
fval = open(txtsavepath + '/val.txt', 'w')
total_files =  glob("./VOC2012/Annotations/*.xml")
total_files = [i.split("/")[-1].split(".xml")[0] for i in total_files]
trainval_files = []
test_files = []
if isUseTest:
    trainval_files, test_files = train_test_split(total_files, test_size=0.2)
else:
    trainval_files = total_files
for file in trainval_files:
    ftrainval.write(file + "\n")

# split
train_files, val_files = train_test_split(trainval_files, test_size=0.2)

# train
for file in train_files:
    ftrain.write(file + "\n")

# val
for file in val_files:
    fval.write(file + "\n")
for file in test_files:
    print(file)
    ftest.write(file + "\n")
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
"""
import os
import numpy as np
import codecs
import json
import glob
import cv2
import shutil
from sklearn.model_selection import train_test_split

# 1.标签路径
labelme_path = "/Volumes/My_Passport/LBT/DATASET/anti-vibration_hammer/timdatasets/json/"  # 原始labelme标注数据路径
saved_path = "VOC2012/"  # 保存路径

# 2.创建要求文件夹
dst_annotation_dir = os.path.join(saved_path, 'Annotations')
if not os.path.exists(dst_annotation_dir):
    os.makedirs(dst_annotation_dir)
dst_image_dir = os.path.join(saved_path, "JPEGImages")
if not os.path.exists(dst_image_dir):
    os.makedirs(dst_image_dir)
dst_main_dir = os.path.join(saved_path, "ImageSets", "Main")
if not os.path.exists(dst_main_dir):
    os.makedirs(dst_main_dir)

# 3.获取待处理文件
org_json_files = sorted(glob.glob(os.path.join(labelme_path, '*.json')))
org_json_file_names = [i.split("\\")[-1].split(".json")[0] for i in org_json_files]
org_img_files = sorted(glob.glob(os.path.join(labelme_path, '*.jpg')))
org_img_file_names = [i.split("\\")[-1].split(".jpg")[0] for i in org_img_files]

# 4.labelme file to voc dataset
for i, json_file_ in enumerate(org_json_files):
    json_file = json.load(open(json_file_, "r", encoding="utf-8"))
    image_path = os.path.join(labelme_path, org_json_file_names[i]+'.jpg')
    img = cv2.imread(image_path , -1)
    height, width, channels = img.shape
    dst_image_path = os.path.join(dst_image_dir, "{:06d}.jpg".format(i))
    cv2.imwrite(dst_image_path, img)
    dst_annotation_path = os.path.join(dst_annotation_dir, '{:06d}.xml'.format(i))
    with codecs.open(dst_annotation_path, "w", "utf-8") as xml:
        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'Pin_detection' + '</folder>\n')
        xml.write('\t<filename>' + "{:06d}.jpg".format(i) + '</filename>\n')
        # xml.write('\t<source>\n')
        # xml.write('\t\t<database>The UAV autolanding</database>\n')
        # xml.write('\t\t<annotation>UAV AutoLanding</annotation>\n')
        # xml.write('\t\t<image>flickr</image>\n')
        # xml.write('\t\t<flickrid>NULL</flickrid>\n')
        # xml.write('\t</source>\n')
        # xml.write('\t<owner>\n')
        # xml.write('\t\t<flickrid>NULL</flickrid>\n')
        # xml.write('\t\t<name>ChaojieZhu</name>\n')
        # xml.write('\t</owner>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>' + str(width) + '</width>\n')
        xml.write('\t\t<height>' + str(height) + '</height>\n')
        xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')
        for multi in json_file["shapes"]:
            points = np.array(multi["points"])
            xmin = min(points[:, 0])
            xmax = max(points[:, 0])
            ymin = min(points[:, 1])
            ymax = max(points[:, 1])
            label = multi["label"]
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                xml.write('\t<object>\n')
                xml.write('\t\t<name>' + label + '</name>\n')
                xml.write('\t\t<pose>Unspecified</pose>\n')
                xml.write('\t\t<truncated>1</truncated>\n')
                xml.write('\t\t<difficult>0</difficult>\n')
                xml.write('\t\t<bndbox>\n')
                xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
                xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
                xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
                xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
                xml.write('\t\t</bndbox>\n')
                xml.write('\t</object>\n')
                print(json_file_, xmin, ymin, xmax, ymax, label)
        xml.write('</annotation>')

# 5.split files for txt
train_file = os.path.join(dst_main_dir, 'train.txt')
trainval_file = os.path.join(dst_main_dir, 'trainval.txt')
val_file = os.path.join(dst_main_dir, 'val.txt')
test_file = os.path.join(dst_main_dir, 'test.txt')

ftrain = open(train_file, 'w')
ftrainval = open(trainval_file, 'w')
fval = open(val_file, 'w')
ftest = open(test_file, 'w')

total_annotation_files = glob.glob(os.path.join(dst_annotation_dir, "*.xml"))
total_annotation_names = [i.split("/")[-1].split(".xml")[0] for i in total_annotation_files]

# test_filepath = ""
for file in total_annotation_names:
    ftrainval.writelines(file + '\n')
# test
# for file in os.listdir(test_filepath):
#    ftest.write(file.split(".jpg")[0] + "\n")
# split
train_files, val_files = train_test_split(total_annotation_names, test_size=0.2)
# train
for file in train_files:
    ftrain.write(file + '\n')
# val
for file in val_files:
    fval.write(file + '\n')

ftrainval.close()
ftrain.close()
fval.close()
# ftest.close()

"""



