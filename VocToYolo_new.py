import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

# sets = ['train', 'val']
#
# classes = ['call', 'no_gesture']  # voc的2个类别 0 call 1 no_gesture
sets=['train','val']
classes=['one','two_up','two_up_inverted','three','three2','four','fist','palm','ok','peace','peace_inverted','like','dislike','stop','stop_inverted','call','mute','rock','no_gesture']

names=['VOC_stop','VOC_rock','VOC_call','VOC_dislike','VOC_fist','VOC_four','VOC_like','VOC_mute','VOC_ok','VOC_one','VOC_palm','VOC_peace','VOC_peace_inverted','VOC_stop_inverted','VOC_three','VOC_three2','VOC_two_up','VOC_two_up_inverted']

for file in names:
    print(file)
    def convert(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)


    def convert_annotation(image_id):
        #in_file = open('OriginHandsData/VOC_call/Annotations/%s.xml' % (image_id))
        in_file = open('OriginHandsData/'+file+'/Annotations/%s.xml' % (image_id))
        # 填原来voc数据集xml标注数据文件所在路径
        out_file = open('myyolodata/yolo_labels/test/%s.txt' % (image_id), 'w')
        # 填转换后的yolov5需要labels文件所在路径
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


    # if __name__ == "__main__":

    wd = getcwd()

    for image_set in sets:
        if not os.path.exists('myyolodata/yolo_labels'):
            os.makedirs('myyolodata/yolo_labels')
        # 如果不存在存放转换后的labels的文件，则创建转换后labels存放文件路径

        #image_ids = open('OriginHandsData/VOC_call/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
        image_ids = open('OriginHandsData/'+file+'/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
        # list_file = open('./%s.txt' % (image_set), 'w')

        for image_id in image_ids:
            # list_file.write('dataset/VOCdevkit/VOC2007/JPEGImages/%s.jpg\n' % (image_id))
            convert_annotation(image_id)
        # list_file.close()