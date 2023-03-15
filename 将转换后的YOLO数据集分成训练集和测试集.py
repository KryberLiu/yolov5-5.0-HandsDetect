import os
import shutil

names=['VOC_stop','VOC_call','VOC_dislike','VOC_fist','VOC_four','VOC_like','VOC_mute','VOC_ok','VOC_one','VOC_palm','VOC_peace','VOC_peace_inverted','VOC_rock','VOC_stop_inverted','VOC_three','VOC_three2','VOC_two_up','VOC_two_up_inverted']

for file in names:
    print(file)
    train_txt = r"OriginHandsData/"+file+"/ImageSets/Main/train.txt"
    test_txt = r"OriginHandsData/"+file+"/ImageSets/Main/val.txt"

    img_folder = r"OriginHandsData/"+file+"/JPEGImages"
    yolo_labels = r"myyolodata/yolo_labels/test"

    #HRSID_dir = r"/home/dwt/MyCode/object detection/yolov5/DataSets/HRSID"
    Dataset=r"myyolodata"

    file_train = open(train_txt)
    file_test = open(test_txt)
    for line in file_train.readlines():
        line = line.strip()
        #print(line)
        shutil.copyfile(os.path.join(img_folder, line + ".jpg"), os.path.join(Dataset, "images", "train",
                                                                              line + ".jpg"))  # 根据train.txt指示的文件名将对应的图片复制到yolo格式整理的数据文件夹中
        shutil.copyfile(os.path.join(yolo_labels, line + ".txt"), os.path.join(Dataset, "labels", "train",
                                                                               line + ".txt"))  # 根据train.txt指示的文件名将对应的标注文件复制到yolo格式整理的数据文件夹中

    for line in file_test.readlines():
        line = line.strip()
        shutil.copyfile(os.path.join(img_folder, line + ".jpg"), os.path.join(Dataset, "images", "val",
                                                                              line + ".jpg"))  # 根据val.txt指示的文件名将对应的图片复制到yolo格式整理的数据文件夹中
        shutil.copyfile(os.path.join(yolo_labels, line + ".txt"), os.path.join(Dataset, "labels", "val",
                                                                               line + ".txt"))  # 根据val.txt指示的文件名将对应的标注文件复制到yolo格式整理的数据文件夹中

