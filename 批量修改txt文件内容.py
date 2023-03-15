import os
import random
import numpy as np
from numpy import *

names=['VOC_call','VOC_dislike','VOC_fist','VOC_four','VOC_like','VOC_mute','VOC_ok','VOC_one','VOC_palm','VOC_peace','VOC_peace_inverted','VOC_rock','VOC_stop','VOC_stop_inverted','VOC_three','VOC_three2','VOC_two_up','VOC_two_up_inverted']
for filename in names:
    txtfilepath = "OriginHandsData/"+filename+"/ImageSets/Main" #原始txt文件所存文件夹，文件夹可以有一个或多个txt文件
    savefilepath = "OriginHandsData/"+filename+"/ImageSets/Main" #更改后txt文件存放的文件夹
    total_txt = os.listdir(txtfilepath) # 返回指定的文件夹包含的文件或文件夹的名字的列表
    num = len(total_txt)
    list = range(num) #创建从0到num的整数列表
    files = os.listdir(savefilepath)
    for i in list: #遍历每一个文件
        name = total_txt[i]
        readfile = open(txtfilepath+"/"+name, 'r') #读取文件,
        fline = readfile.readlines() #读取txt文件中每一行
        savetxt = open(savefilepath+"/"+name,'w')#  必须写w，否则会报“not writable”，w才有写入权限

        for j in fline:
            #if "你查找的内容" in j:
            if ".jpg" in j:
               # b = j.replace('你所查找的内容', '替换成的内容') #替换固定行内容
                b = j.replace('.jpg', '') #替换固定行内容
                savetxt.write(b) #写入新的文件中
