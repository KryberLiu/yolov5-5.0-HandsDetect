import sys
import cv2
import time

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5 import QtCore, QtGui

from MyUI import *
from MyUI.UI import Ui_MainWindow


class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)  # 隐藏边框
        self.timer=QtCore.QTimer()#初始化定时器
        self.cap=cv2.VideoCapture()#初始化摄像头
        self.slot_init() #信号槽绑定初始化
        self.show()

#信号槽绑定
    def slot_init(self):
        self.ui.horizontalSlider.valueChanged.connect(self.doubleSpinBoxChange)
        self.ui.doubleSpinBox.valueChanged.connect(self.horizontalSliderChange)
        self.ui.horizontalSlider_2.valueChanged.connect(self.doubleSpinBox_2Change)
        self.ui.doubleSpinBox_2.valueChanged.connect(self.horizontalSlider_2Change)
        self.ui.horizontalSlider_3.setEnabled(False)
        self.ui.pushButton_4.setEnabled(False)
        self.ui.radioButton.toggled.connect(self.radioButtonClick)
        self.ui.pushButton_2.clicked.connect(self.open_image)#打开待检测图片
        self.ui.camera_Button.clicked.connect(self.button_open_camera_click)#打开摄像头
        self.timer.timeout.connect(self.show_camera)
        self.ui.select_video_Button.clicked.connect(self.button_opne_video_click)#打开视频文件，还在调试
        self.timer.timeout.connect(self.show_video)

    def button_opne_video_click(self):
        if self.timer.isActive()==False:#若定时器未启动
            videoName, videoType = QFileDialog.getOpenFileName(self.centralWidget(), "选择待检测视频", "",
                                                               "*.mp4;;*.avi;;*.mov;;*.flv;;All Files(*)")
            flag=self.cap.open(videoName)
            if flag==False:# flag表示open()成不成功
                msg = QMessageBox.Warning(self, 'Warning', '视频打开失败',  buttons=QMessageBox.Ok)
            else:
                self.timer.start(50)# 定时器开始计时30ms，结果是每过30ms从视频中取一帧显示
        else:
            self.timer.stop()
            self.cap.release()

    def show_video(self):
        flag,self.videoImage=self.cap.read()
        frame = cv2.cvtColor(self.videoImage, cv2.COLOR_BGR2RGB)
        frame = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.ui.label_8.setPixmap(QPixmap.fromImage(frame))
        self.ui.label_8.setScaledContents(True)  # 自适应大小

#启动电脑自带的摄像头，并设置截取捕获图像的时间。按照设置时间区间返回捕获图像。
    def button_open_camera_click(self):
        if self.timer.isActive()==False:# 若定时器未启动
            flag=self.cap.open(0) # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if flag==False:# flag表示open()成不成功
                msg = QMessageBox.Warning(self, 'Warning', '请检测相机与电脑是否连接正确',  buttons=QMessageBox.Ok)
            else:
                self.timer.start(30)# 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
        else:
            self.timer.stop()
            self.cap.release()
#根据设定的定时器时间，逐个展示摄像头捕获的图像，从而形成实时摄像展示
    def show_camera(self):
        flag,self.image=self.cap.read()
        frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        frame=QImage(frame.data,frame.shape[1],frame.shape[0],QImage.Format_RGB888)
        self.ui.label_8.setPixmap(QPixmap.fromImage(frame))
        self.ui.label_8.setScaledContents(True)  # 自适应大小


    def radioButtonClick(self):
        if self.ui.radioButton.isChecked() == True:
            self.ui.horizontalSlider_3.setEnabled(True)
            self.ui.pushButton_4.setEnabled(True)
            self.ui.horizontalSlider_3.setSingleStep(1)
            self.ui.horizontalSlider_3.setRange(0, 20)
            self.ui.pushButton_4.setSingleStep(1)
            self.ui.pushButton_4.setRange(0, 20)
            self.ui.horizontalSlider_3.valueChanged.connect(self.pushButton_4Change)
            self.ui.pushButton_4.valueChanged.connect(self.horizontalSlider_3Change)
        if self.ui.radioButton.isChecked() == False:
            self.ui.horizontalSlider_3.setEnabled(False)
            self.ui.pushButton_4.setEnabled(False)

    def doubleSpinBoxChange(self):
        self.ui.doubleSpinBox.setValue(self.ui.horizontalSlider.value() / 100)

    def horizontalSliderChange(self):
        self.ui.horizontalSlider.setValue(self.ui.doubleSpinBox.value() * 100)

    def doubleSpinBox_2Change(self):
        self.ui.doubleSpinBox_2.setValue(self.ui.horizontalSlider_2.value() / 100)

    def horizontalSlider_2Change(self):
        self.ui.horizontalSlider_2.setValue(self.ui.doubleSpinBox_2.value() * 100)

    def pushButton_4Change(self):
        self.ui.pushButton_4.setValue(self.ui.horizontalSlider_3.value())

    def horizontalSlider_3Change(self):
        self.ui.horizontalSlider_3.setValue(self.ui.pushButton_4.value())

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    # 拖动窗口
    def mousePressEvent(self, event):
        if event.button()==QtCore.Qt.LeftButton and self.isMaximized()==False:
            self.m_flag=True
            self.m_Position=event.globalPos()-self.pos()#获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))#更改鼠标图标

    def mouseMoveEvent(self, mouse_event):
        if QtCore.Qt.LeftButton and self.m_flag:
            self.move(mouse_event.globalPos()-self.m_Position)#更改窗口位置
            mouse_event.accept()

    def mouseReleaseEvent(self,mouse_event):
        self.m_flag=False
        self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

    #打开图片
    def open_image(self):
        imgName,imgType=QFileDialog.getOpenFileName(self,"选择待检测图片","","*.jpg;;*.png;;All Files(*)")
        '''上面一行代码是弹出选择文件的对话框，第一个参数固定，第二个参数是打开后右上角显示的内容
		第三个参数是对话框显示时默认打开的目录，"." 代表程序运行目录
		第四个参数是限制可打开的文件类型。
		返回参数 imgName为G:/xxxx/xxx.jpg，imgType为*.jpg。	
		此时相当于获取到了文件地址 
	    '''
        img = QtGui.QPixmap(imgName)  # 通过文件路径获取图片文件
        self.ui.label_8.setPixmap(img)  # 在label控件上显示选择的图片
        self.ui.label_8.setScaledContents(True)#自适应大小

    # #打开视频文件
    # def open_video(self):
    #     videoName,videoType=QFileDialog.getOpenFileName(self.centralWidget(),"选择待检测视频","","*.mp4;;*.avi;;*.mov;;*.flv;;All Files(*)")
    #     self.cap_video=cv2.VideoCapture(videoName)#打开视频
    #     #print(cap_video.isOpened()) #用来验证是否成功打开视频文件
    #     self.timer.start(50)
    #     flag, frame = self.cap_video.read()  # 获取视频的每一帧
    #     if flag:
    #         frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #         frame=QImage(frame.data,frame.shape[1],frame.shape[0],QImage.Format_RGB888)
    #         self.ui.label_8.setPixmap(QPixmap.fromImage(frame))
    #         self.ui.label_8.setScaledContents(True)  # 自适应大小
    #     else :
    #         self.cap_video.release()
    #         self.timer.stop()#停止读取


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow
    import images  # 导入添加的资源（根据实际情况填写文件名）
    import res

    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = MyMainWindow()

    ui.show()
    sys.exit(app.exec_())