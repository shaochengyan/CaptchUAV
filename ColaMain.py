import time
import cv2
import numpy as np
import torch
import sys
import os
from threading import Thread
# Cola import
from ColaYolov3Detect import ColaYolov3
from ColaDataset import ColaDataset
from ColaKalman import ColaKalmanFilter
from ColaLib import (numpy_image_to_pixmap, 
                     scale_img_with_cons_ration, 
                     draw_box, 
                     draw_circle_with_boxsize)
# Qt import
from Ui_MainWindow import Ui_MainWindow
from PyQt5.QtWidgets import (
    QApplication, 
    QMainWindow, 
    QToolTip,
    QPushButton,
    QMessageBox,
    QTextEdit,
    QDesktopWidget,
    QHBoxLayout,
    QVBoxLayout,
    QDialog,
    QWidget, 
    QFileDialog
)
from PyQt5.QtGui import QFont, QPixmap, QIcon, QImage, QPalette, QColor, QFont
from PyQt5.QtCore import QTimer, QThread, Qt, pyqtSignal



class ColaMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None, is_test=True):
        super(ColaMainWindow, self).__init__(parent)
        self.setupUi(self)
        
        # variable
        self.cola_yolov3 = None
        self.cola_dataset = ColaDataset()
        self.cola_kalman = ColaKalmanFilter(12, 4, dt=5)
        
        # connect signal and slot
        self.init_ui()
        self.connect_signal_slot()
        
    
    def init_ui(self):
        self.pbtn_run.setEnabled(False)
        self.setWindowIcon(QIcon("UAV.png"))
    
    
    def connect_signal_slot(self):
        # Select source
        self.pbtn_slct_source.clicked.connect(
            self.cola_slot_select_source)
        
        # Load model
        self.pbtn_load_model.clicked.connect(
            self.cola_slot_load_model
        )
        
        # run to init cola_datset.pred
        self.pbtn_run.clicked.connect(
            self.cola_slot_run
        )
        
        # change img
        self.pbtn_pre_img.clicked.connect(self.cola_slot_img_change)
        self.pbtn_next_img.clicked.connect(self.cola_slot_img_change)
        
        # clear image source
        self.pbtn_clc_data.clicked.connect(self.cola_slot_clearn_data)
        
        # combbox change
        self.cbbox_max_frame_num.valueChanged.connect(self.cola_slot_dataset_para_change)
        self.cbbox_internal_video_frame.valueChanged.connect(self.cola_slot_dataset_para_change)
        
        # save data
        self.pbtn_save_data.clicked.connect(self.cola_slot_savedata)
        
        # init filter paramters
        self.pbtn_init_filter_parameters.clicked.connect(self.cola_slot_init_filter)
        
        # show video
        self.pbtn_show_video.clicked.connect(self.cola_slot_show_video)
        
        
    def cola_show_video(self):
        while self.cola_dataset.curr_idx != self.cola_dataset.len - 1:
            self.cola_show_curr_img()
            self.cola_dataset.update_idx(1)
            time.sleep(0.5)
            
    
    def cola_slot_show_video(self):
        t = Thread(target=self.cola_show_video)
        t.start()
        
    
    def cola_slot_init_filter(self):
        # init kalmanfilter with current z
        _, z0 = self.cola_dataset.get_curr_data()
        self.cola_kalman.init_x(z0[:, 0].reshape(4, 1))
        
    
    def cola_slot_savedata(self):
        self.cola_dataset.save_data()
        
        
    def cola_slot_dataset_para_change(self):
        self.cola_dataset.init_paramenter(
            self.cbbox_max_frame_num.value(), 
            self.cbbox_internal_video_frame.value()
        )
        
    
    def cola_slot_clearn_data(self):
        self.cola_dataset.clear()


    def cola_slot_img_change(self):
        sender = self.sender().objectName()
        print(type(sender))
        print(sender)
        if sender == "pbtn_pre_img":
            self.cola_dataset.update_idx(-1)
        elif sender == 'pbtn_next_img':
            self.cola_dataset.update_idx(1)
        self.cola_show_curr_img()
    
    
    def cola_show_curr_img(self):
        try:
            img, pred = self.cola_dataset.get_curr_data()
        except Exception as ex:
            print(ex)
        else:
            box = pred.reshape(4, -1).astype(np.float) if pred is not None else None
            # Kalman filter
            self.cola_kalman.predict()
            self.cola_kalman.cola_update(box)
            box_filter = self.cola_kalman.get_box().reshape(-1)
            boxs_next = self.cola_kalman.get_next_steps_z_hat(5)

            # show image
            if self.cbbox_img_option.currentIndex() == 0:  # original
                pass 
            elif self.cbbox_img_option.currentIndex() == 1:  # yolo
                if pred is not None:
                    for i in range(box.shape[1]):
                        img = draw_box(img, box[:, i], 'uav')
            elif self.cbbox_img_option.currentIndex() == 2:  # kalman
                img = draw_box(img, box_filter)
                for i in range(boxs_next.shape[1]):
                    draw_circle_with_boxsize(img, boxs_next[:, i])
            elif self.cbbox_img_option.currentIndex() == 2:  # both
                if pred is not None:
                    for i in range(box.shape[1]):
                        img = draw_box(img, box[:, i], 'uav')
                img = draw_box(img, box_filter)
                for i in range(boxs_next.shape[1]):
                    draw_circle_with_boxsize(img, boxs_next[:, i])
            self.cola_show_img(img)
    

    def cola_run(self):
        self.pbtn_run.setEnabled(False)
        for i, (img, pred) in enumerate(self.cola_dataset):
            if pred is not None:
                continue
            pred, cof = self.cola_yolov3.inference_img_xywh(img)
            self.cola_dataset.pred_list[i] = pred.numpy().reshape(4, -1)
            self.cola_dataset.pred_conf[i] = cof.numpy().reshape(1, -1)
            print(self.cola_dataset.pred_list[i])
        self.pbtn_run.setEnabled(True)

    
    def cola_slot_run(self):
        t = Thread(target=self.cola_run)
        t.start()
        
    
    def cola_load_model(self):
        self.pbtn_load_model.setEnabled(False)
        if self.cola_yolov3 is None:
            self.cola_yolov3 = ColaYolov3()
        self.pbtn_load_model.setEnabled(True)
        self.pbtn_run.setEnabled(True)


    def cola_slot_load_model(self):
        t = Thread(target=self.cola_load_model)
        t.start()
        
    
    def cola_show_img(self, img):
        if img is None:
            return
        img = scale_img_with_cons_ration(img)
        self.label_img.setPixmap((numpy_image_to_pixmap(img)))
    
    
    def cola_init_dataset(self, filename_list, filetype):
        self.pbtn_slct_source.setEnabled(False)
        self.cbbox_img_option.setCurrentIndex(0)
        self.cola_dataset.update(filename_list, filetype)
        self.cola_show_curr_img()
        self.pbtn_slct_source.setEnabled(True)

    
    def cola_slot_select_source(self):
        filename_list, filetype = QFileDialog.getOpenFileNames(
            self, "Select File", 
            "D:/Coding/Study/DL/YOLOv3/yolov3/ColaDo/datasets/mycoco/mycoco/images/train", 
            "ImageFile(*.jpg;*.png);;VideoFile(*.mp4);;")
        if len(filename_list) != 0:
            t = Thread(target=self.cola_init_dataset, args=(filename_list, filetype))
            t.start()
        


def main():
    app = QApplication(sys.argv)
    cola_win = ColaMainWindow()
    cola_win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
    