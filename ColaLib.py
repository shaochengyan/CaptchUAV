import cv2
import numpy as np
from threading import Thread
import os
import sys
# Qt import
from PyQt5.QtGui import QImage, QPixmap


def draw_circle_with_boxsize(img, box, color=(0, 255, 255)):
    raidus = max(int(min(box[2:4]) / 20), 5)
    img = cv2.circle(img, (int(box[0]), int(box[1])), raidus, color, -1)


def draw_box(img, box, label='uav', color=(128, 128, 128), txt_color=(255, 255, 255)):
    """return img with box

    Args:
        box (array like): len = 4
        label (str, optional): uav Defaults to ''.
        color (tuple, optional): _description_. Defaults to (128, 128, 128).
        txt_color (tuple, optional): _description_. Defaults to (255, 255, 255).
    """
    p1, p2 = (int(box[0] - box[2] / 2), int(box[1] - box[3] / 2)), (int(box[0] + box[2] / 2), int(box[1] + box[3] / 2))
    cv2.rectangle(img, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
    return img


def numpy_image_to_pixmap(img):
    img_h, img_w, img_c = img.shape
    num_byte_per_line = img_c * img_w
    q_img  = QImage(img.data, img_w, img_h, num_byte_per_line, QImage.Format_BGR888 )
    return QPixmap.fromImage(q_img)


def scale_img_with_cons_ration(img, win_w=800, win_h=600):
    # get new size    
    h, w = img.shape[0:2]
    ratio_img = w / h
    if ratio_img > win_w / win_h:
        new_w = win_w
        new_h = int(new_w / ratio_img)
    else:
        new_h = win_h
        new_w = int(new_h * ratio_img)
        
    # resize
    img = cv2.resize(img, (new_w, new_h))
    
    # padding
    top_pad_num = round((win_h - new_h) / 2)
    bot_pad_num = win_h - new_h - top_pad_num
    left_pad_num = round((win_w - new_w) / 2)
    right_pad_num = win_w - new_w - left_pad_num
    img = cv2.copyMakeBorder(img, 
                             top_pad_num, 
                             bot_pad_num, 
                             left_pad_num, 
                             right_pad_num, 
                             cv2.BORDER_CONSTANT, value=0)
    return img