from asyncio import trsock
from typing_extensions import Self
import torch
import sys
import cv2
from pathlib import Path
import numpy as np

# Yolov3 import
sys.path.append('..')
from models.common import DetectMultiBackend
from utils.augmentations import letterbox 
from utils.datasets import LoadImages, LoadStreams
from utils.general import (check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from utils.plots import Annotator, Colors, colors, save_one_box
from utils.torch_utils import select_device, time_sync


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

"""
"./datasets/mycoco/mycoco/images/train/00_01_52_to_00_01_58_045.jpg",  # ./data/uav_test.mp4 ./data/zidane.jpg
"""
@torch.no_grad()
class ColaYolov3:
    @torch.no_grad()
    def __init__(
            self, 
            weights="./pretrained/best_for_uav.pt", 
            # "./datasets/mycoco/mycoco/images/train/00_01_52_to_00_01_58_045.jpg"
            source="./datasets/mycoco/mycoco/images/train/00_01_52_to_00_01_58_045.jpg",  # ./data/uav_test.mp4 ./data/zidane.jpg
            imgsz=[640, 640], 
            conf_thres=0.25, 
            iou_thres=0.45, 
            max_det=1000,
            device='', 
            save_txt=False, 
            save_conf=False, 
            nosave=False, 
            classes=None, 
            agnostic_nms=False, 
            update=False, 
            project="./runs_srtp_detect", 
            name='exp', 
            exist_ok=False, 
            line_thickess=3, 
            hide_labels=False, 
            hide_conf=False, 
            half=False, 
            dnn=False):

        # variabel
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.nosave = nosave
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.update = update
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.line_thickess = line_thickess
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.half = half
        self.dnn = dnn
        
        # Load model
        self.device = select_device(device)  # cuda:0
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn)
        self.stride, self.names, self.pt, self.jit, self.onnx = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx
        self.imgsz = check_img_size(imgsz, s=self.stride)
        
        # Half precision, just supported by PyTorch on CUDA
        self.half = half    
        self.half &= self.pt and self.device.type != 'cpu'
        if self.pt:
            self.model.model.half() if half else self.model.model.float()
            
        # Image parameters
        self.img_size=imgsz
        self.auto = self.pt
        

    def preprocess_img(self, img):
        """Imitate from LoadImage
        return: image with (3, h, w) -> rgb
        Args:
            img (_type_): ndarray, (h, w, 3), bgr 
        """        
        # padded resize
        img = letterbox(img, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img

    
    def run_img_file(self, img_filename, view_img=False):
        img = cv2.imread(img_filename)
        assert img is not None
        return self.run_img(img, view_img)
    

    @torch.no_grad()
    def run_img(self, img, view_img=False):
        im0s = img.copy()
        # Preprocess
        img, pred = self.inference_img(img)
        
        # Process predictions
        for i, det in enumerate(pred):  # per image
            im0 = im0s.copy()
            
            if len(det):
                # Rescale
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Results
                for *xyxy, cof, cls in reversed(det):
                    c = int(cls)
                    label = self.names[c]
        return im0
    
    
    def inference_img_file(self, img_filename):
        img = cv2.imread(img_filename)
        return self.inference_img(img)
    
    @torch.no_grad()
    def inference_img(self, img):
        # Preprocess
        img = self.preprocess_img(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        img = img[None]
        
        # Inference
        pred = self.model(img, augment=False, visualize=False)

        # NMS
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det
        )
        
        return img, pred
    

    @torch.no_grad()
    def inference_img_xywh(self, img):
        im0s = img.copy()
        # Preprocess
        img, pred = self.inference_img(img)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            im0 = im0s.copy()
            
            if len(det):
                # Rescale
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                det = det.cpu()
                xywh = xyxy2xywh(det[..., 0:4]).type(torch.int32)
                return xywh, det[..., 4]
        return (None, None)
        
    
    

def test1():
    cola_yolov3 = ColaYolov3()
    cola_yolov3.run_img_file(
        "./datasets/mycoco/mycoco/images/train/00_01_52_to_00_01_58_045.jpg", 
        view_img=True
    )

if __name__ == '__main__':
    test1()