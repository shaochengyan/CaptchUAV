import torch
import sys
import cv2
from pathlib import Path 
# Yolov3 import
sys.path.append('..')
from models.common import DetectMultiBackend
from utils.datasets import LoadImages, LoadStreams
from utils.general import (check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from utils.plots import Annotator, Colors, colors, save_one_box
from utils.torch_utils import select_device, time_sync


"""
Created for study, from detect.py
"""
@torch.no_grad()
def run(weights="./pretrained/best_for_uav.pt", 
        # "./datasets/mycoco/mycoco/images/train/00_01_52_to_00_01_58_045.jpg"
        source="./datasets/mycoco/mycoco/images/train/00_01_52_to_00_01_58_045.jpg",  # ./data/uav_test.mp4 ./data/zidane.jpg
        imgsz=[640, 640], 
        conf_thres=0.25, 
        iou_thres=0.45, 
        max_det=1000,
        device='', 
        view_img=True, 
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
    # ready
    source = str(source)
    
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)
    
    # Half precision, just supported by PyTorch on CUDA
    half &= pt and device.type != 'cpu'
    if pt:
        model.model.half() if half else model.model.float()

    # Dataloader and batch size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
    bs = 1 
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0  # todo?
    for path, im, im0s, vid_cap, s, in dataset:
        # print(im)
        # print(im.shape)
        # import numpy as np
        # im = np.swapaxes(im, 0, 2)
        # print(im.shape)
        # cv2.imshow("WIN", im)
        # cv2.waitKey(0)
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255
        if len(im.shape) == 3:
            im = im[None]  # (1, 3, h, w)
        t2 = time_sync()
        dt[0] += t2 - t1
        
        # Inference
        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
        )
        dt[2] += time_sync() - t3
        
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)
            annotator = Annotator(im0, line_width=line_thickess, example=str(names))
            
            if len(det):
                # Rescale
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                
                # Results
                for *xyxy, cof, cls in reversed(det):
                    c = int(cls)
                    label = names[c]
                    annotator.box_label(xyxy, label, color=colors(c, True))
            
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(0)
                    

def test1():
    run()


if __name__ == '__main__':
    test1()