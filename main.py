import sys
import time
#sys.path.insert(0, './yolov5')
import cv2
import os
os.system('Xvfb :1 -screen 0 1600x1200x16  &')    # create virtual display with size 1600x1200 and 16 bit color. Color can be changed to 24 or 8
os.environ['DISPLAY']=':1.0'    # tell X clients to use our virtual DISPLAY :1.0
import shutil
import torch
import numpy as np
import PIL.Image, PIL.ImageTk
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from yolov5.utils.datasets import letterbox
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort


points = [[43, 543], [480, 330], [620, 350], [580, 550]]
counter = 0
pts = np.array([[43, 543], [480, 330], [620, 350], [580, 550]], np.int32)
pts = pts.reshape((-1, 1, 2))
isClosed = True
j = 0

#os.environ["DISPLAY"] = "localhost:0.0"
os.environ["MPLBACKEND"] = "Agg"


class Page(tk.Frame):

    def __init__(self, parent, window):
        self.count=0
        tk.Frame.__init__(self, parent)
        self.window = window
        self.window.title = "Title"

        self.out = 'inference/output'
        self.source = 'inference/images'
        self.weights = 'yolov5/weights/yolov5s.pt'
        self.view_img = False
        self.save_txt = False
        self.imgsz = 1088
        self.iou_thres = 0.5
        self.classes = [0, 1, 2, 3, 5]
        self.conf_thres = 0.4
        self.fourcc = 'mp4v'
        self.config_deepsort = 'deep_sort_pytorch/configs/deep_sort.yaml'
        self.device = ''
        self.agnostic_nms = False
        self.augment = False
        self.source = "highway.mp4"

        self.count = 0

        # Open camera source
        #self.vid = oneCameraCapture.cameraCapture()
        self.vs = cv2.VideoCapture(self.source)
        # Create a canvas that will fit the camera source
        self.canvas = tk.Canvas(window, width=1000, height=600)
        self.canvas.grid(row=0, column=0)

        menuFrame = ttk.Labelframe(window, text=("Menu"))
        menuFrame.grid(row=3, column=0, sticky="NSW",
                       padx=5, pady=2)
        # Button that lets the user take a snapshot
        self.btnSaveImage = tk.Button(menuFrame, text="Save Image", command=self.saveImage)
        self.btnSaveImage.grid(row=0, column=2, sticky="W")

        self.webcam = self.source == '0' or self.source.startswith(
            'rtsp') or self.source.startswith('http') or self.source.endswith('.txt')

        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(self.config_deepsort)
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                            nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        # Initialize
        self.device = select_device(self.device)
        if os.path.exists(self.out):
            shutil.rmtree(self.out)  # delete output folder
        os.makedirs(self.out)  # make new output folder
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = torch.hub.load("ultralytics/yolov5", 'yolov5s', pretrained=True)
        
        # Set Dataloader
        vid_path, vid_writer = None, None

        view_img = True
        save_img = True
        # dataset = LoadImages(self.source, img_size=self.imgsz)

        # Get names and colors
        # names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # Run inference
        # t0 = time.time()

        # run once
        # _ = model(img.half() if half else img) if device.type != 'cpu' else None

        self.save_path = str(Path(self.out))
        self.txt_path = str(Path(self.out)) + '/results.txt'



        self.delay = 100
        self.update()


    def update(self):

        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
        (grabbed, frame) = self.vs.read()
        fps = self.vs.get(cv2.CAP_PROP_FPS)
        path = self.source
        print("************", path)
        img0 = frame
        img = letterbox(img0, new_shape=640)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # count = self.count+1
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
        pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        t2 = time_synchronized()
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if self.webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, img0[i].copy()
            else:
                p, s, im0 = path, '', img0

            s += '%gx%g ' % img.shape[2:]  # print string
            # save_path = str(Path(self.out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                bbox_xywh = []
                confs = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
                    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
                    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
                    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
                    x_c = (bbox_left + bbox_w / 2)
                    y_c = (bbox_top + bbox_h / 2)
                    bbox_w = bbox_w
                    bbox_h = bbox_h
                    # x_c, y_c, bbox_w, bbox_h = bbox_rel(self, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                outputs = self.deepsort.update(xywhs, confss, im0)

                # draw line
                cv2.polylines(im0, [pts], isClosed, (0, 255, 255), 2)

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    offset = (0, 0)
                    counter = 0
                    for i, box in enumerate(bbox_xyxy):
                        x1, y1, x2, y2 = [int(i) for i in box]
                        x1 += offset[0]
                        x2 += offset[0]
                        y1 += offset[1]
                        y2 += offset[1]
                        # box text and bar
                        id = int(identities[i]) if identities is not None else 0
                        label = '{}{:d}'.format("", id)

                        # check if center points of object is inside the polygon
                        point = Point((int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2)))
                        polygon = Polygon(points)
                        if (polygon.contains(point)) == True:
                            counter = counter + 1
                            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                            cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.rectangle(
                                im0, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), (0, 255, 0), -1)
                            cv2.putText(im0, label, (x1, y1 +
                                                     t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255],
                                        2)
                    cv2.putText(im0, "Count - " + str(counter), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                                (0, 0, 255), 2)

                    self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img0))
                    self.canvas.create_image(500, 300, image=self.photo)
                    tk.Label(self.window, text="Count - " + str(counter), borderwidth=1).grid(row=1, column=0)



            else:
                self.deepsort.increment_ages()
            self.window.after(self.delay, self.update)
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))



    def saveImage(self):
        # Get a frame from the video source
        grabbed,frame = self.vs.read()

        cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg",
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":



    root = tk.Tk()
    
    testWidget = Page(root, root)
    testWidget.grid(row=0, column=0, sticky="W")
    root.mainloop()
