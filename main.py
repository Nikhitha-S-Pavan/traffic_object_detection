import time
import cv2
import os
os.system(
    "Xvfb :1 -screen 0 1600x1200x16  &"
)  # create virtual display with size 1600x1200 and 16 bit color. Color can be changed to 24 or 8
os.environ["DISPLAY"] = ":1.0"  # tell X clients to use our virtual DISPLAY :1.0
import datetime
import shutil
import torch
import numpy as np
import PIL.Image, PIL.ImageTk
import tkinter as tk
from pathlib import Path
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from yolov5.utils.datasets import letterbox
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

os.environ["MPLBACKEND"] = "Agg"


class Page:
    def __init__(self):
        self.count = 0

        self.root = tk.Tk()

        self.out = "inference/output"
        self.source = "inference/images"
        self.weights = "yolov5/weights/yolov5s.pt"
        self.view_img = False
        self.save_txt = False
        self.imgsz = 1088
        self.iou_thres = 0.5
        self.classes = [0, 1, 2, 3, 5, 7]
        self.conf_thres = 0.4
        self.fourcc = "mp4v"
        self.config_deepsort = "deep_sort_pytorch/configs/deep_sort.yaml"
        self.device = ""
        self.agnostic_nms = False
        self.augment = False
        self.two_w, self.three_w, self.four_w, self.truck, self.bus, self.total = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        self.count = 0
        self.fps = None
        font = ("Arial", 25)
        self.root.resizable(0, 0)
        self.panel = tk.Frame(self.root)
        self.panel.pack(side="top", padx=10, pady=10)
        self.canvas = tk.Label(self.panel, text="loading...", anchor="center", font=font, fg="blue")
        self.canvas.pack(side="left", padx=10, pady=10)
        self.counting_result = tk.Frame(self.root)
        self.counting_result.pack(side="bottom", padx=10, pady=10)
        self.Quit_btn = tk.Button(
            self.counting_result,
            text="Quit",
            font=("Arial", 12),
            command=self.onClose,
            bg="red",
            fg="white",
            width=6,
        )
        self.Quit_btn.grid(row=2, column=5)

        # set a callback to handle when the window is closed
        self.root.wm_title("Traffic")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

        # Open camera source
        # self.vid = oneCameraCapture.cameraCapture()
        self.vs = cv2.VideoCapture("traffic3.mp4")

        self.webcam = (
            self.source == "0"
            or self.source.startswith("rtsp")
            or self.source.startswith("http")
            or self.source.endswith(".txt")
        )

        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(self.config_deepsort)
        self.deepsort = DeepSort(
            cfg.DEEPSORT.REID_CKPT,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE,
            n_init=cfg.DEEPSORT.N_INIT,
            nn_budget=cfg.DEEPSORT.NN_BUDGET,
            use_cuda=True,
        )

        # Initialize
        self.device = select_device(self.device)
        if os.path.exists(self.out):
            shutil.rmtree(self.out)  # delete output folder
        os.makedirs(self.out)  # make new output folder
        self.half = self.device.type != "cpu"  # half precision only supported on CUDA

        # Load model
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        self.source = "traffic3.mp4"
        self.save_path = str(Path(self.out))
        self.txt_path = str(Path(self.out)) + "/results.txt"
        self.points = [[5, 100], [400, 100], [730, 300], [5, 300]]
        self.pts = np.array(self.points, np.int32)
        self.pts_arr = self.pts.reshape((-1, 1, 2))
        self.isClosed = True
        self.delay = 100
        self.update()

    def onClose(self):
        print("[INFO] closing...")
        # self.stopEvent.set()
        self.vs.release()
        self.root.quit()
        # self.root.destroy()

    def calculate_fps(self, start_time, framec):
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = framec / elapsed_time
        return fps

    def Obj_counting(self, Id, label, trackIds, count, total):
        if Id in trackIds:
            count = count
            total = total
        else:
            count += 1
            total += 1
            trackIds.append(Id)
        return count, total

    def update(self):
        f = 0
        start_time = datetime.datetime.now()
        today = datetime.date.today()
        # dd/mm/YY
        date = today.strftime("%d/%m/%Y")
        current_time = start_time.strftime("%H:%M:%S")

        trackIds, position, speed_e, fps = [], {}, 0, 0.0
        two_w, three_w, four_w, truck, bus, total = 0, 0, 0, 0, 0, 0
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
        (grabbed, frame) = self.vs.read()

        path = "traffic3.mp4"
        img0 = frame
        names = self.model.module.names if hasattr(self.model, "module") else self.model.names

        if grabbed == True:
            img = letterbox(img0, new_shape=640)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            f = f + 1
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
                pred,
                self.conf_thres,
                self.iou_thres,
                classes=self.classes,
                agnostic=self.agnostic_nms,
            )
            t2 = time_synchronized()
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if self.webcam:  # batch_size >= 1
                    p, s, im0 = path[i], "%g: " % i, img0[i].copy()
                else:
                    p, s, im0 = path, "", img0

                s += "%gx%g " % img.shape[2:]  # print string
                # save_path = str(Path(self.out) / Path(p).name)

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    bbox_xywh = []
                    confs = []
                    labels = []

                    # Adapt detections to deep sort input format
                    for *xyxy, conf, cls in det:
                        label = f"{names[int(cls)]}"
                        bbox_left = min([xyxy[0].item(), xyxy[2].item()])
                        bbox_top = min([xyxy[1].item(), xyxy[3].item()])
                        bbox_w = abs(xyxy[0].item() - xyxy[2].item())
                        bbox_h = abs(xyxy[1].item() - xyxy[3].item())
                        x_c = bbox_left + bbox_w / 2
                        y_c = bbox_top + bbox_h / 2
                        bbox_w = bbox_w
                        bbox_h = bbox_h
                        # x_c, y_c, bbox_w, bbox_h = bbox_rel(self, *xyxy)
                        obj = [x_c, y_c, bbox_w, bbox_h]
                        bbox_xywh.append(obj)
                        confs.append([conf.item()])
                        labels.append(label)

                    confss, labelss = [], []
                    for conf, label in zip(confs, labels):
                        confss.append(conf)
                        labelss.append(label)

                    xywhs = torch.Tensor(bbox_xywh)
                    confss = torch.Tensor(confs)

                    # Pass detections to deepsort
                    outputs = self.deepsort.update(xywhs, confss, im0)

                    # draw line
                    cv2.polylines(im0, [self.pts_arr], self.isClosed, (255, 0, 0), 2)
                    cv2.rectangle(img0,(650,0), (850,170), color = (0, 0, 0) , thickness=-1)
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        offset = (0, 0)
                        counter = 0
                        for i, box in enumerate(bbox_xyxy):
                            if i < (len(labels[::-1]) - 1):
                                x1, y1, x2, y2 = [int(i) for i in box]
                                x1 += offset[0]
                                x2 += offset[0]
                                y1 += offset[1]
                                y2 += offset[1]
                                # box text and bar
                                id = int(identities[i]) if identities is not None else 0
                                label = "{}{:d}".format("", id)

                                cls = labels[::-1][i]

                                # Object counting
                                if cls == "motorcycle":
                                    two_w, total = self.Obj_counting(id, label, trackIds, two_w, total)
                                elif cls == "auto":
                                    three_w, total = self.Obj_counting(id, label, trackIds, three_w, total)
                                elif cls == "car":
                                    four_w, total = self.Obj_counting(id, label, trackIds, four_w, total)
                                elif cls == "truck":
                                    truck, total = self.Obj_counting(id, label, trackIds, truck, total)
                                elif cls == "bus":
                                    bus, total = self.Obj_counting(id, label, trackIds, bus, total)
                                fps = self.calculate_fps(start_time, f)
                                # check if center points of object is inside the polygon
                                point = Point((int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2)))
                                polygon = Polygon(self.points)
                                if (polygon.contains(point)) == True:
                                    counter = counter + 1
                                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                                    cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        if counter > 5:
                            flow = "High"
                        elif counter >=2 and counter <5:
                            flow = "Medium"
                        else:
                            flow = "Low"
                        cv2.putText(
                            im0,
                            "Occupancy - " + str(counter),
                            (650, 30),
                            cv2.FONT_HERSHEY_DUPLEX,
                            .5,
                            (255, 0, 0),
                            1,
                        )
                        cv2.putText(
                            im0,
                            "Date - " + str(date),
                            (650, 60),
                            cv2.FONT_HERSHEY_DUPLEX,
                            .5,
                            (255, 0, 0),
                            1,
                        )
                        cv2.putText(
                            im0,
                            "Time - " + str(current_time),
                            (650, 90),
                            cv2.FONT_HERSHEY_DUPLEX,
                            .5,
                            (255, 0, 0),
                            1,
                        )
                        cv2.putText(
                            im0,
                            "Speed - " + "N A",
                            (650, 120),
                            cv2.FONT_HERSHEY_DUPLEX,
                            .5,
                            (255, 0, 0),
                            1,
                        )
                        cv2.putText(
                            im0,
                            "Flow - " + str(flow),
                            (650, 150),
                            cv2.FONT_HERSHEY_DUPLEX,
                            .5,
                            (255, 0, 0),
                            1,
                        )

                    # img = cv2.resize(img, (650, 360))
                    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    image = PIL.Image.fromarray(img0)
                    image = PIL.ImageTk.PhotoImage(image)
                    font = ("Arial", 12)
                    self.canvas.configure(image=image)
                    self.canvas.image = image
                    result = tk.Label(
                        self.counting_result,
                        text=f"Counting Results",
                        width=12,
                        font=font,
                        anchor="center",
                        fg="blue",
                    )
                    result.grid(row=0, column=2, padx=2)
                    # result.pack(padx=10, pady=10)
                    if self.two_w is None:
                        self.two_w = tk.Label(
                            self.counting_result,
                            text=f"Two Wheeler \n\n{two_w}",
                            width=13,
                            font=font,
                            anchor="center",
                            bg="#8080c0",
                            fg="white",
                        )
                        self.two_w.grid(row=1, column=0, padx=2)
                    else:
                        self.two_w.configure(text=f"Two Wheeler\n\n{two_w}")

                    if self.three_w is None:
                        self.three_w = tk.Label(
                            self.counting_result,
                            text=f"Three Wheeler\n\n{three_w}",
                            font=font,
                            width=13,
                            anchor="center",
                            bg="#8080c0",
                            fg="white",
                        )
                        self.three_w.grid(row=1, column=1, padx=2)
                    else:
                        self.three_w.configure(text=f"Three Wheeler\n\n{three_w}")

                    if self.four_w is None:
                        self.four_w = tk.Label(
                            self.counting_result,
                            text=f"Four Wheeler\n\n{four_w}",
                            width=13,
                            font=font,
                            anchor="center",
                            bg="#8080c0",
                            fg="white",
                        )
                        self.four_w.grid(row=1, column=2, padx=2)
                    else:
                        self.four_w.configure(text=f"Four Wheeler\n\n{four_w}")

                    if self.truck is None:
                        self.truck = tk.Label(
                            self.counting_result,
                            text=f"Truck\n\n{truck}",
                            font=font,
                            width=10,
                            anchor="center",
                            bg="#8080c0",
                            fg="white",
                        )
                        self.truck.grid(row=1, column=3, padx=1)
                    else:
                        self.truck.configure(text=f"Truck\n\n{truck}")

                    if self.bus is None:
                        self.bus = tk.Label(
                            self.counting_result,
                            text=f"Bus\n\n{bus}",
                            font=font,
                            width=10,
                            anchor="center",
                            bg="#8080c0",
                            fg="white",
                        )
                        self.bus.grid(row=1, column=4, padx=2)
                    else:
                        self.bus.configure(text=f"Bus\n\n{bus}")

                    if self.total is None:
                        self.total = tk.Label(
                            self.counting_result,
                            text=f"Total Vehicle\n\n{total}",
                            font=font,
                            width=10,
                            anchor="center",
                            bg="#8080c0",
                            fg="white",
                        )
                        self.total.grid(row=1, column=5, pady=2)
                    else:
                        self.total.configure(text=f"Total Vehicle\n\n{total}")

                    if self.fps is None:
                        self.fps = tk.Label(
                            self.counting_result,
                            text=f"FPS\n\n{fps:.2f}",
                            font=font,
                            width=13,
                            anchor="center",
                            bg="#8080c0",
                            fg="white",
                        )
                        self.fps.grid(row=2, column=0, pady=2)
                    else:
                        self.fps.configure(text=f"FPS\n\n{fps:.2f}")

                else:
                    self.deepsort.increment_ages()
                self.root.after(self.delay, self.update)
                # Print time (inference + NMS)
                print("%sDone. (%.3fs)" % (s, t2 - t1))

        else:
            self.root.quit()
            print(
                "***********************************************FINSHED***********************************************"
            )

    def saveImage(self):
        # Get a frame from the video source
        grabbed, frame = self.vs.read()

        cv2.imwrite(
            "frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg",
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        )


if __name__ == "__main__":
    testWidget = Page()
    testWidget.root.mainloop()