import cv2
from ultralytics import YOLO as YOLO_
import time
import camera
import numpy as np

model = YOLO_(r"D:\yolov8\ultralytics-main\ultralytics-main\ultralytics\runs\detect\train12\weights\best.pt")

class turn_cam(camera.Camera):
    def __init__(self,  frame1, array0):
        super().__init__()
        self.frame = frame1
        self.array = array0
    def yolo_detect(self):
        results = model(self.frame, show=True)
        for result in results:
            # boxes = result.boxes  # Boxes object for bounding box outputs
            # masks = result.masks  # Masks object for segmentation masks outputs
            # keypoints = result.keypoints  # Keypoints object for pose outputs
            # probs = result.probs  # Probs object for classification outputs

            for r in result:
                if r.boxes is not None:
                    self.array.extend(r.boxes)
        print(len(self.array))
        if (len(self.array)) > 3:
            self.stop()
            print("Device 2 closed")
            time.sleep(2)
            self.restart()
            print("Device 2 opened")



                # result.show()  # display to screen
