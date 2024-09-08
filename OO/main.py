import cv2
import time
import camera
import lane
import yolo
import numpy as np
from ultralytics import YOLO as YOLO_

model = YOLO_(r"D:\yolov8\ultralytics-main\ultralytics-main\ultralytics\runs\detect\train12\weights\best.pt")

array = []
def YOLO(frame1):

    model = YOLO_(r"D:\yolov8\ultralytics-main\ultralytics-main\ultralytics\runs\detect\train12\weights\best.pt")
# while True:

    results = model(frame1, show=True)
    for result in results:
        # boxes = result.boxes  # Boxes object for bounding box outputs
        # masks = result.masks  # Masks object for segmentation masks outputs
        # keypoints = result.keypoints  # Keypoints object for pose outputs
        # probs = result.probs  # Probs object for classification outputs

        for r in result:
            if r.boxes is not None:
                array.extend(r.boxes)
    return len(array)




            # result.show()  # display to screen
def main():
    cam1 = camera.Camera(0)
    cam2 = camera.Camera(1)
    while True:
        frame1 = cam1.frame
        frame2 = cam2.frame
        cv2.imshow('1',frame1)
        cv2.imshow('2',frame2)
        frame = lane.Frame_For_Detection(frame2)
        frame.distort()
        try:
             frame.decision()
             len = YOLO(frame1)
             if len > 3:
                 cam2.release()
                 print("Device 2 closed")
                 time.sleep(2)
                 cam2.restart()
                 print("Device 2 opened")
             else:
                 print("no arrival!")

        except:
             pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()