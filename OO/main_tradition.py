import cv2
import time
import camera
import lane
import yolo
import numpy as np
from ultralytics import YOLO as YOLO_

model = YOLO_(r"D:\yolov8\ultralytics-main\ultralytics-main\ultralytics\runs\detect\train12\weights\best.pt")

array = []


def inflection_detect(test_distort_image):
    height, width, channels = test_distort_image.shape
    # print(f"图片的尺寸为：宽度={width}，高度={height}，通道数={channels}")
    # 设置目标尺寸
    target_height = 720
    target_width = 1280
    # 调整图片尺寸
    if width != 1280 or height != 720:
        test_warp_image = cv2.resize(test_distort_image, (target_width, target_height))

    else:
        test_warp_image = test_distort_image
    test_warp_image = cv2.bilateralFilter(test_warp_image, d=5, sigmaColor=40, sigmaSpace=50)
    # 将图像从BGR转换到HSV
    hsv_img = cv2.cvtColor(test_warp_image, cv2.COLOR_BGR2HSV)
    # 定义红色的HSV范围
    # 在HSV中，红色通常在0到10和170到180度之间
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv_img, lower_red, upper_red)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_img, lower_red, upper_red)
    # 合并两个掩码
    red_mask_0 = cv2.add(mask1, mask2)
    kernel1 = np.ones((5, 5), np.uint8)
    red_mask = cv2.erode(red_mask_0, kernel1)
    if not np.all(red_mask == 0):
        # 显示图像
        # cv2.imshow('Red regions', red_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return 1
    else :
        return 0

def inflection_decision(frame1):
    # 读取图像
    test_distort_image = frame1

    decision=inflection_detect(test_distort_image)
    if decision == 1:
        array.append(decision)
    # result.show()  # display to screen


def main():
    cam1 = camera.Camera(0)
    cam2 = camera.Camera(1)
    while True:
        frame1 = cam1.frame
        frame2 = cam2.frame
        cv2.imshow('1', frame1)
        cv2.imshow('2', frame2)
        frame = lane.Frame_For_Detection(frame2)
        frame.distort()
        try:
            frame.decision()
            len = inflection_detect(frame1)
            if len == 1:
                cam2.release()
                print("Device 2 closed")
                time.sleep(2)
                cam2.restart()
                print("Device 2 opened")
            elif len == 0:
                print("no arrival!")

        except:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()