import cv2
import glob
import numpy as np


# Step 1 读入图片、预处理图片、检测交点、标定相机的一系列操作
def getCameraCalibrationCoefficients(chessboardname, nx, ny):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob(chessboardname)
    if len(images) > 0:
        print("images num for calibration : ", len(images))
    else:
        print("No image for calibration.")
        return

    ret_count = 0
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = (img.shape[1], img.shape[0])
        # Finde the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            ret_count += 1
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    print('Do calibration successfully')
    return ret, mtx, dist, rvecs, tvecs


# Step 2 传入计算得到的畸变参数，即可将畸变的图像进行畸变修正处理
def undistortImage(distortImage, mtx, dist):
    return cv2.undistort(distortImage, mtx, dist, None, mtx)


# Step 3 透视变换 : Warp image based on src_points and dst_points
# The type of src_points & dst_points should be like
# np.float32([ [0,0], [100,200], [200, 300], [300,400]])
def warpImage(image, src_points, dst_points):
    image_size = (image.shape[1], image.shape[0])
    # rows = img.shape[0] 720
    # cols = img.shape[1] 1280
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    warped_image = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)

    return warped_image, M, Minv


if __name__ == "__main__":
    nx = 9
    ny = 6

    # Step 1 获取畸变参数
    rets, mtx, dist, rvecs, tvecs = getCameraCalibrationCoefficients('camera_cal/calibration*.jpg', nx, ny)

    # 读取图片
    test_distort_image = cv2.imread('test_img/unity1.png')
    height, width, channels = test_distort_image.shape
    print(f"图片的尺寸为：宽度={width}，高度={height}，通道数={channels}")
    # Step 2 畸变修正
    test_undistort_image = undistortImage(test_distort_image, mtx, dist)

    # Step 3 透视变换
    # “不断调整src和dst的值，确保在直线道路上，能够调试出满意的透视变换图像”
    # 左图梯形区域的四个端点
    src = np.float32([[580, 460], [700, 460], [1096, 720], [200, 720]])
    # 右图矩形区域的四个端点
    dst = np.float32([[300, 0], [950, 0], [950, 720], [300, 720]])

    # 变换
    test_warp_image, M, Minv = warpImage(test_undistort_image, src, dst)

    # 显示
    cv2.imshow('img', test_warp_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
