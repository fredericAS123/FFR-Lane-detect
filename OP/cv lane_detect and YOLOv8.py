import cv2

from ultralytics import YOLO as YOLO_
import time
import numpy as np

array = []
option=0
def hlsSSelect(img, thresh=(125, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    s_channel = s_channel*(255/np.max(s_channel))
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 255
    return binary_output
def labBSelect(img, thresh=(200, 250)):
    # 1) Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    lab_b = lab[:, :, 2]
    # don't normalize if there are no yellows in the image
    if np.max(lab_b) > 100:
        lab_b = lab_b*(255/np.max(lab_b))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 255
    # 3) Return a binary image of threshold result
    return binary_output

def process(img):
    hlsL_binary = hlsSSelect(img)
    labB_binary = labBSelect(img)
    combined_binary_0 = np.zeros_like(hlsL_binary, dtype=np.uint8)
    combined_binary_0[(hlsL_binary == 255) | (labB_binary == 255)] = 255
    gray_blur = cv2.GaussianBlur(combined_binary_0, (5, 5), sigmaX=1, sigmaY=0)
    kernel1 = np.ones((5, 5), np.uint8)
    combined_binary_1 = cv2.erode(gray_blur, kernel1)
    kernel2 = np.ones((5, 5), np.uint8)
    combined_binary = cv2.dilate(combined_binary_1, kernel2)
    # 显示
    edges = cv2.Canny(combined_binary, 50, 150)
    cv2.imshow('img_0', edges)
    return edges
def process_OTSU(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), sigmaX=1, sigmaY=0)
    # OTSU计算阈值
    ret1, binary_otsu = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 腐蚀+膨胀（开运算）
    kernel1 = np.ones((5, 5), np.uint8)
    combined_binary_1 = cv2.erode(binary_otsu, kernel1)
    kernel2 = np.ones((5, 5), np.uint8)
    combined_binary = cv2.dilate(combined_binary_1, kernel2)
    edges = cv2.Canny(combined_binary, 50, 150)
    cv2.imshow('edges',edges)
    return edges
def calculate_slope(line):
    """
    计算线段line的斜率
    :param line: np.array([[x_1, y_1, x_2, y_2]])
    :return:
    """
    x_1, y_1, x_2, y_2 = line[0]
    return (y_2 - y_1) / (x_2 - x_1)

def calculate_point(point1,point2):
    x1, y1 = point1
    x2, y2 = point2

    # 计算斜率
    if x2 - x1 != 0:
        slope = (y2 - y1) / (x2 - x1)
        return slope
    else:
        # 处理斜率不存在的情况（垂直线），可以根据实际需求返回适当的值
        return float('inf')  # 无穷大表示垂直线
def reject_abnormal_lines(lines, threshold):
    """
    剔除斜率不一致的线段
    :param lines: 线段集合, [np.array([[x_1, y_1, x_2, y_2]]),np.array([[x_1, y_1, x_2, y_2]]),...,np.array([[x_1, y_1, x_2, y_2]])]
    """
    slopes = [calculate_slope(line) for line in lines]
    while len(lines) > 0:
        mean = np.mean(slopes)
        diff = [abs(s - mean) for s in slopes]
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            slopes.pop(idx)
            lines.pop(idx)
        else:
            break
    return lines

def least_squares_fit(lines):
    """
    将lines中的线段拟合成一条线段
    :param lines: 线段集合, [np.array([[x_1, y_1, x_2, y_2]]),np.array([[x_1, y_1, x_2, y_2]]),...,np.array([[x_1, y_1, x_2, y_2]])]
    :return: 线段上的两点,np.array([[xmin, ymin], [xmax, ymax]])
    """
    # 1. 取出所有坐标点
    x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
    y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
    # 2. 进行直线拟合.得到多项式系数
    poly = np.polyfit(x_coords, y_coords, deg=1)
    # 3. 根据多项式系数,计算两个直线上的点,用于唯一确定这条直线
    point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
    point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))
    return np.array([point_min, point_max], dtype=np.int_)

def find_start(image_array):
    USER_SIZE_H, USER_SIZE_W = image_array.shape
    # 定义存储起点坐标的数组
    start_point_l = np.zeros(2, dtype=np.int_)
    start_point_r = np.zeros(2, dtype=np.int_)
    t=USER_SIZE_H-20
    # 是否找到的标志
    l_found_0=False
    r_found_0=False
    def find_seed(bin_image, start_row,l_found,r_found):
        # # 是否找到的标志
        # l_found = False
        # r_found = False
        # 从中间开始遍历
        mid = int(len(bin_image[0]) / 2)
        for i in range(mid, 0, -1):
            # 判断当前和左边一点
            if bin_image[start_row][i] == 0 and bin_image[start_row][i - 1] == 255:
                # 记录左边起点
                start_point_l[0] = i
                start_point_l[1] = start_row
                l_found = True
                break

        for i in range(mid, len(bin_image[0])):
            # 判断当前和右边一点
            if bin_image[start_row][i] == 255 and bin_image[start_row][i-1] == 0:
                # 记录右起点
                start_point_r[0] = i
                start_point_r[1] = start_row
                r_found = True
                break

        # 返回是否都找到了
        if l_found and r_found:
            return True
        else:
            return False

    while True:
        result = find_seed(image_array, t,l_found_0,r_found_0)
        if result:
            # 找到起点,退出循环
            # print("yes")
            # print(start_point_l)
            # print(start_point_r)
            print('t=',t)
            break
        elif l_found_0==False and t==0:
            return 0,1,0
            print('t=', 0)
            break
        elif r_found_0==False and t==0:
            print('t=', 0)
            return 0,0,1
            break
        elif l_found_0==False and r_found_0==False and t==0:
            return 0,0,0
            print('t=', 0)
            break
        else:
            # 未找到,行标记减1
            t -= 1
            # print(t)
    return t,start_point_l[0],start_point_r[0]

def operate(cropped):

    cv2.imshow('cropped', cropped)
    # 霍夫直线检测
    lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, threshold=50, minLineLength=13, maxLineGap=1)
    # print('lines:',lines)
    left_lines = [line for line in lines if calculate_slope(line) < 0]
    right_lines = [line for line in lines if calculate_slope(line) > 0]
    # print(left_lines)
    if len(left_lines)!=0 and len(right_lines)!=0:
        # 计算左右直线斜率
        # 按照斜率分成车道线

        left_lines = reject_abnormal_lines(left_lines, threshold=0.2)
        right_lines = reject_abnormal_lines(right_lines, threshold=0.2)
        # 得出两点唯一确定该直线
        left_line = least_squares_fit(left_lines)
        right_line = least_squares_fit(right_lines)

        # print("left lane")
        # print(least_squares_fit(left_lines))
        # print("right lane")
        # print(least_squares_fit(right_lines))
        # 验证这两个点
        line_image0 = np.zeros_like(test_warp_image)
        cv2.line(line_image0, tuple(left_line[0]), tuple(left_line[1]), color=(0, 255, 255), thickness=5)
        cv2.line(line_image0, tuple(right_line[0]), tuple(right_line[1]), color=(0, 255, 255), thickness=5)

        # cv2.imshow('lane', line_image0)
        # 根据上两点计算斜率
        print("left",calculate_point(tuple(left_line[0]),tuple(left_line[1])))
        print("right",calculate_point(tuple(right_line[0]), tuple(right_line[1])))
        print("左右直线斜率和为：")
        sum = calculate_point(tuple(left_line[0]),
                              tuple(left_line[1])) + calculate_point(tuple(right_line[0]), tuple(right_line[1]))
        print(sum)
        # 创建一张空白图像，用于绘制直线
        line_image = np.zeros_like(test_warp_image)
        # 绘制检测到的直线
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

        # # 将直线图与原图叠加
        # result = cv2.addWeighted(test_warp_image, 1, line_image, 1, 0)
        #
        # # 显示结果
        # cv2.imshow('Lane Detection Result', result)
        return sum
    elif len(left_lines)==0:
        print()
        return 20  # 20 找不到左边线
    elif len(right_lines)==0:
        return 30  # 30 找不到右边线

    else :
        return 40   # 两边都找不到


def decision(edg_img):
    t, left_edge, right_edge = find_start(edg_img)
    cropped = edg_img[(t - 100):t, 0:1280]

    if t != 0:

        sum = operate(cropped)
        if sum == 40:
            print("路径模糊未通过霍夫检测，按原路线行驶！")
        elif sum == 20:
            print("找不到左边线,马头严重向右，马立即向左！")
        elif sum == 30:
            print("找不到右边线,马头严重向左，马立即向右！")
        elif abs(sum) < 0.5:
            print("马路径正常，继续直行！")
        elif sum < -0.5:
            print("马偏向左，向右！")
        elif sum > 0.5:
            print(sum)
            print("马偏向右，向左！")  # 正常应该是0左右
    elif t == 0 and left_edge == 0 and right_edge != 0:
        line = cv2.HoughLinesP(cropped, 1, np.pi / 180, threshold=50, minLineLength=25, maxLineGap=1)  # 后续需要调参 保证只有一条线
        if line is not None:
            if 0 < calculate_slope(line) < 1:  # 需要后续调参确定
                print("马严重向右，看不到左边线！向左行驶！")
            else:
                print("进入拐角与直线赛道交界区域！")
        else:
            print("右边线存在，但没有被检测到！仍要向左！")
    elif t == 0 and right_edge == 0 and left_edge != 0:
        line = cv2.HoughLinesP(cropped, 1, np.pi / 180, threshold=50, minLineLength=5, maxLineGap=1)  # 后续需要调参 保证只有一条线
        if line is not None:
            if -1 < calculate_slope(line) < 0:  # 需要后续调参确定
                print("马严重向左，看不到右边线！向右行驶")
            else:
                print("进入拐角与直线赛道交界区域！")
        else:
            print("左边线存在，但没有被检测到！仍要向右！")
    else:
        print("左右边线都看不到，进入十字交界区域！")

# def YOLO(cap1,cap2,frame1):
#
#     model = YOLO_(r"D:\yolov8\ultralytics-main\ultralytics-main\ultralytics\runs\detect\train12\weights\best.pt")
# # while True:
#
#     results = model(frame1, show=True)
#     for result in results:
#         # boxes = result.boxes  # Boxes object for bounding box outputs
#         # masks = result.masks  # Masks object for segmentation masks outputs
#         # keypoints = result.keypoints  # Keypoints object for pose outputs
#         # probs = result.probs  # Probs object for classification outputs
#
#         for r in result:
#             if r.boxes is not None:
#                 array.extend(r.boxes)
#     print(len(array))
#     if (len(array)) > 3:
#         cap2.release()
#         print("Device 2 closed")
#         time.sleep(2)
#         cap2 = cv2.VideoCapture(0)
#         print("Device 2 opened")
#
#
#
#             # result.show()  # display to screen

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




if __name__ == "__main__":
    cap1=cv2.VideoCapture(2)
    cap2=cv2.VideoCapture(1)
    while True:

    # 读取图片
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        # cv2.imshow('Camera1',frame1)
        cv2.imshow('Camera2', frame2)
        cv2.imshow('Camera1',frame1)
        test_distort_image = frame2
        height, width, channels = test_distort_image.shape
        # print(f"图片的尺寸为：宽度={width}，高度={height}，通道数={channels}")
        # 设置目标尺寸
        target_height = 720
        target_width = 1280
        # 调整图片尺寸
        if width != 1280 or height != 720:
            test_warp_image = cv2.resize(test_distort_image, (target_width, target_height))

        else:
            test_warp_image=test_distort_image
        edges=process_OTSU(test_warp_image)

        try:
            decision(edges)
            # YOLO(cap1, cap2, frame1)

        except:
            pass
        # inflection_decision(frame1)
        # if len(array) >= 30:
        #     option=option+1
        #
        #     if option==2 or option==3 :
        #         print("检测到拐角！向右拐")
        #         array=[]
        #         cap2.release()
        #         print("Device 2 closed")
        #         time.sleep(10)
        #         cap2 = cv2.VideoCapture(1)
        #         print("Device 2 opened")
        #
        #
        #     elif option==6:
        #         print("检测到拐角！向左拐")
        #         array = []
        #         cap2.release()
        #         print("Device 2 closed")
        #         time.sleep(10)
        #         cap2 = cv2.VideoCapture(1)
        #         print("Device 2 opened")
        #     elif option==7:
        #         print("到达终点，停止运行！")
        #         cap1.release()
        #         cap2.release()
        #     elif option==1 or option==4 or option==5:
        #         print("************************* time*******************************************************")
        #         array=[]

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap1.release()
    cap2.release()

