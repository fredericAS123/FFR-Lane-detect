import cv2
import time
import camera
import numpy as np

class Frame_For_Detection:
    def __init__(self,frame2):
        self.test_warp_image = None
        self.edges = None
        self.option = 0
        self.frame = frame2
        self.height, width, channels = frame2.shape
        self.lines = None
        self.left_lines = None
        self.right_lines = None
        self.cropped = None
        self.sum = 0


    def process_OTSU(self):

        """
        大津法（OTSU）分割阈值，大津法详细见learning basis 文件夹
        """
        gray = cv2.cvtColor(self.test_warp_image, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), sigmaX=1, sigmaY=0)
        # OTSU计算阈值
        ret1, binary_otsu = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 腐蚀+膨胀（开运算）
        kernel1 = np.ones((5, 5), np.uint8)
        combined_binary_1 = cv2.erode(binary_otsu, kernel1)
        kernel2 = np.ones((5, 5), np.uint8)
        combined_binary = cv2.dilate(combined_binary_1, kernel2)
        self.edges = cv2.Canny(combined_binary, 50, 150)
        cv2.imshow('edges', self.edges)



    def distort(self):
        """
        保持图片尺寸一致，提升检测和处理判断准确度
        """
        target_height = 720
        target_width = 1280
        if self.width != 1280 or self.height != 720:
            self.test_warp_image = cv2.resize(self.frame, (target_width, target_height))

        else:
            self.test_warp_image = self.frame

    @staticmethod
    def calculate_slope(line):
        """
        计算线段line的斜率
        :param line: np.array([[x_1, y_1, x_2, y_2]])
        :return:
        """
        x_1, y_1, x_2, y_2 = line[0]
        return (y_2 - y_1) / (x_2 - x_1)

    @staticmethod
    def calculate_point(point1, point2):
        x1, y1 = point1
        x2, y2 = point2

        # 计算斜率
        if x2 - x1 != 0:
            slope = (y2 - y1) / (x2 - x1)
            return slope
        else:
            # 处理斜率不存在的情况（垂直线），可以根据实际需求返回适当的值
            return float('inf')  # 无穷大表示垂直线

    def reject_abnormal_lines(self,lines, threshold):
        """
        剔除斜率不一致的线段
        :param threshold:
        :param lines: 线段集合, [np.array([[x_1, y_1, x_2, y_2]]),np.array([[x_1, y_1, x_2, y_2]]),...,np.array([[x_1, y_1, x_2, y_2]])]
        """
        slopes = [lines.calculate_slope(line) for line in lines]
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

    def find_start(self):
        USER_SIZE_H, USER_SIZE_W = self.edges.shape
        # 定义存储起点坐标的数组
        start_point_l = np.zeros(2, dtype=np.int_)
        start_point_r = np.zeros(2, dtype=np.int_)
        t = USER_SIZE_H - 20
        # 是否找到的标志
        l_found_0 = False
        r_found_0 = False

        def find_seed(bin_image, start_row, l_found, r_found):
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
                if bin_image[start_row][i] == 255 and bin_image[start_row][i - 1] == 0:
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
            result = find_seed(self.edges, t, l_found_0, r_found_0)
            if result:
                # 找到起点,退出循环
                # print("yes")
                # print(start_point_l)
                # print(start_point_r)
                print('t=', t)
                break
            elif l_found_0 == False and t == 0:
                return 0, 1, 0
            elif r_found_0 == False and t == 0:
                return 0, 0, 1
            elif l_found_0 == False and r_found_0 == False and t == 0:
                return 0, 0, 0
            else:
                # 未找到,行标记减1
                t -= 1
                # print(t)
        return t, start_point_l[0], start_point_r[0]

    def operate(self):

        cv2.imshow('cropped', self.cropped)
        # 霍夫直线检测
        lines = cv2.HoughLinesP(self.cropped, 1, np.pi / 180, threshold=50, minLineLength=13, maxLineGap=1)
        # print('lines:',lines)
        self.left_lines = [line for line in lines if self.calculate_slope(line) < 0]
        self.right_lines = [line for line in lines if self.calculate_slope(line) > 0]
        # print(left_lines)
        if len(self.left_lines) != 0 and len(self.right_lines) != 0:
            # 计算左右直线斜率
            # 按照斜率分成车道线

            self.left_lines = self.reject_abnormal_lines(self, self.left_lines, threshold=0.2)
            self.right_lines = self.reject_abnormal_lines(self, self.right_lines, threshold=0.2)
            # 得出两点唯一确定该直线
            left_line = self.least_squares_fit()
            right_line = self.least_squares_fit()

            # print("left lane")
            # print(least_squares_fit(left_lines))
            # print("right lane")
            # print(least_squares_fit(right_lines))
            # 验证这两个点
            line_image0 = np.zeros_like(self.test_warp_image)
            cv2.line(line_image0, tuple(left_line[0]), tuple(left_line[1]), color=(0, 255, 255), thickness=5)
            cv2.line(line_image0, tuple(right_line[0]), tuple(right_line[1]), color=(0, 255, 255), thickness=5)

            # cv2.imshow('lane', line_image0)
            # 根据上两点计算斜率
            print("left", self.calculate_point(tuple(left_line[0]), tuple(left_line[1])))
            print("right", self.calculate_point(tuple(right_line[0]), tuple(right_line[1])))
            print("左右直线斜率和为：")
            self.sum = self.calculate_point(tuple(left_line[0]),
                                       tuple(left_line[1])) + self.calculate_point(tuple(right_line[0]), tuple(right_line[1]))
            print("self.sum="+self.sum)
            # 创建一张空白图像，用于绘制直线
            line_image = np.zeros_like(self.test_warp_image)
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
        elif len(self.left_lines) == 0:
            print()
            self.sum = 20  # 20 找不到左边线
        elif len(self.right_lines) == 0:
            self.sum = 30  # 30 找不到右边线

        else:
            self.sum = 40  # 两边都找不到

    def decision(self):
        t, left_edge, right_edge = self.find_start()
        cropped = self.edges[(t - 100):t, 0:1280]

        if t != 0:

            self.operate()
            if self.sum == 40:
                print("路径模糊未通过霍夫检测，按原路线行驶！")
            elif self.sum == 20:
                print("找不到左边线,马头严重向右，马立即向左！")
            elif self.sum == 30:
                print("找不到右边线,马头严重向左，马立即向右！")
            elif abs(self.sum) < 0.5:
                print("马路径正常，继续直行！")
            elif self.sum < -0.5:
                print("马偏向左，向右！")
            elif self.sum > 0.5:
                print("马偏向右，向左！")  # 正常应该是0左右
        elif t == 0 and left_edge == 0 and right_edge != 0:
            line = cv2.HoughLinesP(cropped, 1, np.pi / 180, threshold=50, minLineLength=25,
                                   maxLineGap=1)  # 后续需要调参 保证只有一条线
            if line is not None:
                if 0 < self.calculate_slope(line) < 1:  # 需要后续调参确定
                    print("马严重向右，看不到左边线！向左行驶！")
                else:
                    print("进入拐角与直线赛道交界区域！")
            else:
                print("右边线存在，但没有被检测到！仍要向左！")
        elif t == 0 and right_edge == 0 and left_edge != 0:
            line = cv2.HoughLinesP(cropped, 1, np.pi / 180, threshold=50, minLineLength=5,
                                   maxLineGap=1)  # 后续需要调参 保证只有一条线
            if line is not None:
                if -1 < self.calculate_slope(line) < 0:  # 需要后续调参确定
                    print("马严重向左，看不到右边线！向右行驶")
                else:
                    print("进入拐角与直线赛道交界区域！")
            else:
                print("左边线存在，但没有被检测到！仍要向右！")
        else:
            print("左右边线都看不到，进入十字交界区域！")
