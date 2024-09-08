import cv2
import threading

class Camera:
    def __init__(self, src=0):
        self.src = src
        self.reopen()
        self.start()
    def reopen(self):
        self.stream = cv2.VideoCapture(self.src)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.stream.set(3, 1920)
        self.stream.set(4, 1200)
        # ... 其他设置 ...

        if self.stream.isOpened():
            print("cam is ready")
        self.stopped = False

        self.thread = threading.Thread(target=self.update, args=())
        for _ in range(10):  # warm up the camera
            (self.grabbed, self.frame) = self.stream.read()

    def close(self):
        self.release()
        cv2.destroyAllWindows()

    def start(self):
        self.thread.daemon = True
        self.thread.start()

    def restart(self):
        self.release()  # 释放资源
        self.reopen()   # 重新打开相机
        self.start()    # 重新启动线程

    def update(self):
        while True:
            if self.stopped:
                return
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True

    def release(self):
        self.stop()
        if self.stream.isOpened():
            self.stream.release()


if __name__ == '__main__':
    cam0 = Camera(0)


    while True:
        ret, frame = cam0.read()
        if not ret:
            break
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam0.stop()
        if cv2.waitKey(1) & 0xFF == ord('z'):
            cam0.restart()
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break
    cam0.close()
    print("cam is closed")
