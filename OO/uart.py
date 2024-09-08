import serial
import threading
import time


class MyUart:
    def __init__(self, baudrate, port="/dev/ttyUSB0"):
        self.ser = serial.Serial(port, baudrate)
        self.read_thread = threading.Thread(target=self.read_thread)
        self.read_thread.start()
        self.msg = "none"

    def write(self, data):
        self.ser.write(data.encode("utf-8"))
        # time.sleep(0.1)

    def read_thread(self):
        while True:
            time.sleep(0.002)

            num = self.ser.inWaiting()
            if num:
                self.msg = self.ser.read(num)
            else:
                self.msg = "none"

    def close(self):
        self.ser.close()

    def uart_send_data(self, data):
        self.ser.write(data.encode("utf-8"))
        time.sleep(0.1)


if __name__ == '__main__':
    uart = MyUart(115200, "COM7")
    while True:
        uart.write("hello")
        print(uart.msg)
        time.sleep(1)
    uart.close()


    print("uart is closed")
