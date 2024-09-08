import serial
import threading
import time
from restart import *
f = open('run.txt','a+')
class MyUart:
    def __init__(self, baudrate, port="/dev/ttyUSB0"):
        self.baud = baudrate
        self.port = port
        self.ser = serial.Serial(port, baudrate)
        # while self.ser.is_open() != True:
        
        #     self.ser = serial.Serial(port, baudrate)
        
        self.read_thread = threading.Thread(target=self.read_thread)
        self.read_thread.start()
        self.mode = 4

    def write(self, data):
        if self.ser.isOpen():
            try:
                self.ser.write(data.encode("utf-8"))
                #f.write(f"{time.time()}+{data}\n")
            except:
                time.sleep(5)
                self.ser.close()
                time.sleep(1)
                self.ser = serial.Serial(self.port, self.baud)
        else:
            self.ser = serial.Serial(self.port, self.baud)
        # time.sleep(0.1)

    def read_thread(self):
        while True:
            rx_buf = self.ser.read(1).hex()
            if rx_buf:
                if int(rx_buf) != 9:
                    self.mode = int(rx_buf)
                    print("uart",rx_buf)
                else:
                    restart()
        # while True:
        #     time.sleep(0.02)
        #     if self.ser.inWaiting() >= 1:
        #         rx_buf = self.ser.read(1).hex()
        #         self.mode = int(rx_buf)
        #         print("uart rx",self.mode)
                
    def close(self):
        self.ser.close()

    def uart_send_data(self, data):
        self.ser.write(data.encode("utf-8"))
        time.sleep(0.1)


if __name__ == '__main__':
    uart = MyUart(115200, "/dev/ttyUSB0")
    while True:
        #uart.write("hello")
        #print(uart.mode)
        pass
        #time.sleep(1)
    uart.close()
    print("uart is closed")
