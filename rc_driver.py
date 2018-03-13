import socket
import threading
import socketserver
import serial
import cv2
import numpy as np
import math

# distance data measured by ultrasonic sensor
sensor_data = " "


class NeuralNetwork(object):

    def __init__(self):
        self.model = cv2.ml.ANN_MLP_load('mlp_xml/mlp.xml')

    def predict(self, samples):
        ret, resp = self.model.predict(samples)
        print(resp)
        return resp.argmax(-1)



class Driver(object):
    def __init__(self):
        self.server_socket = socket.socket()
        self.server_socket.bind(('192.168.0.100', 9999))
        self.server_socket.listen(0)
        self.conn, self.addr = self.server_socket.accept()
        self.connection = self.conn.makefile('rb')


        # create neural network
        self.model = NeuralNetwork()


        self.collect_image()

    def collect_image(self):



        try:
            stream_bytes = b' '
            while True:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    # roi - select lower half of the image
                    roi = image[120:240, :]

                    cv2.imshow('image', roi)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    temp_array = roi.reshape(1, 38400).astype(np.float32)

                    # neural network makes prediction
                    prediction = self.model.predict(temp_array)
                    print(prediction)
                    if prediction == 0:
                        self.conn.send("forward".encode('ascii'))
                        print("Forward")
                    elif prediction == 3:
                        self.conn.send("left".encode('ascii'))
                        print("Left")
                    elif prediction == 2:
                        self.conn.send("right".encode('ascii'))
                        print("Right")
                    elif prediction == 1:
                        self.conn.send("reverse".encode('ascii'))
                    else:
                        self.conn.send("stop".encode('ascii'))

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.conn.send("4".encode('ascii'))
                        break

        finally:
            self.connection.close()
            self.server_socket.close()



if __name__ == '__main__':
    Driver()
