import time
import pygame
import os
import numpy as np
import cv2
import socket




class KeyboardControl(object):
    def __init__(self):
        self.server_socket = socket.socket()
        self.server_socket.bind(('192.168.8.178', 8888))
        self.server_socket.listen(0)
        self.conn, self.addr = self.server_socket.accept()
        self.connection = self.conn.makefile('rb')

        self.k = np.zeros((4, 4), 'float')
        for i in range(4):
            self.k[i, i] = 1

        self.image_array = np.zeros((1, 38400), dtype=np.float32)
        self.label_array = np.zeros((1, 4), dtype=np.float32)


        self.isWaiting = True
        pygame.init()
        screen = pygame.display.set_mode([300, 300])
        pygame.display.set_caption("Keyboad Control")
        self.collect_image()

    def collect_image(self):
        print("Start collecting images...")
        try:
            stream_bytes = b' '
            pressed_up = False
            pressed_left = False
            pressed_right = False
            pressed_down = False
            while self.isWaiting:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    image = image[120:240, :]
                    cv2.imshow('image', image)
                    temp_array = image.reshape(1, 38400).astype(np.float32)
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            key_input = pygame.key.get_pressed()
                            if key_input[pygame.K_UP]:
                                pressed_up = True
                                self.conn.send("forward".encode("ascii"))
                            elif key_input[pygame.K_DOWN]:
                                self.conn.send("reverse".encode('ascii'))
                                pressed_down = True
                            elif key_input[pygame.K_RIGHT]:
                                self.conn.send("right".encode('ascii'))
                                self.save_frame(temp_array, 2)
                            elif key_input[pygame.K_LEFT]:
                                self.conn.send("left".encode('ascii'))
                                self.save_frame(temp_array, 3)
                            elif key_input[pygame.K_q]:
                                self.isWaiting = False
                                self.conn.send("stop".encode('ascii'))
                                self.conn.send("steer".encode('ascii'))
                                break
                        elif event.type == pygame.KEYUP:
                            self.conn.send("stop".encode('ascii'))
                            self.conn.send("steer".encode('ascii'))
                            if key_input[pygame.K_UP]:
                                pressed_up = False
                            elif key_input[pygame.K_DOWN]:
                                pressed_down = False
                            elif key_input[pygame.K_RIGHT]:
                                pressed_right = False
                            elif key_input[pygame.K_LEFT]:
                                pressed_left = False
                    if pressed_up:
                        print("Forward")
                        self.save_frame(temp_array, 0)
                    if pressed_right:
                        self.save_frame(temp_array, 2)
                        print("Right")
                    if pressed_left:
                        self.save_frame(temp_array, 3)
                        print("Left")
                    if pressed_down:
                        print("Down")
                        self.save_frame(temp_array, 1)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break


            train = self.image_array[1:, :]
            train_labels = self.label_array[1:, :]

            file_name = str(int(time.time()))
            directory = "training_data"

            if not os.path.exists(directory):
                os.makedirs(directory)
            try:
                np.savez(directory + '/' + file_name + '.npz', train=train, train_labels=train_labels)
            except IOError as e:
                print(e)
        finally:
            self.connection.close()
            self.server_socket.close()


    def save_frame(self, temp_array, label):
        self.image_array = np.vstack((self.image_array, temp_array))
        self.label_array = np.vstack((self.label_array, self.k[label]))


if __name__ == '__main__':
    KeyboardControl()
