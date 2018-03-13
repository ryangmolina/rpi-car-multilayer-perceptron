import numpy as np
import cv2
import socket
import warnings


class VideoStreamingTest(object):
    def __init__(self):
        address = '192.168.8.178'
        port = 8888
        self.server_socket = socket.socket()
        self.server_socket.bind((address, port))
        print("Binding to address: {} port: {} complete".format(address, port))
        self.server_socket.listen(0)
        print("Socket now listening")
        self.connection, self.client_address = self.server_socket.accept()
        print("Connection accepted")
        self.connection = self.connection.makefile('rb')
        self.streaming()

    def streaming(self):
        try:
            print("Connection from: ", self.client_address)
            print("Streaming...")
            print("Press 'q' to exit")
            stream_bytes = b' '
            while True:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), 1)
                    # image = self.cascade_classifier(image,  cv2.CascadeClassifier('trained_xml/left/cascade.xml'))
                    image = image[140:240, :]
                    image = self.lane_detection(image)
                    # image = find_blob(image)
                    cv2.imshow('image', image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            self.connection.close()
            self.server_socket.close()

    def lane_detection(self, img):

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel_size = (3, 3)
        gauss_gray = cv2.GaussianBlur(img_gray, ksize=kernel_size, sigmaX=0)
        canny_edge = cv2.Canny(np.uint8(gauss_gray), 50, 150, apertureSize=3)

        masked_edges = canny_edge

        lines = cv2.HoughLinesP(masked_edges, 2, np.pi / 180, 5, np.array([]))
        lines_img = np.zeros((masked_edges.shape[0], masked_edges.shape[1], 3), dtype=np.uint8)

        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if x1 > 160 or x2 > 160:
                        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    else:
                        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.line(img, (80, 0), (0, 240), (255, 255, 100), 2)
        cv2.line(img, (160, 0), (160, 240), (255, 255, 255), 3)
        cv2.line(img, (240, 0), (320, 240), (55, 255, 0), 2)

        return img

def cascade_classifier(self, frame, left_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    left = left_cascade.detectMultiScale(gray)

    for (x, y, w, h) in left:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame


def find_blob(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    # ih, iw = image.shape
    #
    # # Taking a matrix of size 5 as the kernel
    # kernel = np.ones((5, 5), np.uint8)
    #
    # image = cv2.dilate(image, kernel, iterations=1)
    # image = cv2.erode(image, kernel, iterations=1)


    detector = cv2.SimpleBlobDetector_create()

    # Detect blobs.
    keypoints = detector.detect(image)

    image = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # black = np.zeros((ih + 300, iw + 300), np.uint8)
    # cv2.imwrite('black.jpg', black)
    #
    # bh, bw = black.shape
    # pts_src = np.array([[0.0, 0.0], [float(iw), 0.0], [float(iw), float(ih)], [0.0, float(ih)]])
    # pts_dst = np.array([[0, 0], [float(bw), 0], [float(bw) * 0.65, float(bh)], [bw * 0.35, float(bh)]])
    #
    # h, status = cv2.findHomography(pts_src, pts_dst)
    #
    # im_out = cv2.warpPerspective(image, h, (black.shape[1], black.shape[0]))
    # return im_out
    return image


if __name__ == '__main__':
    VideoStreamingTest()
