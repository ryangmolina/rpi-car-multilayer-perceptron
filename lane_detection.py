import numpy as np
import cv2
import os


cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def lane_detection(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_size = (3, 3)
    gauss_gray = cv2.GaussianBlur(img_gray, ksize=kernel_size, sigmaX=0)
    canny_edge = cv2.Canny(np.uint8(gauss_gray), 50, 150, apertureSize=3)

    masked_edges = canny_edge

    lines = cv2.HoughLinesP(masked_edges, 2, np.pi / 180, 5, np.array([]))
    lines_img = np.zeros((masked_edges.shape[0], masked_edges.shape[1], 3), dtype=np.uint8)

    img = draw_line(img=lines_img, lines=lines)
    return img


def draw_line(img, lines, color=[0, 0, 255], thickness=10):
    # Calculae slopes and sizes of lanes
    lines = np.squeeze(lines)
    slopes = (lines[:, 3] - lines[:, 1]) / (lines[:, 2] - lines[:, 0])
    line_size = np.sqrt((lines[:, 2] - lines[:, 0]) ** 2 + (lines[:, 3] - lines[:, 1]) ** 2)

    # Get rid of outlier lines
    slope_threshold = 0.5
    lines = lines[np.abs(slopes) > slope_threshold]
    line_size = line_size[np.abs(slopes) > slope_threshold]
    slopes = slopes[np.abs(slopes) > slope_threshold]

    # Seperate positive and negative slopes, lines, and sizes
    left_slopes, right_slopes = slopes[slopes > 0], slopes[slopes < 0]
    left_lines, right_lines = lines[slopes > 0, :], lines[slopes < 0, :]
    left_lane_sizes, right_lane_sizes = line_size[slopes > 0], line_size[slopes < 0]

    # lanes sorted by size
    left_lane_sizes_sorted = np.argsort(left_lane_sizes)
    right_lane_sizes_sorted = np.argsort(right_lane_sizes)

    # Calculate average slope on the biggest 6 lines.
    left_slope_avg = left_slopes[left_lane_sizes_sorted][-6::].mean()
    right_slope_avg = right_slopes[right_lane_sizes_sorted][-6::].mean()

    # find y intercept
    # b = y - m * x
    left_x_values, left_y_values = np.concatenate([left_lines[:, 0], left_lines[:, 2]]), np.concatenate(
        [left_lines[:, 1], left_lines[:, 3]])
    right_x_values, right_y_values = np.concatenate([right_lines[:, 0], right_lines[:, 2]]), np.concatenate(
        [right_lines[:, 1], right_lines[:, 3]])

    left_y_intercept = left_y_values - (left_slope_avg * left_x_values)
    right_y_intercept = right_y_values - (right_slope_avg * right_x_values)

    # find mean y-intercept based on n biggest lines
    left_y_intercept = left_y_intercept[left_lane_sizes_sorted][-6::]
    right_y_intercept = right_y_intercept[right_lane_sizes_sorted][-6::]
    left_y_intercept_avg, right_y_intercept_avg = np.mean(left_y_intercept), np.mean(right_y_intercept)

    imshape = img.shape
    img_floor = imshape[0]
    horizon_line = imshape[0] / 1.5

    # calculate new points x = (y - b) / m

    # Right lane points
    max_right_x = (img_floor - right_y_intercept_avg) / right_slope_avg
    min_right_x = (horizon_line - right_y_intercept_avg) / right_slope_avg

    # Left lane points
    min_left_x = (img_floor - left_y_intercept_avg) / left_slope_avg
    max_left_x = (horizon_line - left_y_intercept_avg) / left_slope_avg



    l1 = (int(min_left_x), int(img_floor))
    l2 = (int(max_left_x), int(horizon_line))
    # print('Right points l1 and l2,', l1, l2)
    cv2.line(img, l1, l2, [0, 255, 0], 8)



    r1 = (int(max_right_x), int(img_floor))
    r2 = (int(min_right_x), int(horizon_line))
    # print('Left points l1 and l2,', l1, l2)
    cv2.line(img, r1, r2, [255, 0, 0], 8)
    return img


while (True):
    # Capture frame-by-frame
    img = cv2.imread('solidWhiteRight.jpg')
    img = cv2.resize(img, (320, 240))
    img = img[140:240, :]
    img = lane_detection(img)

    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




cap.release()
cv2.destroyAllWindows()
