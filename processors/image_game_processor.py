from copy import deepcopy
from math import sqrt
from statistics import mean, stdev

import cv2
import numpy as np
from PIL import Image

from datastructures import barbecue as ba


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def get_image(path):
    """
    get image ndarray from image
    :param path: path to image
    :return: ndarray image
    """
    image = cv2.imread(path)

    if image is None:
        print("Image not found!")
        exit(1)

    return rescale_frame(image)


def custom_threshold(image):
    """
    custom threshold for input image
    :param image: rgb image
    :return: binary image
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 1)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 55, 5)
    return thresh


def distance(p1, p2):
    return sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def nearest_point(src, dst):
    dist = float('inf')
    nearest = src[0]
    for p in src:
        tmp = distance(p, dst)
        if tmp < dist:
            nearest = p
            dist = tmp
    return nearest


def closest_pair(lp1, lp2):
    dist = float('inf')
    nearest = 0
    for i in range(len(lp1)):
        tmp = distance(lp1[i], lp2[i])
        if tmp < dist:
            dist = tmp
            nearest = i
    return nearest


def rotating_index(i, i_min, i_max):
    if i > i_max:
        return rotating_index(i - (i_max + 1), i_min, i_max)
    if i < i_min:
        return rotating_index(i_max + (i + 1), i_min, i_max)
    return i


def form_square_perspective(i_anchor, old_perspective):
    new_perspective = deepcopy(old_perspective)
    if i_anchor % 2 == 0:
        new_perspective[rotating_index(i_anchor + 1, 0, 3)][0] = old_perspective[i_anchor][0]
        new_perspective[rotating_index(i_anchor - 1, 0, 3)][1] = old_perspective[i_anchor][1]
        new_perspective[rotating_index(i_anchor + 2, 0, 3)] = [
            new_perspective[rotating_index(i_anchor - 1, 0, 3)][0],
            new_perspective[rotating_index(i_anchor + 1, 0, 3)][1]
        ]
    else:
        new_perspective[rotating_index(i_anchor + 1, 0, 3)][1] = old_perspective[i_anchor][1]
        new_perspective[rotating_index(i_anchor - 1, 0, 3)][0] = old_perspective[i_anchor][0]
        new_perspective[rotating_index(i_anchor + 2, 0, 3)] = [
            new_perspective[rotating_index(i_anchor + 1, 0, 3)][0],
            new_perspective[rotating_index(i_anchor - 1, 0, 3)][1]
        ]
    return new_perspective


def fit_grid_perspective(binary, cnt):
    cnt_points = [x[0] for x in cnt.tolist()]

    image_edges = [[0, 0], [0, binary.shape[1]], [binary.shape[0], binary.shape[1]], [binary.shape[0], 0]]

    old_grid_edges = [
        nearest_point(cnt_points, image_edges[0]),
        nearest_point(cnt_points, image_edges[1]),
        nearest_point(cnt_points, image_edges[2]),
        nearest_point(cnt_points, image_edges[3])
    ]

    anchor_point = closest_pair(
        old_grid_edges,
        image_edges
    )

    new_grid_edges = form_square_perspective(anchor_point, old_grid_edges)

    w = max([x[0] for x in new_grid_edges])
    h = max([x[1] for x in new_grid_edges])

    if w > binary.shape[0]:
        binary = np.pad(binary, ((0, w - binary.shape[0]), (0, 0)), mode='constant', constant_values=0)
    if h > binary.shape[1]:
        binary = np.pad(binary, ((0, 0), (0, h - binary.shape[1])), mode='constant', constant_values=0)

    pts_from = np.float32(old_grid_edges)
    pts_to = np.float32(new_grid_edges)
    matrix = cv2.getPerspectiveTransform(pts_from, pts_to)
    result = cv2.warpPerspective(binary, matrix, (binary.shape[1], binary.shape[0]))

    return result


def max_contour(cnts):
    max_area = 0
    best_cnt = cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area > max_area:
            best_cnt = c
            max_area = area
    return best_cnt


def crop_fit_binary(binary):
    min_x, min_y, max_x, max_y = 0, 0, binary.shape[1], binary.shape[0]

    for column in range(binary.shape[1]):
        if any(binary[:, column]):
            min_x = column

    for column in range(binary.shape[1] - 1, -1, -1):
        if any(binary[:, column]):
            max_x = column

    for row in range(binary.shape[0]):
        if any(binary[row]):
            min_y = row

    for row in range(binary.shape[0] - 1, -1, -1):
        if any(binary[row]):
            max_y = row

    return max_x, max_y, min_x - max_x, min_y - max_y


def filter_play_field(binary):
    """
    get grid in binary image and format grid perspective
    :param binary: image containing grid
    :return: list of ndarrays where first element is binary
    image with grid only and second element is input image
    with same perspective
    """

    uncleaned = binary.copy()
    cnts = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    mean_area = mean([cv2.contourArea(c) for c in cnts])
    stdev_area = stdev([cv2.contourArea(c) for c in cnts])

    # wipe small elements
    for c in cnts:
        if cv2.contourArea(c) < mean_area + stdev_area:
            cv2.drawContours(binary, [c], -1, (0, 0, 0), -1)

    best_cnt = max_contour(cnts)

    binary = fit_grid_perspective(binary, best_cnt)
    uncleaned = fit_grid_perspective(uncleaned, best_cnt)

    # x, y, w, h = cv2.boundingRect(best_cnt)
    # binary = binary[y:y + h, x:x + w]
    # uncleaned = uncleaned[y:y + h, x:x + w]

    x, y, w, h = crop_fit_binary(binary)
    binary = binary[y:y + h, x:x + w]
    uncleaned = uncleaned[y:y + h, x:x + w]

    return [binary, uncleaned]


def get_cells(binary, area_alpha):
    """
    Get cells from binary image, ignoring small holes.
    :param binary: ndarray containing grid only.
    :param area_alpha: coefficient for recognizing holes as cells.
    If mean_hole_area - mean_hole_area * area_alpha < hole_area < mean_hole_area + mean_hole_area * area_alpha,
    then hole is recognized as cell.
    :return: list of contours of cells
    """
    cells = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cells = cells[0] if len(cells) == 2 else cells[1]

    mean_area_cells = mean([cv2.contourArea(c) for c in cells])
    delta = mean_area_cells * area_alpha
    filtered_cells = []

    for c in cells:
        if mean_area_cells - delta < cv2.contourArea(c) < mean_area_cells + delta:
            filtered_cells.append(c)

    return filtered_cells


def order_cells(cells, delta):
    sorted_row = []

    grill = ba.Grill(delta)
    for c in cells:
        obj = ba.Shape(c)
        grill.put(obj)

    matrix = grill.get_lunch()

    for row in matrix:
        for obj in row:
            sorted_row.append(obj.shape)

    return sorted_row


# i = 128
def get_sign(binary_image, contour):
    # global i
    mask = np.zeros((binary_image.shape[0], binary_image.shape[1]), np.uint8)
    masked = np.zeros((binary_image.shape[0], binary_image.shape[1]), np.uint8)

    cv2.drawContours(mask, [contour], -1, 255, -1)

    idx = (mask != 0)
    masked[idx] = binary_image[idx]
    cv2.drawContours(masked, [contour], 0, 0, 2)

    x, y, w, h = cv2.boundingRect(contour)
    masked = masked[y:y + h, x:x + w]

    masked = cv2.resize(masked, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)

    # img = Image.fromarray(masked)
    # img.save('C:/Users/Den/PycharmProjects/sokoban-image-processing/images/samples/training-symbols/filled-squares/{}.jpg'.format(str(i)))
    # i += 1

    return masked
