import cv2
import numpy as np
from collections import Counter
from dataset import idd_lite
import dataset
MIN_BLOB_AREA = 10
# output = cv2.imread('output.tiff', 0)
# output = cv2.imread('output.tiff', 0)
# output_blob_matrix = output_cc[1]
# label = cv2.imread('label.jpg', 0)
# label_blob_matrix = output_cc[1]

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()


def calc_iou(output, label, threshold):
    count_blob = 0
    recall = 0
    precision = 0
    output = cv2.threshold(output, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    label_cc = cv2.connectedComponentsWithStats(label, connectivity=8)
    # imshow_components(output_cc[1]) #to comment out
    # imshow_components(label_cc[1]) #to comment out
    square_image=np.zeros(label.shape)
    for blob in label_cc[2][1:]: #first is always!! background
        square_image[blob[1]:blob[1]+blob[3], blob[0]:blob[0]+blob[2]] = 1

    cv2.imwrite('square_image.png', square_image)
    square_image = cv2.imread('square_image.png', 0)
    # imshow_components(square_image)
    output[output < 128] = 0
    output[output >= 128] = 1
    label[label < 128] = 0
    label[label >= 128] = 1
    square_cc = cv2.connectedComponentsWithStats(square_image, connectivity=8)
    # imshow_components(square_cc[1])
    iou_array=[] #array of iou for each blob. return average of this.
    for i in range(1,square_cc[0]):
        if square_cc[2][i][4] > MIN_BLOB_AREA:
            diff = output[square_cc[1] == i] - label[square_cc[1] == i] #since type is unsighned int, 255=-1
            diff = Counter(diff)
            iou = diff[0]/sum(diff.values())
            iou_array.append(iou)
            if iou >= threshold:
                count_blob += 1
    if len(iou_array) != 0:
        recall = count_blob / len(iou_array)

    # calc precision
    count_blobs = 0
    found_blobs = 0
    output_cc = cv2.connectedComponentsWithStats(output, connectivity=8)
    square_image = np.zeros(label.shape)
    for blob in output_cc[2][1:]:  # first is always!! background
        square_image[blob[1]:blob[1] + blob[3], blob[0]:blob[0] + blob[2]] = 1

    cv2.imwrite('square_image.png', square_image)
    square_image = cv2.imread('square_image.png', 0)
    square_cc = cv2.connectedComponentsWithStats(square_image, connectivity=8)
    for i in range(1, square_cc[0]):
        if square_cc[2][i][4] > MIN_BLOB_AREA:
            count_blobs += 1
            diff = output[square_cc[1] == i] - label[square_cc[1] == i]  # since type is unsighned int, 255=-1
            diff = Counter(diff)
            iou = diff[0] / sum(diff.values())
            if iou >= threshold:
                found_blobs += 1
    if count_blobs != 0:
        precision = found_blobs / count_blobs
    return np.average(iou_array), recall, precision #returns average iou, recall, precision(not yet)


# print(calc_iou(label, output))
# calc_iou(output, label, 0.3)


# print(label_cc[0]) #number of blobs+1 (with background)
# print(label_cc[1]) #label matrix for showing image
# print(label_cc[2]) #most left square, most top square, W, H, area
# print(label_cc[3]) #blob centroid
