import cv2
import numpy as np

# output = cv2.imread('output.tiff', 0)
output = cv2.imread('00564.jpg', 0)
output = cv2.threshold(output, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
output_cc = cv2.connectedComponentsWithStats(output,connectivity=8)

label = cv2.imread('label.tiff', 0)
label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
label_cc = cv2.connectedComponentsWithStats(label,connectivity=8)


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

imshow_components(output_cc[1])
imshow_components(label_cc[1])

# for blob in range(1,output_cc[0]): #first is always!! background
# TODO for each blob calc iou here



print(output_cc[0]) #number of blobs+1 (with background)
print(output_cc[1]) #label matrix for showing image
print(output_cc[2]) #most left square, most top square, W/H, H/W, area
print(output_cc[3]) #blob centroid
