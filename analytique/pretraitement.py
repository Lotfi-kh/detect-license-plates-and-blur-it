import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import imutils


fig, axes = plt.subplots(1, 3, figsize=(20, 10))


img = cv.imread("data/A2.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # from RGB to Grayscale


axes[0].imshow(
    cv.cvtColor(gray, cv.COLOR_BGR2RGB)
)  # n3awdo nktboha ldakhl prsq imshow expect 3 channel tsma haka rana n9olo is one channel
axes[0].set_title("gray picture ")


bfilter = cv.bilateralFilter(
    gray, 11, 17, 17
)  # bilateralFilter jay kima gauss bs7 maykhrbch edges , ydir blur ghir l same color , 3la 7sab value li ndiroh fel code


"""
how canny work :
noise reduction using gaussian
yl9a win pixel change rapidly , it use sobel
yrj3 edge thin , ydi max values fel edges et lba9i ydirhom 0
Hysteresis Thresholding : ch9wa nchr7ha
   
"""


edged = cv.Canny(
    bfilter, 150, 200
)  # canny , first value is min value of capable edge , the second value it mean if it is above it is for sure edge


axes[1].imshow(cv.cvtColor(edged, cv.COLOR_BGR2RGB))
axes[1].set_title("after bfilter and canny ")


points = cv.findContours(
    edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
)  # yl9a contours et zid y3rf child conteur (using RETR_TREE , Y3ni ida kayn conteur dakhl conteur )and he save the only corner points ta3 counteur using CHAIN_APPROX_SIMPLE


contours = imutils.grab_contours(
    points
)  # ydi counteurs brk , 3la 7sab version ta3 opencv prsq maymdoch ga3 same results
contours = sorted(contours, key=cv.contourArea, reverse=True)[
    :30
]  # yrtb contours men lkbir l sghir and it keep only the the 30 biggest


location = None
for contour in contours:

    approx = cv.approxPolyDP(contour, 20, True)
    if len(approx) == 4:
        location = approx
        break


mask = np.zeros(
    gray.shape, np.uint8
)  # nkhdmo array wla image same size ta3 gray bs7 ga3 pixels rahom 0


new_image = cv.drawContours(mask, [location], 0, 255, -1)
new_image = cv.bitwise_and(img, img, mask=mask)


axes[2].imshow(cv.cvtColor(new_image, cv.COLOR_BGR2RGB))


plt.show()
