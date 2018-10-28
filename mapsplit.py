#coding:utf-8
import cv2
import numpy as np


def cutBlank(img):
    # img = Image.fromarray(binaryInv)
    # img.save('cut.png')
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    t, binaryInv = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    hVec = np.sum(binaryInv, axis=1)
    wVec = np.sum(binaryInv, axis=0)
    hstart, hend = None, None
    wstart, wend = None, None
    h = len(hVec)
    w = len(wVec)
    # print 'cut h,w:',h,w
    for i in range(h):
        if not hstart and hVec[i] != 0:
            hstart = i
        if not hend and hVec[h - i - 1]:
            hend = h - i - 1
        if hstart and hend:
            break
    for i in range(w):
        if not wstart and wVec[i] != 0:
            wstart = i
        if not wend and wVec[w - i - 1]:
            wend = w - i - 1
        if wstart and wend:
            break
    if hstart == None or hend == None or wstart == None or wend == None:
        return None
    elif hstart > 0.5 * h or hend < 0.5 * h:
        hstart = 0
        hend = h

    cut = img[hstart:hend + 1, wstart:wend + 1]
    return cut

def kmeans(img,K):
    h,w = img.shape[:2]
    imgList = []
    if len(img.shape) == 3:
        Z = img.reshape((-1, 3))
    else:
        Z = img.reshape((-1, 1))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    countList = []
    imgBackupList = []
    targetImg = None
    maxPixelCount = 0
    for i in range(K):
        centerCopy = [(0,0,0)] * K
        centerCopy[i] = center[i]
        centerCopy = np.array(centerCopy,dtype=np.uint8)
        res = centerCopy[label.flatten()]
        res2 = res.reshape((img.shape))
        pixelCount = np.sum(label==i)
        if pixelCount > maxPixelCount:
            maxPixelCount = pixelCount
            targetImg = res2

    targetGray = cv2.cvtColor(targetImg,cv2.COLOR_BGR2GRAY)
    thres,targetBinary = cv2.threshold(targetGray,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(targetBinary, cv2.MORPH_OPEN, kernel)
    image, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for j in range(len(contours)):
        tmpImg = np.zeros(img.shape)
        tmpImg = cv2.drawContours(tmpImg, contours, j, (0, 255, 0), cv2.FILLED)
        cv2.imwrite('out/{0}.jpg'.format(j),tmpImg)



if __name__ == '__main__':
    path = r'C:\workspace\map_calc_area\data\B0FFF0EVLU_1.png'
    img = cv2.imread(path)
    img = cutBlank(img)
    targetImg = kmeans(img,K=5)  # K为聚类的种类数目，根据图片中的颜色数目来调整这个值
    print('=====finish')