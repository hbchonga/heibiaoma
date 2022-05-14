# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 08:23:50 2021
源代码是画出日历上的线
@author: Administrator
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("ceshi001.jpg",0)
edges = cv.Canny(img, 0, 127)
plt.imshow(edges, cmap= plt.cm.gray)
#

lines = cv.HoughLines(edges, 0.8, np.pi/180, 150)

for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = rho*a
    y0 = rho*b
    x1 = int(x0+1000*(-b))
    y1 = int(y0+1000*a)
    x2 = int(x0-1000*(-b))
    y2 = int(y0-1000*a)
    cv.line(img,(x1,y1),(x2,y2),(0,255,0))#绘制线条指定为绿色

plt.imshow(img[:,:,::-1])

'''
#霍夫变换圆检测
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
#1.读图像
planets = cv.imread("ceshi001.jpg")
gay_img = cv.cvtColor(planets, cv.COLOR_BGRA2GRAY)
#2.进行中值模糊，去噪点
img = cv.medianBlur(gay_img, 7)
#3.霍夫圆检测  HOUGH_GRADIENT(算法) 1(分辨率) 200(距离) param(阈值) Radius(半径)
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 200, param1=100, param2=30, minRadius=0, maxRadius=100)
#4.将检测结果绘制在图像上
for i in circles[0, :]:
    #绘制图形
    cv.circle(planets,(i[0], i[1]), i[2], (0, 255, 0),2)
    #绘制圆心
    cv.circle(planets, (i[0], i[1]), 2, (0, 0, 255), 3)
#5.图像显示
plt.figure(figsize=(10,8), dpi=100)
plt.imshow(planets[:,:,::-1]),plt.title('霍夫变换圆检测')
plt.xticks([]), plt.yticks([])
plt.show()
'''

