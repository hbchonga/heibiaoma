# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:10:52 2021

@author: Administrator
"""

import cv2
import matplotlib.pyplot as plt

#读取图片
image1 = cv2.imread('ceshi001.jpg')
# 灰度图像
gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# 二值化
ret, binary = cv2.threshold(gray,130, 255, cv2.THRESH_BINARY)
(h,w) = binary.shape#返回高和宽

#垂直投影
vproject = binary.copy()
a = [0 for x in range(0,w)]
#记录每一列的波峰
for j in range(0,w):#遍历一列
    for i in range(0,h):#遍历一行
        if vproject[i,j]==0:#如果改点为黑点
            a[j]+=1#该列的计数器加1计数
            vproject[i,j]=255#记录完后将其变为白色
for j in range(0,w):#遍历每一列
    for i in range((h-a[j]),h):#从该列应该变黑的最顶部的点开始向最底部涂黑
        vproject[i,j]=0 #涂黑
cv2.putText(vproject,"verticality",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(100,100,100),4)

#水平投影
hproject = binary.copy()
b = [0 for x in range(0,h)]
for j in range(0,h):
    for i in range(0,w):
        if hproject[j,i] == 0:
            b[j] += 1
            hproject[j,i] = 255
for j in range(0,h):
    for i in range(0,b[j]):
        hproject[j,i]=0
cv2.putText(hproject,"horizontal",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(100,100,100),4)

'''
cv2.imshow("image1",image1)
cv2.imshow("vproject",vproject)
cv2.imshow("hproject",hproject)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#plt.imshow(image1[:,:,::1])
#plt.imshow(hproject)
#plt.imshow(vproject)
#cv2.imwrite('chzhity.png', vproject)
cv2.imwrite('shzhty.png', hproject)