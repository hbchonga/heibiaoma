# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 15:30:55 2021

@author: Administrator
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def findone():
    img = cv2.imread('data/Camera1_BlackBlock_GlassDown/20210707094407.bmp',1)
    #equalhistimage = cv2.equalizeHist(img)
    tozero_img = cv2.threshold(img,50,255,cv2.THRESH_TOZERO)[1]
    
    #高斯去噪
    #gao_img = cv2.GaussianBlur(tozero_img, (3,3), 0)
    
    #cv2.imshow('tozero_img', tozero_img)
    #cv2.waitKey(0)
    #cv2.destoryAllWindows()
    
    #灰度处理
    gray1 = cv2.cvtColor(tozero_img, cv2.COLOR_BGR2GRAY)
    #二值化
    binary1 = cv2.threshold( gray1, 50, 255,cv2.THRESH_BINARY)[1]
    #ret,binary1 = cv2.threshold(gray1, 60, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #binary1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)[1]
    
    #1.filename：需要打开图片的路径，可以是绝对路径或者相对路径，路径中不能出现中文。
    #2.flag：图像的通道和色彩信息（默认值为1）。
    #flag = -1,   8位深度，原通道
    #flag = 0，   8位深度，1通道
    #flag = 1，   8位深度，3通道
    #flag = 2，   原深度， 1通道
    #flag = 3，   原深度， 3通道
    #flag = 4，   8位深度，3通道 

    #equalhistimage = cv2.equalizeHist(binary1)

    template = cv2.imread('muban3.jpg',0)
    h,w = template.shape[:2]
    print('h:',h,'w:',w)
    template2 = cv2.imread('muban.jpg',0)
    h2,w2 = template2.shape[:2]
    print('h2:',h2,'w2:',w2)
    
    binary2 = cv2.threshold(template,127,255,cv2.THRESH_BINARY)[1]
    
    
    #相关系数匹配方法：cv2.TM_CCOEFF
    #对于方法SQDIFF和SQDIFF_NORMED，越小的数值代表更高的匹配结果
    #而对于其他方法，数值越大匹配越好
    res = cv2.matchTemplate(binary1, binary2, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    left_top = max_loc #左上角
    
    #left_top = min_loc 
    
    right_bottom = (left_top[0] + w2, left_top[1] + h2) #右下角
    cv2.rectangle(binary1, left_top, right_bottom, (0,0,255),2) #画出矩形位置
    
    #将定位到的矩阵图截出来
    #dwq=binary1[left_top[0] : left_top[0]+w2, left_top[1] : left_top[1]+h2]
    dwq=binary1[left_top[1] : left_top[1]+h2, left_top[0] : left_top[0]+w2]
    #cv2.imwrite("20210707094407_059.png", binary1)
    cv2.imwrite('dwq.png', dwq)
    
    '''对定位区分别进行垂直投影img1和水平投影img2，
    对img1的列像素进行遍历，
    '''
    #def chzhity1():
        
    #垂直投影  dwqh:定位区高  dwqw:定位区宽
    (dwqh,dwqw) = dwq.shape#返回高和宽
    vproject = dwq.copy()
    a = [0 for x in range(0,dwqw)]
    #记录每一列的波峰
    for j in range(0,dwqw):#遍历一列
        for i in range(0,dwqh):#遍历一行
            if vproject[i,j]==0:#如果该点为黑点
                a[j]+=1#该列的计数器加1计数
                vproject[i,j]=255#记录完后将其变为白色
    for j in range(0,dwqw):#遍历每一列
        for i in range((dwqh-a[j]),dwqh):#从该列应该变黑的最顶部的点开始向最底部涂黑
            vproject[i,j]=0 #涂黑
    cv2.putText(vproject,"verticality",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(100,100,100),4)
    cv2.imwrite('chzhity.png', vproject)
    
    #遍历最下面一行的像素值，定位横坐标
    #创建数组,46个点定位23个横坐标
    #w_w = [0]*w
    w_w=[]
    w_wd=[]#横坐标中点数组
    for i in range(5,dwqw-5):
        if vproject[dwqh-7,i] == 255 and vproject[dwqh-7,i+1] == 0:
            w_w += [i]
        elif vproject[dwqh-7,i] == 0 and vproject[dwqh-7,i+1] == 255:
            w_w += [i]
    for i in range(0,len(w_w),2):
        w_wd +=[round((w_w[i]+w_w[i+1])/2)]
    
    print('len(w_wd)有',len(w_wd),"个点")
    '''
    #测试代码
    print(len(w_wd),"个点")
    ceshi=[dwqh-5 for i in range(len(w_wd))]
    plt.plot(w_wd,ceshi,'b*')#w_wd:x轴  ceshi:y轴
    plt.show()
    
    for i in range(0, len(w_wd)):
        print(w_wd[i])
    '''
    
    
    #竖直投影
    hproject = dwq.copy()
    b = [0 for x in range(0,dwqh)]
    for j in range(0,dwqh):
        for i in range(0,dwqw):
            if hproject[j,i] == 0:
                b[j] += 1
                hproject[j,i] = 255
    for j in range(0,dwqh):
        for i in range(0,b[j]):
            hproject[j,i]=0
    cv2.putText(hproject,"horizontal",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(100,100,100),4)
    cv2.imwrite('shzhty.png', hproject)
    
    #遍历最左面一列的像素值，定位纵坐标
    h_h=[]
    h_hd=[]#纵坐标中点数组
    print('dwqh:',dwqh)
    for i in range(3,dwqh-3):#遍历第3列的像素点
        if hproject[i,3] == 255 and hproject[i+1,3] == 0:
            h_h += [i]
        elif hproject[i,3] == 0 and hproject[i+1,3] == 255:
            h_h += [i]
    
    '''
    #测试代码
    print('len(h_h)有',len(h_h),'个点')
    ceshi=[3 for i in range(len(h_h))]
    plt.plot(ceshi,h_h,'b*')
    plt.show()
    for i in range(0,len(h_h)):
        print('h_h ',i,':',h_h[i])
    '''
    
    for i in range(0,len(h_h),2):
        h_hd +=[round((h_h[i]+h_h[i+1])/2)]
    
    print('len(h_hd)有',len(h_hd),'个点')
    
    '''
    #测试代码
    for i in range(0,len(h_hd)):
        print('h_hd ',i,':',h_hd[i])
    ceshi=[3 for i in range(len(h_hd))]
    plt.plot(ceshi,h_hd,'b*')
    plt.show()
    '''
    
    #用c存储2号区域的值
    c=[[0]*22 for i in range(4)]#全置0，代表空白矩形块
    #aList = [[0] * cols for i in range(rows)]
    for j in range(1,len(w_wd)):
        for i in range(1,len(h_hd)):
            if dwq[h_hd[i],w_wd[j]]==0:#遇到像素点为黑则置为1
                c[i-1][j-1]=1
    
    print(c)
    
    #将二进制转化为十六进制
    shl=[]
    for j in range(0,len(c[0])):
        a=''
        for i in range(0,len(c)):
             a=a+str(c[i][j])
             
        print(a)
        #将二进制转化为十进制再转化为十六进制
        j=hex(int(a,2))
        shl.append(j[2:])#从第二个开始截掉，然后放入shl
    print(shl)
        
def er_to_shi(num):
    sum=0
    length=len(num)
    for x in range(length):
        sum += int(num[length-1-x])*pow(2, x)
    print(sum)

def chzhty() :
    image1 = cv2.imread('dwq_001.png')
    #灰度图像
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
    cv2.imwrite('chzhity.png', vproject)
    
def findall():
    #1.读入原图和模板
    img_rgb = cv2.imread('mario.png')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BAYER_BG2BGR)
    template = cv2.imread('mario_coin.png',0)
    h,w = template.shape[:2]
    
    #2.标准相关模板匹配
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    
    #3.这边是python/numpy的只是
    loc = np.where(res >= threshold) #匹配程度大于%80的坐标y,x
    for pt in zip(*loc[::-1]): # *号表示可选参数
        right_bottom = (pt[0] + w, pt[1] + h)
        cv2.rectangle(img_rgb, pt, right_bottom, (0, 0, 255), 2)
    cv2.imwrite("coinloc.png", img_rgb)
    
if __name__ == "__main__":
    time_start = time.time()
    findone()
    #chzhity()
    #findalll()
    time_end = time.time()
    print('总时间:',time_end-time_start,"秒")