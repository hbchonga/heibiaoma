# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:32:17 2021

@author: Administrator
"""
import cv2
import numpy as np

def findone():
    img = cv2.imread('data/Camera1_BlackBlock_GlassDown/20210707094407.bmp',0)
    
    #1.filename：需要打开图片的路径，可以是绝对路径或者相对路径，路径中不能出现中文。
    #2.flag：图像的通道和色彩信息（默认值为1）。
    #flag = -1,   8位深度，原通道
    #flag = 0，   8位深度，1通道
    #flag = 1，   8位深度，3通道
    #flag = 2，   原深度， 1通道
    #flag = 3，   原深度， 3通道
    #flag = 4，   8位深度，3通道 

    template = cv2.imread('muban3.jpg',0)
    h,w = template.shape[:2]
    
    #相关系数匹配方法：cv2.TM_CCOEFF
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    left_top = max_loc #左上角
    right_bottom = (left_top[0] + w, left_top[1] + h) #右下角
    cv2.rectangle(img, left_top, right_bottom, (0,0,255),2) #画出矩形位置
    cv2.imwrite("find.png", img)
    
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
    findone()
    #findalll()