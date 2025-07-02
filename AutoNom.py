#-*- coding : utf-8-*-

import cv2
import numpy as np

import os
import sys
from typing import List, Tuple
import pytesseract, re
# https://github.com/UB-Mannheim/tesseract/wiki   for ocr install
pytesseract.pytesseract.tesseract_cmd = (r'C:\Program Files\Tesseract-OCR\tesseract.exe')
'''
ROI就不再进行放缩了
直线均为水平线段，
'''
#lsd = cv2.line_descriptor.LSDDetector.createLSDDetector(scale=1.0, ang_th=3.0)
lsd = cv2.ximgproc.createFastLineDetector(
    length_threshold=70,  # 最小线段长度
    distance_threshold=1.41421356,  # 点到直线的最大距离
    canny_th1=30,  # Canny 边缘检测的第一个阈值
    canny_th2=100,  # Canny 边缘检测的第二个阈值
    canny_aperture_size=3,  # Canny 边缘检测的 Sobel 算子孔径大小
    do_merge=True  # 是否合并共线的线段
)

# parameters
CONFIG = {
    "SHOW_IMAGE": False,
    "RESIZE_RATE": 1.0,
    "BAR_TYPE": "bd",
    "EXPAND_RADIUS": 2,
    "MORPH_KERNEL_SIZE": (3,3),
    "dpi": 200,
    "tail": 'fx',
}

def ocr(gray):
    _, binary = cv2.threshold(gray, 50, 127, cv2.THRESH_BINARY_INV)
    denoised = cv2.medianBlur(binary, 3)
    cv2.imshow("roi", denoised)
    cv2.waitKey(0)
    num_text = pytesseract.image_to_string(denoised,
                                       config='--psm 6 --oem 3 --dpi %d -c tessedit_char_whitelist=0123456789.'%CONFIG["dpi"])
    numbers = ''.join(re.findall(r'[0-9.]+', num_text))
    unit_text = pytesseract.image_to_string(denoised,
                                       config='--psm 6 --oem 3 -c tessedit_char_whitelist=uμknmcdv')
    units = ['μm', 'um', 'nm']  # , 'cm', 'dm', 'm', 'km']  # not include

    potential_units = re.findall(r'[uμknmcdv]{1,2}', unit_text)
    potential_units.sort(key=len, reverse=True)
    unit = ''
    for potential_unit in potential_units:
        if potential_unit in units:
            unit = potential_unit
            break
    return num_text, unit

def show(lines, img):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0].astype(int)
            start_point = (x1, y1)
            end_point = (x2, y2)
            cv2.line(img, start_point, end_point, (255, 255, 255), 2)

    cv2.imshow('LSD Result', img)
    cv2.waitKey(0)

def get_background_color(img):
    """
    计算图像的背景色，通过统计灰度值的众数来确定
    :param img: 输入的灰度图像
    :return: 背景色的灰度值
    """
    #hist = np.bincount(img.flatten())
    #print(hist)
    #background_color = np.argmax(hist)
    background_color = np.median(img)
    return background_color

def fill_lines_with_background(img, lines):
    """
    将检测到的直线填充为背景色
    :param img: 输入的灰度图像
    :param lines: 检测到的直线列表
    :return: 填充后的图像
    """
    background_color = int(np.clip(get_background_color(img), 0, 255))
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0].astype(int)
            start_point = (x1, y1)
            end_point = (x2, y2)
            # 使用背景色填充直线
            cv2.line(img, start_point, end_point, background_color, 2)
    return img

def find_longest_valid_number(s):
    numbers = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', s)
    if numbers:
        longest_number_str = max(numbers, key=len)
        try:
            return float(longest_number_str)
        except ValueError:
            pass
    return 0

def main():
    imgList = []
    mskList = []
    name = ''
    for root, dirs, files in os.walk(sys.path[0]):
        for f in files:
            if f[-4:] == ".tif" or f[-4:] == ".TIF":  # source image
                imgList.append(f)
            elif len(f)>8 and (f[-8:] == "_Msk.png" or f[-8:] == "_Msk..TIF"):  # mask image
                mskList.append(f)

    print(imgList)
    count = 0
    for imgPath in imgList:
        im = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

        '''
        1. 两种分辨率： 
            1.1 683×1024(h,w)： 对应标尺在 左下角  30: 300, 580:640, 对应：-bd
            1.2 1086×1376(h,w)： 对应标尺在 右下角 760: 1370 , 1060:1086, 对应: -fx
            
            对应数值需要获取
            对应线段的像素值 需要？ 文件名没体现
            直线似乎对检测OCR结果影响很大，先检测直线            
        '''
        if im.shape == (683, 1024):
            CONFIG["dpi"] = 72
            CONFIG["tail"] = 'bd'  # bd 是基于单位？还是基于传感器？
            roi = im[580:650, 30:300]
            line = lsd.detect(roi)
            #name = ocr(roi)
            #print(name)
            #show(line, roi)
            filled_roi = fill_lines_with_background(roi.copy(), line)
            num, unit = ocr(filled_roi)

        elif im.shape == (1086, 1376):
            CONFIG["dpi"] = 200
            CONFIG["tail"] = 'fx'
            roi = im[1060:1086, 760:1370]
            line = lsd.detect(roi)
            #name = ocr(roi)
            #print(name)
            #show(line, roi)
            filled_roi = fill_lines_with_background(roi.copy(), line)
            num, unit = ocr(filled_roi)
        else:
            print("None")
        postfix = CONFIG["tail"]

        # 这里的输出名 有两个问题，就是 单位似乎没统一，我不知道定义，所以直接输出了
        # 第二个就是文件名序号，没有给定义，依次进行的编号，但实际上最好用固定长度的count.zfill(n)
        name = f"TEM_image{count}-{find_longest_valid_number(num)}-{unit}-{postfix}.png"  #
        count += 1
        # print(name)
        cv2.imwrite(name, im)
if __name__ == "__main__":
    main()