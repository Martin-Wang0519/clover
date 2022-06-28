# import os
#
# imagesFolderPath = os.path.join('./dataset/stock_images', 'test')
# imageFiles = os.listdir(imagesFolderPath)
#
# for imageName in imageFiles:
#     print(imageName[0])
#     # imagePath = os.path.join(imagesFolderPath, imageName)
#     # image = Image.open(imagePath)
#     # grayImage = image.convert("L")
#     #
#     # imageTensor = self.trans(grayImage)
#     # self.imagesData.append(imageTensor)
#     #
#     # self.labelsData.append(imageName[0])

# from kmeans import k_means;
# import os
# from PIL import Image
# import numpy as np
#
#
#
# ImagesFolderPath = "positive_samples";
# for imageName in os.listdir(ImagesFolderPath):
#     imagePath = os.path.join(ImagesFolderPath, imageName)
#
#     image = Image.open(imagePath)
#     grayImage = image.convert("1")
#
#     grayImage.save(imagePath)

# import psutil
# print(list(psutil.process_iter()))
# -*-coding:utf-8-*-
import csv
import os
from datetime import date, timedelta

import win32con
import win32gui

from utils.utils import clear_folder, get_window_handle, show_window, screen_clicked, keyboard_input, ocr
from config import settings
from PIL import Image
from PIL import ImageGrab
import pandas as pd
from openpyxl import load_workbook
from aggregation import Aggregation

if __name__ == '__main__':
    # import win32api
    # from time import sleep
    #
    # win32api.ShellExecute(0, 'open', r'C:\海王星金融终端-中国银河证券\TdxW.exe', '', '', 0)
    # sleep(5)
    # screen_clicked((1267, 624))
    #
    # # excel_path = settings.get('statistics_info').get('prediction_excel_info').get('path')
    # # agg = Aggregation('daily_screenshot',
    # #                   settings.get('statistics_info').get('folder_path'),
    # #                   settings.get('statistics_info').get('prediction_excel_info').get('header'),
    # #                   settings.get('statistics_info').get('prediction_excel_info').get('time_curve_type_encode'))
    # # agg.to_excel('gang', 'tongji.xlsx')
    a = [-1.12, 0, -1.3, 0.79, -1.51, -0.39, 1.11, -2.14, 1.35]
    b = [x for x in a if x >= 0]
    print(sum(a) / len(a))
    print(len(b) / len(a))

    # handle = get_window_handle('海王星')
    # print(handle)
    #
    # win32gui.PostMessage(handle, win32con.WM_CLOSE, 0, 0)
