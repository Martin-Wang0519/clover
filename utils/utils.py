import hashlib
import os
import random
import re
import shutil
import time
import logging
from logging.handlers import RotatingFileHandler

import numpy as np
import pytesseract
import win32api
import win32con
import win32gui
from PIL import Image
from shutil import copy, rmtree


def random_select_sample(src_path, dst_path, select_num):
    image_paths = os.listdir(src_path)
    # 随机采样验证集的索引
    random_path = random.sample(image_paths, k=select_num)

    for path in random_path:
        # 将分配至测试集中的文件复制到相应目录
        image_path = os.path.join(src_path, path)
        shutil.copy(image_path, dst_path)


def get_set_OCR_path():
    key = win32api.RegOpenKey(win32con.HKEY_LOCAL_MACHINE, 'SOFTWARE\Tesseract-OCR', 0, win32con.KEY_READ)
    info = win32api.RegQueryInfoKey(key)
    k = filter(lambda x: win32api.RegEnumValue(key, x)[0] == 'InstallDir', range(0, info[1]))
    k = list(k)[0]
    install_path = win32api.RegEnumValue(key, k)[1]
    os.environ['PATH'] = os.environ['PATH'] + ';{}'.format(install_path)


def ocr(image, crop_zone):
    image = image.convert("L")
    ocr_crop = image.crop(crop_zone)  # (left, upper, right, lower)
    text = pytesseract.image_to_string(ocr_crop, lang="eng")
    return text.rstrip()


def binary_array(x):
    ans = ""
    avg = np.sum(x) / np.size(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            temp = '1' if x[i][j] > avg else '0'
            ans += temp
    return ans


def sha256_str(image_path):
    with Image.open(image_path, 'r') as img:
        img = img.resize((32, 32))
        img = img.convert('L')
        x = np.array(img)
        binary_str = binary_array(x)

        binary_str = binary_str.encode('utf-8')
        sha = hashlib.sha256()
        sha.update(binary_str)
        return sha.hexdigest()[:25]


def clear_folder(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            clear_folder(c_path)
            os.rmdir(c_path)
        else:
            os.remove(c_path)


def copy_file(src_path, dst_path):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    ls = os.listdir(src_path)
    for i in ls:
        sub_path = os.path.join(src_path, i)
        if os.path.isdir(sub_path):
            copy_file(sub_path, dst_path)
        else:
            shutil.copy(sub_path, dst_path)


def RMZ_crop_save(image, save_path, zone):
    cropped = image.crop(zone)
    cropped.save(save_path)


def folder_RMZ_crop_save(folder_path, zone):
    for sub_path in os.listdir(folder_path):
        path = os.path.join(folder_path, sub_path)
        if os.path.isdir(path):
            folder_RMZ_crop_save(path, zone)
        else:
            try:
                with Image.open(path, 'r') as img:
                    RMZ_crop_save(img, path, zone)
            except Exception:
                pass


def init_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        clear_folder(path)


def get_window_handle(name_begin):
    hwnd_list = []
    win32gui.EnumWindows(lambda hWnd, param: param.append(hWnd), hwnd_list)
    for handle in hwnd_list:
        match = re.match(name_begin, win32gui.GetWindowText(handle))
        if match is not None:
            return handle
    return None


def show_window(handle, isMaxShow):
    if isMaxShow is False:
        show_mode = win32con.SW_SHOW
    else:
        show_mode = win32con.SW_MAXIMIZE
    # 最大化并显示在最前面
    win32gui.SendMessage(handle, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)
    win32gui.ShowWindow(handle, show_mode)
    win32gui.SetForegroundWindow(handle)
    time.sleep(1)



def screen_clicked(point):
    if isinstance(point, str):
        point = eval(point)
    win32api.SetCursorPos(point)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
    time.sleep(1)


def screen_double_clicked(point):
    if isinstance(point, str):
        point = eval(point)
    win32api.SetCursorPos(point)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
    time.sleep(1)


def keyboard_input(key, sleep_time):
    win32api.keybd_event(key, 0, 0, 0)
    win32api.keybd_event(key, 0, win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(sleep_time)


def find_indent_click(point):
    win32api.SetCursorPos(point)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
    time.sleep(1)


def split_dataset(dataset_root):
    # 保证随机可复现
    random.seed(0)

    # 将数据集中10%的数据划分到验证集中
    split_rate = 0.1

    # 指向你解压后的flower_photos文件夹
    origin_flower_path = os.path.join(dataset_root, "stock_images")
    assert os.path.exists(origin_flower_path), "path '{}' does not exist.".format(origin_flower_path)

    flower_class = [cla for cla in os.listdir(origin_flower_path)
                    if os.path.isdir(os.path.join(origin_flower_path, cla))]

    # 清空文件夹

    # 建立保存训练集的文件夹
    train_root = os.path.join(dataset_root, "train")
    if os.path.exists(train_root):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(train_root)
    os.makedirs(train_root)

    for cla in flower_class:
        # 建立每个类别对应的文件夹
        os.mkdir(os.path.join(train_root, cla))

    # 建立保存验证集的文件夹
    val_root = os.path.join(dataset_root, "val")
    if os.path.exists(val_root):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(val_root)
    os.makedirs(val_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        os.mkdir(os.path.join(val_root, cla))

    test_root = os.path.join(dataset_root, "test")
    if os.path.exists(test_root):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(test_root)
    os.makedirs(test_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        os.mkdir(os.path.join(test_root, cla))

    for cla in flower_class:
        cla_path = os.path.join(origin_flower_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        # 随机采样验证集的索引
        test_index = random.sample(images, k=int(num * split_rate))

        val_and_train = list(set(images).difference(test_index))
        val_index = random.sample(val_and_train, k=int(num * split_rate))

        for index, image in enumerate(images):
            if image in test_index:
                # 将分配至测试集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(test_root, cla)
                copy(image_path, new_path)
            elif image in val_index:
                # 将分配至验证集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                # 将分配至训练集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
        print()

    print("processing done!")


def logger():
    logger = logging.getLogger()

    logger.setLevel("INFO")  # 选择自己的日志等级

    # 配置打印的日志格式和内容
    fmt = '%(asctime)s - %(process)d - %(threadName)s - %(levelname)s - %(filename)s - %(lineno)d  %(message)s'
    formatter = logging.Formatter(fmt)
    file_handler = RotatingFileHandler('log.txt', backupCount=3, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def generate_path_name(root, dir_name_tree):
    global _root, _dir_name_tree
    ans = []
    for dir_name in dir_name_tree:
        if isinstance(dir_name, dict):
            for k, v in dir_name.items():
                _root, _dir_name_tree = k, v

            for child in generate_path_name(_root, _dir_name_tree):
                ans.append(os.path.join(root, child))
        else:
            ans.append(os.path.join(root, dir_name))
    return ans


def code_sum(code):
    _sum = 0
    for ch in code:
        _sum = _sum + int(ch)
    return _sum


if __name__ == '__main__':
    print(code_sum('002590'))
